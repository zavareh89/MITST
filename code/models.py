import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from torch.utils.checkpoint import checkpoint

from einops import rearrange, repeat


# feedforward and attention
class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


def FeedForward(dim, mult=4, output_dim=None):
    if output_dim is None:
        output_dim = dim
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, dim * mult * 2),
        GEGLU(),
        nn.Linear(dim * mult, output_dim),
    )


class Fusion(nn.Module):
    def __init__(self, input_dim, hid_dim):
        super(Fusion, self).__init__()
        self.hid_dim = hid_dim
        self.linear = nn.Linear(input_dim, hid_dim)
        self.context_vector = nn.Linear(hid_dim, 1, bias=False)

    def forward(self, x):
        attn_raw = torch.tanh(self.linear(x))
        # Multiply by the context vector (u)
        attn_weights = self.context_vector(attn_raw).squeeze(-1)
        # Apply softmax over the sequence dimension
        attn_weights = F.softmax(attn_weights, dim=-1).unsqueeze(-1)
        return attn_weights


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5

        self.norm = nn.LayerNorm(dim)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, src_key_padding_mask=None, return_attn=False):
        h = self.heads
        x = self.norm(x)

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))
        q = q * self.scale

        sim = einsum("b h i d, b h j d -> b h i j", q, k)

        # apply masking
        if src_key_padding_mask is not None:
            # Expand mask to cover heads dimension: [Batch Size, 1, 1, Key Len]
            src_key_padding_mask = src_key_padding_mask.unsqueeze(1).unsqueeze(2)
            # Set masked positions to a large negative value, ensuring they get close to zero probability after the softmax
            sim = sim.masked_fill(src_key_padding_mask, float("-inf"))

        attn = sim.softmax(dim=-1)

        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)", h=h)
        out = self.to_out(out)
        if return_attn:
            return out, attn
        return out


# transformer


class Transformer(nn.Module):

    def __init__(self, dim, depth, heads, dim_head):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.dim_head = dim_head

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(dim, heads=heads, dim_head=dim_head),
                        FeedForward(dim),
                    ]
                )
            )

    @staticmethod
    def generate_sin_cos_pos_emb(positions, dim):
        # position = torch.arange(0, max_seq_len).unsqueeze(1)
        batch_size, max_seq_len = positions.shape[0], positions.shape[1]
        positions = positions.unsqueeze(-1)
        div_term = torch.exp(torch.arange(0, dim, 2) * -(math.log(10000.0) / dim))
        div_term = div_term.unsqueeze(0).unsqueeze(0).to(positions.device)
        pos_emb = torch.zeros(batch_size, max_seq_len, dim, device=positions.device)
        pos_emb[:, :, 0::2] = torch.sin(positions * div_term)
        pos_emb[:, :, 1::2] = torch.cos(positions * div_term)
        return pos_emb

    def forward(self, x, src_key_padding_mask=None, return_attn=False, positions=None):
        # positions shape is (batch_size, max_seq_len) and for masked tokens, the value is 0
        post_softmax_attns = []

        # Add positional embeddings to input if enabled
        if positions is not None:
            pos_emb = self.generate_sin_cos_pos_emb(positions, self.dim)
            x = x + pos_emb

        for attn, ff in self.layers:
            if return_attn:
                attn_out, post_softmax_attn = attn(
                    x, src_key_padding_mask=src_key_padding_mask, return_attn=True
                )
                post_softmax_attns.append(post_softmax_attn)
            else:
                attn_out = attn(x, src_key_padding_mask=src_key_padding_mask)

            x = attn_out + x
            x = ff(x) + x

        if not return_attn:
            return x

        return x, torch.stack(post_softmax_attns)


# numerical embedder


class NumericalEmbedder(nn.Module):
    def __init__(self, dim, num_numerical_types):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(num_numerical_types, dim))
        self.biases = nn.Parameter(torch.randn(num_numerical_types, dim))

    def forward(self, x):
        x = rearrange(x, "b n -> b n 1")
        return x * self.weights + self.biases


# main class


class FTTransformer(nn.Module):
    def __init__(
        self,
        *,
        categories,
        num_continuous,
        dim,
        depth,
        heads,
        dim_head=16,
        num_special_tokens=2,
    ):
        super().__init__()

        # categories related calculations

        self.num_categories = len(categories)
        self.num_unique_categories = sum(categories)

        # create category embeddings table

        self.num_special_tokens = num_special_tokens
        total_tokens = self.num_unique_categories + num_special_tokens

        # for automatically offsetting unique category ids to the correct position in the categories embedding table

        if self.num_unique_categories > 0:
            categories_offset = F.pad(
                torch.tensor(list(categories)), (1, 0), value=num_special_tokens
            )
            categories_offset = categories_offset.cumsum(dim=-1)[:-1]
            self.register_buffer("categories_offset", categories_offset)

            # categorical embedding

            self.categorical_embeds = nn.Embedding(total_tokens, dim)

        # continuous

        self.num_continuous = num_continuous

        if self.num_continuous > 0:
            self.numerical_embedder = NumericalEmbedder(dim, self.num_continuous)

        # cls token

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        # transformer

        self.transformer = Transformer(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
        )

    def forward(self, x_categ, x_numer, return_attn=False):

        xs = []
        if self.num_unique_categories > 0:
            assert (
                x_categ.shape[-1] == self.num_categories
            ), f"you must pass in {self.num_categories} values for your categories input"
            x_categ = x_categ + self.categories_offset

            x_categ = self.categorical_embeds(x_categ)

            xs.append(x_categ)

        # add numerically embedded tokens
        if self.num_continuous > 0:
            x_numer = self.numerical_embedder(x_numer)

            xs.append(x_numer)

        # concat categorical and numerical

        x = torch.cat(xs, dim=1)

        # append cls tokens
        b = x.shape[0]
        cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)

        # attend and return CLS token
        if return_attn:
            x, attns = self.transformer(x, return_attn=return_attn)
            return x[:, 0], attns
        else:
            x = self.transformer(x)
            return x[:, 0]


class EHRTransformer(nn.Module):
    def __init__(
        self,
        *,
        shared_dim=32,
        tf_n_heads=(8, 8, 8),
        tf_depths=(4, 4, 4),
        tf_dim_head=(8, 8, 8),
        n_classes=3,
        source_indicator=None,
    ):
        super().__init__()

        self.sources = [
            "static",
            "unit_info",
            "addx",
            "diagnosis",
            "lab",
            "IO",
            "IO_num_reg",
            "past_history",
            "treatment",
            "med",
            "infusion",
            "GCS",
            "sedation",
            "HR",
            "RR",
            "SpO2",
            "Temp",
            "nibp_mean",
            "ibp_mean",
            "nibp_systolic",
            "ibp_systolic",
            "nibp_diastolic",
            "ibp_diastolic",
        ]
        self.n_sources = len(self.sources)
        self.source_indicator = source_indicator
        if source_indicator is None:
            self.source_indicator = torch.ones(len(self.sources), dtype=bool)

        # Define config for each source as a tuple of (categories, num_continuous, s_dim)
        # which categories is a tuple containing the number of unique values within each category
        # and s_dim is the source-specific dimension.
        self.source_configs = [
            ((3, 6, 208, 5, 5, 15, 2), 3, 32),  # static
            ((8, 4, 15), 0, 32),  # unit_info
            ((16,), 0, 16),  # addx
            ((170, 3), 0, 32),  # diagnosis
            ((49,), 1, 32),  # lab
            ((13,), 1, 16),  # IO
            ((), 4, 16),  # IO_num_reg
            ((16,), 0, 16),  # past_history
            ((14,), 0, 16),  # treatment
            ((21, 10, 6, 2, 2), 1, 32),  # med
            ((3,), 1, 16),  # infusion
            ((14,), 0, 16),  # GCS
            ((10,), 0, 16),  # sedation
            ((), 1, 16),  # HR
            ((), 1, 16),  # RR
            ((), 1, 16),  # SpO2
            ((10,), 1, 16),  # Temp
            ((), 1, 16),  # nibp_mean
            ((), 1, 16),  # ibp_mean
            ((), 1, 16),  # nibp_systolic
            ((), 1, 16),  # ibp_systolic
            ((), 1, 16),  # nibp_diastolic
            ((), 1, 16),  # ibp_diastolic
        ]

        # FTtransformers (Aggregation at the feature level for each source)
        self.FTtransformers = nn.ModuleList(
            [
                FTTransformer(
                    categories=categories,
                    num_continuous=num_continuous,
                    dim=s_dim,
                    depth=tf_depths[0],
                    heads=tf_n_heads[0],
                    dim_head=tf_dim_head[0],
                    num_special_tokens=1,
                )
                for categories, num_continuous, s_dim in self.source_configs
            ]
        )

        # transformer for aggregation at the offset level for each source (except static)
        ## define cls token for each source
        self.cls_tokens = nn.ParameterList(
            [
                nn.Parameter(torch.randn(1, 1, s_dim))
                for _, _, s_dim in self.source_configs[1:]
            ]
        )
        self.TStransformers = nn.ModuleList(
            [
                Transformer(
                    dim=s_dim,
                    depth=tf_depths[1],
                    heads=tf_n_heads[1],
                    dim_head=tf_dim_head[1],
                )
                for _, _, s_dim in self.source_configs[1:]
            ]
        )

        # FF network for each source to map to the same dimensionality (i.e. shared_dim)
        self.FFnetworks = nn.ModuleList(
            [
                FeedForward(s_dim, output_dim=shared_dim)
                for i, (_, _, s_dim) in enumerate(self.source_configs)
                if self.source_indicator[i]
            ]
        )

        # transformer for aggregation at the source level
        self.Stransformer = Transformer(
            dim=shared_dim,
            depth=tf_depths[2],
            heads=tf_n_heads[2],
            dim_head=tf_dim_head[2],
        )

        # fusion layer
        self.fusion = Fusion(shared_dim, shared_dim)

        # output layer
        self.to_logits = nn.Sequential(
            nn.LayerNorm(shared_dim), nn.ReLU(), nn.Linear(shared_dim, n_classes)
        )

    def forward(self, source_data, mask, seq_len_per_source, return_attn=False):
        sources, offsets = source_data

        # calculate the representations for each source ( 2 levels of aggregation)
        source_repr = []
        for i, k in enumerate(self.sources):
            if not self.source_indicator[i]:
                continue
            # aggregate at the feature level
            x_cat, x_cont = sources[k]["categorical"], sources[k]["continuous"]
            x = self.FTtransformers[i](x_cat, x_cont)

            if k == "static":
                source_repr.append(x)
                continue

            # form the sequence for each source
            s_offset = offsets[k]
            s_mask = mask[k]
            s_seq_len = seq_len_per_source[k]
            batch_size, max_seq_len = s_mask.shape
            z = torch.zeros(batch_size, max_seq_len, x.shape[-1], device=x.device)
            p = torch.zeros(batch_size, max_seq_len, device=x.device)
            count = 0
            for b in range(batch_size):
                l = s_seq_len[b].int()
                z[b, :l] = x[count : count + l]
                o = s_offset[count : count + l]
                o = o - o[0]
                p[b, :l] = o  # positions
                count += l

            # aggregate at the offset level
            ## append cls tokens and update the mask and position accordingly
            cls_tokens = repeat(self.cls_tokens[i - 1], "1 1 d -> b 1 d", b=batch_size)
            z = torch.cat((cls_tokens, z), dim=1)
            s_mask = F.pad(s_mask, (1, 0), value=False)
            p = F.pad(p, (1, 0), value=0)
            ## apply TStransformer
            z = self.TStransformers[i - 1](z, src_key_padding_mask=s_mask, positions=p)
            ## extract the cls token
            source_repr.append(z[:, 0])

        # apply FF network for each source
        projs = [ff(x) for ff, x in zip(self.FFnetworks, source_repr)]
        projs = torch.stack(projs, dim=1)

        # aggregate at the source level
        a = self.Stransformer(projs)

        # compute fusion weights
        weights = self.fusion(a)

        # combine source representations and get final irregular TS representation
        ts = torch.sum(weights * projs, dim=1)

        # output layer
        logits = self.to_logits(ts)

        return logits
