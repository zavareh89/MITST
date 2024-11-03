import pandas as pd
import psycopg

# database configuration
db_params = {"dbname": "eicu", "user": "eicu", "password": "eicu"}


def fetch_from_db(query):
    with psycopg.connect(**db_params) as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            df = pd.DataFrame(
                cur.fetchall(), columns=[desc[0] for desc in cur.description]
            )

    return df
