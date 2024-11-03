# MITST: Multi-source Irregular Time-Series Transformer for ICU Glucose Level Prediction

MITST is a novel, attention-based neural network framework for predicting blood glucose (BG) levels in ICU patients. It leverages heterogeneous clinical time-series data, such as lab results, medications, and vital signs, to capture the complex dynamics of ICU patients. MITST's modular, hierarchical Transformer architecture processes multi-source and irregular time-series data, offering a scalable and adaptable approach for accurate BG level predictions and other ICU prediction tasks. Below is an illustration of the high-level architecture.

![MITST Architecture](./Appendix.pdf) <!-- Replace with actual path -->

---

## Getting Started

To set up and run the code for the MITST model, follow the steps below.

### Prerequisites

1. **Database Setup**  
   First, ensure that the eICU database is properly set up as described by the eICU data maintainers. Follow their guidelines at [eICU Data Setup](https://github.com/MIT-LCP/eicu-code/tree/main/build-db/postgres).

2. **Environment Setup**  
   Create a virtual environment using Python 3.10. Then, install the required packages by running:

   ```bash
   pip install -r requirements.txt

3. **Update Patient Table**  
    Update the `patient` table to include previous and next visits (this will be used for concatenation of consecutive stays of a single patient in a specific hospital). Use the command below:

    ```bash
   psql -U your_username -d your_db_name -f concepts/add_previous_next_visits.sql


## Running the Code
Once the environment is set up, activate it and navigate to the `code/` directory. Update the database parameters if necessary by adjusting `db_params` values including `dbname`, `user`, and `password` in the `db_conn.py` file.

1. **Preprocess Database** 
    The information from each stay is extracted and at the end, they are outputed as a JSONL file, where each line corresponds to a single stay:
    ```bash
   python3 preprocess_database.py

2. **Concatenate Stays and Process Labels** 
    Consecutive stays for each patient are concatenated. Also, only the stays with at least six BG measurements are included. This step creates another JSONL file with the processed data:
    ```bash
   python3 concatenate_admissions_and_preprocess_labels.py

3. **Feature Generation** 
    The data is transformed into a format compatible with the MITST model:
    ```bash
   python3 feature_generation.py

4. **Train the MITST Model** 
    Then, the MNIST model is trained using generated features:
    ```bash
   python3 train.py

5. **Model Evaluation** 
    Finally, the model is evaluated on test set and AUROC and AUPRC are computed for each class:
    ```bash
   python3 evaluation.py

## Citation
If you find MITST helpful in your research, please consider citing our preprint:

```plaintext
@article{mehdizavareh2024mitst,
  title={Enhancing Glucose Level Prediction of ICU Patients through Irregular Time-Series Analysis and Integrated Representation},
  author={Mehdizavareh, Hadi and Khan, Arijit and Cichosz, Simon Lebech},
  journal={arXiv preprint arXiv:2400.00000},
  year={2024}
}
```

## Acknowledgements
1. Special thanks to the eICU team for providing access to this large-scale EHR dataset. Their work enables impactful research in healthcare analytics. Learn more at [eICU GitHub Repository](https://github.com/MIT-LCP/eicu-code).
2. We extend our appreciation to the teams behind the FT-Transformer and Tab-Transformer for their open-source implementations, which inspired aspects of MITST. See their project at [FT-Transformer GitHub Repository](https://github.com/lucidrains/tab-transformer-pytorch/tree/main).

