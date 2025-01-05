# Material Recognition Model with ChemBERTa

This project uses the ChemBERTa model for molecular property predictions based on SMILES strings. The model is fine-tuned using a dataset of SMILES strings and associated molecular properties.

## Prerequisites

- Python 3.8 or later
- `pip` package manager


## Installation 

1. **Clone the Repository**: First, clone the repository to your local machine.

    ```bash
    git clone https://github.com/Yassa122/Material-Recognition-Using-XAI-Bachelor-Thesis.git
    cd Material-Recognition-Using-XAI-Bachelor-Thesis/Model/src/Transformer_model
    ```

2. **Set up a Python virtual environment**: Create and activate a virtual environment to install dependencies.

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install Dependencies**: Install the necessary Python packages. These are listed in `requirements.txt`.

    ```bash
    pip install -r requirements.txt
    ```

   If the `requirements.txt` file is not available, manually install the necessary packages:

    ```bash
    pip install pandas torch transformers scikit-learn tqdm
    ```

## Files and Folder Structure

- `Model/src/Transformer_model/`: Contains scripts and datasets for fine-tuning and prediction.
- `fine_tuned_chemberta/`: Folder where the fine-tuned model will be saved.
- `SMILES_Big_Data_Set.csv`: Dataset for fine-tuning the model.
- `test.csv`: Dataset for making predictions.

## Running the Model

### 1. Prepare the Datasets

Ensure you have the following CSV files in the specified directory:

- `SMILES_Big_Data_Set.csv`: This should contain SMILES strings and molecular properties for training.
- `test.csv`: This should contain SMILES strings without properties for prediction.

### 2. Fine-Tune the Model

Run the following script to fine-tune the ChemBERTa model using the SMILES strings in `SMILES_Big_Data_Set.csv`.

```bash
python Model/src/Transformer_model/Training_fine_tuning.py
