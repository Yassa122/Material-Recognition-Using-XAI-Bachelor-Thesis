import os
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm
import numpy as np
import random
import warnings
from rdkit import Chem
from rdkit.Chem import Descriptors
import joblib

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")


# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Compute molecular descriptors
def compute_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None  # Handle invalid SMILES strings
    descriptors = {
        "MolWt": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "NumHDonors": Descriptors.NumHDonors(mol),
        "NumHAcceptors": Descriptors.NumHAcceptors(mol),
        "TPSA": Descriptors.TPSA(mol),
        "NumRotatableBonds": Descriptors.NumRotatableBonds(mol),
        # Add more descriptors as needed
    }
    return descriptors


# Define the Dataset class
class LogPDataset(Dataset):
    def __init__(self, df, tokenizer, descriptor_list, scaler, max_length=128):
        self.smiles = df["SMILES"].values
        self.logp = df["logP"].values
        self.descriptors = df[descriptor_list].values
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.scaler = scaler

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        smiles = self.smiles[idx]
        logp = self.logp[idx]
        descriptor = self.descriptors[idx]

        # Tokenize the SMILES string
        encoding = self.tokenizer(
            smiles,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
        )

        # Squeeze to remove the batch dimension
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "descriptors": torch.tensor(descriptor, dtype=torch.float),
            "logp": torch.tensor(logp, dtype=torch.float),
        }


# Define the Regression Model with Knowledge-Based Layer
class ChemBERTaForLogP(nn.Module):
    def __init__(self, base_model, descriptor_dim):
        super(ChemBERTaForLogP, self).__init__()
        self.base_model = base_model
        self.hidden_size = self.base_model.config.hidden_size
        self.descriptor_dim = descriptor_dim

        # Define layers to process ChemBERTa embeddings
        self.embedding_fc = nn.Linear(self.hidden_size, 256)
        self.embedding_relu = nn.ReLU()
        self.embedding_dropout = nn.Dropout(0.1)

        # Define layers to process descriptors
        self.descriptor_fc = nn.Linear(self.descriptor_dim, 128)
        self.descriptor_relu = nn.ReLU()
        self.descriptor_dropout = nn.Dropout(0.1)

        # Combine embeddings and descriptors
        self.combined_fc = nn.Linear(256 + 128, 128)
        self.combined_relu = nn.ReLU()
        self.combined_dropout = nn.Dropout(0.1)
        self.regressor = nn.Linear(128, 1)

    def forward(self, input_ids, attention_mask, descriptors):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        # Use the [CLS] token representation
        cls_output = outputs.last_hidden_state[
            :, 0, :
        ]  # Shape: (batch_size, hidden_size)

        # Process embeddings
        embedding = self.embedding_fc(cls_output)
        embedding = self.embedding_relu(embedding)
        embedding = self.embedding_dropout(embedding)

        # Process descriptors
        descriptor = self.descriptor_fc(descriptors)
        descriptor = self.descriptor_relu(descriptor)
        descriptor = self.descriptor_dropout(descriptor)

        # Concatenate embeddings and descriptors
        combined = torch.cat(
            (embedding, descriptor), dim=1
        )  # Shape: (batch_size, 256 + 128)

        # Combined layers
        combined = self.combined_fc(combined)
        combined = self.combined_relu(combined)
        combined = self.combined_dropout(combined)

        # Regression output
        logp = self.regressor(combined)
        return logp.squeeze()


# Define Training and Evaluation Functions
def train_epoch(model, loader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss = 0

    for batch in tqdm(loader, desc="Training"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        descriptors = batch["descriptors"].to(device)
        logp = batch["logp"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, descriptors)
        loss = criterion(outputs, logp)
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    return avg_loss


def eval_model(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    preds = []
    targets = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            descriptors = batch["descriptors"].to(device)
            logp = batch["logp"].to(device)

            outputs = model(input_ids, attention_mask, descriptors)
            loss = criterion(outputs, logp)

            total_loss += loss.item()
            preds.extend(outputs.cpu().numpy())
            targets.extend(logp.cpu().numpy())

    avg_loss = total_loss / len(loader)
    mse = mean_squared_error(targets, preds)
    r2 = r2_score(targets, preds)
    return avg_loss, mse, r2


# Inference Function
def predict_logp(smiles, model, tokenizer, scaler, descriptor_list, device):
    # Compute descriptors
    descriptor = compute_descriptors(smiles)
    if descriptor is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")
    descriptor_values = [descriptor[desc] for desc in descriptor_list]
    descriptor_tensor = np.array(descriptor_values).reshape(1, -1)
    descriptor_tensor = scaler.transform(descriptor_tensor)
    descriptor_tensor = torch.tensor(descriptor_tensor, dtype=torch.float).to(device)

    # Tokenize SMILES
    encoding = tokenizer(
        smiles,
        add_special_tokens=True,
        truncation=True,
        max_length=128,
        padding="max_length",
        return_attention_mask=True,
        return_tensors="pt",
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        prediction = model(input_ids, attention_mask, descriptor_tensor)

    return prediction.cpu().item()


def main():
    set_seed()

    # Define descriptor list
    descriptor_list = [
        "MolWt",
        "LogP",
        "NumHDonors",
        "NumHAcceptors",
        "TPSA",
        "NumRotatableBonds",
    ]

    # Check if GPU is available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    # 1. Load the Dataset
    data_path = "S.csv"
    df = pd.read_csv(data_path)

    # Display basic information about the dataset
    print(f"Dataset contains {len(df)} samples.")
    print(df.head())

    # 2. Compute Descriptors
    print("Computing molecular descriptors...")
    for descriptor in descriptor_list:
        df[descriptor] = df["SMILES"].apply(
            lambda x: (
                compute_descriptors(x)[descriptor] if compute_descriptors(x) else 0.0
            )
        )

    # Handle any missing or invalid descriptors
    df.fillna(0.0, inplace=True)

    # 3. Normalize Descriptors
    scaler_path = "descriptor_scaler.joblib"
    if not os.path.exists(scaler_path):
        scaler = StandardScaler()
        df[descriptor_list] = scaler.fit_transform(df[descriptor_list])
        joblib.dump(scaler, scaler_path)
        print("Descriptors normalized and scaler saved.")
    else:
        scaler = joblib.load(scaler_path)
        df[descriptor_list] = scaler.transform(df[descriptor_list])
        print("Descriptors normalized using existing scaler.")

    # 4. Split the Dataset into Training and Validation Sets
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")

    # 5. Initialize the Tokenizer and Model
    model_name = "seyonec/ChemBERTa-zinc-base-v1"

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        base_model = AutoModel.from_pretrained(model_name)
    except OSError as e:
        print(f"Error loading model '{model_name}': {e}")
        print(
            "Please check if the model name is correct and you have internet connectivity."
        )
        return

    # 6. Initialize the Regression Model
    descriptor_dim = len(descriptor_list)
    model = ChemBERTaForLogP(base_model, descriptor_dim)
    model.to(device)

    # 7. Create DataLoaders
    batch_size = 32

    train_dataset = LogPDataset(train_df, tokenizer, descriptor_list, scaler)
    val_dataset = LogPDataset(val_df, tokenizer, descriptor_list, scaler)

    # Set num_workers=0 to avoid multiprocessing issues on Windows
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    # 8. Define the Optimizer and Scheduler
    epochs = 10
    learning_rate = 2e-5
    warmup_steps = 0

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    total_steps = len(train_loader) * epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # 9. Define the Loss Function
    criterion = nn.MSELoss()

    # 10. Training Loop
    best_r2 = -np.inf
    best_model_path = "best_chemberta_logp_model.pt"

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, criterion, device
        )
        print(f"Training Loss: {train_loss:.4f}")

        val_loss, val_mse, val_r2 = eval_model(model, val_loader, criterion, device)
        print(f"Validation Loss: {val_loss:.4f}, MSE: {val_mse:.4f}, R2: {val_r2:.4f}")

        # Save the best model
        if val_r2 > best_r2:
            best_r2 = val_r2
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved with R2: {best_r2:.4f}")

    print("\nTraining complete.")
    print(f"Best Validation R2: {best_r2:.4f}")

    # 11. Load the Best Model for Inference
    model.load_state_dict(torch.load(best_model_path))
    model.to(device)
    model.eval()

    # 12. Example Usage
    example_smiles = [
        "CCO",  # Ethanol
        "CC(C)O",  # Isopropanol
        "CCC(C)O",  # 2-Butanol
        "c1ccccc1O",  # Phenol
        "CC(=O)O",  # Acetic acid
    ]

    for smiles in example_smiles:
        try:
            predicted_logp = predict_logp(
                smiles, model, tokenizer, scaler, descriptor_list, device
            )
            print(f"SMILES: {smiles} => Predicted logP: {predicted_logp:.2f}")
        except ValueError as e:
            print(e)


if __name__ == "__main__":
    main()
