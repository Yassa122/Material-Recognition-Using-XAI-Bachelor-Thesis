import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from tqdm import tqdm
import time
from torch import nn
from rdkit import Chem
from rdkit.Chem import Descriptors


# Function to extract knowledge-based features
def extract_knowledge_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        molecular_weight = Descriptors.MolWt(mol)
        num_h_donors = Descriptors.NumHDonors(mol)
        num_h_acceptors = Descriptors.NumHAcceptors(mol)
        return [molecular_weight, num_h_donors, num_h_acceptors]
    else:
        return [0, 0, 0]  # Default values for invalid SMILES


# Function to load SMILES data
def load_smiles_data(csv_file, properties_present=True):
    data = pd.read_csv(csv_file)
    smiles = data["SMILES"].tolist()

    # Extract knowledge-based features
    knowledge_features = [extract_knowledge_features(sm) for sm in smiles]

    if properties_present:
        targets = data.drop(columns=["SMILES", "mol"], errors="ignore")
        print("Initial number of samples:", len(targets))

        # Convert targets to numeric and drop NaNs
        targets = targets.apply(pd.to_numeric, errors="coerce")
        non_numeric_rows = targets[targets.isnull().any(axis=1)]
        if not non_numeric_rows.empty:
            print("Non-numeric rows found:")
            print(non_numeric_rows)

        targets = targets.dropna()
        valid_indices = targets.index
        targets = targets.values
        smiles = [smiles[i] for i in valid_indices]
        knowledge_features = [knowledge_features[i] for i in valid_indices]

        print("Number of samples with NaN values removed:", len(targets))
        print(f"Final number of SMILES: {len(smiles)}")
        print(f"Final number of targets: {len(targets)}")

        return smiles, targets, knowledge_features
    else:
        return smiles, None, knowledge_features


# SMILESDataset class
class SMILESDataset(Dataset):
    def __init__(
        self,
        smiles_list,
        knowledge_features,
        target_list=None,
        tokenizer=None,
        max_length=128,
    ):
        self.smiles_list = smiles_list
        self.knowledge_features = knowledge_features
        self.target_list = target_list
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]

        # Tokenize the SMILES string
        inputs = self.tokenizer(
            smiles,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        item = {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "knowledge_features": torch.tensor(
                self.knowledge_features[idx], dtype=torch.float
            ),
        }

        # Include targets if available
        if self.target_list is not None:
            targets = self.target_list[idx]
            item["labels"] = torch.tensor(targets, dtype=torch.float)

        return item


# Custom model class
class KnowledgeAugmentedModel(nn.Module):
    def __init__(self, base_model, knowledge_dim, num_labels):
        super(KnowledgeAugmentedModel, self).__init__()
        self.base_model = base_model

        # Process knowledge-based features
        self.knowledge_fc = nn.Sequential(
            nn.Linear(knowledge_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        # Adjust the classifier to accept combined features
        hidden_size = base_model.config.hidden_size  # Typically 768 for RoBERTa-base
        self.classifier = nn.Linear(hidden_size + 64, num_labels)

    def forward(
        self, input_ids=None, attention_mask=None, knowledge_features=None, labels=None
    ):
        outputs = self.base_model.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=True,
        )
        pooled_output = outputs.last_hidden_state[:, 0, :]  # Get [CLS] token output

        knowledge_output = self.knowledge_fc(knowledge_features)

        combined_output = torch.cat((pooled_output, knowledge_output), dim=1)

        logits = self.classifier(combined_output)

        loss = None
        if labels is not None:
            loss_fn = nn.MSELoss()
            loss = loss_fn(logits, labels)

        return {"loss": loss, "logits": logits}


# Custom data collator
def data_collator(batch):
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    knowledge_features = torch.stack([item["knowledge_features"] for item in batch])

    batch_dict = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "knowledge_features": knowledge_features,
    }

    if "labels" in batch[0]:
        labels = torch.stack([item["labels"] for item in batch])
        batch_dict["labels"] = labels

    return batch_dict


# Main function
def main():
    # Load the pre-trained tokenizer and base model (ChemBERTa)
    tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")

    # Define the number of labels
    num_labels = 3  # Update this to match the number of columns in your target data

    # Load the base model
    base_model = AutoModelForSequenceClassification.from_pretrained(
        "seyonec/ChemBERTa-zinc-base-v1",
        num_labels=num_labels,
        problem_type="regression",
    )

    # Create the custom model
    knowledge_dim = 3
    model = KnowledgeAugmentedModel(base_model, knowledge_dim, num_labels)

    # Load the fine-tuning dataset (with properties)
    fine_tune_file = "SMILES_Big_Data_Set.csv"  # Training dataset path
    smiles_train, targets_train, knowledge_features_train = load_smiles_data(
        fine_tune_file, properties_present=True
    )

    # Split data into training and validation sets
    from sklearn.model_selection import train_test_split

    (
        train_smiles,
        val_smiles,
        train_targets,
        val_targets,
        train_knowledge_features,
        val_knowledge_features,
    ) = train_test_split(
        smiles_train,
        targets_train,
        knowledge_features_train,
        test_size=0.2,
        random_state=42,
    )

    # Prepare training and validation datasets
    train_dataset = SMILESDataset(
        train_smiles, train_knowledge_features, train_targets, tokenizer, max_length=128
    )
    val_dataset = SMILESDataset(
        val_smiles, val_knowledge_features, val_targets, tokenizer, max_length=128
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",  # Output directory
        evaluation_strategy="epoch",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
    )

    # Trainer for fine-tuning
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    # Train the model
    trainer.train()

    # Save the fine-tuned model
    trainer.save_model("fine_tuned_chemberta_with_knowledge")

    # Load the dataset for prediction (only SMILES strings)
    predict_file = "test.csv"  # Prediction dataset path
    smiles_predict, _, knowledge_features_predict = load_smiles_data(
        predict_file, properties_present=False
    )

    # Prepare prediction dataset
    predict_dataset = SMILESDataset(
        smiles_predict, knowledge_features_predict, tokenizer=tokenizer, max_length=128
    )

    # Generate predictions
    predictions = []
    model.eval()  # Set model to evaluation mode
    dataloader = DataLoader(predict_dataset, batch_size=16, collate_fn=data_collator)

    # Time tracking start
    start_time = time.time()

    # Using tqdm for progress bar
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting", unit="batch"):
            inputs = {
                "input_ids": batch["input_ids"].to(trainer.model.device),
                "attention_mask": batch["attention_mask"].to(trainer.model.device),
                "knowledge_features": batch["knowledge_features"].to(
                    trainer.model.device
                ),
            }
            outputs = model(**inputs)
            logits = outputs["logits"]
            predictions.extend(logits.cpu().numpy())

    # Time tracking end
    end_time = time.time()
    elapsed_time = end_time - start_time

    predictions_df = pd.DataFrame(
        predictions,
        columns=["Predicted_pIC50", "Predicted_logP", "Predicted_num_atoms"],
    )
    predictions_df["SMILES"] = smiles_predict
    predictions_df.to_csv("predicted_properties.csv", index=False)

    print(
        f"Predictions saved to 'predicted_properties.csv'. Total prediction time: {elapsed_time:.2f} seconds."
    )


if __name__ == "__main__":
    main()
