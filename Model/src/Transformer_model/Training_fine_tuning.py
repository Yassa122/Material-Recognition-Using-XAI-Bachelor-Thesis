import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
import torch
from torch.utils.data import Dataset, DataLoader


# Define the PyTorch Dataset for training and prediction
class SMILESDataset(Dataset):
    def __init__(self, smiles_list, target_list=None, tokenizer=None, max_length=128):
        self.smiles_list = smiles_list
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
        }

        # Include targets if available (for training)
        if self.target_list is not None:
            targets = self.target_list[idx]
            item["targets"] = torch.tensor(targets, dtype=torch.float)

        return item


# Load the dataset
def load_smiles_data(csv_file, properties_present=True):
    """
    Load a CSV file containing SMILES strings and optionally, molecular properties.
    """
    data = pd.read_csv(csv_file)
    smiles = data["SMILES"].tolist()  # Column containing SMILES strings

    if properties_present:
        targets = data.drop(
            columns=["SMILES"]
        ).values  # Drop SMILES column, use properties as targets
        return smiles, targets
    else:
        return smiles, None


# Main function for fine-tuning and generating predictions
def main():
    # Load the pre-trained tokenizer and model (ChemBERTa)
    tokenizer = AutoTokenizer.from_pretrained("seyonechandra/chemberta")
    model = AutoModelForSequenceClassification.from_pretrained(
        "seyonechandra/chemberta",
        num_labels=4,  # Adjust this based on the number of properties
    )

    # Load the fine-tuning dataset (with properties)
    fine_tune_file = "SMILES_Big_Data_Set.csv"  # Training dataset path
    smiles_train, targets_train = load_smiles_data(
        fine_tune_file, properties_present=True
    )

    # Split data into training and validation sets
    from sklearn.model_selection import train_test_split

    train_smiles, val_smiles, train_targets, val_targets = train_test_split(
        smiles_train, targets_train, test_size=0.2, random_state=42
    )

    # Prepare training and validation datasets
    train_dataset = SMILESDataset(
        train_smiles, train_targets, tokenizer, max_length=128
    )
    val_dataset = SMILESDataset(val_smiles, val_targets, tokenizer, max_length=128)

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
    )

    # Train the model
    trainer.train()

    # Save the fine-tuned model
    trainer.save_model("fine-tuned-chemberta")

    # Load the dataset for prediction (only SMILES strings)
    predict_file = "test.csv"  # Prediction dataset path
    smiles_predict, _ = load_smiles_data(predict_file, properties_present=False)

    # Prepare prediction dataset
    predict_dataset = SMILESDataset(smiles_predict, tokenizer=tokenizer, max_length=128)

    # Generate predictions
    predictions = []
    model.eval()  # Set model to evaluation mode
    dataloader = DataLoader(predict_dataset, batch_size=16)

    with torch.no_grad():
        for batch in dataloader:
            inputs = {
                "input_ids": batch["input_ids"].to(trainer.model.device),
                "attention_mask": batch["attention_mask"].to(trainer.model.device),
            }
            outputs = model(**inputs)
            logits = outputs.logits
            predictions.extend(logits.cpu().numpy())

    # Save predictions
    predictions_df = pd.DataFrame(
        predictions, columns=["Property_1", "Property_2", "Property_3", "Property_4"]
    )  # Adjust columns as needed
    predictions_df["SMILES"] = smiles_predict
    predictions_df.to_csv("predicted_properties.csv", index=False)

    print("Predictions saved to 'predicted_properties.csv'.")


if __name__ == "__main__":
    main()
