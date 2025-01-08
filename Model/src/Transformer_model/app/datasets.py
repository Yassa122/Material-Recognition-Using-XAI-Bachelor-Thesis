# app/datasets.py

import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors
import logging
from torch import nn


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
        if self.target_list is not None:
            item["labels"] = torch.tensor(self.target_list[idx], dtype=torch.float)
        return item


class KnowledgeAugmentedModel(nn.Module):
    def __init__(self, base_model, knowledge_dim, num_labels):
        super(KnowledgeAugmentedModel, self).__init__()
        self.base_model = base_model
        self.knowledge_fc = nn.Sequential(
            nn.Linear(knowledge_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        hidden_size = base_model.config.hidden_size
        combined_size = hidden_size + 64

        self.classifier_pIC50 = nn.Linear(combined_size, 1)
        self.classifier_logP = nn.Linear(combined_size, 1)
        self.classifier_num_atoms = nn.Linear(combined_size, 1)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        knowledge_features=None,
        labels=None,
        output_attentions=False,
    ):
        outputs = self.base_model.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            output_attentions=output_attentions,
        )
        pooled_output = outputs.last_hidden_state[:, 0, :]
        knowledge_output = self.knowledge_fc(knowledge_features)
        combined_output = torch.cat((pooled_output, knowledge_output), dim=1)

        pred_pIC50 = self.classifier_pIC50(combined_output).squeeze(-1)
        pred_logP = self.classifier_logP(combined_output).squeeze(-1)
        pred_num_atoms = self.classifier_num_atoms(combined_output).squeeze(-1)

        loss = None
        if labels is not None:
            loss_fn = nn.MSELoss()
            loss_pIC50 = loss_fn(pred_pIC50, labels[:, 0])
            loss_logP = loss_fn(pred_logP, labels[:, 1])
            loss_num_atoms = loss_fn(pred_num_atoms, labels[:, 2])
            loss = (loss_pIC50 + loss_logP + loss_num_atoms) / 3  # Average the losses

        ret = {
            "loss": loss,
            "logits_pIC50": pred_pIC50,
            "logits_logP": pred_logP,
            "logits_num_atoms": pred_num_atoms,
        }
        if output_attentions:
            ret["attentions"] = outputs.attentions
        return ret
