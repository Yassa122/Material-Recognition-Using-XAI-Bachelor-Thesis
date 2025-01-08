import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import get_scheduler
from tqdm import tqdm
import random
import numpy as np
from rdkit import Chem
from rdkit import RDLogger  # Suppress RDKit warnings
from lime.lime_text import LimeTextExplainer
import shap
import matplotlib.pyplot as plt
import base64
from io import BytesIO

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')

# Configure logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set random seed for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# Parameters
input_file = r'C:\Users\ASUS\Desktop\moses\data\train.csv'
local_model_path = r"C:\Users\ASUS\Desktop\moses\chemberta_finetuned"
new_model_path = r"C:\Users\ASUS\Desktop\moses\fine_tuned_model"
output_dir = "./fine_tuned_model"
batch_size = 16
epochs = 3
max_length = 128
learning_rate = 5e-5
input_sample_size = 1000
target_novel_smiles = 10
max_new_tokens = 50

substructure_explanations = {
    # Basic elements
    "C": "Carbon atom: fundamental building block of organic molecules, determines backbone structure.",
    "H": "Hydrogen atom: critical for molecular interactions, often involved in hydrogen bonding.",
    "O": "Oxygen atom: contributes to polarity and hydrogen bonding, affects solubility.",
    "N": "Nitrogen atom: introduces polarity, basicity, and potential hydrogen bonding sites.",
    "S": "Sulfur atom: increases molecular complexity, affects reactivity and binding properties.",
    "P": "Phosphorus atom: common in phosphates, affects polarity and enzyme interactions.",
    "F": "Fluorine atom: improves metabolic stability and binding affinity but can affect toxicity.",
    "Cl": "Chlorine atom: can improve binding affinity but may increase toxicity risk.",
    "Br": "Bromine atom: similar to chlorine, affects binding affinity and reactivity.",
    "I": "Iodine atom: increases molecular weight and hydrophobicity, can affect binding affinity.",

    # Hydrocarbons
    "CH": "Part of alkyl chain: contributes to hydrophobicity and molecular volume.",
    "CH3": "Methyl group: increases hydrophobicity, common in many organic molecules.",
    "CH2": "Methylene group: forms part of hydrocarbon chains and rings, contributes to flexibility.",
    "CC": "Ethyl group: contributes to hydrophobicity, common in drug scaffolds.",
    "CCC": "Propyl group: moderate hydrophobicity, impacts lipophilicity and solubility.",
    "CCCC": "Butyl group: increases lipophilicity, often affects solubility and binding.",
    "C(C)C": "Isopropyl group: increases hydrophobicity and steric bulk.",
    "CC(C)C": "Tert-butyl group: highly hydrophobic, increases metabolic stability.",
    "C(C)(C)C": "Branched alkyl chain: increases steric hindrance, affects receptor binding.",
    "C(C)(C)(C)": "Quaternary carbon: adds steric bulk, affects bioavailability.",

    # Halogen-containing groups
    "CF3": "Trifluoromethyl group: enhances metabolic stability, affects lipophilicity.",
    "CCl": "Chloromethyl group: increases reactivity and binding affinity.",
    "CBr": "Bromomethyl group: contributes to hydrophobicity, affects reactivity.",
    "CI": "Iodomethyl group: increases molecular weight, may enhance binding affinity.",

    # Oxygen-containing groups
    "OH": "Hydroxyl group: increases solubility and hydrogen bonding potential.",
    "CO": "Methoxy group: can alter electronic properties and affect solubility.",
    "OCC": "Oxygen-carbon chain: may affect polarity and reactivity.",
    "COC": "Ether group: increases polarity and flexibility.",
    "C=O": "Ketone group: contributes to polarity and reactivity, can act as a hydrogen bond acceptor.",
    "COOH": "Carboxylic acid group: improves solubility but may decrease membrane permeability.",
    "C(O)O": "Lactone group: cyclic ester, common in natural products and antibiotics.",
    "O=C(O)": "Ester group: increases reactivity and can act as a prodrug moiety.",
    "O=C(N)": "Amide group: contributes to hydrogen bonding and stability in proteins.",
    "O=C-S": "Thioester group: enhances reactivity, common in biochemical processes.",
    "C-O-C": "Acetal group: increases hydrophilicity, often used in sugars and prodrugs.",
    "C(O)OH": "Hydroxy acid: increases both hydrogen bonding and solubility.",

    # Nitrogen-containing groups
    "NH2": "Primary amine: contributes to basicity and hydrogen bonding.",
    "NH": "Amine group: increases basicity and hydrogen bonding potential, common in drugs.",
    "CN": "Methylamino group: contributes to basicity and affects metabolic stability.",
    "N(C)": "Methylamine group: may increase hydrophilicity but could impact bioavailability.",
    "N=N": "Azo group: can increase conjugation but may introduce toxicity risks.",
    "C#N": "Nitrile group: contributes to polarity, affects metabolic stability.",
    "C(NO2)": "Nitro group: increases electron density, often associated with toxicity risks.",
    "N=C=O": "Isocyanate group: highly reactive, affects protein interactions.",
    "C(N)C=O": "Amide group: enhances hydrogen bonding and polarity.",
    "C(O)N": "Hydroxylamine: enhances hydrogen bonding, reactive in biological systems.",

    # Sulfur-containing groups
    "SH": "Thiol group: increases reactivity, common in metal ion chelation.",
    "C(S)": "Thiol group: highly reactive, impacts enzyme activity.",
    "C(S)C": "Thioether group: contributes to hydrophobicity, can enhance lipophilicity.",
    "S(=O)(=O)": "Sulfonyl group: highly polar, increases solubility and hydrogen bonding.",
    "C=S": "Thione group: enhances reactivity and electron density.",
    "S(=O)": "Sulfoxide group: increases polarity and can enhance water solubility.",
    "S(=O)(N)": "Sulfone group: enhances polarity and reactivity, used in pharmaceuticals.",

    # Phosphorus-containing groups
    "P(=O)(O)O": "Phosphate group: highly polar, commonly used in prodrugs and biological molecules.",
    "OP(O)(O)": "Phosphonate group: increases polarity, commonly used in enzyme inhibitors.",
    "P(C)(C)": "Phosphine group: increases reactivity and electron density.",
    "P=S": "Thionophosphate group: used in pesticides and enzyme inhibitors.",

    # Aromatic rings
    "c1cccs1": "Thiophene ring: sulfur-containing aromatic rings may reduce stability or drug-likeness.",
    "c1ccc": "Benzene ring: common aromatic scaffold; impacts hydrophobicity and binding.",
    "c1ccc2c(c1)ccc2": "Naphthalene ring: increases hydrophobicity, common in hydrophobic drug scaffolds.",
    "c1ccn": "Pyridine ring: increases polarity and can enhance drug-likeness.",
    "c1cccnc1": "Pyrimidine ring: contributes to hydrogen bonding and polarity, often found in drugs.",
    "c1ccco1": "Furan ring: oxygen-containing aromatic ring, can affect stability and reactivity.",
    "c1c[nH]cn1": "Imidazole ring: contributes to hydrogen bonding and polarity, common in enzyme inhibitors.",
    "c2cccs2": "Thiophene derivative: contributes to electronic properties and aromaticity.",
    "c1nnc2c1ncnc2": "Purine scaffold: key component of nucleotides and many bioactive molecules.",

    # Cyclic groups
    "C1CC1": "Cyclopropane: adds steric bulk, affects binding affinity and rigidity.",
    "C1CCC1": "Cyclobutane: increases steric strain and rigidity, affects hydrophobicity.",
    "C1CCCC1": "Cyclopentane: hydrophobic scaffold, adds moderate flexibility.",
    "C1CCCCC1": "Cyclohexane: increases hydrophobicity and rigidity.",
    "C1C=CC=C1": "Cyclohexadiene: contributes to electronic conjugation, adds rigidity.",

    # Double and triple bonds
    "C=C": "Double bond: increases reactivity but can reduce chemical stability.",
    "C#C": "Triple bond: highly reactive, contributes to molecular rigidity.",
    "C=C-C": "Conjugated system: enhances electronic delocalization, common in chromophores.",
    "C=C-O": "Enol group: enhances reactivity, often seen in keto-enol tautomerism.",
    "C=C(C)": "Alkene group: increases lipophilicity and reactivity.",

    # Miscellaneous
    "NO2": "Nitro group: increases electron density, often associated with toxicity risks.",
    "SO2": "Sulfonyl group: highly polar, increases solubility and hydrogen bonding.",
    "C(F)(F)(F)": "Trifluoromethyl group: enhances metabolic stability and lipophilicity.",
    "C(O)(N)": "Hydroxamic acid: acts as a chelator and hydrogen bond donor/acceptor.",
    "C=CC": "Alkene group: increases lipophilicity and reactivity.",
    "CO": "Carbon monoxide-like fragment: contributes to polarity, rare in stable compounds.",

    "n3cncn3": "Triazine ring: nitrogen-rich aromatic ring, often used in agrochemicals and dyes.",
    "n1": "Part of an aromatic heterocycle containing nitrogen, such as pyridine or triazole.",
    "Cc1ccc": "Substituted benzene ring: impacts hydrophobicity and binding.",
    "cc1S": "Sulfur-containing aromatic ring: affects electronic properties and reactivity.",
    "c2ccncc2": "Pyridazine ring: nitrogen-rich heterocycle, enhances polarity.",
    "NC1CC1": "Aminocyclopropane: introduces steric bulk and affects bioavailability.",
    "nH": "Nitrogen with a hydrogen atom: contributes to hydrogen bonding and basicity.",
    "nc2c1": "Fused heterocyclic rings: impacts electronic and binding properties.",
    "Nc1ccc2ncccc2c1": "Quinoline derivative: enhances aromaticity and potential binding affinity.",
    "s2": "Part of a thiophene-like structure: sulfur-containing aromatic ring affects stability.",
    "cs2": "Thiocarbonyl group: contributes to reactivity and sulfur-based interactions.",
    "c3nc": "Part of a nitrogen-containing aromatic system: enhances polarity and binding.",
    "C1": "Cyclized carbon structure: contributes to rigidity and hydrophobicity.",
    "1": "Possible ring closure marker: indicates cyclic structures.",

    "c1cccs1": "Thiophene ring: sulfur-containing aromatic rings may reduce stability or drug-likeness.",
    "COc1ccc": "Methoxy group: can alter electronic properties and affect solubility.",
    "OC": "Hydroxyl group: increases solubility but may introduce metabolic liabilities.",
    "OCC": "Oxygen-carbon chain: may affect polarity and reactivity.",
    "c1C": "Part of aromatic ring system: contributes slightly to the molecule's overall behavior.",
    "N(C)": "Methylamine group: may increase hydrophilicity but could impact bioavailability.",
    "CC(=O)": "Carbonyl group: often improves reactivity but may reduce stability.",
    "Cl": "Chlorine atom: can improve binding affinity but may increase toxicity risk.",
    "CN": "Methylamino group: contributes to basicity and affects metabolic stability.",
    "C=C": "Double bond: increases reactivity but can reduce chemical stability.",
    "C#N": "Nitrile group: contributes to polarity but may decrease synthetic accessibility.",
    "COOH": "Carboxylic acid group: improves solubility but may decrease membrane permeability.",
    "c1ccn": "Pyridine ring: increases polarity and can enhance drug-likeness.",
    "F": "Fluorine atom: improves metabolic stability and binding affinity but can affect toxicity.",
    "Br": "Bromine atom: similar to chlorine, affects binding affinity and reactivity.",
    "c1ccc": "Benzene ring: common aromatic scaffold; impacts hydrophobicity and binding."
}

default_explanation = "No specific insights available for this substructure."

# Save plot as base64-encoded image
def save_plot_to_base64(fig):
    buffer = BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    buffer.seek(0)
    encoded_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
    buffer.close()
    return encoded_image

# Validate SMILES strings
def is_valid_smiles(smiles):
    """
    Check if the given SMILES string is valid using RDKit.
    Args:
        smiles (str): A SMILES string to validate.
    Returns:
        bool: True if valid, False otherwise.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except Exception as e:
        logger.error(f"Error validating SMILES: {e}")
        return False

# Detailed substructure explanations
def generate_detailed_explanations(smiles, lime_features, lime_weights):
    """
    Generate detailed explanations for LIME features using substructure insights.

    Args:
        smiles (str): The SMILES string being analyzed.
        lime_features (list): Substructures identified by LIME.
        lime_weights (list): Corresponding importance weights for the substructures.

    Returns:
        list: A list of detailed textual explanations for each substructure with specific insights.
    """
    explanations = []
    if not lime_features or not lime_weights:
        return ["Error: Missing features or weights for explanation generation."]
    
    for feature, weight in zip(lime_features, lime_weights):
        try:
            # Fetch substructure insight
            substructure_insight = substructure_explanations.get(feature)
            
            # Skip substructures without specific insights
            if not substructure_insight:
                continue
            
            # Determine impact
            impact = "positive" if weight > 0 else "negative"
            
            # Construct explanation
            explanations.append(
                f"Substructure: {feature} | Impact: {impact} | Magnitude: {abs(weight):.2f} | Insight: {substructure_insight}"
            )
        except Exception as e:
            # Handle unexpected errors
            explanations.append(f"Error processing feature {feature}: {e}")
    
    return explanations if explanations else ["No substructures with specific insights found."]


from rdkit import Chem
from rdkit import Chem
from rdkit.Chem import BondType
from rdkit.Chem import rdchem
from rdkit.Chem import PeriodicTable

def modify_atoms(mol):
    rw_mol = Chem.RWMol(mol)
    try:
        atom_idx = random.randint(0, rw_mol.GetNumAtoms() - 1)
        atom = rw_mol.GetAtomWithIdx(atom_idx)
        current_atomic_num = atom.GetAtomicNum()

        valid_atomic_nums = [6, 7, 8, 9, 15, 16, 17]  # C, N, O, F, P, S, Cl
        valid_atomic_nums = [num for num in valid_atomic_nums if num != current_atomic_num]
        if not valid_atomic_nums:
            return None

        new_atomic_num = random.choice(valid_atomic_nums)
        atom.SetAtomicNum(new_atomic_num)
        Chem.SanitizeMol(rw_mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        return rw_mol
    except Exception as e:
        logger.warning(f"Failed to modify atom: {e}")
        return None
def modify_bonds(mol):
    rw_mol = Chem.RWMol(mol)
    try:
        atom_indices = random.sample(range(rw_mol.GetNumAtoms()), 2)
        bond = rw_mol.GetBondBetweenAtoms(atom_indices[0], atom_indices[1])

        if bond:
            # Modify bond type
            current_bond_type = bond.GetBondType()
            new_bond_type = random.choice(
                [Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE]
            )
            if new_bond_type != current_bond_type:
                rw_mol.GetBondBetweenAtoms(atom_indices[0], atom_indices[1]).SetBondType(new_bond_type)
        else:
            # Add a new bond if none exists
            rw_mol.AddBond(atom_indices[0], atom_indices[1], Chem.BondType.SINGLE)

        Chem.SanitizeMol(rw_mol)
        return rw_mol
    except Exception as e:
        logger.warning(f"Failed to modify bond: {e}")
    return None

def generate_counterfactuals(smile, max_changes=10):
    counterfactuals = set()
    original_mol = Chem.MolFromSmiles(smile)
    if not original_mol:
        logger.warning(f"Invalid original SMILES: {smile}")
        return []

    for _ in range(max_changes * 10):  # Retry multiple times for diversity
        modified_mol = None
        if random.random() < 0.5:
            modified_mol = modify_bonds(original_mol)
        else:
            modified_mol = modify_atoms(original_mol)

        if modified_mol:
            try:
                new_smiles = Chem.MolToSmiles(modified_mol, isomericSmiles=True)
                if is_valid_smiles(new_smiles) and new_smiles != smile:
                    counterfactuals.add(new_smiles)
                    if len(counterfactuals) >= max_changes:
                        break
            except Exception as e:
                logger.warning(f"Failed to generate valid counterfactual SMILES: {e}")

    return list(counterfactuals)

from rdkit import Chem

def compare_structures(original_smiles, counterfactual_smiles):
    original_mol = Chem.MolFromSmiles(original_smiles)
    counterfactual_mol = Chem.MolFromSmiles(counterfactual_smiles)
    
    if original_mol is None or counterfactual_mol is None:
        return [{"action": "Invalid SMILES", "substructure": None}]
    
    differences = []

    # Compare atoms
    original_atoms = [atom.GetSymbol() for atom in original_mol.GetAtoms()]
    counterfactual_atoms = [atom.GetSymbol() for atom in counterfactual_mol.GetAtoms()]
    for atom in set(counterfactual_atoms):
        if counterfactual_atoms.count(atom) > original_atoms.count(atom):
            differences.append({"action": "Added", "substructure": atom})
    for atom in set(original_atoms):
        if original_atoms.count(atom) > counterfactual_atoms.count(atom):
            differences.append({"action": "Removed", "substructure": atom})

    # Compare bonds
    original_bonds = {(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond.GetBondType()) for bond in original_mol.GetBonds()}
    counterfactual_bonds = {(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond.GetBondType()) for bond in counterfactual_mol.GetBonds()}

    added_bonds = counterfactual_bonds - original_bonds
    removed_bonds = original_bonds - counterfactual_bonds

    for bond in added_bonds:
        differences.append({"action": "Added bond", "substructure": f"Bond between {bond[0]} and {bond[1]}: {bond[2]} bond"})
    for bond in removed_bonds:
        differences.append({"action": "Removed bond", "substructure": f"Bond between {bond[0]} and {bond[1]}: {bond[2]} bond"})

    return differences



def prediction_fn(smiles_list):
    """
    Generate predictions for a list of SMILES strings.

    Args:
        smiles_list (list): List of SMILES strings.

    Returns:
        list: List of prediction scores.
    """
    inputs = tokenizer(
        smiles_list,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    ).to(model.device)

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
        return probs[:, 1].tolist()  # Assuming class 1 is the target


def generate_counterfactual_explanation(original_smiles, counterfactuals, prediction_fn):
    explanations = []
    original_pred = prediction_fn([original_smiles])[0] if prediction_fn else None

    for cf_smiles in counterfactuals:
        try:
            cf_pred = prediction_fn([cf_smiles])[0] if prediction_fn else None
            changes = compare_structures(original_smiles, cf_smiles)

            detailed_changes = []
            for change in changes:
                substructure = change.get('substructure', 'Unknown')
                action = change.get('action', 'Unknown')
                insight = substructure_explanations.get(substructure, "No specific insights available.")
                detailed_changes.append(f"{action} {substructure}: {insight}")

            explanation = (
                f"Original: {original_smiles} (Prediction: {original_pred:.3f})\n"
                f"Counterfactual: {cf_smiles} (Prediction: {cf_pred:.3f})\n"
                f"Changes:\n" + "\n".join(detailed_changes) + "\n"
                f"Impact: {'Increased' if cf_pred > original_pred else 'Decreased'} prediction by {abs(cf_pred - original_pred):.3f}."
            )
            explanations.append(explanation)
        except Exception as e:
            logger.error(f"Error generating counterfactual explanations: {e}")
            explanations.append(f"Failed to generate explanation for counterfactual: {cf_smiles}")

    return explanations if explanations else ["No counterfactuals generated or explanation unavailable."]



# Load SMILES dataset
try:
    data = pd.read_csv(input_file)
    logger.info(f"Loaded {len(data)} rows from the dataset.")
except FileNotFoundError:
    logger.error(f"Input file not found at {input_file}. Please check the path.")
    raise

# Sample input SMILES
if input_sample_size < len(data):
    data = data.sample(n=input_sample_size, random_state=42)
    logger.info(f"Sampled {input_sample_size} rows from the dataset.")

# Tokenizer and Model
tokenizer = AutoTokenizer.from_pretrained(local_model_path)
model = AutoModelForCausalLM.from_pretrained(local_model_path).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# Dataset
class SMILESDataset(Dataset):
    def __init__(self, smiles_strings, tokenizer, max_length):
        self.smiles_strings = smiles_strings
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.smiles_strings)

    def __getitem__(self, idx):
        smiles = self.smiles_strings[idx]
        encoded = self.tokenizer(
            smiles,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
        }

# Split dataset into train and validation sets

train_size = int(0.8 * len(data))
val_size = len(data) - train_size
train_data, val_data = data[:train_size], data[train_size:]
train_dataset = SMILESDataset(train_data["SMILES"].tolist(), tokenizer, max_length)
val_dataset = SMILESDataset(val_data["SMILES"].tolist(), tokenizer, max_length)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=learning_rate)
num_training_steps = epochs * len(train_loader)
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

# Loss function
def calculate_loss(model, batch):
    input_ids = batch["input_ids"].to(model.device)
    attention_mask = batch["attention_mask"].to(model.device)
    labels = input_ids.clone()
    labels[labels == tokenizer.pad_token_id] = -100
    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
    return outputs.loss

# Training loop
logger.info("Starting training...")
for epoch in range(epochs):
    model.train()
    train_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
    for batch in progress_bar:
        optimizer.zero_grad()
        loss = calculate_loss(model, batch)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        train_loss += loss.item()
        progress_bar.set_postfix({"loss": loss.item()})
    logger.info(f"Epoch {epoch+1} Training Loss: {train_loss / len(train_loader):.4f}")

    # Validation loop
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            loss = calculate_loss(model, batch)
            val_loss += loss.item()
    logger.info(f"Epoch {epoch+1} Validation Loss: {val_loss / len(val_loader):.4f}")

# Save fine-tuned model
os.makedirs(output_dir, exist_ok=True)
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
logger.info(f"Fine-tuned model saved to {output_dir}")

from rdkit import Chem
def generate_smiles(model, tokenizer, seed_text, max_new_tokens=50, temperature=1.2, top_p=0.9):
    """
    Generate a SMILES string using the model and validate its chemical structure.
    """
    model.eval()
    try:
        # Tokenize the seed text
        inputs = tokenizer(
            seed_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=128
        ).to(model.device)

        # Generate SMILES
        output = model.generate(
            inputs["input_ids"],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )

        # Decode the output
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True).strip()

        # Validate the generated SMILES
        if is_valid_smiles(generated_text):
            return generated_text
        else:
            logger.warning(f"Generated invalid SMILES: {generated_text}")
            return None
    except Exception as e:
        logger.error(f"Error during SMILES generation: {e}")
        return None


# Generate explanations
def generate_explanations(smiles):
    explanations = {"smiles": smiles, "lime_plot": None, "shap_plot": None, "counterfactual": None}

    # Counterfactuals
    try:
        logger.info(f"Generating counterfactuals for SMILES: {smiles}")
        counterfactuals = generate_counterfactuals(smiles, max_changes=5)
        if counterfactuals:
            logger.info(f"Generated {len(counterfactuals)} counterfactuals: {counterfactuals}")
            counterfactual_explanations = generate_counterfactual_explanation(smiles, counterfactuals, prediction_fn)
            explanations["counterfactual"] = counterfactual_explanations
        else:
            explanations["counterfactual"] = ["No counterfactuals generated."]
            logger.warning(f"No counterfactuals generated for SMILES: {smiles}")
    except Exception as e:
        explanations["counterfactual"] = f"Error: {e}"
        logger.error(f"Error generating counterfactual explanations: {e}")

    # Other explanations (LIME, SHAP, etc.)
    try:
        explainer = LimeTextExplainer()
        lime_exp = explainer.explain_instance(smiles, lambda x: np.random.rand(len(x), 10), num_features=5)
        lime_features, lime_weights = zip(*lime_exp.as_list())
        explanations["detailed_explanations"] = generate_detailed_explanations(smiles, lime_features, lime_weights)

        # Save LIME plot
        fig, ax = plt.subplots()
        ax.barh(lime_features, lime_weights)
        ax.set_xlabel("Feature Importance")
        ax.set_title(f"LIME Explanation for {smiles}")
        explanations["lime_plot"] = save_plot_to_base64(fig)
        plt.close(fig)
    except Exception as e:
        explanations["lime_plot"] = f"Error: {e}"
        logger.error(f"Error generating LIME explanations: {e}")

    # SHAP Explanation
    try:
        shap_explainer = shap.Explainer(lambda x: np.random.rand(len(x), 10))
        shap_values = shap_explainer([smiles])
        explanations["shap_plot"] = shap_values
    except Exception as e:
        explanations["shap_plot"] = f"Error: {e}"
        logger.error(f"Error generating SHAP explanations: {e}")

    return explanations


# Generate SMILES with explanations
logger.info("Generating SMILES with explanations...")
results = []
novel_smiles_set = set()

with tqdm(total=target_novel_smiles, desc="Generating SMILES") as pbar:
    print(f"Using model: {model.__class__.__name__}")
    print(f"Model configuration: {model.config}")
    print(f"Using tokenizer: {tokenizer.__class__.__name__}")
    print(f"Tokenizer configuration: {tokenizer}")

    while len(results) < target_novel_smiles:
        # Select a seed SMILES randomly
        seed_smiles = random.choice(data["SMILES"].tolist())
        
        # Generate SMILES
        generated = generate_smiles(model, tokenizer, seed_smiles)
        if generated and is_valid_smiles(generated) and generated not in novel_smiles_set:
            logger.info(f"Generated valid SMILES: {generated}")
            
            # Use the existing generate_explanations function
            explanations = generate_explanations(generated)

            # Log the results for debugging
            logger.info(f"Explanations for {generated}: {explanations}")

            # Append explanations and generated SMILES to results
            results.append({
                "generated_smiles": generated,
                "explanations": explanations
            })
            
            # Add to the novel SMILES set
            novel_smiles_set.add(generated)
            pbar.update(1)



# Save results as an HTML file
output_file = "generated_smiles_with_explanations.html"
html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SMILES Explanations</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
        }
        img {
            max-width: 600px;
            height: auto;
            border: 1px solid #ddd;
            margin-top: 10px;
        }
        h1 {
            color: #003366;
            text-align: center;
        }
        h2 {
            color: #003366;
        }
        ul {
            padding-left: 20px;
        }
        li {
            margin-bottom: 10px;
        }
        .section {
            margin-bottom: 30px;
        }
    </style>
</head>
<body>
    <h1>Generated SMILES with Explanations</h1>
"""

# Loop through the results and add them to the HTML content
for res in results:
    generated_smiles = res.get("generated_smiles", "N/A")
    explanations = res.get("explanations", {})

    html_content += f"<div class='section'><h2>SMILES: {generated_smiles}</h2>"
    
    # Add LIME Plot
    lime_plot = explanations.get("lime_plot")
    if lime_plot:
        html_content += f'<h3>LIME Explanation</h3>'
        html_content += f'<img src="data:image/png;base64,{lime_plot}" alt="LIME Explanation"><br>'
    else:
        html_content += "<p>No LIME explanation available.</p>"

    # Add Detailed Substructure Analysis
    detailed_explanations = explanations.get("detailed_explanations", [])
    if detailed_explanations:
        html_content += "<h3>Detailed Substructure Analysis</h3><ul>"
        for explanation in detailed_explanations:
            html_content += f"<li>{explanation}</li>"
        html_content += "</ul>"
    else:
        html_content += "<p>No detailed substructure analysis available.</p>"

    # Add Counterfactual Explanation
    counterfactuals = explanations.get("counterfactual", [])
    html_content += "<h3>Counterfactual Explanation</h3>"
    if isinstance(counterfactuals, list) and counterfactuals:
        html_content += "<ul>"
        for cf in counterfactuals:
            html_content += f"<li>{cf}</li>"
        html_content += "</ul>"
    else:
        html_content += "<p>No counterfactuals generated or explanation unavailable.</p>"
    
    html_content += "</div>"

# Finalize the HTML content
html_content += "</body></html>"

# Save the HTML file
with open(output_file, "w", encoding="utf-8") as f:
    f.write(html_content)

logger.info(f"Results saved to {output_file}")

