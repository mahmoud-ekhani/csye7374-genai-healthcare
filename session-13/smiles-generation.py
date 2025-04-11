# Jupyter Notebook: end_to_end_conditional_gpt_smiles.ipynb

###############################################################################
#                                                                             #
#   Title: End-to-End Conditional SMILES Generation Using a GPT Model         #
#   Author: Your Name                                                         #
#   Date:   2025-04-08                                                        #
#                                                                             #
###############################################################################

################################################################################
# 1. Introduction & Environment Setup
################################################################################

"""
In this notebook, we will:
1. Load the MOSES dataset, which contains pre-filtered drug-like molecules
2. Compute molecular descriptors (scaffolds, logP, QED, and TPSA)
3. Format data for conditional generation: condition on (scaffold, logP, QED, TPSA)
4. Train a GPT-like decoder-only transformer to generate SMILES given these conditions
5. Evaluate generation by:
   - Validity of SMILES
   - Uniqueness
   - Tanimoto similarity of generated molecules to those in the training set
   - Distribution of conditional properties
"""

# If running on Google Colab, uncomment the next lines to install necessary packages:
# !pip install rdkit-pypi transformers torch tqdm scikit-learn matplotlib moses

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, QED, Lipinski, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    GPT2Config,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)

from moses.dataset import get_dataset

################################################################################
# 2. Data Loading and Filtering
################################################################################

"""
We'll use the MOSES dataset, which is a curated set of drug-like molecules specifically
designed for machine learning applications. It's much smaller than ChEMBL but still
contains high-quality, drug-like compounds.
"""

def load_moses_data():
    """
    Load the MOSES dataset and return a DataFrame with SMILES strings.
    The dataset is already filtered for drug-like compounds.
    """
    # Get the training split of MOSES dataset
    smiles_list = get_dataset('train')
    
    # Convert to DataFrame
    df = pd.DataFrame(smiles_list, columns=['SMILES'])
    
    # Basic validation of SMILES
    valid_mols = []
    for smi in tqdm(smiles_list, desc="Validating SMILES"):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            valid_mols.append(smi)
    
    df = pd.DataFrame(valid_mols, columns=['SMILES'])
    print(f"Loaded {len(df)} valid molecules from MOSES dataset")
    return df

# Load the data
filtered_df = load_moses_data()
filtered_df.head()

################################################################################
# 3. Descriptor Calculation (Scaffolds, logP, QED, TPSA)
################################################################################

"""
Now, let's compute additional descriptors needed:
- Murcko Scaffolds (for the structural motif)
- QED
- TPSA
- LogP

We'll store these in the DataFrame alongside the SMILES.
"""

def calculate_descriptors(smiles: str):
    """
    Calculate molecular descriptors for a given SMILES string.
    Returns None for all descriptors if the molecule is invalid.
    """
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None, None, None, None
    
    try:
        # Scaffolds
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
        
        # QED
        qed_val = QED.qed(mol)
        
        # TPSA
        tpsa_val = Descriptors.TPSA(mol)
        
        # LogP
        logp_val = Descriptors.MolLogP(mol)
        
        return scaffold, logp_val, qed_val, tpsa_val
    except:
        return None, None, None, None

# Calculate descriptors for all molecules
print("Calculating molecular descriptors...")
scaffolds = []
logps = []
qeds = []
tpsas = []

for smi in tqdm(filtered_df['SMILES'], desc="Calculating descriptors"):
    scaffold, lp, qd, tp = calculate_descriptors(smi)
    scaffolds.append(scaffold)
    logps.append(lp)
    qeds.append(qd)
    tpsas.append(tp)

# Add descriptors to DataFrame
filtered_df['Scaffold'] = scaffolds
filtered_df['LogP'] = logps
filtered_df['QED'] = qeds
filtered_df['TPSA'] = tpsas

# Remove any rows where descriptor calculation failed
filtered_df = filtered_df.dropna(subset=['Scaffold', 'LogP', 'QED', 'TPSA'])
print(f"Final dataset size after descriptor calculation: {len(filtered_df)}")

# Display some basic statistics
print("\nDescriptor Statistics:")
print(f"LogP range: {filtered_df['LogP'].min():.2f} to {filtered_df['LogP'].max():.2f}")
print(f"QED range: {filtered_df['QED'].min():.2f} to {filtered_df['QED'].max():.2f}")
print(f"TPSA range: {filtered_df['TPSA'].min():.2f} to {filtered_df['TPSA'].max():.2f}")

filtered_df.head()

################################################################################
# 4. Preparing the Data for Conditional Generation
################################################################################

"""
We will train a GPT model to generate the full SMILES given:
1. The scaffold SMILES
2. The desired LogP
3. The desired QED
4. The desired TPSA

One straightforward approach is to serialize these conditions into a single text string.
For example:

    "SCAFFOLD: Cc1ccccn1 | LOGP: 2.3 | QED: 0.72 | TPSA: 32.4 => FULL_SMILES"

We then train the model in a language modeling fashion to predict FULL_SMILES from 
these inputs. Alternatively, you could build a more sophisticated approach that 
incorporates the numeric data differently, but for demonstration, we'll do it text-based.
"""

def create_conditional_text(row):
    """
    Convert row data into a text prompt for the model. 
    We'll keep it simple: 
      "Scaffold: <scaffold> | LogP: <logp> | QED: <qed> | TPSA: <tpsa> => <full_smiles>"
    """
    scaffold_str = row['Scaffold']
    logp_str = f"{row['LogP']:.2f}"
    qed_str = f"{row['QED']:.2f}"
    tpsa_str = f"{row['TPSA']:.2f}"
    full_smiles = row['SMILES']
    
    input_text = (
        f"Scaffold: {scaffold_str} | "
        f"LogP: {logp_str} | "
        f"QED: {qed_str} | "
        f"TPSA: {tpsa_str} => "
        f"{full_smiles}"
    )
    return input_text

filtered_df['conditional_text'] = filtered_df.apply(create_conditional_text, axis=1)
filtered_df.head()

################################################################################
# 5. Train/Validation/Test Split
################################################################################

from sklearn.model_selection import train_test_split

train_df, valtest_df = train_test_split(filtered_df, test_size=0.2, random_state=42)
val_df, test_df = train_test_split(valtest_df, test_size=0.5, random_state=42)

print(f"Train size: {len(train_df)}")
print(f"Val size: {len(val_df)}")
print(f"Test size: {len(test_df)}")

################################################################################
# 6. Defining the GPT Model and Tokenizer
################################################################################

"""
We'll create a small GPT2 model from scratch (or you could fine-tune a pretrained GPT2).
For demonstration, we'll use a smaller config to train quickly in a teaching environment.
"""

# Create a GPT tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# GPT2 doesn't have a padding token by default; let's set it to the EOS token
tokenizer.pad_token = tokenizer.eos_token

# Create a small GPT2 config
config = GPT2Config(
    vocab_size=tokenizer.vocab_size,
    n_positions=256,
    n_ctx=256,
    n_embd=128,
    n_layer=4,
    n_head=4
)

model = GPT2LMHeadModel(config)

################################################################################
# 7. Create a Custom Dataset for Language Modeling
################################################################################

class SmilesConditionalDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=256):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        txt = self.data.iloc[idx]['conditional_text']
        encoding = self.tokenizer(
            txt, 
            truncation=True, 
            max_length=self.max_length, 
            return_tensors='pt'
        )
        # This returns input_ids and attention_mask
        # For language modeling, we typically set labels = input_ids
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': input_ids
        }

train_dataset = SmilesConditionalDataset(train_df, tokenizer)
val_dataset   = SmilesConditionalDataset(val_df, tokenizer)
test_dataset  = SmilesConditionalDataset(test_df, tokenizer)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

################################################################################
# 8. Training the Model
################################################################################

training_args = TrainingArguments(
    output_dir="./conditional_gpt_smiles",
    overwrite_output_dir=True,
    num_train_epochs=1,             # Increase for real training
    per_device_train_batch_size=2,  # Increase for real training
    per_device_eval_batch_size=2,
    evaluation_strategy="steps",
    eval_steps=50,
    save_steps=50,
    logging_steps=50,
    save_total_limit=1,
    learning_rate=1e-4,
    warmup_steps=100,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# Uncomment to train:
# trainer.train()

################################################################################
# 9. Inference (Generating SMILES from Conditions)
################################################################################

"""
During inference, we'll supply the scaffold and desired property values. For example:

"Scaffold: <scaffold> | LogP: <val> | QED: <val> | TPSA: <val> =>"

The model should complete the sequence by generating a SMILES string.

We'll generate multiple samples to test uniqueness and validity.
"""

def generate_smiles_from_conditions(model, tokenizer, scaffold, logp, qed, tpsa, 
                                    max_length=256, num_return_sequences=1):
    prompt = f"Scaffold: {scaffold} | LogP: {logp:.2f} | QED: {qed:.2f} | TPSA: {tpsa:.2f} =>"
    
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    input_ids = input_ids.to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            do_sample=True,           # Use sampling
            top_k=50,                 # Adjust as desired
            top_p=0.95,               # Adjust as desired
            temperature=0.7,          # Adjust as desired
            pad_token_id=tokenizer.eos_token_id
        )
        
    generated_texts = []
    for output in outputs:
        text = tokenizer.decode(output, skip_special_tokens=True)
        # We want only the SMILES part after the "=>"
        if "=>" in text:
            smiles_part = text.split("=>")[-1].strip()
            generated_texts.append(smiles_part)
        else:
            generated_texts.append(text)
    
    return generated_texts

# Example usage with the first row in the test set:
# trainer.model.eval()
# row = test_df.iloc[0]
# generated_smiles = generate_smiles_from_conditions(
#     model=trainer.model,
#     tokenizer=tokenizer,
#     scaffold=row['Scaffold'],
#     logp=row['LogP'],
#     qed=row['QED'],
#     tpsa=row['TPSA'],
#     num_return_sequences=5
# )
# print(generated_smiles)

################################################################################
# 10. Evaluation
################################################################################

"""
We will:
1. Generate 1000 molecules with random conditions from the test set (or a random range).
2. Check:
   - Valid SMILES: Can RDKit parse them?
   - Unique SMILES: How many are duplicates?
   - Tanimoto similarity to training set.
   - Distribution of predicted properties.
"""

def is_valid_smiles(smi):
    mol = Chem.MolFromSmiles(smi)
    return mol is not None

def compute_tanimoto_similarity(smi1, smi2, radius=2, nBits=2048):
    mol1 = Chem.MolFromSmiles(smi1)
    mol2 = Chem.MolFromSmiles(smi2)
    if mol1 is None or mol2 is None:
        return None
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius, nBits=nBits)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius, nBits=nBits)
    return rdMolDescriptors.TanimotoSimilarity(fp1, fp2)

# Let's define a quick function to evaluate. 
# In practice, you might do more robust analysis.
def evaluate_model(
    model, 
    tokenizer, 
    reference_df, 
    n_samples=1000
):
    model.eval()
    
    valid_count = 0
    unique_smiles = set()
    similarities = []
    
    # We'll store the property differences if we want to check property distribution.
    requested_logps, generated_logps = [], []
    requested_qeds, generated_qeds = [], []
    requested_tpsas, generated_tpsas = [], []
    
    # Convert train_df SMILES to a list for similarity reference
    train_smiles_list = train_df['SMILES'].tolist()
    
    for i in range(n_samples):
        # Randomly select a row from the reference set or sample property values
        row = reference_df.sample(n=1).iloc[0]
        scaffold = row['Scaffold']
        logp_req = row['LogP']
        qed_req = row['QED']
        tpsa_req = row['TPSA']
        
        gen_smiles_list = generate_smiles_from_conditions(
            model, tokenizer, scaffold, logp_req, qed_req, tpsa_req, 
            num_return_sequences=1
        )
        
        gen_smi = gen_smiles_list[0]
        
        if is_valid_smiles(gen_smi):
            valid_count += 1
            unique_smiles.add(Chem.MolToSmiles(Chem.MolFromSmiles(gen_smi)))  # canonical
            
            # Tanimoto similarity (just to the original training data)
            # We'll compute the max similarity to any molecule in the training set 
            # as a measure of novelty.
            best_sim = 0
            for train_smi in train_smiles_list[:1000]:  # limit to 1000 for speed
                sim = compute_tanimoto_similarity(gen_smi, train_smi)
                if sim is not None and sim > best_sim:
                    best_sim = sim
            similarities.append(best_sim)
            
            # Check the property distribution if desired
            gen_mol = Chem.MolFromSmiles(gen_smi)
            if gen_mol:
                gen_logp = Descriptors.MolLogP(gen_mol)
                gen_qed = QED.qed(gen_mol)
                gen_tpsa = Descriptors.TPSA(gen_mol)
                
                requested_logps.append(logp_req)
                generated_logps.append(gen_logp)
                requested_qeds.append(qed_req)
                generated_qeds.append(gen_qed)
                requested_tpsas.append(tpsa_req)
                generated_tpsas.append(gen_tpsa)
    
    validity_ratio = valid_count / n_samples
    uniqueness_ratio = len(unique_smiles) / n_samples
    avg_similarity = np.mean(similarities) if similarities else 0
    
    print(f"Validity: {validity_ratio:.2f}")
    print(f"Uniqueness: {uniqueness_ratio:.2f}")
    print(f"Average Tanimoto similarity to training set: {avg_similarity:.2f}")
    
    # Plot property distribution comparisons
    # For example, requested vs generated LogP
    if len(requested_logps) > 0:
        plt.figure()
        plt.scatter(requested_logps, generated_logps, alpha=0.5)
        plt.xlabel("Requested LogP")
        plt.ylabel("Generated LogP")
        plt.title("Requested vs. Generated LogP")
        plt.show()

        plt.figure()
        plt.scatter(requested_qeds, generated_qeds, alpha=0.5)
        plt.xlabel("Requested QED")
        plt.ylabel("Generated QED")
        plt.title("Requested vs. Generated QED")
        plt.show()

        plt.figure()
        plt.scatter(requested_tpsas, generated_tpsas, alpha=0.5)
        plt.xlabel("Requested TPSA")
        plt.ylabel("Generated TPSA")
        plt.title("Requested vs. Generated TPSA")
        plt.show()

# Uncomment to run evaluation (this can take time for large n_samples):
# evaluate_model(trainer.model, tokenizer, test_df, n_samples=1000)

################################################################################
# 11. Conclusion
################################################################################

"""
In this notebook, we demonstrated:
1. Loading the MOSES dataset, which contains pre-filtered drug-like molecules
2. Computing scaffolds, logP, QED, and TPSA.
3. Creating a conditional text prompt that includes scaffold and property conditions.
4. Training a small GPT-like model to generate SMILES.
5. Evaluating the generation on validity, uniqueness, Tanimoto similarity, and
   property distribution matching.

This pipeline illustrates the core steps for conditional generative modeling
of drug-like molecules. In practice, you may want to:
- Train with more epochs or with a larger model
- Use more advanced property conditioning methods
- Include more thorough data cleaning/curation steps
- Optimize hyperparameters (batch size, learning rate, etc.)
- Explore advanced sampling methods (e.g., nucleus sampling, beam search, etc.)

We hope this example serves as a valuable educational demonstration for 
generative chemistry using Transformer-based architectures!
"""
