import numpy as np
import torch
import json
import sys
from pathlib import Path
# Add the project root to Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)
from torch.utils.data import DataLoader
from evaluate import load
from datasets import Audio
from types import SimpleNamespace
from model.model import Whisper
from transformers import WhisperProcessor
from model.load_model import load_model
from model.accent_model import ModifiedWhisper
from operator import attrgetter
from tqdm import tqdm
import wandb
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
import pandas as pd
from datasets import Audio, concatenate_datasets, load_dataset
import io
from scipy.io import wavfile
import librosa
from add_new_accents import load_new_dataset
from train import load_datasets, prepare_data
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sns
from huggingface_hub import hf_hub_download
import os


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
wer_metric = load("wer")
model_variant = "small"
# name_of_run = "finetune-decoder-only and compare english scores to scottish"
# margin = 0.13
num_accent_classes = 54
ID_TO_ACCENT = {0: 'Scottish', 1: 'English', 2: 'Indian', 3: 'Irish', 4: 'Welsh', 5: 'NewZealandEnglish', 6: 'AustralianEnglish', 7: 'SouthAfrican', 8: 'Canadian', 9: 'NorthernIrish', 10: 'American', 11: 'Austria', 12: 'Greece', 13: 'Brazil', 14: 'Mexico/Central America', 15: 'Bangladesh', 16: 'Russia', 17: 'France', 18: 'Czech Republic', 19: 'Nigeria', 20: 'Kenya', 21: 'Romania', 22: 'Poland', 23: 'Hungary', 24: 'West Indies and Bermuda', 25: 'China', 26: 'Sweden', 27: 'Nepal', 28: 'Ukraine', 29: 'Indonesia', 30: 'East Africa', 31: 'Japan', 32: 'Vietnam', 33: 'Latvia', 34: 'Israel', 35: 'Spain', 36: 'Hong Kong', 37: 'Turkey', 38: 'Germany', 39: 'Ghana', 40: 'Bulgaria', 41: 'Netherlands', 42: 'Italy', 43: 'Malaysia', 44: 'South Korea', 45: 'Norway', 46: 'Finland', 47: 'Singapore', 48: 'Slovakia', 49: 'Croatia', 50: 'Thailand', 51: 'Kazakhstan', 52: 'Denmark', 53: 'Philippines'}
lora = True
# finetuned_model_path = "best_model_2025-03-16-05-38-01.pt"

def setup_model(checkpoint_file = None, num_accent_classes=num_accent_classes):
    base_whisper_model = load_model(model_variant, device=DEVICE)
    model = ModifiedWhisper(base_whisper_model.dims, num_accent_classes, base_whisper_model)
    
    peft_config = LoraConfig(
        inference_mode=False,
        r=8,
        target_modules=["out", "token_embedding", "query", "key", "value", "proj_out"],
        lora_alpha=32,
        lora_dropout=0.1
    )
    model = get_peft_model(model, peft_config)
    if checkpoint_file:
        checkpoint_path = str(Path(__file__).parent / checkpoint_file)
        checkpoint = torch.load(checkpoint_path, map_location=torch.device(DEVICE))
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
            print('Model loaded successfully')
    model.print_trainable_parameters()
    
    model.to(DEVICE)
    
    return model





def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    
    correct = 0
    total = 0
    progress_bar = tqdm(dataloader, desc="Evaluating")
    with torch.no_grad():
        for batch_idx, batch in enumerate(progress_bar):
            mel = batch['mel'].to(device)
            text = batch['text'].to(device)
            target = batch['target'].to(device)
            output = model(mel, text)
            print(output.shape, 'output shape')
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
            print(pred.shape, 'pred shape')
            all_preds.extend(pred.cpu().numpy())
            print(all_preds, 'all preds')
            all_targets.extend(target.cpu().numpy())
            
    accuracy = correct / total if total > 0 else 0
    return accuracy, all_preds, all_targets

def plot_confusion_matrix(y_true, y_pred, class_names=None):
    """
    Plot confusion matrix for model evaluation.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names (optional)
    """

    
    # Compute confusion matrix
    print(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print(cm)
    
    # Normalize the confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Plot with seaborn for better styling
    if class_names is not None:
        # If there are too many classes, use a subset for readability
        if len(class_names) > 20:
            # Find the classes that appear in the data
            used_indices = sorted(list(set(y_true) | set(y_pred)))
            used_class_names = [class_names[i] for i in used_indices]
            # Filter confusion matrix to only include used classes
            cm_normalized = cm_normalized[used_indices][:, used_indices]
            sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', 
                        xticklabels=used_class_names, yticklabels=used_class_names)
        else:
            sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', 
                        xticklabels=class_names, yticklabels=class_names)
    else:
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues')
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    print(f"Confusion matrix saved to 'confusion_matrix.png'")


def load_all_data():
    if DEVICE == "cuda":
        new_dataset, unique_accents = load_new_dataset(['Amirjab21/commonvoice'])
    else:
        # Read all parquet files in dataframes directory and concatenate them
        dataframes_dir = Path("dataframes")
        parquet_files = list(dataframes_dir.glob("*.parquet"))
        
        if parquet_files:
            dataframes = [pd.read_parquet(file) for file in parquet_files]
            new_dataset = pd.concat(dataframes, ignore_index=True)
            print(f"Loaded {len(parquet_files)} parquet files from dataframes directory")
        else:
            # Fallback to CSV if no parquet files found
            new_dataset = pd.read_csv("dataframes/commonvoice_dataframe.csv")
            print("No parquet files found, loaded CSV file instead")

    dataset_df = load_datasets(sample_size=100)
    concatenated_df = pd.concat([dataset_df, new_dataset], ignore_index=True)
    dialects_to_limit = ID_TO_ACCENT.values()
    MAX_SAMPLES = 150
    
    for dialect in dialects_to_limit:
        dialect_samples = concatenated_df[concatenated_df['dialect'] == dialect]
        if len(dialect_samples) > MAX_SAMPLES:
            print(f"Limiting {dialect} from {len(dialect_samples)} to {MAX_SAMPLES} samples")
            # Get indices of samples to keep (random selection)
            keep_indices = dialect_samples.sample(n=MAX_SAMPLES, random_state=42).index
            # Get indices to drop
            drop_indices = dialect_samples.index.difference(keep_indices)
            # Drop the excess samples
            concatenated_df = concatenated_df.drop(index=drop_indices)
    return concatenated_df


def main():
    # model_path = "best_model_2025-03-16-05-38-01.pt"
    hf_token = os.getenv("HF_TOKEN")
    hf_filename = "best_model_2025-03-16-05-38-01.pt"
    checkpoint_path = hf_hub_download(
        repo_id="Amirjab21/accent-classifier",
        filename=hf_filename,
        token=hf_token
    )

    data = load_all_data()
    model = setup_model(checkpoint_path, num_accent_classes=num_accent_classes)
    model.to(DEVICE)
    
    train_loader, test_loader, train_df = prepare_data(data, ID_TO_ACCENT, 16)

    accuracy, all_preds, all_targets = evaluate(model, train_loader, DEVICE)
    plot_confusion_matrix(all_targets, all_preds, class_names=list(ID_TO_ACCENT.values()))

######################################################################################
###################################### best_model_both.pt ###########################
########################total accuracy 0.90#######################################

if __name__ == "__main__":
    main()