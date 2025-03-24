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
from training.accent_dataset import ContrastiveDataset
import ipdb


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
            target = batch['target'].to(device)
            output = model(mel)
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
    # cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create figure
    plt.figure(figsize=(36, 30))
    
    # Plot with seaborn for better styling
    if class_names is not None:
        # If there are too many classes, use a subset for readability
        # if len(class_names) > 20:
        #     # Find the classes that appear in the data
        #     used_indices = sorted(list(set(y_true) | set(y_pred)))
        #     used_class_names = [class_names[i] for i in used_indices]
        #     # Filter confusion matrix to only include used classes
        #     cm_normalized = cm_normalized[used_indices][:, used_indices]
        #     sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', 
        #                 xticklabels=used_class_names, yticklabels=used_class_names)
        # else:
        sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', 
                        xticklabels=class_names, yticklabels=class_names, annot_kws={"size": 8})
    else:
        sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues')
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()

    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    
    plt.tight_layout()
    # Save the figure
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    print(f"Confusion matrix saved to 'confusion_matrix.png'")

def do_mapping(df):

    category_mapping = {
        # United States
        "United States English": "American",
        "United States English, Midwestern, Low, Demure": "American",
        "United States English, Midwestern USA": "American",
        "United States English, Midwestern": "American",
        "United States English, Southern United States, New Orleans dialect": "American",
        "United States English, Southwestern United States English": "American",
        "United States English, CARIBBEAN AND BRITISH MIXED WITH SOME NEW YORK ACCENTS": "American",
        "United States English, southern midwest hippie english, grateful deadhead": "American",
        "United States English, Upstate New York": "American",
        "United States English, Like a Kennedy, Boston": "American",
        "United States English, Washington State": "American",
        "United States English, New York English": "American",
        "United States English, New Yorker": "American",
        "United States English, South Texas, Slightly effeminate, Conversational": "American",
        "United States English, southern draw": "American",
        "United States English, Delaware Valley, Pennsylvania": "American",
        "United States English, Southern Californian English": "American",
        "United States English, North Indiana": "American",
        "United States English, California, Valley": "American",
        "United States English, Pennsylvanian, Neutral": "American",
        "United States English, Texas": "American",
        "United States English, California English, Southern US English": "American",
        "United States English, Southern Ohio English": "American",
        "United States English, Mid-west United States English": "American",
        "United States English, Pacific North West United States": "American",
        "United States English, Midwestern US English (United States)": "American",
        "American South East Georgia Dialect": "American",
        "American Kansas": "American",
        
        # England
        "England English": "English",
        "England English, Well spoken softly spoken Home Counties gay, softly spoken male, well spoken male, gay male": "English",
        "England English, south German / Swiss accent": "English",
        "England English, yorkshire": "English",
        "England English, femalepublic school accent, quiet and under articulated finals": "English",
        "England English, southern UK, male": "English",
        "England English, southern english, sussex": "English",
        "England English, Very British": "English",
        "England English, Slightly lazy Midlands English, Oxford English": "English",
        "England English, Received Pronunciation": "English",
        "British English / Received Pronunciation (RP)": "English",
        "British accent": "English",
        "British English - North West": "English",
        
        # Canada
        "Canadian English": "Canadian",
        "Canadian English, Ontario": "Canadian",
        
        # Australia
        "Australian English": "AustralianEnglish",
        "Educated Australian Accent": "AustralianEnglish",
        "South Australia": "AustralianEnglish",
        
        # New Zealand
        "New Zealand English": "NewZealandEnglish",
        
        # Ireland
        "Irish English": "Irish",
        "Northern Irish": "Irish",
        "Northern Irish English": "Irish",
        "Irish English, Northern Irish English, belfast": "Irish",
        
        # Scotland
        "Scottish English": "Scottish",
        "Scottish English (West Coast), Scottish English (Ayrshire)": "Scottish",
        
        # Wales
        "Welsh English": "Welsh",
        "Welsh English, Shropshire": "Welsh",
        
        # South Africa
        "South African English Accent": "SouthAfrican",
        "South African accent": "SouthAfrican",
        "2nd language, South African (first language Afrikaans)": "SouthAfrican",
        
        # Southern Africa (South Africa, Zimbabwe, Namibia)
        "Southern African (South Africa, Zimbabwe, Namibia)": "SouthAfrican",
        
        # India and South Asia (India, Pakistan, Sri Lanka)
        "India and South Asia (India, Pakistan, Sri Lanka)": "Indian",
        "Indian English": "Indian",
        "International Indian Accent": "Indian",
        
        # Philippines
        "Filipino": "Philippines",
        
        # Hong Kong
        "Hong Kong English": "Hong Kong",
        
        # Malaysia
        "Malaysian English": "Malaysia",
        
        # Singapore
        "Singaporean English": "Singapore",
        
        # Japan
        "Japanese English": "Japan",
        "Japan English": "Japan",
        "Japanese": "Japan",
        
        # South Korea
        "Korean": "South Korea",
        
        # China
        "Chinese English": "China",
        "Chinese": "China",
        "chinese accent": "China",
        "Chinese accent of English": "China",
        "Chinese-English": "China",
        
        # Vietnam
        "Vietnam": "Vietnam",
        
        # Thailand
        "Thai": "Thailand",
        
        # Indonesia
        "Indonesian": "Indonesia",
        "indonesia": "Indonesia",
        
        # Nepal
        "Nepali": "Nepal",
        
        # Bangladesh
        "Bangladeshi": "Bangladesh",
        
        # Nigeria
        "nigeria english": "Nigeria",
        "Nigerian English": "Nigeria",
        "Nigerian": "Nigeria",
        
        # Kenya
        "Kenyan English": "Kenya",
        "Kenyan English accent": "Kenya",
        "Kenyan": "Kenya",
        
        # Ghana
        "Ghanaian english Accent, african regular reader": "Ghana",
        
        # East Africa
        "East African Khoja": "East Africa",
        
        # West Indies and Bermuda (Bahamas, Bermuda, Jamaica, Trinidad)
        "West Indies and Bermuda (Bahamas, Bermuda, Jamaica, Trinidad)": "West Indies and Bermuda",
        
        # Germany
        "German": "Germany",
        "German English": "Germany",
        
        # France
        "French": "France",

        # Spain
        "spanish english": "Spain",
        "Spanish bilinguals": "Spain",
        "Spanish": "Spain",
        "Spanglish, England English": "Spain",
        "Spanish accent": "Spain",
        "Spanish from the Canary Islands": "Spain",

        
        # Italy
        "Italian": "Italy",
        
        # Netherlands
        "Dutch": "Netherlands",
        "Slight Dutch accent": "Netherlands",
        
        # Austria
        "Austrian": "Austria",
        "United States English, American english with austrian accent": "Austria",
        
        # Sweden
        "Swedish English": "Sweden",
        "Swedish accent": "Sweden",
        
        # Finland
        "Finnish": "Finland",
        
        # Norway
        "Northern Irish, Norwegian, yorkshire": "Norway",
        
        # Denmark
        "United States English, Irish English, England English, Scottish English, Danish English": "Denmark",
        
        # Russia
        "Russian": "Russia",
        "Russian Accent": "Russia",
        
        # Ukraine
        "Ukrainian": "Ukraine",
        
        # Poland
        "polish": "Poland",
        "Slavic, polish": "Poland",
        "Slavic, East European, polish": "Poland",
        
        # Czech Republic
        "Czech": "Czech Republic",
        
        # Slovakia
        "Slovak": "Slovakia",
        
        # Hungary
        "Hungarian": "Hungary",
        
        # Romania
        "Eastern European, Romanian": "Romania",
        
        # Bulgaria
        "Bulgarian": "Bulgaria",
        
        # Croatia
        "Croatian English": "Croatia",
        
        # Greece
        "Greek": "Greece",
        
        # Turkey
        "Turkish": "Turkey",
        
        # Israel
        "Israeli": "Israel",
        
        # Brazil
        "Brazilian": "Brazil",
        "Brazillian Accent": "Brazil",
        
        # Mexico/Central America
        "Central American, United States English": "Mexico/Central America",
        "little latino, United States English, second language": "Mexico/Central America",
        
        # Kazakhstan
        "Kazakhstan English": "Kazakhstan",
        
        # Latvia
        "strong Latvian accent": "Latvia",
        "Latvian": "Latvia",

    }


    
    # Apply the mapping to the 'accents' column
    df['dialect'] = df['dialect'].map(category_mapping).fillna(df['dialect'])
    unique_accents = list(set(category_mapping.values()))


    return df, unique_accents



def load_all_data():
    if DEVICE == "cuda":
        new_dataset, unique_accents = load_new_dataset(['Amirjab21/commonvoice'])
    else:
        # Read all parquet files in dataframes directory and concatenate them
        dataframes_dir = Path("dataframes")
        parquet_files = list(dataframes_dir.glob("*.parquet"))
        
        if parquet_files:
            dataframes = pd.read_parquet(parquet_files[0])
            new_dataset = dataframes
            print(len(new_dataset), 'new dataset length')
            # new_dataset = pd.concat(dataframes, ignore_index=True)
            # new_dataset.drop(columns=['down_votes', 'quality_check', '__index_level_0__'], inplace=True)
            new_dataset.rename(columns={'accents': 'dialect', 'sentence': 'text'}, inplace=True)
            print(f"Loaded {len(parquet_files)} parquet files from dataframes directory")
            new_dataset, unique_accents = do_mapping(new_dataset)
        else:
            # Fallback to CSV if no parquet files found
            new_dataset = pd.read_csv("dataframes/commonvoice_dataframe.csv")
            print("No parquet files found, loaded CSV file instead")

    dataset_df = load_datasets(sample_size=1000)
    concatenated_df = pd.concat([ new_dataset, dataset_df], ignore_index=True)
    dialects_to_limit = ID_TO_ACCENT.values()
    MAX_SAMPLES = 100
    
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

def prepare_data(dataset_df,ID_TO_ACCENT, batch_size=16):
    train_size = int(0.98 * len(dataset_df))
    test_size = len(dataset_df) - train_size
    
    train_df, test_df = torch.utils.data.random_split(dataset_df, [train_size, test_size])
    train_df = train_df.dataset
    test_df = test_df.dataset
    
    # Create datasets
    train_dataset = ContrastiveDataset(train_df, ID_TO_ACCENT, device=DEVICE)
    test_dataset = ContrastiveDataset(test_df, ID_TO_ACCENT, device=DEVICE)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=6 if torch.cuda.is_available() else 0, 
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=6 if torch.cuda.is_available() else 0, 
        pin_memory=True
    )
    
    return train_loader, test_loader, train_df


def main():
    # model_path = "best_model_2025-03-16-05-38-01.pt"
    hf_token = os.getenv("HF_TOKEN")
    hf_filename = "best_model_2025-03-16-05-38-01.pt"
    if DEVICE == "cuda":
        checkpoint_path = hf_hub_download(
            repo_id="Amirjab21/accent-classifier",
            filename=hf_filename,
            token=hf_token
        )
    else:
        checkpoint_path = "best_model_2025-03-16-05-38-01.pt"

    data = load_all_data()
    data = data.dropna(subset=['text'])
    model = setup_model(checkpoint_path, num_accent_classes=num_accent_classes)
    model.to(DEVICE)
    
    train_loader, test_loader, train_df = prepare_data(data, ID_TO_ACCENT, 16)
    ipdb.set_trace()
    accuracy, all_preds, all_targets = evaluate(model, train_loader, DEVICE)
    plot_confusion_matrix(all_targets, all_preds, class_names=list(ID_TO_ACCENT.values()))

######################################################################################
###################################### best_model_both.pt ###########################
########################total accuracy 0.90#######################################

if __name__ == "__main__":
    main()