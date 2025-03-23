import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import Audio, concatenate_datasets, load_dataset
import pandas as pd
from peft import get_peft_config, get_peft_model, LoraConfig
from model.load_model import load_model
from model.accent_model import ModifiedWhisper
from training.accent_dataset import ContrastiveDataset
from training.train_utils import train_model
from pathlib import Path
from training.train_utils import evaluate
from train import load_datasets, prepare_data, calculate_class_weights, setup_model
from huggingface_hub import HfApi
import os
from datetime import datetime
from huggingface_hub import hf_hub_download
import datasets
datasets.disable_caching()

# export HF_TOKEN=hf_token
#wandb login prior to script

hf_token = os.getenv("HF_TOKEN")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_VARIANT = "small"
NUM_ACCENT_CLASSES = 11
ID_TO_ACCENT = {
    0: "Scottish", 1: "English", 2: "Indian", 3: "Irish", 4: "Welsh",
    5: "NewZealandEnglish", 6: "AustralianEnglish", 7: "SouthAfrican",
    8: "Canadian", 9: "NorthernIrish", 10: "American"
}


hf_filename = "model_11_accents_epoch_2.pt"
checkpoint_path = hf_hub_download(
    repo_id="Amirjab21/accent-classifier",
    filename=hf_filename,
    token=hf_token
)


def add_new_accents(model, number_existing_accents, existing_mapping, accent_names):
    
    number_of_new_accents = len(accent_names)
    new_id_to_accent = existing_mapping.copy()  # Create a copy of existing mapping

    # Check for existing accents
    existing_accents = set(existing_mapping.values())
    filtered_accents = [accent for accent in accent_names if accent not in existing_accents]

    if len(filtered_accents) != len(accent_names):
        skipped_accents = set(accent_names) - set(filtered_accents)
        print(f"Warning: Skipping already existing accents: {skipped_accents}")

    number_of_new_accents = len(filtered_accents)
    for i, accent in enumerate(filtered_accents):
        new_index = number_existing_accents + i
        new_id_to_accent[new_index] = accent

    print(new_id_to_accent, 'new mapping')

    new_number_accents = number_existing_accents + number_of_new_accents  # e.g., 11 + 5 = 16
    model.base_model.model.accent_classifier = torch.nn.Linear(model.dims.n_text_state, new_number_accents)

    # Freeze all layers except the classifier
    # for param in model.parameters():
    #     param.requires_grad = False
    
    # # Unfreeze the classifier layer
    # for param in model.base_model.model.accent_classifier.parameters():
    #     param.requires_grad = True
    

    # # Initialize new weights for the new classes
    torch.nn.init.xavier_uniform_(model.base_model.model.accent_classifier.weight[number_existing_accents:, :])
    torch.nn.init.zeros_(model.base_model.model.accent_classifier.bias[number_existing_accents:])

    # Define optimizer with different learning rates (optional)
    # optimizer = torch.optim.Adam([
    #     {"params": model.base_model.parameters(), "lr": 1e-5},  # Low LR for pretrained layers
    #     {"params": model.classifier.parameters(), "lr": 1e-3}   # Higher LR for new output layer
    # ])
    return model, new_id_to_accent, new_number_accents


def load_new_dataset(dataset_names):
    all_dfs = []

    for dataset_name in dataset_names:
        dataset = load_dataset(dataset_name)
        df = dataset['train'].to_pandas()
        temporary = df['accents'].value_counts()
        print(temporary, 'temporary')
        df.drop(columns=['down_votes', 'quality_check', '__index_level_0__'], inplace=True)
        df.rename(columns={'accents': 'dialect', 'sentence': 'text'}, inplace=True)
        all_dfs.append(df)
    
    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_df, unique_accents = do_mapping(combined_df)

    if DEVICE != "cuda":
        combined_df = combined_df.sample(n=100)
    return combined_df, unique_accents

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





def main():


    # Load and prepare data
    new_dataset, unique_accents = load_new_dataset(['Amirjab21/commonvoice'])
    # print(len(new_dataset), 'length of new dataset')
    # columns = new_dataset.columns
    # print(columns, 'columns')
    # unique_accents = new_dataset['dialect'].value_counts()
    # print(unique_accents, 'unique accents')
    dataset_df = load_datasets(sample_size=100)
    # train_loader, test_loader, train_df = prepare_data(dataset_df, ID_TO_ACCENT)
    # class_weights = calculate_class_weights(train_df, NUM_ACCENT_CLASSES, ID_TO_ACCENT)
    
    # Setup model and optimizer
    model = setup_model(checkpoint_path)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


    
    # new_accents = ['Scottish', 'American', 'SouthAfrican', 'Indian', 'China',
    #    'Canadian', 'English', 'Singapore', 'AustralianEnglish',
    #    'Philippines', 'Welsh', 'Germany', 'West Indies and Bermuda',
    #    'Irish', 'Greece', 'Israel', 'Ukraine', 'Malaysia', 'Brazil',
    #    'Hong Kong', 'NewZealandEnglish', 'Turkey', 'Poland', 'Thailand',
    #    'Sweden', 'Nepal', 'Netherlands', 'Kenya', 'Bangladesh', 'Russia',
    #    'Finland', 'Italy', 'France', 'Hungary', 'Austria', 'Spain',
    #    'Nigeria', 'Czech Republic', 'Indonesia', 'South']
    model, new_id_to_accent, new_number_accents = add_new_accents(model, NUM_ACCENT_CLASSES, ID_TO_ACCENT, unique_accents)


    # new_dataset = load_new_dataset(['Amirjab21/commonvoice'])

    subsample_of_old_data = dataset_df

    concatenated_df = pd.concat([subsample_of_old_data, new_dataset], ignore_index=True)
    print(len(concatenated_df), 'length of concatenated df')

    dialects_to_limit = ["American", "SouthAfrican", "English", "Scottish", "Canadian", "Philippines"]
    MAX_SAMPLES = 10000
    
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

    all_accents = concatenated_df['dialect'].value_counts()
    print(all_accents, "all accents")
    #First, well try to just use the new data to see if training this alone can help.
    #If this doesnt work, you have to prepare_data with the concatented df above.
    if DEVICE != 'cuda':
        concatenated_df = concatenated_df.sample(n=100)

        
    train_loader, test_loader, train_df = prepare_data(concatenated_df, new_id_to_accent)
    # train_loader_both, test_loader_both, train_df_both = prepare_data(concatenated_df, new_id_to_accent)

    class_weights_new = calculate_class_weights(train_df, new_number_accents, new_id_to_accent)

    with open('results.txt', 'a') as f:
        f.write(f"ADD 39 ACCENTS TO TRAINING SET\n")
        f.write(f"mapping: {new_id_to_accent}\n")
        f.write(f"number of accents: {new_number_accents}\n")
    api = HfApi()
    api.upload_file(
            path_or_fileobj="results.txt",
            path_in_repo=f"results/results_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}initial.txt",
            repo_id="Amirjab21/accent-classifier",
            token=hf_token  # Replace with your actual token
    )

    
    
    model.to(DEVICE)
    model = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        class_weights=class_weights_new,
        device=DEVICE,
        number_epochs=3,
        run_name="add 39 accents to training set"
    )

    # model = setup_model('best_model.pt', new_number_accents)
    

    #final eval
    avg_loss, accuracy = evaluate(model, test_loader, class_weights_new, DEVICE, save_model=False)
    with open('results.txt', 'a') as f:
        f.write(f"FINAL EVALUATION on new weights\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
    

    

    # Upload the model file
    if DEVICE == "cuda":
        api.upload_file(
            path_or_fileobj="best_model.pt",
            path_in_repo=f"best_model_{datetime.now().strftime('%Y-%m-%d')}.pt",
            repo_id="Amirjab21/accent-classifier",
            token=hf_token  # Replace with your actual token
        )
        api.upload_file(
            path_or_fileobj="results.txt",
            path_in_repo=f"results/results_{datetime.now().strftime('%Y-%m-%d')}.txt",
            repo_id="Amirjab21/accent-classifier",
            token=hf_token  # Replace with your actual token
        )

if __name__ == "__main__":
    main()

