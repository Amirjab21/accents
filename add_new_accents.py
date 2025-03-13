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
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze the classifier layer
    for param in model.base_model.model.accent_classifier.parameters():
        param.requires_grad = True
    

    # Initialize new weights for the new classes
    torch.nn.init.xavier_uniform_(model.accent_classifier.weight[number_existing_accents:, :])
    torch.nn.init.zeros_(model.accent_classifier.bias[number_existing_accents:])

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
        df.drop(columns=['down_votes', 'quality_check', '__index_level_0__'], inplace=True)
        df.rename(columns={'accents': 'dialect', 'sentence': 'text'}, inplace=True)
        all_dfs.append(df)
    
    combined_df = pd.concat(all_dfs, ignore_index=True)

    if DEVICE != "cuda":
        combined_df = combined_df.sample(n=100)
    return combined_df





def main():
    # Load and prepare data
    dataset_df = load_datasets(sample_size=100)
    train_loader, test_loader, train_df = prepare_data(dataset_df, ID_TO_ACCENT)
    class_weights = calculate_class_weights(train_df, NUM_ACCENT_CLASSES, ID_TO_ACCENT)
    
    # Setup model and optimizer
    model = setup_model(checkpoint_path)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    avg_loss, accuracy = evaluate(model, test_loader, class_weights, DEVICE, save_model=False)

    with open('results.txt', 'a') as f:
        f.write(f"INITIAL EVALUATION on existing weights\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")

    new_accents = ['Scottish', 'NewZealandEnglish', 'Japanese', 'Canadian', 'AustralianEnglish', 'Irish', 'American', 'SouthAfrican', 'Indian', 'English', 'German', 'Filipino', 'Korean']
    model, new_id_to_accent, new_number_accents = add_new_accents(model, NUM_ACCENT_CLASSES, ID_TO_ACCENT, new_accents)


    new_dataset = load_new_dataset(['Amirjab21/commonvoice'])

    # concatenated_df = pd.concat([dataset_df, new_dataset], ignore_index=True)

    #First, well try to just use the new data to see if training this alone can help.
    #If this doesnt work, you have to prepare_data with the concatented df above.
    train_loader_new_only, test_loader_new_only, train_df_new_only = prepare_data(new_dataset, new_id_to_accent)
    # train_loader, test_loader, train_df = prepare_data(concatenated_df, new_id_to_accent)

    class_weights_new = calculate_class_weights(train_df_new_only, new_number_accents, new_id_to_accent)

    model = train_model(
        model=model,
        train_loader=train_loader_new_only,
        test_loader=test_loader_new_only,
        optimizer=optimizer,
        class_weights=class_weights_new,
        device=DEVICE,
        number_epochs=2,
        run_name="add corp20 to training set"
    )

    model = setup_model('best_model.pt', new_number_accents)

    with open('results.txt', 'a') as f:
        f.write(f"ADD CORP20 TO TRAINING SET\n")
        f.write(f"mapping: {new_id_to_accent}\n")
        f.write(f"number of accents: {new_number_accents}\n")

    #final eval
    avg_loss, accuracy = evaluate(model, test_loader, class_weights_new, DEVICE, save_model=False)
    with open('results.txt', 'a') as f:
        f.write(f"FINAL EVALUATION on new weights\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
    

    api = HfApi()

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

