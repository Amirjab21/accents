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
# import ipdb


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_VARIANT = "small"
NUM_ACCENT_CLASSES = 11
ID_TO_ACCENT = {
    0: "Scottish", 1: "English", 2: "Indian", 3: "Irish", 4: "Welsh",
    5: "NewZealandEnglish", 6: "AustralianEnglish", 7: "SouthAfrican",
    8: "Canadian", 9: "NorthernIrish", 10: "American"
}


checkpoint_file = "model_11_accents_epoch_2.pt"

def load_datasets(sample_size=None):
    # Load English dialects datasets
    dataset_hf_scottish = load_dataset("ylacombe/english_dialects", "scottish_male")
    dataset_hf_scottish['train'] = dataset_hf_scottish['train'].cast_column("audio", Audio(sampling_rate=16000))
    dataset_hf_scottish_women = load_dataset("ylacombe/english_dialects", "scottish_female")
    dataset_hf_scottish_women['train'] = dataset_hf_scottish_women['train'].cast_column("audio", Audio(sampling_rate=16000))
    dataset_hf_southern = load_dataset("ylacombe/english_dialects", "southern_male")
    dataset_hf_southern['train'] = dataset_hf_southern['train'].cast_column("audio", Audio(sampling_rate=16000))
    dataset_hf_southern_women = load_dataset("ylacombe/english_dialects", "southern_female")
    dataset_hf_southern_women['train'] = dataset_hf_southern_women['train'].cast_column("audio", Audio(sampling_rate=16000))

    # Add dialect labels
    scottish_dataset = dataset_hf_scottish['train'].add_column("dialect", ["Scottish"] * len(dataset_hf_scottish['train']))
    scottish_dataset_women = dataset_hf_scottish_women['train'].add_column("dialect", ["Scottish"] * len(dataset_hf_scottish_women['train']))
    southern_dataset = dataset_hf_southern['train'].add_column("dialect", ["English"] * len(dataset_hf_southern['train']))
    southern_dataset_women = dataset_hf_southern_women['train'].add_column("dialect", ["English"] * len(dataset_hf_southern_women['train']))

    # Combine all datasets
    combined_dataset = concatenate_datasets([scottish_dataset, southern_dataset, scottish_dataset_women, southern_dataset_women])
    df = combined_dataset.to_pandas()

    # Load VCTK dataset shards
    shard1 = load_dataset("Amirjab21/vctk-accents", data_files="accent_dataset_shard_1.parquet", split="train")
    shard2 = load_dataset("Amirjab21/vctk-accents", data_files="accent_dataset_shard_2.parquet", split="train")
    shard3 = load_dataset("Amirjab21/vctk-accents", data_files="accent_dataset_shard_3.parquet", split="train")
    combined_vctk = concatenate_datasets([shard1, shard2, shard3])
    df_vctk = combined_vctk.to_pandas()

    # Combine all data
    both_datasets = pd.concat([df, df_vctk])
    
    if DEVICE != "cuda":
        both_datasets = both_datasets.sample(n=sample_size)
    
    return both_datasets

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

def calculate_class_weights(train_df, NUM_ACCENT_CLASSES, ID_TO_ACCENT):
    class_counts = {i: 0 for i in range(NUM_ACCENT_CLASSES)}
    dialect_to_id = {v: k for k, v in ID_TO_ACCENT.items()}
    for dialect in train_df['dialect']:
        print(dialect, 'dialect')
        dialect_id = dialect_to_id[dialect]
        class_counts[dialect_id] += 1

    total_samples = len(train_df)
    class_weights = torch.FloatTensor([
        total_samples / (NUM_ACCENT_CLASSES * count) if count > 0 else 0.0 
        for count in [class_counts[i] for i in range(NUM_ACCENT_CLASSES)]
    ])
    
    return class_weights.to(DEVICE)

def setup_model(checkpoint_file = None, num_accent_classes=NUM_ACCENT_CLASSES):
    base_whisper_model = load_model(MODEL_VARIANT, device=DEVICE)
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


def main():
    # Load and prepare data
    dataset_df = load_datasets()
    train_loader, test_loader, train_df = prepare_data(dataset_df)
    class_weights = calculate_class_weights(train_df, NUM_ACCENT_CLASSES)
    
    # Setup model and optimizer
    model = setup_model(None)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    avg_loss, accuracy = evaluate(model, test_loader, class_weights, DEVICE, save_model=False)

        # Write results to file
    with open('results.txt', 'w') as f:
        f.write(f"INITIAL EVALUATION on existing weights\n")
        f.write(f"Average Loss: {avg_loss:.4f}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")

    

    # Train the model
    model = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        class_weights=class_weights,
        device=DEVICE,
        number_epochs=1,
        run_name="finetune-decoder-only and compare english scores to scottish"
    )
    

if __name__ == "__main__":
    main()
