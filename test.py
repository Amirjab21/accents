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
from operator import attrgetter
from tqdm import tqdm
import wandb
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
import pandas as pd
from datasets import Audio, concatenate_datasets, load_dataset
import io
from scipy.io import wavfile
import librosa


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
wer_metric = load("wer")
model_variant = "small"
name_of_run = "finetune-decoder-only and compare english scores to scottish"
margin = 0.13
num_accent_classes = 2
id_to_accent = {0: "scottish", 1: "southern"}
lora = True
finetuned_model_path = "best_model0.04.pt"




base_whisper_model = load_model(model_variant, device=DEVICE)
# dataset_hf_scottish =load_dataset("ylacombe/english_dialects", "scottish_male")
# dataset_hf_scottish['train'] = dataset_hf_scottish['train'].cast_column("audio", Audio(sampling_rate=16000))
# dataset_hf_scottish_women = load_dataset("ylacombe/english_dialects", "scottish_female")
# dataset_hf_scottish_women['train'] = dataset_hf_scottish_women['train'].cast_column("audio", Audio(sampling_rate=16000))
# dataset_hf_southern =load_dataset("ylacombe/english_dialects", "southern_male")
# dataset_hf_southern['train'] = dataset_hf_southern['train'].cast_column("audio", Audio(sampling_rate=16000))
# dataset_hf_southern_women = load_dataset("ylacombe/english_dialects", "southern_female")
# dataset_hf_southern_women['train'] = dataset_hf_southern_women['train'].cast_column("audio", Audio(sampling_rate=16000))

# scottish_dataset = dataset_hf_scottish['train'].add_column("dialect", ["scottish"] * len(dataset_hf_scottish['train']))
# scottish_dataset_women = dataset_hf_scottish_women['train'].add_column("dialect", ["scottish"] * len(dataset_hf_scottish_women['train']))
# # Prepare Southern dataset with source column
# southern_dataset = dataset_hf_southern['train'].add_column("dialect", ["southern"] * len(dataset_hf_southern['train']))
# southern_dataset_women = dataset_hf_southern_women['train'].add_column("dialect", ["southern"] * len(dataset_hf_southern_women['train']))

# # Combine the datasets
# combined_dataset = concatenate_datasets([scottish_dataset, southern_dataset, scottish_dataset_women, southern_dataset_women])

# # Convert to pandas dataframe
# df = combined_dataset.to_pandas()

# # Get line_ids for each dialect
# scottish_line_ids = set(df[df['dialect'] == 'scottish']['line_id'])
# southern_line_ids = set(df[df['dialect'] == 'southern']['line_id'])

# # Find line_ids that appear in both dialects
# matching_line_ids = scottish_line_ids.intersection(southern_line_ids)

# # Filter dataframe to only include rows with matching line_ids
# matched_df = df[df['line_id'].isin(matching_line_ids)]

# # Sort by line_id to make it easier to compare
# matched_df = matched_df.sort_values(['line_id', 'dialect'])

class ModifiedWhisper(torch.nn.Module):
    def __init__(self, dims: int, num_accent_classes: int, whisper: Whisper):
        super().__init__()
        self.dims = dims
        self.whisper = whisper
        self.accent_classifier = torch.nn.Linear(self.dims.n_text_state, num_accent_classes)
    
    def forward(self, mel: torch.Tensor, tokens: torch.Tensor):
        encoder_output = self.whisper.encoder(mel)
        #in the future, we could calculate a score for every timestep
        pooled_output = torch.mean(encoder_output, dim=1)
        
        accent_output = self.accent_classifier(pooled_output)
        return accent_output









def bytes_to_array(audio_bytes):
    # Create a BytesIO object from the bytes
    byte_io = io.BytesIO(audio_bytes['bytes'])
    
    # Read the WAV file from BytesIO
    sample_rate, audio_array = wavfile.read(byte_io)
    
    # Convert to float32 and normalize to [-1, 1]
    audio_array = audio_array.astype(np.float32) / 32768.0
    
    return sample_rate, audio_array

class ContrastiveDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, device=DEVICE, padding_token_id=50257):
        self.device = device
        self.processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="English", task="transcribe")
        self.feature_extractor = self.processor.feature_extractor
        self.tokenizer = self.processor.tokenizer
        self.padding_token_id = padding_token_id
        
        # Get all wav files and their corresponding txt files
        data_path = Path(data_dir)
        wav_files = list(data_path.glob("*.wav"))[:90]  # Limit to 10 files
        self.samples = []
        
        for wav_file in wav_files:
            txt_file = wav_file.with_suffix('.txt')
            if txt_file.exists():
                self.samples.append({
                    'wav_path': str(wav_file),
                    'txt_path': str(txt_file)
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load audio file
        sample_rate, audio_array = wavfile.read(sample['wav_path'])
        # Convert to float32 and normalize to [-1, 1]
        audio_array = audio_array.astype(np.float32) / 32768.0
        # print(sample_rate, 'sample_rate')
        if sample_rate != 16000:
            audio_array = librosa.core.resample(
                y=audio_array,
                orig_sr=sample_rate,
                target_sr=16000
            )
        # print(audio_array, 'audio_array.shape')
        
        # Load text file
        with open(sample['txt_path'], 'r') as f:
            text = f.read().strip()
        
        # Process audio and text
        mel = self.feature_extractor(audio_array, sampling_rate=16000)
        text_tokens = self.tokenizer(text, 
                                   return_tensors="pt", 
                                   padding="max_length", 
                                   truncation=True, 
                                   max_length=400)
        
        # Since this is Scottish data, we'll use 0 as the target (based on id_to_accent mapping)
        target = torch.tensor(0, dtype=torch.long)  # 0 for Scottish

        return {
            'mel': mel.input_features.squeeze(0),
            'text': text_tokens.input_ids.squeeze(0),
            'original': text,
            'target': target
        }


model = ModifiedWhisper(base_whisper_model.dims, num_accent_classes, base_whisper_model)

# peft_config = LoraConfig(
#      inference_mode=False, r=8,target_modules=["out", "token_embedding", "query", "key", "value", "proj_out"] , lora_alpha=32, lora_dropout=0.1
# )
# model = get_peft_model(model, peft_config)
# model.print_trainable_parameters()

if lora:
    print('lora', lora)
    peft_config = LoraConfig(
        inference_mode=False, r=8, 
        target_modules=["out", "token_embedding", "query", "key", "value", "proj_out"],
        lora_alpha=32, lora_dropout=0.1
    )
    model = get_peft_model(model, peft_config)
    checkpoint_path = str(Path(__file__).parent / finetuned_model_path)
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    try:
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print('Model loaded successfully')
    except Exception as e:
        print(f"Error loading state dict:{e}")
elif finetuned_model_path is not None:
        print('no lora')
        # model = get_peft_model(model, peft_config)
        checkpoint_path = str(Path(__file__).parent / finetuned_model_path)
        print(checkpoint_path, 'checkpoint path')
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        # print(checkpoint, 'checkpoint')
        try:
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print('Model loaded successfully')
        except Exception as e:
            print(f"Error loading state dict:{e}")
else:
    print('standard whisper model')


dataset = ContrastiveDataset("data/scottish/p262")
# dataset = ContrastiveDataset("data/english/p268")

test_loader = DataLoader(dataset, batch_size=10, shuffle=True)


def evaluate(model, dataloader):
    model.eval()
    total_loss = 0
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch in dataloader:
            # Get embeddings for both Scottish and English samples
            mel = batch['mel'].to(DEVICE)
            text = batch['text'].to(DEVICE)
            target = batch['target'].to(DEVICE)
            output = model(mel, text)
            print(output.shape, 'output.shape')
            print(output, 'output')
            probabilities = torch.nn.functional.softmax(output, dim=1)
            print(probabilities, 'probabilities')  # This will show values between 0 and 1 that sum to 1

            predictions = torch.argmax(probabilities, dim=1)
            scottish_count = (predictions == 0).sum().item()
            
            print(probabilities, 'probabilities')
            print(f"Number of samples classified as Scottish: {scottish_count} out of {len(predictions)}")


    
val_loss = evaluate(model, test_loader)



