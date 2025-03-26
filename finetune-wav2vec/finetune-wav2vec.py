import numpy as np
import torch
import json
from torch.utils.data import DataLoader
from evaluate import load
from datasets import Audio, concatenate_datasets, load_dataset
from tqdm import tqdm
import wandb
# from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from transformers import AutoProcessor, Wav2Vec2BertForCTC
import io
from scipy.io import wavfile
import librosa
import os
from datetime import datetime
from peft import get_peft_config, get_peft_model, LoraConfig
from huggingface_hub import HfApi
from pathlib import Path


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
wer_metric = load("wer")
processor = AutoProcessor.from_pretrained("hf-audio/wav2vec2-bert-CV16-en")
model = Wav2Vec2BertForCTC.from_pretrained("hf-audio/wav2vec2-bert-CV16-en")

def bytes_to_array(audio_bytes):
    # Create a BytesIO object from the bytes
    byte_data = audio_bytes['bytes'] if isinstance(audio_bytes, dict) else audio_bytes
    
    # Create a BytesIO object from the bytes
    byte_io = io.BytesIO(byte_data)
    
    # Read the WAV file from BytesIO
    sample_rate, audio_array = wavfile.read(byte_io)
    
    audio_array = audio_array.astype(np.float32) / 32768.0
    if sample_rate != 16000:
        audio_array = librosa.core.resample(
            y=audio_array,
            orig_sr=sample_rate,
            target_sr=16000
        )
    
    return 16000, audio_array

def load_datasets():
    # Load English dialects datasets
    dataset_hf_scottish = load_dataset("ylacombe/english_dialects", "scottish_male")
    dataset_hf_scottish['train'] = dataset_hf_scottish['train'].cast_column("audio", Audio(sampling_rate=16000))
    dataset_hf_scottish_women = load_dataset("ylacombe/english_dialects", "scottish_female")
    dataset_hf_scottish_women['train'] = dataset_hf_scottish_women['train'].cast_column("audio", Audio(sampling_rate=16000))
    scottish_dataset = dataset_hf_scottish['train'].add_column("dialect", ["Scottish"] * len(dataset_hf_scottish['train']))
    scottish_dataset_women = dataset_hf_scottish_women['train'].add_column("dialect", ["Scottish"] * len(dataset_hf_scottish_women['train']))


    combined_dataset = concatenate_datasets([scottish_dataset, scottish_dataset_women])
    df = combined_dataset.to_pandas()

    if DEVICE != "cuda":
        df = df.sample(n=100)
    
    return df




class Dataset(torch.utils.data.Dataset):
    """
    A simple class to wrap LibriSpeech and trim/pad the audio to 30 seconds.
    It will drop the last few seconds of a very small portion of the utterances.
    """
    def __init__(self, dataset, device=DEVICE, padding_token_id=0, processor=processor):
        self.dataset = dataset
        self.processor = processor
        self.device = device
        self.feature_extractor = self.processor.feature_extractor
        self.tokenizer = self.processor.tokenizer
        self.padding_token_id = padding_token_id

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset.iloc[idx]
        audio = data['audio']
        sample_rate, audio_array = bytes_to_array(audio)
        text = data['text']
        
        # input_values = self.processor.feature_extractor(audio_array, sampling_rate=sample_rate, return_tensors="pt", truncation=True, padding="max_length", max_length=200)
        inputs = self.processor(audio_array, sampling_rate=sample_rate, return_tensors="pt", padding="max_length", max_length=300, truncation=True)
        
        
        audio_mel = inputs.input_features
        
        # Normalize text to ensure it's clean
        text = text.strip().lower()
        
        inputs["labels"] = self.processor(text=text, return_tensors="pt", padding="max_length", max_length=200, truncation=True, padding_token_id=self.padding_token_id).input_ids
        labels = inputs["labels"]
        labels = labels.masked_fill(labels.eq(0), -100)
        
        return {
            'mel': audio_mel.squeeze(0), 
            'labels': labels.squeeze(0), 
            'text': data['text']
        }


def train_model(model, train_loader, optimizer, device, number_epochs=1, run_name="accent-wav2vec"):
    if device == "cuda":
        wandb.init(project="finetune-wav2vec-scottish", name=run_name)
    model.train()
    model = model.to(device)  # Ensure model is on the correct device

    for epoch in range(number_epochs):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{number_epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            optimizer.zero_grad()
            mel = batch['mel'].to(device)
            labels = batch['labels'].to(device)

            loss = model(mel, labels=labels).loss

            progress_bar.set_postfix({"loss": loss.item()})
            
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            if device == "cuda":
                wandb.log({"batch loss": loss.item()})
            total_loss += loss.item()


def setup_model(model, checkpoint_file=None):

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


def main():
    df = load_datasets()
    processor = AutoProcessor.from_pretrained("hf-audio/wav2vec2-bert-CV16-en")
    model = Wav2Vec2BertForCTC.from_pretrained("hf-audio/wav2vec2-bert-CV16-en")
    model = setup_model(model, checkpoint_file=None)
    batch_size = 16

    dataset = Dataset(df,device=DEVICE, padding_token_id=-100, processor=processor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # model.train()
    # model = model.to(DEVICE)
    # criterion = torch.nn.CrossEntropyLoss()

    train_model(model, dataloader, optimizer, DEVICE)

    # for batch in dataloader:
    #     labels = batch['labels'].to(DEVICE)
    #     mel = batch['mel'].to(DEVICE)
    #     loss = model(mel, labels=labels).loss
    #     print(loss)

    


if __name__ == "__main__":
    main()