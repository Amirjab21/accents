# !pip install transformers
# !pip install datasets
# import soundfile as sf
# import torch
# from datasets import load_dataset
# from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, AutoProcessor, Wav2Vec2BertForCTC
# import pandas as pd
# import io
# import librosa
# import numpy as np
# from scipy.io import wavfile
# import ipdb
# import os
# from datetime import datetime
# from huggingface_hub import HfApi

# load pretrained model
# processor = AutoProcessor.from_pretrained("hf-audio/wav2vec2-bert-CV16-en")
# model = Wav2Vec2BertForCTC.from_pretrained("hf-audio/wav2vec2-bert-CV16-en")


# def bytes_to_array(audio_bytes):
#     # Create a BytesIO object from the bytes
#     byte_data = audio_bytes['bytes'] if isinstance(audio_bytes, dict) else audio_bytes
    
#     # Create a BytesIO object from the bytes
#     byte_io = io.BytesIO(byte_data)
    
#     # Read the WAV file from BytesIO
#     sample_rate, audio_array = wavfile.read(byte_io)
#     # ipdb.set_trace()
    
#     # Convert to float32 and normalize to [-1, 1]
#     audio_array = audio_array.astype(np.float32) / 32768.0
#     if sample_rate != 16000:
#         audio_array = librosa.core.resample(
#             y=audio_array,
#             orig_sr=sample_rate,
#             target_sr=16000
#         )
    
#     return 16000, audio_array


# # librispeech_samples_ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")
# newtest = pd.read_parquet('both_accents.parquet')
# target_transcription = newtest.iloc[0]['text'].lower()

# # load audio
# sample_rate, audio_input = bytes_to_array(newtest.iloc[0]['audio']['bytes'])

# # pad input values and return pt tensor
# input_values = processor.feature_extractor(audio_input, sampling_rate=sample_rate, return_tensors="pt", padding=True)


# # INFERENCE
# logits = model(input_values.input_features).logits
# predicted_ids = torch.argmax(logits, dim=-1)
# transcription = processor.decode(predicted_ids[0])

# # FINE-TUNE
# labels = processor.tokenizer(target_transcription, return_tensors="pt", padding=True)
# labels = labels['input_ids'].masked_fill(labels['attention_mask'] == 0, -100)
# print(labels, 'label')
# outputs = model(input_values.input_features, labels=labels)
# loss = outputs.loss
# loss.backward()

##################################
##################################
########## REAL ATTEMPT ##########
##################################
##################################
import numpy as np
import torch
import json
from torch.utils.data import DataLoader
from evaluate import load
from datasets import Audio, concatenate_datasets, load_dataset
from tqdm import tqdm
import wandb
# from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from transformers import Wav2Vec2BertForCTC, AutoProcessor
import io
from scipy.io import wavfile
import librosa
import os
from datetime import datetime
from huggingface_hub import HfApi

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
wer_metric = load("wer")
processor = AutoProcessor.from_pretrained("hf-audio/wav2vec2-bert-CV16-en")
model = Wav2Vec2BertForCTC.from_pretrained("hf-audio/wav2vec2-bert-CV16-en")

def bytes_to_array(audio_bytes):
    # Create a BytesIO object from the bytes
    byte_data = audio_bytes['bytes'] if isinstance(audio_bytes, dict) else audio_bytes
    
    # Create a BytesIO object from the bytes
    # print(byte_data, 'byte data')
    byte_io = io.BytesIO(byte_data)
    
    # Read the WAV file from BytesIO
    sample_rate, audio_array = wavfile.read(byte_io)
    # ipdb.set_trace()
    
    # Convert to float32 and normalize to [-1, 1]
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

        
        input_values = self.processor.feature_extractor(audio_array, sampling_rate=sample_rate, return_tensors="pt", truncation=True,padding="max_length", max_length=200)
        audio_mel = input_values.input_features
        print(audio_mel.shape, 'audio mel')
        
        text = self.tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=100)
        labels = text.input_ids
        labels = labels.masked_fill(text.attention_mask.eq(0), -100)
            
        
        return  {
            'mel': audio_mel.squeeze(0), 
            'labels': labels.squeeze(0), 
            'text': data['text']
        }






# print(model.wav2vec2_bert)

for param in model.wav2vec2_bert.encoder.parameters():
    # param.requires_grad = False
    param.requires_grad = True
# model.print_trainable_parameters()
# peft_config = LoraConfig(
#      inference_mode=False, r=8,target_modules=["out", "token_embedding", "query", "key", "value", "proj_out"] , lora_alpha=32, lora_dropout=0.1
# )
# model = get_peft_model(model, peft_config)
# model.print_trainable_parameters()



hf_token = os.getenv("HF_TOKEN")
api = HfApi()

def evaluate(model, dataloader, device, save_model=True):
    model.eval()
    
    print(hf_token)
    
    total_loss = 0
    best_loss = float('inf')
    correct = 0
    total = 0
    # criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    progress_bar = tqdm(dataloader, desc="Evaluating")
    with torch.no_grad():
        for batch_idx, batch in enumerate(progress_bar):
            mel = batch['mel'].to(device)
            labels = batch['labels'].to(device)
            # text = batch['text'].to(device)
            output = model(mel, labels)
            total_loss = output.loss
            progress_bar.set_postfix({"loss": total_loss.item()})
            total += labels.size(0)
            
    avg_loss = total_loss / len(dataloader)
    # accuracy = correct / total
    # print(f"avg_loss: {avg_loss}, accuracy: {accuracy}")
    if avg_loss < best_loss and save_model == True:
        best_loss = avg_loss
        torch.save(model.state_dict(), 'best_model.pt')
        if device == "cuda":
            try:
                api.upload_file(
                    path_or_fileobj="best_model.pt",
                    path_in_repo=f"best_model_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.pt",
                    repo_id="Amirjab21/accent-classifier",
                    token=hf_token  # Replace with your actual token
                )
            except Exception as e:
                print(f"Error uploading model: {e}")
    
    model.train()
    return avg_loss

def train_model(model, train_loader, test_loader, optimizer,
                device, number_epochs=1, run_name="accent-training"):
    if device == "cuda":
        wandb.init(project="finetune-wav2vec-scottish", name=run_name)
    model.train()
    
    
    for epoch in range(number_epochs):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{number_epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            optimizer.zero_grad()
            mel = batch['mel'].to(device)
            labels = batch['labels'].to(device)
            import ipdb
            ipdb.set_trace()
            print(mel[0], 'mel')
            print(labels[0], 'labels')
            # text = text.to(device)
            output = model(mel, labels)
            print(output, 'output')
            batch_loss = output.loss

            progress_bar.set_postfix({"loss": batch_loss.item()})
            
            batch_loss.backward()
            optimizer.step()
            if device == "cuda":
                wandb.log({"batch loss": batch_loss.item()})
            total_loss += batch_loss.item()
            
        val_loss = evaluate(model, test_loader, device)
        if device == "cuda":
            wandb.log({"val loss": val_loss, "epoch": epoch})
        print(f"Epoch {epoch} loss: {total_loss / len(train_loader)}")
    
    return model




def main():

    dataset_hf = load_datasets()
    batch_size=1
    train_size = int(0.95 * len(dataset_hf))  # 5% for training
    test_size = len(dataset_hf) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset_hf, [train_size, test_size])
    train_dataset = train_dataset.dataset
    test_dataset = test_dataset.dataset

    train_dataset = Dataset(train_dataset)
    test_dataset = Dataset(test_dataset)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                            num_workers=6 if torch.cuda.is_available() else 0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=6 if torch.cuda.is_available() else 0, pin_memory=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    train_model(model, train_loader, test_loader, optimizer, DEVICE)


if __name__ == "__main__":
    main()
