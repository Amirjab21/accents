import numpy as np
import torch
import json
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


base_whisper_model = load_model(model_variant, device=DEVICE)
dataset_hf_scottish =load_dataset("ylacombe/english_dialects", "scottish_male")
dataset_hf_scottish['train'] = dataset_hf_scottish['train'].cast_column("audio", Audio(sampling_rate=16000))
dataset_hf_scottish_women = load_dataset("ylacombe/english_dialects", "scottish_female")
dataset_hf_scottish_women['train'] = dataset_hf_scottish_women['train'].cast_column("audio", Audio(sampling_rate=16000))
dataset_hf_southern =load_dataset("ylacombe/english_dialects", "southern_male")
dataset_hf_southern['train'] = dataset_hf_southern['train'].cast_column("audio", Audio(sampling_rate=16000))
dataset_hf_southern_women = load_dataset("ylacombe/english_dialects", "southern_female")
dataset_hf_southern_women['train'] = dataset_hf_southern_women['train'].cast_column("audio", Audio(sampling_rate=16000))

scottish_dataset = dataset_hf_scottish['train'].add_column("dialect", ["scottish"] * len(dataset_hf_scottish['train']))
scottish_dataset_women = dataset_hf_scottish_women['train'].add_column("dialect", ["scottish"] * len(dataset_hf_scottish_women['train']))
# Prepare Southern dataset with source column
southern_dataset = dataset_hf_southern['train'].add_column("dialect", ["southern"] * len(dataset_hf_southern['train']))
southern_dataset_women = dataset_hf_southern_women['train'].add_column("dialect", ["southern"] * len(dataset_hf_southern_women['train']))

# Combine the datasets
combined_dataset = concatenate_datasets([scottish_dataset, southern_dataset, scottish_dataset_women, southern_dataset_women])

# Convert to pandas dataframe
df = combined_dataset.to_pandas()

dataset_vctk = pd.read_pickle("accent_dataset.pkl")


combined_full = pd.concat([df, dataset_vctk])

both_datasets = pd.concat([df, dataset_vctk]).drop(columns=['speaker_id', 'line_id'])


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
    if sample_rate != 16000:
        audio_array = librosa.core.resample(
            y=audio_array,
            orig_sr=sample_rate,
            target_sr=16000
        )
    
    return sample_rate, audio_array

class ContrastiveDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, device=DEVICE, padding_token_id=50257):
        self.df = dataframe
        self.processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="English", task="transcribe")
        self.device = device
        self.feature_extractor = self.processor.feature_extractor
        self.tokenizer = self.processor.tokenizer
        self.padding_token_id = padding_token_id
        self.accent_to_id = {accent: id for id, accent in id_to_accent.items()}
        


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        audio_sample = item['audio']
        audio_array = bytes_to_array(audio_sample)
        
        mel = self.feature_extractor(audio_array[1], sampling_rate=16000)
        text = self.tokenizer(item['text'], 
                                     return_tensors="pt", 
                                     padding="max_length", 
                                     truncation=True, 
                                     max_length=400)
        target = torch.tensor(self.accent_to_id[item['dialect']], dtype=torch.long)

        return {
            'mel': mel.input_features.squeeze(0),
            'text': text.input_ids.squeeze(0),
            'original': item['text'],
            'target': target
        }


model = ModifiedWhisper(base_whisper_model.dims, num_accent_classes, base_whisper_model)

peft_config = LoraConfig(
     inference_mode=False, r=8,target_modules=["out", "token_embedding", "query", "key", "value", "proj_out"] , lora_alpha=32, lora_dropout=0.1
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

model.to(DEVICE)


batch_size=16
# Update the data loading section
scottish_df = df[df['dialect'] == 'scottish']
southern_df = df[df['dialect'] == 'southern']

# Calculate split sizes for each dialect
scottish_train_size = int(0.95 * len(scottish_df))
scottish_test_size = len(scottish_df) - scottish_train_size
southern_train_size = int(0.95 * len(southern_df))
southern_test_size = len(southern_df) - southern_train_size

# Split each dialect separately
scottish_train, scottish_test = torch.utils.data.random_split(
    scottish_df, [scottish_train_size, scottish_test_size])
southern_train, southern_test = torch.utils.data.random_split(
    southern_df, [southern_train_size, southern_test_size])

# Combine the splits
train_df = pd.concat([scottish_train.dataset.iloc[scottish_train.indices], 
                     southern_train.dataset.iloc[southern_train.indices]])
test_df = pd.concat([scottish_test.dataset.iloc[scottish_test.indices], 
                    southern_test.dataset.iloc[southern_test.indices]])

train_dataset = ContrastiveDataset(train_df)
test_dataset = ContrastiveDataset(test_df)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                         num_workers=6 if torch.cuda.is_available() else 0, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True,
                        num_workers=6 if torch.cuda.is_available() else 0, pin_memory=True)


# def choose_layers_to_finetune(model):
#     for name, param in model.encoder.named_parameters():
#         param.requires_grad = False
#         # if 'conv1' in name or 'conv2' in name:
#         #     param.requires_grad = True
#     # Freeze most of decoder except token embeddings and last 2 attention layers
#     for name, param in model.decoder.named_parameters():
#         param.requires_grad = False  # First freeze everything
        
#         # # Unfreeze token embeddings
#         if 'token_embedding' in name:
#             param.requires_grad = True
            
#         # # Unfreeze last 2 attention layers
#         # Whisper decoder has 16 layers (0-15), so we want layers 14 and 15
#         if any(f'layers.{i}.' in name for i in [11,12,13,14, 15]):
#             param.requires_grad = True
#     trainable_params = 0
#     all_param = 0
#     for name, param in model.named_parameters():
#         all_param += param.numel()
#         if param.requires_grad:
#             print(f"Trainable: {name}")
#             trainable_params += param.numel()
#     print(
#         f"trainable params: {trainable_params:,d} || all params: {all_param:,d} "
#         f"|| trainable%: {100 * trainable_params / all_param:.2f}%"
#     )
# choose_layers_to_finetune(model)


def evaluate(model, dataloader):
    model.eval()
    total_loss = 0
    best_loss = float('inf')
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch in dataloader:
            # Get embeddings for both Scottish and English samples
            mel = batch['mel'].to(DEVICE)
            text = batch['text'].to(DEVICE)
            target = batch['target'].to(DEVICE)
            output = model(mel, text)
            batch_loss = criterion(output, target)
            total_loss += batch_loss.item()
    
    avg_loss = total_loss / len(dataloader)
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), 'best_model.pt')
    model.train()
    return avg_loss


def train(model, dataloader, optimizer, number_epochs):
    wandb.init(project="finetune-contrastive-scottish", name=name_of_run)
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(number_epochs):
        total_loss = 0
        progress_bar = tqdm(
                dataloader, desc=f"Epoch {epoch + 1}/{number_epochs}"
        )
        for batch_idx, batch in enumerate(progress_bar):
            optimizer.zero_grad()
            mel = batch['mel'].to(DEVICE)
            text = batch['text'].to(DEVICE)
            target = batch['target'].to(DEVICE)
            output = model(mel, text)
            batch_loss = criterion(output, target)

            progress_bar.set_postfix({"loss": batch_loss.item()})
            
            batch_loss.backward()
            optimizer.step()
            
            wandb.log({"batch loss": batch_loss.item()})
            total_loss += batch_loss.item()
        val_loss = evaluate(model, test_loader)
        wandb.log({"val loss": val_loss, "epoch": epoch})
        print(f"Epoch {epoch} loss: {total_loss / len(train_loader)}")




processor = train_dataset.processor
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
number_epochs = 3

train(model, train_loader, optimizer, number_epochs)




checkpoint_path = 'best_model.pt'
torch.save({ 'model_state_dict': model.state_dict()}, checkpoint_path)
# artifact = wandb.Artifact('model-weights', type='model')
# artifact.add_file(checkpoint_path)
# wandb.log_artifact(artifact)



scottish_predictions = []
scottish_references = []
southern_predictions = []
southern_references = []
scottish_wer = 0
southern_wer = 0
n_scottish = 0
n_southern = 0

model.eval()
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Evaluating"):
        mel = batch['mel'].to(DEVICE)
        outputs = model.whisper.generate(mel)
        batch_predictions = processor.batch_decode(outputs, skip_special_tokens=True)
        
        # Split predictions by dialect
        for i, text in enumerate(batch['original']):
            if batch['dialect'][i] == 'scottish':
                scottish_predictions.append(batch_predictions[i])
                scottish_references.append(text)
                scottish_wer += wer_metric.compute(predictions=[batch_predictions[i]], 
                                                 references=[text])
                n_scottish += 1
            else:  # southern
                southern_predictions.append(batch_predictions[i])
                southern_references.append(text)
                southern_wer += wer_metric.compute(predictions=[batch_predictions[i]], 
                                                 references=[text])
                n_southern += 1

        # Save first few predictions for logging
        if len(scottish_predictions) >= 2 and len(southern_predictions) >= 2:
            scottish_predictions = scottish_predictions[:2]
            scottish_references = scottish_references[:2]
            southern_predictions = southern_predictions[:2]
            southern_references = southern_references[:2]

print(f"Scottish WER: {scottish_wer/n_scottish:.4f}")
print(f"Southern WER: {southern_wer/n_southern:.4f}")

with open('wer_results_initial.txt', 'w') as f:
    f.write("Initial run\n\n")
    f.write(f"First 2 Scottish predictions:\n")
    f.write('\n'.join(scottish_predictions[:2]) + '\n\n')
    f.write(f"First 2 Scottish original texts:\n")
    f.write('\n'.join(scottish_references[:2]) + '\n\n')
    f.write(f"First 2 Southern predictions:\n")
    f.write('\n'.join(southern_predictions[:2]) + '\n\n')
    f.write(f"First 2 Southern original texts:\n")
    f.write('\n'.join(southern_references[:2]) + '\n\n')
    f.write(f"Scottish WER: {scottish_wer/n_scottish:.4f}\n")
    f.write(f"Southern WER: {southern_wer/n_southern:.4f}\n")