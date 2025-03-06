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
num_accent_classes = 11
id_to_accent = {0: "Scottish", 1: "English", 2: "Indian", 3: "Irish", 4: "Welsh", 5: "NewZealandEnglish", 6: "AustralianEnglish", 7: "SouthAfrican", 8: "Canadian", 9: "NorthernIrish", 10: "American"}


base_whisper_model = load_model(model_variant, device=DEVICE)
dataset_hf_scottish =load_dataset("ylacombe/english_dialects", "scottish_male")
dataset_hf_scottish['train'] = dataset_hf_scottish['train'].cast_column("audio", Audio(sampling_rate=16000))
dataset_hf_scottish_women = load_dataset("ylacombe/english_dialects", "scottish_female")
dataset_hf_scottish_women['train'] = dataset_hf_scottish_women['train'].cast_column("audio", Audio(sampling_rate=16000))
dataset_hf_southern =load_dataset("ylacombe/english_dialects", "southern_male")
dataset_hf_southern['train'] = dataset_hf_southern['train'].cast_column("audio", Audio(sampling_rate=16000))
dataset_hf_southern_women = load_dataset("ylacombe/english_dialects", "southern_female")
dataset_hf_southern_women['train'] = dataset_hf_southern_women['train'].cast_column("audio", Audio(sampling_rate=16000))

scottish_dataset = dataset_hf_scottish['train'].add_column("dialect", ["Scottish"] * len(dataset_hf_scottish['train']))
scottish_dataset_women = dataset_hf_scottish_women['train'].add_column("dialect", ["Scottish"] * len(dataset_hf_scottish_women['train']))
# Prepare Southern dataset with source column
southern_dataset = dataset_hf_southern['train'].add_column("dialect", ["English"] * len(dataset_hf_southern['train']))
southern_dataset_women = dataset_hf_southern_women['train'].add_column("dialect", ["English"] * len(dataset_hf_southern_women['train']))


combined_dataset = concatenate_datasets([scottish_dataset, southern_dataset, scottish_dataset_women, southern_dataset_women])
df = combined_dataset.to_pandas()


# Load each shard individually
shard1 = load_dataset("Amirjab21/vctk-accents", data_files="accent_dataset_shard_1.parquet", split="train")
shard2 = load_dataset("Amirjab21/vctk-accents", data_files="accent_dataset_shard_2.parquet", split="train")
shard3 = load_dataset("Amirjab21/vctk-accents", data_files="accent_dataset_shard_3.parquet", split="train")

combined_vctk = concatenate_datasets([shard1, shard2, shard3])
df_vctk = combined_vctk.to_pandas()
print("df_vctk", len(df_vctk))
print("df", len(df))
both_datasets = pd.concat([df, df_vctk])

print(len(both_datasets))

if DEVICE != "cuda":
    both_datasets = both_datasets.sample(n=54)


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
train_size = int(0.95 * len(both_datasets))
test_size = len(both_datasets) - train_size


train_df, test_df = torch.utils.data.random_split(
    both_datasets, [train_size, test_size])


train_df = train_df.dataset
test_df = test_df.dataset


scottish_in_test = test_df[test_df['dialect'] == 'Scottish']
english_in_test = test_df[test_df['dialect'] == 'English']

print("scottish_in_test", len(scottish_in_test))
print("English_in_test", len(english_in_test))



train_dataset = ContrastiveDataset(train_df)
test_dataset = ContrastiveDataset(test_df)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                         num_workers=6 if torch.cuda.is_available() else 0, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True,
                        num_workers=6 if torch.cuda.is_available() else 0, pin_memory=True)


class_counts = {i: 0 for i in range(num_accent_classes)}
for dialect in train_df['dialect']:
    # Convert dialect name to ID using the same mapping as in ContrastiveDataset
    dialect_id = {v: k for k, v in id_to_accent.items()}[dialect]
    class_counts[dialect_id] += 1

total_samples = len(train_df)
class_weights = torch.FloatTensor([
    total_samples / (num_accent_classes * count) if count > 0 else 0.0 
    for count in [class_counts[i] for i in range(num_accent_classes)]
])
class_weights = class_weights.to(DEVICE)

# Add verification prints
print("\nClass distribution and weights:")
for class_id, count in class_counts.items():
    weight = total_samples / (num_accent_classes * count) if count > 0 else 0.0
    print(f"Dialect: {id_to_accent[class_id]}")
    print(f"Count: {count}")
    print(f"Weight: {weight:.4f}\n")

print("Class weights tensor:", class_weights)


def evaluate(model, dataloader):
    model.eval()
    total_loss = 0
    best_loss = float('inf')
    correct = 0
    total = 0
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    with torch.no_grad():
        for batch in dataloader:
            # Get embeddings for both Scottish and English samples
            mel = batch['mel'].to(DEVICE)
            text = batch['text'].to(DEVICE)
            target = batch['target'].to(DEVICE)
            output = model(mel, text)
            # Calculate weighted loss
            batch_loss = criterion(output, target)
            total_loss += batch_loss.item()
            
            # Calculate accuracy
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), 'best_model.pt')
    
    model.train()
    return avg_loss, accuracy


def train(model, dataloader, optimizer, number_epochs):
    wandb.init(project="finetune-contrastive-scottish", name=name_of_run)
    model.train()
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
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
        val_loss, val_acc = evaluate(model, test_loader)
        wandb.log({"val loss": val_loss, "epoch": epoch, "val acc": val_acc})
        print(f"Epoch {epoch} loss: {total_loss / len(train_loader)}")




processor = train_dataset.processor
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
number_epochs = 1

train(model, train_loader, optimizer, number_epochs)




checkpoint_path = 'best_model.pt'
torch.save({ 'model_state_dict': model.state_dict()}, checkpoint_path)
# artifact = wandb.Artifact('model-weights', type='model')
# artifact.add_file(checkpoint_path)
# wandb.log_artifact(artifact)



# scottish_predictions = []
# scottish_references = []
# southern_predictions = []
# southern_references = []
# scottish_wer = 0
# southern_wer = 0
# n_scottish = 0
# n_southern = 0

# model.eval()
# with torch.no_grad():
#     for batch in tqdm(test_loader, desc="Evaluating"):
#         mel = batch['mel'].to(DEVICE)
#         outputs = model.whisper.generate(mel)
#         batch_predictions = processor.batch_decode(outputs, skip_special_tokens=True)
        
#         # Split predictions by dialect
#         for i, text in enumerate(batch['original']):
#             if batch['dialect'][i] == 'scottish':
#                 scottish_predictions.append(batch_predictions[i])
#                 scottish_references.append(text)
#                 scottish_wer += wer_metric.compute(predictions=[batch_predictions[i]], 
#                                                  references=[text])
#                 n_scottish += 1
#             else:  # southern
#                 southern_predictions.append(batch_predictions[i])
#                 southern_references.append(text)
#                 southern_wer += wer_metric.compute(predictions=[batch_predictions[i]], 
#                                                  references=[text])
#                 n_southern += 1

#         # Save first few predictions for logging
#         if len(scottish_predictions) >= 2 and len(southern_predictions) >= 2:
#             scottish_predictions = scottish_predictions[:2]
#             scottish_references = scottish_references[:2]
#             southern_predictions = southern_predictions[:2]
#             southern_references = southern_references[:2]

# print(f"Scottish WER: {scottish_wer/n_scottish:.4f}")
# print(f"Southern WER: {southern_wer/n_southern:.4f}")

# with open('wer_results_initial.txt', 'w') as f:
#     f.write("Initial run\n\n")
#     f.write(f"First 2 Scottish predictions:\n")
#     f.write('\n'.join(scottish_predictions[:2]) + '\n\n')
#     f.write(f"First 2 Scottish original texts:\n")
#     f.write('\n'.join(scottish_references[:2]) + '\n\n')
#     f.write(f"First 2 Southern predictions:\n")
#     f.write('\n'.join(southern_predictions[:2]) + '\n\n')
#     f.write(f"First 2 Southern original texts:\n")
#     f.write('\n'.join(southern_references[:2]) + '\n\n')
#     f.write(f"Scottish WER: {scottish_wer/n_scottish:.4f}\n")
#     f.write(f"Southern WER: {southern_wer/n_southern:.4f}\n")