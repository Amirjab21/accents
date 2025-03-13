import torch
import numpy as np
from transformers import WhisperProcessor
import io
from scipy.io import wavfile
import librosa

def bytes_to_array(audio_bytes):
    # Create a BytesIO object from the bytes
    byte_data = audio_bytes['bytes'] if isinstance(audio_bytes, dict) else audio_bytes
    
    # Create a BytesIO object from the bytes
    byte_io = io.BytesIO(byte_data)
    
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
    def __init__(self, dataframe, id_to_accent, device="cuda", padding_token_id=50257):
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