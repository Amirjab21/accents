import torch
from model.model import Whisper

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