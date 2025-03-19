import io
from pydub import AudioSegment
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torchaudio
from datasets import load_dataset
import torch
from transformers import AutoProcessor, AutoModelForCTC
import pandas as pd
import io
from scipy.io import wavfile
import librosa
import numpy as np
import ipdb

from transformers import Wav2Vec2BertForCTC

def calculate_word_scores(token_spans, transcript):
    """
    Calculate average score for each word in the transcript based on token spans.
    
    Args:
        token_spans: List of token spans with token, start, end, and score
        transcript: List of words in the transcript
    
    Returns:
        Dictionary mapping words to their average scores
    """
    # Initialize variables
    word_scores = {}
    current_word = ""
    current_scores = []
    word_index = 0
    
    # Process each token span
    for span in token_spans:
        token = LABELS[span.token]
        score = span.score
        
        # Skip padding or unknown tokens
        if token in ["[PAD]", "[UNK]"]:
            continue
            
        # Add letter to current word
        current_word += token
        current_scores.append(score)
        
        # Check if we've completed a word
        if len(current_word) == len(transcript[word_index]):
            # Calculate average score for the word
            word_scores[transcript[word_index]] = sum(current_scores) / len(current_scores)
            
            # Reset for next word
            current_word = ""
            current_scores = []
            word_index += 1
            
            # Break if we've processed all words
            if word_index >= len(transcript):
                break
    
    return word_scores

def print_word_scores(word_scores):
    """
    Print each word alongside its average score.
    
    Args:
        word_scores: Dictionary mapping words to their average scores
    """
    print("\nWord Scores:")
    print("-" * 30)
    print("Word\t\tAverage Score")
    print("-" * 30)
    
    for word, score in word_scores.items():
        # Add padding for better formatting
        padding = "\t\t" if len(word) < 8 else "\t"
        print(f"{word}{padding}{score:.4f}")



# vocab = {

# }
# for i, letter in enumerate("abcdefghijklmnopqrstuvwxyz"):
#     vocab[letter] = i
# # Add space character
# vocab[" "] = 26
# vocab["|"] = vocab[" "]
# del vocab[" "]
# vocab["[UNK]"] = len(vocab)
# vocab["[PAD]"] = len(vocab)
# import json
# with open('vocab.json', 'w') as vocab_file:
#     json.dump(vocab, vocab_file)


# load model and processor
# export PHONEMIZER_ESPEAK_LIBRARY=/opt/homebrew/lib/libespeak-ng.dylib
# export PHONEMIZER_ESPEAK_PATH=/opt/homebrew/bin


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

processor = AutoProcessor.from_pretrained("hf-audio/wav2vec2-bert-CV16-en")
model = Wav2Vec2BertForCTC.from_pretrained("hf-audio/wav2vec2-bert-CV16-en")

tokenizer = processor.tokenizer
vocab = tokenizer.get_vocab()

# tokenizer = processor.tokenizer
# vocab = tokenizer.get_vocab()
# print(vocab)
LABELS = {v: k for k, v in vocab.items()}
# print(LABELS)

df = pd.read_parquet("dataframes/accent_corp10.parquet")

def align(emission, tokens):
    targets = torch.tensor([tokens], dtype=torch.int32, device="cpu")
    alignments, scores = torchaudio.functional.forced_align(emission, targets, blank=0)

    alignments, scores = alignments[0], scores[0]  # remove batch dimension for simplicity
    # print(scores, 'scores')
    # scores = scores.exp()  # convert back to probability
    # print(scores, 'scores')
    return alignments, scores



text = 'Nuclear fusion on a large scale in an explosion was first carried out in the Ivy Mike hydrogen bomb test'
TRANSCRIPT = text.lower().split()
oneline = pd.read_parquet('temp.parquet')

first = oneline.iloc[0]['audio']['bytes']
second = oneline.iloc[1]['audio']['bytes']
third = oneline.iloc[2]['audio']['bytes']

def process_audio(audio, default_tokenized_transcript=None):
    sample_rate, array = bytes_to_array(audio)
    input_values = processor(array, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        output = model(input_values.input_features)
    
    predicted_ids = torch.argmax(output.logits, dim=-1)
    emission = output.logits
    emission_normalized = torch.nn.functional.softmax(emission, dim=-1)
    tokenized_transcript = [vocab[letter] for word in TRANSCRIPT for letter in word]
    if default_tokenized_transcript is not None:
        tokenized_transcript = default_tokenized_transcript

    aligned_tokens, alignment_scores = align(emission_normalized, tokenized_transcript)

    token_spans = torchaudio.functional.merge_tokens(aligned_tokens, alignment_scores)

    word_scores = calculate_word_scores(token_spans, TRANSCRIPT)
    print_word_scores(word_scores)

    # print("Token\tTime\tScore")
    # for s in token_spans:
    #     print(f"{LABELS[s.token]}\t[{s.start:3d}, {s.end:3d})\t{s.score:.2f}")
    
    return tokenized_transcript


tokenized_transcript = process_audio(first)


def m4a_to_bytes(file_path, target_sample_rate=16000):
    """
    Convert an M4A file to a bytes array with the specified sample rate.
    
    Args:
        file_path: Path to the M4A file
        target_sample_rate: Desired sample rate for the output audio
        
    Returns:
        Tuple of (audio_bytes, sample_rate)
    """
    # Load the M4A file
    audio = AudioSegment.from_file(file_path, format="m4a")
    
    # Convert to the target sample rate if needed
    if audio.frame_rate != target_sample_rate:
        audio = audio.set_frame_rate(target_sample_rate)
    
    # Convert to WAV format in memory
    buffer = io.BytesIO()
    audio.export(buffer, format="wav")
    
    # Get the bytes
    buffer.seek(0)
    audio_bytes = buffer.read()
    
    return audio_bytes, target_sample_rate

# Example usage:
audio_bytes, sample_rate = m4a_to_bytes("test.m4a")
process_audio(audio_bytes, tokenized_transcript)

