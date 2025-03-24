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
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")
processor = AutoProcessor.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")

tokenizer = processor.tokenizer
vocab = tokenizer.get_vocab()
LABELS = {v: k for k, v in vocab.items()}
# print(LABELS)

df = pd.read_parquet("dataframes/accent_corp10.parquet")

def align(emission, tokens):
    targets = torch.tensor([tokens], dtype=torch.int32, device="cpu")
    alignments, scores = torchaudio.functional.forced_align(emission, targets)

    alignments, scores = alignments[0], scores[0]  # remove batch dimension for simplicity
    scores = scores.exp()  # convert back to probability
    return alignments, scores


# sample_rate, array = bytes_to_array(df.iloc[0]["audio"])
# input_values = processor(array, sampling_rate=16000, return_tensors="pt")
 
# with torch.no_grad():
#     output = model(input_values.input_values)
# print(output)
# predicted_ids = torch.argmax(output.logits, dim=-1)
# transcription = processor.batch_decode(predicted_ids)

# # transcription = ['ɪ t k ə n t eɪ n z p æ n əl z ʌ v f ɪ l ɪ ɡ ɹ i æ n d s t æ m p t s ɪ l v ɚ ɹ ɪ b ə n z']
# emission = output.logits
# # print(emission.shape)



# tokenized_transcript = [vocab[phoneme] for phoneme in transcription[0].split(" ")]
# print(tokenized_transcript)

# aligned_tokens, alignment_scores = align(emission, tokenized_transcript)

# token_spans = torchaudio.functional.merge_tokens(aligned_tokens, alignment_scores)

# print("Token\tTime\tScore")
# for s in token_spans:
#     print(f"{LABELS[s.token]}\t[{s.start:3d}, {s.end:3d})\t{s.score:.2f}")

#We can sort of put it into words although its phonemes

#So the next experiment is to get different audios of the same sentence and see how they compare.

#If we can find a pattern, then we can use it for accent alignment.



text = 'Nuclear fusion on a large scale in an explosion was first carried out in the Ivy Mike hydrogen bomb test'
oneline = pd.read_parquet('temp.parquet')

first = oneline.iloc[0]['audio']['bytes']
second = oneline.iloc[1]['audio']['bytes']
third = oneline.iloc[2]['audio']['bytes']

def process_audio(audio, default_tokenized_transcript=None):
    sample_rate, array = bytes_to_array(audio)
    input_values = processor(array, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        output = model(input_values.input_values)
    predicted_ids = torch.argmax(output.logits, dim=-1)
    print(predicted_ids)
    transcription = processor.batch_decode(predicted_ids)
    print(transcription, 'transcription')
    emission = output.logits
    tokenized_transcript = [vocab[phoneme] for phoneme in transcription[0].split(" ")]
    if default_tokenized_transcript is not None:
        tokenized_transcript = default_tokenized_transcript
    # print(tokenized_transcript)

    aligned_tokens, alignment_scores = align(emission, tokenized_transcript)

    token_spans = torchaudio.functional.merge_tokens(aligned_tokens, alignment_scores)

    print("Token\tTime\tScore")
    for s in token_spans:
        print(f"{LABELS[s.token]}\t[{s.start:3d}, {s.end:3d})\t{s.score:.2f}")
    
    return tokenized_transcript


tokenized_transcript = process_audio(first)
# process_audio(second)
# process_audio(third)



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
audio_bytes, sample_rate = m4a_to_bytes("amir.m4a")

process_audio(audio_bytes, tokenized_transcript)