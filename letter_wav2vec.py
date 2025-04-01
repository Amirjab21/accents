







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
from transformers import Wav2Vec2BertForCTC, Wav2Vec2PhonemeCTCTokenizer, Wav2Vec2BertProcessor
from transformers import AutoProcessor, AutoModelForPreTraining
from phonemizer import phonemize
from torchaudio.datasets import CMUDict
from phonemizer.separator import Separator
from phonemizer.backend.espeak.wrapper import EspeakWrapper
from phonemizer.backend import EspeakBackend

import json


_ESPEAK_LIBRARY = '/opt/homebrew/Cellar/espeak-ng/1.52.0/lib/libespeak-ng.1.dylib'  #use the Path to the library.
EspeakWrapper.set_library(_ESPEAK_LIBRARY)
language = 'en-gb-scotland'#'en-gb-scotland'
backend = EspeakBackend(language)
# returned = backend.phonemize(['Hello world'], separator=Separator(phone='-', word=' ', syllable='|'))

# Get the full phoneme dictionary
def get_phoneme_dictionary():
    # Create a comprehensive word list - you can expand this
    word_list = [
        "hello", "world", "the", "quick", "brown", "fox", "jumps", "over", 
        "lazy", "dog", "phoneme", "dictionary", "speech", "recognition",
        "artificial", "intelligence", "machine", "learning", "audio", "processing"
    ]
    
    # # You could also load a larger word list from a file
    # with open('text8 dataset', 'r') as f:
    #     word_list = [line.strip() for line in f]

    with open('text8 dataset', "r") as f:
        wikipedia_data = f.read(100000000)

    word_list = wikipedia_data.strip().split()

    
    # Phonemize all words
    phonemized = backend.phonemize(word_list, separator=Separator(phone='-', word=' ', syllable='|'))
    
    # Extract all unique phonemes
    all_phonemes = set()
    for phoneme_sequence in phonemized:
        # Split by word separator and then by phone separator
        for word in phoneme_sequence.split(' '):
            for phoneme in word.split('-'):
                if phoneme:  # Skip empty strings
                    all_phonemes.add(phoneme)
    
    # Create a dictionary mapping phonemes to indices
    phoneme_dict = {phoneme: idx for idx, phoneme in enumerate(sorted(all_phonemes))}
    
    # Add special tokens
    phoneme_dict["[UNK]"] = len(phoneme_dict)
    phoneme_dict["[PAD]"] = len(phoneme_dict)
    
    return phoneme_dict

# Get and print the phoneme dictionary








model_variant = "small"

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
        if token in ["<pad>", "<unk>"]:
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
    
    average_score = sum(word_scores.values()) / len(word_scores)
    print(f"Average Score: {average_score:.4f}")
    for word, score in word_scores.items():
        # Add padding for better formatting
        padding = "\t\t" if len(word) < 8 else "\t"
        print(f"{word}{padding}{score:.4f}")



vocab = json.load(open('phoneme_vocab.json'))
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
# tokenizer = Wav2Vec2PhonemeCTCTokenizer(vocab_file='phoneme_vocab.json')
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")
tokenizer = processor.tokenizer
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")
vocab = tokenizer.get_vocab()
with open('wav2vec-phoneme-vocab.json', 'w') as f:
    json.dump(vocab, f, indent=2)

# vocab['[UNK]'] = len(vocab)
# vocab['[PAD]'] = len(vocab)

# print(tokenizer)

# # tokenizer = processor.tokenizer
# # vocab = tokenizer.get_vocab()
# # print(vocab)
LABELS = {v: k for k, v in vocab.items()}
# # print(LABELS)

df = pd.read_parquet("dataframes/accent_corp10.parquet")

def align(emission, tokens):
    targets = torch.tensor([tokens], dtype=torch.int32, device="cpu")
    print(targets.shape, 'targets')
    print(emission.shape, 'emission')
    alignments, scores = torchaudio.functional.forced_align(emission, targets)

    alignments, scores = alignments[0], scores[0]  # remove batch dimension for simplicity
    # print(scores, 'scores')
    # scores = scores.exp()  # convert back to probability
    # print(scores, 'scores')
    return alignments, scores

def process_audio(audio, default_tokenized_transcript=None, transcript=None):
    sample_rate, array = bytes_to_array(audio)
    if len(array) < 400:  # Minimum length needed for wav2vec2 model
        print(f"Warning: Audio too short ({len(array)} samples). Padding to minimum length.")
        array = np.pad(array, (0, max(400 - len(array), 0)), mode='constant')
    input_values = processor(array, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        output = model(input_values.input_values)
    
    predicted_ids = torch.argmax(output.logits, dim=-1)
    predicted_tokens = [LABELS[id.item()] for id in predicted_ids[0]]
    predicted_text = ''.join([token for token in predicted_tokens if token not in ['<pad>', '<unk>']])
    print(f"Predicted text: {predicted_text}")
    emission = output.logits
    emission_normalized = torch.nn.functional.softmax(emission, dim=-1)
    
    # tokenized_transcript = [vocab[letter] for word in TRANSCRIPT for letter in word]
    if default_tokenized_transcript is not None:
        tokenized_transcript = default_tokenized_transcript
    print(tokenized_transcript, 'tokenized_transcript')
    aligned_tokens, alignment_scores = align(emission_normalized, tokenized_transcript)

    token_spans = torchaudio.functional.merge_tokens(aligned_tokens, alignment_scores)
    
    word_scores = calculate_word_scores(token_spans, transcript)
    ipdb.set_trace()
    print_word_scores(word_scores)
    
    return tokenized_transcript


# tokenized_transcript = process_audio(first)


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



def main():
    text = 'The width of the coloured band increases as the size of the drops increases'
    TRANSCRIPT = text.lower().split()
    phenomized = backend.phonemize([text], separator=Separator(phone='-', word='', syllable=None))
    wordphonemized = backend.phonemize([text], separator=Separator(phone='', word=' ', syllable=None))
    separate_words = wordphonemized[0].split(' ')
    print(separate_words, 'separate_words')
    print(phenomized, 'phenomized')
    returned = phenomized[0].split('-')
    print(returned, 'returned')
    tokenized_transcript = [vocab.get(phenome, vocab['<unk>']) for phenome in returned]


    newtest = pd.read_parquet('both_accents.parquet')
    scottish = newtest[newtest['accent'] == 'scottish']
    southern = newtest[newtest['accent'] == 'southern']
    print(len(scottish), len(southern))
    firstscottish = scottish.iloc[0]['audio']['bytes']
    firstsouthern = southern.iloc[0]['audio']['bytes']

    process_audio(firstscottish, tokenized_transcript, separate_words)
    process_audio(firstsouthern, tokenized_transcript, separate_words)


def main_iranian():
    textir = 'چکار میکنی لامصب کثافت'
    backend_ir = EspeakBackend('fa')
    phenomized = backend_ir.phonemize([textir], separator=Separator(phone='-', word=' ', syllable=None))
    
    wordphonemized = backend_ir.phonemize([textir], separator=Separator(phone='', word=' ', syllable=None))
    separate_words = wordphonemized[0].split(' ')
    print(separate_words, 'separate_words')
    print(phenomized, 'phenomized')
    returned = phenomized[0].split('-')
    print(returned, 'returned')
    tokenized_transcript = [vocab.get(phenome, vocab['<unk>']) for phenome in returned]

    audio_file_path = "iranian_audio2.m4a"
    audio_bytes, sample_rate = m4a_to_bytes(audio_file_path)
    process_audio(audio_bytes, tokenized_transcript, separate_words)



if __name__ == "__main__":
    # main()
    main_iranian()

# phoneme_dict = get_phoneme_dictionary()
#     print("\nFull Phoneme Dictionary:")
#     for phoneme, idx in phoneme_dict.items():
#         print(f"{phoneme}: {idx}")

#     # Optionally save to a file

#     with open('phoneme_vocab.json', 'w') as f:
#         json.dump(phoneme_dict, f, indent=2)
        
#     print(f"\nSaved phoneme dictionary with {len(phoneme_dict)} entries to phoneme_vocab.json")
