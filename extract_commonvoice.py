import pandas as pd
import librosa
import numpy as np
import os
import soundfile as sf
import io
from tqdm import tqdm




def create_accent_dataframe_and_resample(tsv_file, audio_dir, corp, sample_rate=16000):
    df = pd.read_parquet(tsv_file)
    # df = pd.read_csv(tsv_file, sep='\t')
    df = df[['path', 'accents', 'down_votes', 'sentence', 'source']]
    audio_data = []
    df = df[df['source'] == corp]
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing audio files"):
        try:            
            audio_path = os.path.join(audio_dir, row['source'],'clips', row['path'])
            # Read and process audio
            audio_array, sample_rate = librosa.load(audio_path, sr=sample_rate)
            # Convert to int16 format
            audio_array = (audio_array * 32768).astype(np.int16)
            # Convert to bytes
            buffer = io.BytesIO()
            sf.write(buffer, audio_array, sample_rate, format='WAV')
            audio_bytes = buffer.getvalue()
            
            audio_data.append(audio_bytes)
            
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            audio_data.append(None)
            continue
    
    # Add audio column to dataframe
    df['audio'] = audio_data
    
    return df


def analyze_audio_quality(audio_bytes, energy_threshold=0.005, zcr_threshold=0.5):
    """
    Analyze audio quality using various metrics.
    Returns True if audio quality is acceptable, False otherwise.
    """
    # Convert bytes to numpy array
    audio_array, _ = sf.read(io.BytesIO(audio_bytes))
    
    # Calculate RMS energy
    rms = librosa.feature.rms(y=audio_array)[0]
    mean_rms = np.mean(rms)
    # Calculate zero-crossing rate
    zcr = librosa.feature.zero_crossing_rate(audio_array)[0]
    mean_zcr = np.mean(zcr)
    
    # Check if the audio meets our quality criteria
    if mean_rms < energy_threshold:  # Too quiet
        return False
    if mean_zcr > zcr_threshold:  # Too noisy
        return False
    
    return True

def preprocess_df(df):
    # df = df[df['accents'].isin(relevant_accent_list)]
    
    print("Number of rows before processing: ", len(df))
    df = df.dropna(subset=['audio'])
    df = df[df['down_votes'] < 1]
    print("Number of rows after down_votes filter: ", len(df))
    # df['accents'] = df['accents'].map(lambda x: accent_mapping.get(x, x))

    df['quality_check'] = df.apply(
    lambda row: analyze_audio_quality(row['audio']) 
    if row['audio'] is not None else False, 
    axis=1)

    print("Number of rows before quality check: ", len(df))
    df = df[df['quality_check'] == True]
    
    print("Number of rows after quality check: ", len(df))
    return df

def main():

    # corp20/clips
    # EXISTING_MAPPING = {
    #     0: "Scottish", 1: "English", 2: "Indian", 3: "Irish", 4: "Welsh",
    #     5: "NewZealandEnglish", 6: "AustralianEnglish", 7: "SouthAfrican",
    #     8: "Canadian", 9: "NorthernIrish", 10: "American"
    # }


    # relevant_accent_list = [
    #     'United States English', 'Scottish English', 'England English', 'Filipino',
    #     'Japaenese English', 'Japan English', 'Korean', 'German English',
    #     'India and South Asia (India, Pakistan, Sri Lanka)', 'Canadian English',
    #     'Australian English', 'New Zealand English', 'Southern African (South Africa, Zimbabwe, Namibia)',
    #     'Irish English', 'Indian English'
    # ]
    
    # accent_mapping = {
    #     'United States English': 'American',
    #     'Scottish English': 'Scottish',
    #     'England English': 'English',
    #     'India and South Asia (India, Pakistan, Sri Lanka)': 'Indian',
    #     'Canadian English': 'Canadian',
    #     'Australian English': 'AustralianEnglish',
    #     'New Zealand English': 'NewZealandEnglish',
    #     'Southern African (South Africa, Zimbabwe, Namibia)': 'SouthAfrican',
    #     'Irish English': 'Irish',
    #     'Indian English': 'Indian',
    #     "Japanese English": "Japanese",
    #     "Japan English": "Japanese",
    #     "German English": "German",
    #     "Filipino": "Filipino",
    #     "Korean": "Korean"
    # }

    # accents = ['Scottish', 'NewZealandEnglish', 'Japanese', 'Canadian', 'AustralianEnglish', 'Irish', 'American', 'SouthAfrican', 'Indian', 'English', 'German', 'Filipino', 'Korean']
    corps = ['corp18', 'corp17', 'corp16', 'corp15', 'corp14', 'corp13', 'corp12', 'corp10']
    tsv_file = 'filtered_corp20-11.parquet'
    audio_dir = 'commonvoice-data'
    for corp in corps:
        df = create_accent_dataframe_and_resample(tsv_file, audio_dir, corp)
        df = preprocess_df(df)
        df.to_parquet(f'dataframes/accent_{corp}.parquet')

if __name__ == "__main__":
    main()