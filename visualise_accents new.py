from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from model.accent_model import ModifiedWhisper
import torch
from torch.utils.data import DataLoader
from model.load_model import load_model
from train import setup_model, load_datasets
from training.accent_dataset import ContrastiveDataset
import pandas as pd
from datasets import load_dataset, Audio, concatenate_datasets
from peft import get_peft_config, get_peft_model, LoraConfig
from pathlib import Path

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_VARIANT = "small"
NUM_ACCENT_CLASSES = 21
ID_TO_ACCENT = {0: 'Scottish', 1: 'English', 2: 'Indian', 3: 'Irish', 4: 'Welsh', 5: 'NewZealandEnglish', 6: 'AustralianEnglish', 7: 'SouthAfrican', 8: 'Canadian', 9: 'NorthernIrish', 10: 'American', 11: 'South East Asia', 12: 'Eastern Europe', 13: 'East Asia', 14: 'Nordic', 15: 'France', 16: 'Southern Europe', 17: 'Germany', 18: 'West Indies and Bermuda', 19: 'Western Africa', 20: 'South Asia'}

model_path = "best_model_2025-03-24-22-48-00.pt"

# def load_model():
#     model = setup_model()
#     checkpoint = torch.load(model_path, map_location=torch.device(DEVICE))
#     try:
#         if 'model_state_dict' in checkpoint:
#             model.load_state_dict(checkpoint['model_state_dict'])
#             print('Model loaded successfully')
#         else:
#             model.load_state_dict(checkpoint)
#             print('Model loaded successfully')
#     except Exception as e:
#         raise e
#     return model

def setup_model(checkpoint_file = None, num_accent_classes=NUM_ACCENT_CLASSES):
    base_whisper_model = load_model(MODEL_VARIANT, device=DEVICE)
    model = ModifiedWhisper(base_whisper_model.dims, num_accent_classes, base_whisper_model)
    
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
    
    model.to(DEVICE)
    
    return model

def prepare_data(dataset_df, samples_per_accent=100):

    print(dataset_df['dialect'].unique())
    balanced_df = []
    for accent_id in ID_TO_ACCENT.keys():
        accent_data = dataset_df[dataset_df['dialect'] == ID_TO_ACCENT[accent_id]]
        if len(accent_data) >= samples_per_accent:
            balanced_df.append(accent_data.sample(n=samples_per_accent, random_state=42))
        else:
            print(f"Warning: {ID_TO_ACCENT[accent_id]} has only {len(accent_data)} samples")
            balanced_df.append(accent_data)

    balanced_df = pd.concat(balanced_df, ignore_index=True)
    dataset = ContrastiveDataset(balanced_df, ID_TO_ACCENT, device=DEVICE)
    loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=6 if torch.cuda.is_available() else 0, pin_memory=True)
    return loader

def do_mapping(df):

    category_mapping = {
        # United States
        "United States English": "American",
        "United States English, Midwestern, Low, Demure": "American",
        "United States English, Midwestern USA": "American",
        "United States English, Midwestern": "American",
        "United States English, Southern United States, New Orleans dialect": "American",
        "United States English, Southwestern United States English": "American",
        "United States English, CARIBBEAN AND BRITISH MIXED WITH SOME NEW YORK ACCENTS": "American",
        "United States English, southern midwest hippie english, grateful deadhead": "American",
        "United States English, Upstate New York": "American",
        "United States English, Like a Kennedy, Boston": "American",
        "United States English, Washington State": "American",
        "United States English, New York English": "American",
        "United States English, New Yorker": "American",
        "United States English, South Texas, Slightly effeminate, Conversational": "American",
        "United States English, southern draw": "American",
        "United States English, Delaware Valley, Pennsylvania": "American",
        "United States English, Southern Californian English": "American",
        "United States English, North Indiana": "American",
        "United States English, California, Valley": "American",
        "United States English, Pennsylvanian, Neutral": "American",
        "United States English, Texas": "American",
        "United States English, California English, Southern US English": "American",
        "United States English, Southern Ohio English": "American",
        "United States English, Mid-west United States English": "American",
        "United States English, Pacific North West United States": "American",
        "United States English, Midwestern US English (United States)": "American",
        "American South East Georgia Dialect": "American",
        "American Kansas": "American",
        
        # England
        "England English": "English",
        "England English, Well spoken softly spoken Home Counties gay, softly spoken male, well spoken male, gay male": "English",
        "England English, south German / Swiss accent": "English",
        "England English, yorkshire": "English",
        "England English, femalepublic school accent, quiet and under articulated finals": "English",
        "England English, southern UK, male": "English",
        "England English, southern english, sussex": "English",
        "England English, Very British": "English",
        "England English, Slightly lazy Midlands English, Oxford English": "English",
        "England English, Received Pronunciation": "English",
        "British English / Received Pronunciation (RP)": "English",
        "British accent": "English",
        "British English - North West": "English",
        
        # Canada
        "Canadian English": "Canadian",
        "Canadian English, Ontario": "Canadian",
        
        # Australia
        "Australian English": "AustralianEnglish",
        "Educated Australian Accent": "AustralianEnglish",
        "South Australia": "AustralianEnglish",
        
        # New Zealand
        "New Zealand English": "NewZealandEnglish",
        
        # Ireland
        "Irish English": "Irish",
        "Northern Irish": "Irish",
        "Northern Irish English": "Irish",
        "Irish English, Northern Irish English, belfast": "Irish",
        
        # Scotland
        "Scottish English": "Scottish",
        "Scottish English (West Coast), Scottish English (Ayrshire)": "Scottish",
        
        # Wales
        "Welsh English": "Welsh",
        "Welsh English, Shropshire": "Welsh",
        
        # South Africa
        "South African English Accent": "SouthAfrican",
        "South African accent": "SouthAfrican",
        "2nd language, South African (first language Afrikaans)": "SouthAfrican",
        
        # Southern Africa (South Africa, Zimbabwe, Namibia)
        "Southern African (South Africa, Zimbabwe, Namibia)": "SouthAfrican",
        
        # India and South Asia (India, Pakistan, Sri Lanka)
        "India and South Asia (India, Pakistan, Sri Lanka)": "Indian",
        "Indian English": "Indian",
        "International Indian Accent": "Indian",
        
        # Philippines
        "Filipino": "Philippines",
        
        # Hong Kong
        "Hong Kong English": "Hong Kong",
        
        # Malaysia
        "Malaysian English": "Malaysia",
        
        # Singapore
        "Singaporean English": "Singapore",
        
        # Japan
        "Japanese English": "Japan",
        "Japan English": "Japan",
        "Japanese": "Japan",
        
        # South Korea
        "Korean": "South Korea",
        
        # China
        "Chinese English": "China",
        "Chinese": "China",
        "chinese accent": "China",
        "Chinese accent of English": "China",
        "Chinese-English": "China",
        
        # Vietnam
        "Vietnam": "Vietnam",
        
        # Thailand
        "Thai": "Thailand",
        
        # Indonesia
        "Indonesian": "Indonesia",
        "indonesia": "Indonesia",
        
        # Nepal
        "Nepali": "Nepal",
        
        # Bangladesh
        "Bangladeshi": "Bangladesh",
        
        # Nigeria
        "nigeria english": "Nigeria",
        "Nigerian English": "Nigeria",
        "Nigerian": "Nigeria",
        
        # Kenya
        "Kenyan English": "Kenya",
        "Kenyan English accent": "Kenya",
        "Kenyan": "Kenya",
        
        # Ghana
        "Ghanaian english Accent, african regular reader": "Ghana",
        
        # East Africa
        "East African Khoja": "East Africa",
        
        # West Indies and Bermuda (Bahamas, Bermuda, Jamaica, Trinidad)
        "West Indies and Bermuda (Bahamas, Bermuda, Jamaica, Trinidad)": "West Indies and Bermuda",
        
        # Germany
        "German": "Germany",
        "German English": "Germany",
        
        # France
        "French": "France",
        "french accent": "France",
        "light french accent": "France",
        "not native English, originally French native language": "France",
        "french english": "France",
        "non native French speaker": "France",

        # Spain
        "spanish english": "Spain",
        "Spanish bilinguals": "Spain",
        "Spanish": "Spain",
        "Spanglish, England English": "Spain",
        "Spanish accent": "Spain",
        "Spanish from the Canary Islands": "Spain",

        
        # Italy
        "Italian": "Italy",
        "Italian,United States English": "Italy",
        "Italian accent from Northern Italy (Milan)": "Italy",
        
        # Netherlands
        "Dutch": "Netherlands",
        "Slight Dutch accent": "Netherlands",
        
        # Austria
        "Austrian": "Austria",
        "United States English, American english with austrian accent": "Austria",
        
        # Sweden
        "Swedish English": "Sweden",
        "Swedish accent": "Sweden",
        
        # Finland
        "Finnish": "Finland",
        
        # Norway
        "Northern Irish, Norwegian, yorkshire": "Norway",
        
        # Denmark
        "United States English, Irish English, England English, Scottish English, Danish English": "Denmark",
        
        # Russia
        "Russian": "Russia",
        "Russian Accent": "Russia",
        
        # Ukraine
        "Ukrainian": "Ukraine",
        
        # Poland
        "polish": "Poland",
        "Slavic, polish": "Poland",
        "Slavic, East European, polish": "Poland",
        
        # Czech Republic
        "Czech": "Czech Republic",
        
        # Slovakia
        "Slovak": "Slovakia",
        
        # Hungary
        "Hungarian": "Hungary",
        
        # Romania
        "Eastern European, Romanian": "Romania",
        
        # Bulgaria
        "Bulgarian": "Bulgaria",
        
        # Croatia
        "Croatian English": "Croatia",
        
        # Greece
        "Greek": "Greece",
        
        # Turkey
        "Turkish": "Turkey",
        
        # Israel
        "Israeli": "Israel",
        
        # Brazil
        "Brazilian": "Brazil",
        "Brazillian Accent": "Brazil",
        "Brazilian English": "Brazil",
        "Brazilian portuguese native speaker": "Brazil",
        "Brazilian Accent,slight british accent": "Brazil",
        "portuguese": "Brazil",
        "United States English,portuguese": "Brazil",
        "I'm from Brazil,  my Native language is portuguese": "Brazil",
        "colombian": "Brazil",



        
        # Mexico/Central America
        "Central American, United States English": "Mexico/Central America",
        "Mexican English": "Mexico/Central America",
        "Mexican": "Mexico/Central America",
        "Latino English (Mexico)": "Mexico/Central America",
        "Hispanic": "Mexico/Central America",
        "I am Hispanic": "Mexico/Central America",
        "Hispanic/Latino": "Mexico/Central America",
        "little latino,United States English,second language": "Mexico/Central America",
        "Latino": "Mexico/Central America",
        "Latin American accent": "Mexico/Central America",
        "Latin American accent influenced by American English": "Mexico/Central America",

        
        # Kazakhstan
        "Kazakhstan English": "Kazakhstan",
        
        # Latvia
        "strong Latvian accent": "Latvia",
        "Latvian": "Latvia",
        
        # Kazakhstan
        "Kazakhstan English": "Kazakhstan",
        
        # Latvia
        "strong Latvian accent": "Latvia",
        "Latvian": "Latvia",

    }


    
    # Apply the mapping to the 'accents' column
    df['dialect'] = df['dialect'].map(category_mapping).fillna(df['dialect'])
    unique_accents = list(set(category_mapping.values()))


    return df, unique_accents



def load_new_dataset(dataset_names):
    all_dfs = []

    for dataset_name in dataset_names:
        dataset = load_dataset(dataset_name)
        df = dataset['train'].to_pandas()
        temporary = df['accents'].value_counts()
        print(temporary, 'temporary')
        df.drop(columns=['down_votes', 'quality_check', '__index_level_0__'], inplace=True)
        df.rename(columns={'accents': 'dialect', 'sentence': 'text'}, inplace=True)
        all_dfs.append(df)
    
    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_df, unique_accents = do_mapping(combined_df)

    if DEVICE != "cuda":
        combined_df = combined_df.sample(n=100)
    return combined_df, unique_accents

def load_datasets(sample_size=None):
    # Load English dialects datasets
    dataset_hf_scottish = load_dataset("ylacombe/english_dialects", "scottish_male")
    dataset_hf_scottish['train'] = dataset_hf_scottish['train'].cast_column("audio", Audio(sampling_rate=16000))
    dataset_hf_scottish_women = load_dataset("ylacombe/english_dialects", "scottish_female")
    dataset_hf_scottish_women['train'] = dataset_hf_scottish_women['train'].cast_column("audio", Audio(sampling_rate=16000))
    dataset_hf_southern = load_dataset("ylacombe/english_dialects", "southern_male")
    dataset_hf_southern['train'] = dataset_hf_southern['train'].cast_column("audio", Audio(sampling_rate=16000))
    dataset_hf_southern_women = load_dataset("ylacombe/english_dialects", "southern_female")
    dataset_hf_southern_women['train'] = dataset_hf_southern_women['train'].cast_column("audio", Audio(sampling_rate=16000))

    # Add dialect labels
    scottish_dataset = dataset_hf_scottish['train'].add_column("dialect", ["Scottish"] * len(dataset_hf_scottish['train']))
    scottish_dataset_women = dataset_hf_scottish_women['train'].add_column("dialect", ["Scottish"] * len(dataset_hf_scottish_women['train']))
    southern_dataset = dataset_hf_southern['train'].add_column("dialect", ["English"] * len(dataset_hf_southern['train']))
    southern_dataset_women = dataset_hf_southern_women['train'].add_column("dialect", ["English"] * len(dataset_hf_southern_women['train']))

    # Combine all datasets
    combined_dataset = concatenate_datasets([scottish_dataset, southern_dataset, scottish_dataset_women, southern_dataset_women])
    df = combined_dataset.to_pandas()

    # Load VCTK dataset shards
    shard1 = load_dataset("Amirjab21/vctk-accents", data_files="accent_dataset_shard_1.parquet", split="train")
    shard2 = load_dataset("Amirjab21/vctk-accents", data_files="accent_dataset_shard_2.parquet", split="train")
    shard3 = load_dataset("Amirjab21/vctk-accents", data_files="accent_dataset_shard_3.parquet", split="train")
    combined_vctk = concatenate_datasets([shard1, shard2, shard3])
    df_vctk = combined_vctk.to_pandas()

    # Combine all data
    both_datasets = pd.concat([df, df_vctk])
    
    if DEVICE != "cuda":
        both_datasets = both_datasets.sample(n=sample_size)
    
    return both_datasets

def limit_accents(df, accents_to_limit, max_samples=10000):

    for dialect in accents_to_limit:
        dialect_samples = df[df['dialect'] == dialect]
        if len(dialect_samples) > max_samples:
            print(f"Limiting {dialect} from {len(dialect_samples)} to {max_samples} samples")
            # Get indices of samples to keep (random selection)
            keep_indices = dialect_samples.sample(n=max_samples, random_state=42).index
            # Get indices to drop
            drop_indices = dialect_samples.index.difference(keep_indices)
            # Drop the excess samples
            df = df.drop(index=drop_indices)

    return df

def load_and_process_commonvoice():

    new_dataset, unique_accents = load_new_dataset(['Amirjab21/commonvoice'])
    grouped_dataset, dialect_to_region, unique_regions = group_accents(new_dataset)
    unique_accents = grouped_dataset['dialect'].value_counts()
    filtered_dataset = filter_out_accents(grouped_dataset)

    dialects_to_limit = ["American", "SouthAfrican", "English", "Scottish", "Canadian", "Philippines"]
    MAX_SAMPLES = 10000
    
    for dialect in dialects_to_limit:
        dialect_samples = filtered_dataset[filtered_dataset['dialect'] == dialect]
        if len(dialect_samples) > MAX_SAMPLES:
            print(f"Limiting {dialect} from {len(dialect_samples)} to {MAX_SAMPLES} samples")
            # Get indices of samples to keep (random selection)
            keep_indices = dialect_samples.sample(n=MAX_SAMPLES, random_state=42).index
            # Get indices to drop
            drop_indices = dialect_samples.index.difference(keep_indices)
            # Drop the excess samples
            filtered_dataset = filtered_dataset.drop(index=drop_indices)
    
    return filtered_dataset, unique_accents


def group_accents(df):
    # Group accents by Region
    region_mapping = {
        # East Asia
        "Japan": "East Asia",
        "South Korea": "East Asia",
        "Hong Kong": "East Asia",
        "China": "East Asia",
        
        # South East Asia
        "Indonesia": "South East Asia",
        "Malaysia": "South East Asia",
        "Thailand": "South East Asia",
        "Vietnam": "South East Asia",
        "Singapore": "South East Asia",
        "Philippines": "South East Asia",
        
        # Eastern Europe
        "Russia": "Eastern Europe",
        "Hungary": "Eastern Europe",
        "Poland": "Eastern Europe",
        "Austria": "Eastern Europe",
        "Slovakia": "Eastern Europe",
        "Greece": "Eastern Europe",
        "Ukraine": "Eastern Europe",
        "Czech Republic": "Eastern Europe",
        "Turkey": "Eastern Europe",
        "Bulgaria": "Eastern Europe",
        "Romania": "Eastern Europe",
        "Croatia": "Eastern Europe",
        "Latvia": "Eastern Europe",
        
        # Nordic
        "Sweden": "Nordic",
        "Finland": "Nordic",
        "Norway": "Nordic",
        "Denmark": "Nordic",
        "Netherlands": "Nordic",
        
        # Southern Europe
        "Spain": "Southern Europe",
        "Italy": "Southern Europe",
        
        # Western Africa
        "Ghana": "Western Africa",
        "Nigeria": "Western Africa",
        
        # East Africa
        "Kenya": "East Africa",
        "East Africa": "East Africa",

        "Nepal": "South Asia",
        "Bangladesh": "South Asia",
        
    }
    
    # Apply the mapping to the dialect column
    df['region'] = df['dialect'].map(region_mapping)
    
    # For accents that don't have a region mapping, keep the original dialect
    df['region'] = df['region'].fillna(df['dialect'])
    
    # Create a new mapping from original dialects to their regions
    dialect_to_region = df[['dialect', 'region']].drop_duplicates().set_index('dialect')['region'].to_dict()
    
    # Get unique regions
    unique_regions = df['region'].unique().tolist()
    
    return df, dialect_to_region, unique_regions
    
    
def filter_out_accents(df, min_samples=100):
    """
    Filter out accents that have fewer than min_samples samples.
    
    Args:
        df: DataFrame containing the accent data
        min_samples: Minimum number of samples required to keep an accent (default: 100)
    
    Returns:
        Filtered DataFrame with only accents that have at least min_samples samples
    """
    value_counts = df['region'].value_counts()
    print(value_counts, 'value counts')
    
    # Get accents with at least min_samples samples
    valid_accents = value_counts[value_counts >= min_samples].index.tolist()
    
    # Filter the DataFrame to keep only rows with valid accents
    filtered_df = df[df['region'].isin(valid_accents)]
    
    print(f"Original dataset had {len(df)} samples across {len(value_counts)} accents")
    print(f"Filtered dataset has {len(filtered_df)} samples across {len(valid_accents)} accents")

    print(filtered_df['region'].unique(), 'remaining dialects')
    print("\nDialects filtered out (fewer than {min_samples} samples):")
    print(value_counts[value_counts < min_samples].sort_index())
    filtered_df.drop(columns=['dialect'], inplace=True)
    filtered_df = filtered_df.rename(columns={'region': 'dialect'})
    
    return filtered_df




def main():


    dataset_df = load_datasets(sample_size=100)
    accent_counts = dataset_df['dialect'].value_counts()
    # print(accent_counts, 'accent counts of commonvoice dataset')
    dataset_df = limit_accents(dataset_df, accents_to_limit=["English"])
    new_dataset, unique_accents = load_and_process_commonvoice()
    

    concatenated_df = pd.concat([dataset_df, new_dataset], ignore_index=True)
    # print("accent counts after concat", concatenated_df['dialect'].value_counts())
    # number_of_accents = len(concatenated_df['dialect'].unique())

    model = setup_model(checkpoint_file=model_path, num_accent_classes=NUM_ACCENT_CLASSES)
    dataloader = prepare_data(dataset_df, samples_per_accent=100)
    model.eval()
    # After running your model:
    embeddings = []
    accent_labels = []
    for batch in dataloader:
        mel = batch['mel'].to(DEVICE)
        # text = batch['text'].to(DEVICE)
        target = batch['target'].to(DEVICE)

        accent_preds, emb = model(mel)
        embeddings.append(emb.detach().cpu().numpy())
        accent_labels.append(target.cpu().numpy())
        # ... collect corresponding accent labels
    print(len(embeddings), "embeddings.shape")
    print(len(accent_labels), "accent_labels.shape")
    embeddings = np.vstack(embeddings)
    accent_labels = np.concatenate(accent_labels)

    print(embeddings.shape, "embeddings.shape")
    print(accent_labels.shape, "accent_labels.shape")
    # PCA visualization
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(embeddings)

    distinct_colors = {
        0: '#000080',  # Scottish - Navy Blue
        1: '#1E90FF',  # English - Dodger Blue
        2: '#FFA500',  # Indian - Orange
        3: '#4169E1',  # Irish - Royal Blue
        4: '#87CEEB',  # Welsh - Sky Blue
        5: '#90EE90',  # NewZealandEnglish - Light Green
        6: '#228B22',  # AustralianEnglish - Forest Green
        7: '#32CD32',  # SouthAfrican - Lime Green
        8: '#8B0000',  # Canadian - Dark Red
        9: '#800080',  # NorthernIrish - Purple
        10: '#FF0000', # American - Bright Red
        # 10 additional distinct colors
        11: '#FF1493', # Deep Pink
        12: '#FFD700', # Gold
        13: '#00CED1', # Dark Turquoise
        14: '#9400D3', # Dark Violet
        15: '#FF4500', # Orange Red
        16: '#2E8B57', # Sea Green
        17: '#DAA520', # Goldenrod
        18: '#FF00FF', # Magenta
        19: '#1E90FF', # Dodger Blue
        20: '#8A2BE2', # Blue Violet

    }
    
    # PCA visualization
    fig = plt.figure(figsize=(20, 8))
    
    # PCA visualization - 2D
    # plt.subplot(121)
    # for i in range(NUM_ACCENT_CLASSES):
    #     mask = accent_labels == i
    #     plt.scatter(pca_result[mask, 0], pca_result[mask, 1], 
    #                color=distinct_colors[i], label=ID_TO_ACCENT[i], alpha=0.6)
    # plt.grid(True, alpha=0.3)
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    # plt.title('PCA visualization of accent embeddings (2D)')
    
    # # PCA visualization - 3D
    # pca3d = PCA(n_components=3)
    # pca3d_result = pca3d.fit_transform(embeddings)
    
    # ax = fig.add_subplot(122, projection='3d')
    # for i in range(NUM_ACCENT_CLASSES):
    #     mask = accent_labels == i
    #     ax.scatter(pca3d_result[mask, 0], pca3d_result[mask, 1], pca3d_result[mask, 2],
    #               color=distinct_colors[i], label=ID_TO_ACCENT[i], alpha=0.6)
    # ax.grid(True, alpha=0.3)
    # ax.set_title('PCA visualization of accent embeddings (3D)')
    # plt.tight_layout()
    # plt.show()

    # # t-SNE visualizations
    # fig = plt.figure(figsize=(20, 8))
    
    # t-SNE - 2D
    plt.figure(figsize=(20, 16), dpi=300)  # Increased figure size and DPI for high resolution
    tsne = TSNE(n_components=2, random_state=42)
    tsne_result = tsne.fit_transform(embeddings)
    for i in range(NUM_ACCENT_CLASSES):
        mask = accent_labels == i
        plt.scatter(tsne_result[mask, 0], tsne_result[mask, 1], 
                   color=distinct_colors[i], label=ID_TO_ACCENT[i], alpha=0.6)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
    plt.title('t-SNE visualization of accent embeddings (2D)', fontsize=16)
    plt.tight_layout()
    plt.savefig('tsne_2d_accents.png', dpi=300, bbox_inches='tight')
    
    # t-SNE - 3D
    fig = plt.figure(figsize=(20, 16), dpi=300)  # Increased figure size and DPI for high resolution
    tsne3d = TSNE(n_components=3, random_state=42)
    tsne3d_result = tsne3d.fit_transform(embeddings)
    
    ax = fig.add_subplot(111, projection='3d')
    for i in range(NUM_ACCENT_CLASSES):
        mask = accent_labels == i
        ax.scatter(tsne3d_result[mask, 0], tsne3d_result[mask, 1], tsne3d_result[mask, 2],
                  color=distinct_colors[i], label=ID_TO_ACCENT[i], alpha=0.6)
    ax.grid(True, alpha=0.3)
    ax.set_title('t-SNE visualization of accent embeddings (3D)', fontsize=16)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
    plt.tight_layout()
    plt.savefig('tsne_3d_accents.png', dpi=300, bbox_inches='tight')
    
    # Show the plots after saving
    plt.show()



if __name__ == "__main__":
    main()