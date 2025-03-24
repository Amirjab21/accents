import torch
from datasets import load_dataset
import pandas as pd
from pathlib import Path
from tqdm import tqdm
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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

        # Spain
        "spanish english": "Spain",
        "Spanish bilinguals": "Spain",
        "Spanish": "Spain",
        "Spanglish, England English": "Spain",
        "Spanish accent": "Spain",
        "Spanish from the Canary Islands": "Spain",

        
        # Italy
        "Italian": "Italy",
        
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
        # "Northern Irish, Norwegian, yorkshire": "Norway",
        
        # Denmark
        # "United States English, Irish English, England English, Scottish English, Danish English": "Denmark",
        
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

    }

    df['dialect'] = df['dialect'].map(category_mapping).fillna(df['dialect'])
    unique_accents = list(set(category_mapping.values()))


    return df, unique_accents

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
    
    tuple_examples = df[df['dialect'].apply(lambda x: isinstance(x, tuple))]['dialect'].head(5).tolist()
    if tuple_examples:
        print(f"Found tuples in dialect column. Examples: {tuple_examples}")
        print(f"Number of tuple entries: {df['dialect'].apply(lambda x: isinstance(x, tuple)).sum()}")
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
    
    return filtered_df


def load_new_dataset():
    if DEVICE == "cuda":
        new_dataset, unique_accents = load_new_dataset(['Amirjab21/commonvoice'])
    else:
        # Read all parquet files in dataframes directory and concatenate them
        dataframes_dir = Path("dataframes")
        parquet_files = list(dataframes_dir.glob("*.parquet"))
        concat_df = pd.DataFrame()
        if parquet_files:
            for file in tqdm(parquet_files):
                dataframes = pd.read_parquet(file)
                concat_df = pd.concat([concat_df, dataframes], ignore_index=True)
                # new_dataset = pd.concat(dataframes, ignore_index=True)
            # new_dataset.drop(columns=['down_votes', 'quality_check', '__index_level_0__'], inplace=True)
            concat_df.rename(columns={'accents': 'dialect', 'sentence': 'text'}, inplace=True)
            print(f"Loaded {len(parquet_files)} parquet files from dataframes directory")
            # new_dataset, unique_accents = do_mapping(new_dataset)
        else:
            # Fallback to CSV if no parquet files found
            raise ValueError("No parquet files found in dataframes directory")
    dialect_value_counts = concat_df['dialect'].value_counts()
    dialect_value_counts.to_csv('dialect_value_counts.csv', index=True)
    return concat_df

if __name__ == "__main__":
    try:
        dataset = load_new_dataset()
        mapped_dataset, unique_accents = do_mapping(dataset)
        grouped_dataset, dialect_to_region, unique_regions = group_accents(mapped_dataset)
        filtered = filter_out_accents(grouped_dataset)
        
    except Exception as e:
        import ipdb
        print(f"Error occurred: {str(e)}")
        ipdb.post_mortem()