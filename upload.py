from huggingface_hub import HfApi
from datasets import Dataset
import pandas as pd
import os
# Initialize the API
# api = HfApi()
hf_token = os.environ.get('HF_TOKEN')

# # Delete the existing repository
# api.delete_repo(repo_id="Amirjab21/commonvoice", repo_type="dataset")

# # Create a fresh repository (optional - will be created automatically on first push)
# api.create_repo(repo_id="Amirjab21/commonvoice", repo_type="dataset")


# parquet_files = [f for f in os.listdir("dataframes/") if f.endswith(".parquet")]

# for file in parquet_files:
#     processed = pd.read_parquet(f"dataframes/{file}")
#     dataset = Dataset.from_pandas(processed)
#     dataset.push_to_hub("Amirjab21/commonvoice", token=hf_token)


from datasets import Dataset

# Load multiple Parquet files into a dataset
dataset = Dataset.from_parquet("dataframes/*.parquet")

dataset.push_to_hub("Amirjab21/commonvoice", token=hf_token)
