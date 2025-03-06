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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_VARIANT = "small"
NUM_ACCENT_CLASSES = 11
ID_TO_ACCENT = {
    0: "Scottish", 1: "English", 2: "Indian", 3: "Irish", 4: "Welsh",
    5: "NewZealandEnglish", 6: "AustralianEnglish", 7: "SouthAfrican",
    8: "Canadian", 9: "NorthernIrish", 10: "American"
}

model_path = "model_11_accents.pt"

def load_model():
    model = setup_model()
    checkpoint = torch.load(model_path, map_location=torch.device(DEVICE))
    try:
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print('Model loaded successfully')
        else:
            model.load_state_dict(checkpoint)
            print('Model loaded successfully')
    except Exception as e:
        raise e
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












def main():


    dataset_df = load_datasets()
    model = load_model()
    dataloader = prepare_data(dataset_df, samples_per_accent=5)
    model.eval()
    # After running your model:
    embeddings = []
    accent_labels = []
    for batch in dataloader:
        mel = batch['mel'].to(DEVICE)
        text = batch['text'].to(DEVICE)
        target = batch['target'].to(DEVICE)

        accent_preds, emb = model(mel, text)
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

    distinct_colors = [
        '#e41a1c',  # Red
        '#377eb8',  # Blue
        '#4daf4a',  # Green
        '#984ea3',  # Purple
        '#ff7f00',  # Orange
        '#ffff33',  # Yellow
        '#a65628',  # Brown
        '#f781bf',  # Pink
        '#00ffff',  # Cyan
        '#808080',  # Gray
        '#000000',  # Black
    ]
    
    # PCA visualization
    plt.figure(figsize=(12, 8))
    for i in range(NUM_ACCENT_CLASSES):
        mask = accent_labels == i
        plt.scatter(pca_result[mask, 0], pca_result[mask, 1], 
                   color=distinct_colors[i], label=ID_TO_ACCENT[i], alpha=0.6)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title('PCA visualization of accent embeddings')
    plt.tight_layout()
    plt.show()

    # t-SNE visualization
    tsne = TSNE(n_components=2, random_state=42)
    tsne_result = tsne.fit_transform(embeddings)

    plt.figure(figsize=(12, 8))
    for i in range(NUM_ACCENT_CLASSES):
        mask = accent_labels == i
        plt.scatter(tsne_result[mask, 0], tsne_result[mask, 1], 
                   color=distinct_colors[i], label=ID_TO_ACCENT[i], alpha=0.6)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title('t-SNE visualization of accent embeddings')
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    main()