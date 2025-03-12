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
    dataloader = prepare_data(dataset_df, samples_per_accent=100)
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
    plt.subplot(121)
    tsne = TSNE(n_components=2, random_state=42)
    tsne_result = tsne.fit_transform(embeddings)
    for i in range(NUM_ACCENT_CLASSES):
        mask = accent_labels == i
        plt.scatter(tsne_result[mask, 0], tsne_result[mask, 1], 
                   color=distinct_colors[i], label=ID_TO_ACCENT[i], alpha=0.6)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title('t-SNE visualization of accent embeddings (2D)')
    
    # t-SNE - 3D
    tsne3d = TSNE(n_components=3, random_state=42)
    tsne3d_result = tsne3d.fit_transform(embeddings)
    
    ax = fig.add_subplot(122, projection='3d')
    for i in range(NUM_ACCENT_CLASSES):
        mask = accent_labels == i
        ax.scatter(tsne3d_result[mask, 0], tsne3d_result[mask, 1], tsne3d_result[mask, 2],
                  color=distinct_colors[i], label=ID_TO_ACCENT[i], alpha=0.6)
    ax.grid(True, alpha=0.3)
    ax.set_title('t-SNE visualization of accent embeddings (3D)')
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    main()