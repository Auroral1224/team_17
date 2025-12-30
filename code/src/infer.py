# Designed by 314511063
import os
import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from common import CustomDataset, LitDeepfakeModule, BATCH_SIZE

def find_best_checkpoint(search_dirs):
    for d in search_dirs:
        # Search for best-checkpoint.ckpt in all subdirectories of lightning_logs
        # Pattern: directory/lightning_logs/version_*/checkpoints/best-checkpoint.ckpt
        candidates = list(d.rglob("best-checkpoint.ckpt"))
        if candidates:
            # Return the most recently modified one
            return max(candidates, key=os.path.getmtime)
    return None

def main():
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.cuda.empty_cache()

    # Paths
    possible_dataset_paths = [
        Path("../dataset"),       # If running from src/
        Path("./dataset")        # If running from project root
    ]
    DATASET_DIR = None
    for p in possible_dataset_paths:
        if p.exists():
            DATASET_DIR = p
            break
            
    if DATASET_DIR is None:
        raise FileNotFoundError("Could not find 'dataset' directory.")
        
    TEST_DIR = DATASET_DIR / "test"
    
    if not TEST_DIR.exists():
        raise FileNotFoundError(f"Test directory not found at {TEST_DIR}")
        
    test_paths = sorted(list(TEST_DIR.glob("*.jpg")), key=lambda p: int(p.stem))
    print(f"Test images found: {len(test_paths)}")

    # Locate Checkpoint
    # Look in ../lightning_logs (if running from src) or ./lightning_logs
    possible_log_dirs = [
        Path("../lightning_logs"),
        Path("./lightning_logs")
    ]
    
    checkpoint_path = find_best_checkpoint([p for p in possible_log_dirs if p.exists()])
    
    if not checkpoint_path:
        # Fallback: try to find any checkpoint or ask user
        print("WARNING: Could not auto-locate 'best-checkpoint.ckpt'.")
        # You might need to specify it manually if not found
        # checkpoint_path = "path/to/checkpoint.ckpt" 
        return

    print(f"Loading best checkpoint: {checkpoint_path}")
    
    # Load Model
    # Note: We instantiate module structure first or just load from checkpoint if class is available
    best_model = LitDeepfakeModule.load_from_checkpoint(checkpoint_path)
    best_model.eval()
    best_model.to(DEVICE)

    # Prepare Test Loader
    test_ds = CustomDataset(test_paths, transform=best_model.preprocess_val)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    ids = []
    labels = []

    print("Starting Inference on Test Set...")
    with torch.no_grad():
        for imgs, filenames in tqdm(test_loader, desc="Inference"):
            imgs = imgs.to(DEVICE)
            logits = best_model(imgs).squeeze(1)
            preds = torch.sigmoid(logits)
            
            ids.extend(filenames)
            labels.extend(preds.cpu().numpy())

    # Create Submission
    threshold = np.percentile(labels, 50)
    final_labels = ["fake" if p > threshold else "real" for p in labels]

    df_sub = pd.DataFrame({"filename": ids, "label": final_labels})

    # Sort numerically
    df_sub["filename_int"] = df_sub["filename"].astype(int)
    df_sub = df_sub.sort_values("filename_int").drop(columns=["filename_int"])

    df_sub.to_csv("submission.csv", index=False)
    print("Saved submission.csv")
    print(df_sub.head())

if __name__ == "__main__":
    main()
