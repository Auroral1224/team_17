# Designed by 314511063
import torch
import pytorch_lightning as pl
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from pytorch_lightning.callbacks import ModelCheckpoint
from common import CustomDataset, LitDeepfakeModule, SEED, BATCH_SIZE, GRAD_ACCUM, EPOCHS

pl.seed_everything(SEED)

def main():
    # Paths
    # Modified to look relative to src/ or absolute assuming execution from src/
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
        
    print(f"Using dataset at: {DATASET_DIR}")

    TRAIN_REAL_DIR = DATASET_DIR / "train" / "real"
    TRAIN_FAKE_DIR = DATASET_DIR / "train" / "fake"
    TEST_DIR = DATASET_DIR / "test"

    # Collect Files
    train_real_paths = sorted(list(TRAIN_REAL_DIR.glob("*.jpg")))
    train_fake_paths = sorted(list(TRAIN_FAKE_DIR.glob("*.jpg")))
    test_paths = sorted(list(TEST_DIR.glob("*.jpg")), key=lambda p: int(p.stem))

    print(f"Train Real: {len(train_real_paths)}")
    print(f"Train Fake: {len(train_fake_paths)}")
    print(f"Test: {len(test_paths)}")

    # Labels: 0 for Real, 1 for Fake
    all_paths = train_real_paths + train_fake_paths
    all_labels = [0.0] * len(train_real_paths) + [1.0] * len(train_fake_paths)

    # Split Train/Val
    tr_paths, va_paths, tr_labels, va_labels = train_test_split(
        all_paths, all_labels, test_size=0.15, stratify=all_labels, random_state=SEED
    )

    print(f"Training Set: {len(tr_paths)}")
    print(f"Validation Set: {len(va_paths)}")

    torch.cuda.empty_cache()

    # Initialize Module
    lit_model = LitDeepfakeModule()

    # Create DataLoaders
    train_ds = CustomDataset(tr_paths, tr_labels, transform=lit_model.preprocess_train)
    val_ds = CustomDataset(va_paths, va_labels, transform=lit_model.preprocess_val)
    # test_ds = CustomDataset(test_paths, transform=lit_model.preprocess_val) # Not needed for train

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    # test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",
        mode="max",
        filename="best-checkpoint",
        save_top_k=1,
        verbose=True
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator="auto",
        devices="auto",
        precision="16-mixed",
        accumulate_grad_batches=GRAD_ACCUM,
        callbacks=[checkpoint_callback],
        log_every_n_steps=10
    )

    # Train
    trainer.fit(lit_model, train_loader, val_loader)

if __name__ == "__main__":
    main()
