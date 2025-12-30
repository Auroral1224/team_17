# Designed by 314511063
import torch
import torch.nn as nn
import open_clip
import pytorch_lightning as pl
from peft import LoraConfig, get_peft_model
from PIL import Image
from torch.utils.data import Dataset
from torchvision.ops import sigmoid_focal_loss

# Config
MODEL_NAME = 'hf-hub:laion/CLIP-convnext_xxlarge-laion2B-s34B-b82K-augreg-soup'
BATCH_SIZE = 8
GRAD_ACCUM = 2  # Effective Batch Size = 16
EPOCHS = 5
LR = 1e-4
SEED = 42
LORA_R = 16
LORA_ALPHA = 16
LORA_TARGET_MODULES = ["fc1", "fc2"]
LORA_DROPOUT = 0.1
USE_DORA = False

class CustomDataset(Dataset):
    def __init__(self, paths, labels=None, transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        try:
            img = Image.open(path).convert("RGB")
            if self.transform:
                img = self.transform(img)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            img = torch.zeros(3, 256, 256)

        if self.labels is not None:
            return img, torch.tensor(self.labels[idx], dtype=torch.float32)
        else:
            return img, path.stem

class LitDeepfakeModule(pl.LightningModule):
    def __init__(self, model_name=MODEL_NAME, lr=LR):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        
        # Load Model & Transforms
        print(f"Loading {model_name}...")
        clip_model, self.preprocess_train, self.preprocess_val = open_clip.create_model_and_transforms(model_name)
        
        # Define Backbone (Visual)
        self.backbone = clip_model.visual.trunk
        
        # Modify Backbone to output spatial features (Identity head) - Matching AIDE
        self.backbone.head.global_pool = nn.Identity()
        self.backbone.head.flatten = nn.Identity()
        
        # Explicit pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # AIDE Projection Layer (Matching AIDE.py Line 242)
        self.convnext_proj = nn.Linear(3072, 256)
            
        # Final Binary Classifier
        self.head = nn.Linear(256, 1)
        
        # Apply LoRA to Backbone
        print("Applying LoRA to backbone...")
        lora_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules=LORA_TARGET_MODULES,
            lora_dropout=LORA_DROPOUT,
            use_dora=USE_DORA,
            bias="none"
        )
        self.backbone = get_peft_model(self.backbone, lora_config)
        self.backbone.print_trainable_parameters()

    def forward(self, x):
        features = self.backbone(x) # [B, 3072, H, W]
        x = self.avgpool(features)  # [B, 3072, 1, 1]
        x = x.view(x.size(0), -1)   # [B, 3072]
        
        # Projection (AIDE style)
        x = self.convnext_proj(x)   # [B, 256]
        
        # Classifier
        return self.head(x)

    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        logits = self(imgs).squeeze(1)
        loss = sigmoid_focal_loss(logits, labels, reduction='mean')
        
        acc = ((torch.sigmoid(logits) > 0.5) == labels).float().mean()
        
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        logits = self(imgs).squeeze(1)
        loss = sigmoid_focal_loss(logits, labels, reduction='mean')
        
        acc = ((torch.sigmoid(logits) > 0.5) == labels).float().mean()
        
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_acc", acc, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
