import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from torchmetrics.classification import BinaryAUROC
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

# ---------------------------------------------------------------------
# Fractions Skill Score (FSS)
# ---------------------------------------------------------------------
def compute_fss(preds: torch.Tensor, targets: torch.Tensor, window: int = 9) -> torch.Tensor:
    pool = nn.AvgPool2d(window, 1, window // 2)
    p = pool(preds)
    t = pool(targets)
    mse = torch.mean((p - t) ** 2)
    ref = torch.mean(p ** 2) + torch.mean(t ** 2)
    fss = 1 - mse / (ref + 1e-6)
    return fss.clamp(min=0.0, max=1.0)


class FSSLoss(nn.Module):
    def __init__(self, window_size=9):
        super().__init__()
        self.pool = nn.AvgPool2d(window_size, 1, window_size // 2)

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        return F.mse_loss(self.pool(probs), self.pool(targets))

# ---------------------------------------------------------------------
# Dataset with lag scaling
# ---------------------------------------------------------------------
class ShardDataset(Dataset):
    def __init__(self, shard_dir: str):
        super().__init__()
        files = sorted(f for f in os.listdir(shard_dir) if f.endswith(".pt"))
        if not files:
            raise RuntimeError(f"No shards found in {shard_dir}")

        self.X, self.G, self.Y = [], [], []
        for f in files:
            data = torch.load(os.path.join(shard_dir, f), map_location="cpu")
            self.X.append(data["X"].half())
            self.G.append(data["G"].half())
            self.Y.append(data["Y"])

        self.X = torch.cat(self.X, dim=0)
        self.G = torch.cat(self.G, dim=0)
        self.Y = torch.cat(self.Y, dim=0)

        print(f"Loaded {len(self.X):,} samples from {len(files)} shards "
              f"({self.X.shape[1]}x{self.X.shape[2]})")

        # Optional: pre-apply lag scaling once here
        lag_col = 9
        self.X[..., lag_col] = 2 * (self.X[..., lag_col] / 120.0) - 1.0  # 0–120 → −1…1

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x, g, y = self.X[i].float(), self.G[i].float(), self.Y[i].unsqueeze(0).float()
        return x, g, y

# ---------------------------------------------------------------------
# Smooth Decoder for 64×64 latent
# ---------------------------------------------------------------------
class SmoothDecoder(nn.Module):
    def __init__(self, embed_dim: int, out_hw=(350, 370)):
        super().__init__()
        self.out_hw = out_hw
        self.layers = nn.Sequential(
            nn.Conv2d(embed_dim, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 1, 1)
        )

    def forward(self, x):
        x = F.interpolate(x, size=self.out_hw, mode="bilinear", align_corners=False)
        return self.layers(x)

# ---------------------------------------------------------------------
# Model (64×64 latent + lag scaling)
# ---------------------------------------------------------------------
class Core2MapModel(pl.LightningModule):
    def __init__(self, embed_dim=128, num_heads=4, num_layers=4, lr=1e-4, out_hw=(350, 370)):
        super().__init__()
        self.save_hyperparameters()

        # Transformer encoder
        self.in_proj = nn.Linear(10, embed_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=4 * embed_dim, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # Global projection
        self.global_proj = nn.Linear(4, embed_dim)

        # Map projection to 64×64 latent
        self.map_proj = nn.Linear(embed_dim, embed_dim * 64 * 64)

        # Decoder
        self.decoder = SmoothDecoder(embed_dim, out_hw=out_hw)

        # FSS loss and metric
        self.criterion = FSSLoss(window_size=9)
        self.val_auc = BinaryAUROC()
        self.mask_col = 8

    def forward(self, x, g):
        b, s, f = x.shape
        mask = (x[..., self.mask_col] <= 0.0)
        x = self.in_proj(x)
        x = self.transformer(x, src_key_padding_mask=mask)
        valid = (~mask).float().unsqueeze(-1)
        pooled = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        g_emb = self.global_proj(g)
        combined = pooled + g_emb
        x = self.map_proj(combined).view(b, -1, 64, 64)
        return self.decoder(x)

    def training_step(self, batch, _):
        x, g, y = (t.to(self.device) for t in batch)
        logits = self(x, g)
        loss = self.criterion(logits, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, _):
        x, g, y = (t.to(self.device) for t in batch)
        preds = torch.sigmoid(self(x, g))
        for w in [3, 5, 9]:
            fss_val = compute_fss(preds, y, window=w)
            self.log(f"val_fss_{w}", fss_val, on_epoch=True, prog_bar=(w == 9), sync_dist=True)
        self.val_auc.update(preds.flatten(), y.flatten().int())

    def on_validation_epoch_end(self):
        auc = self.val_auc.compute()
        self.log("val_auc", auc, prog_bar=True, sync_dist=True)
        self.val_auc.reset()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    torch.set_float32_matmul_precision("high")

    LEAD_TIME = "1"
    BASE_DIR = f"/gws/nopw/j04/wiser_ewsa/mrakotomanga/Intercomparison/preprocessed/t{LEAD_TIME}"
    TRAIN_DIR = f"{BASE_DIR}/train_t{LEAD_TIME}"
    VAL_DIR   = f"{BASE_DIR}/val_t{LEAD_TIME}"
    BATCH_SIZE = 32
    CHECKPOINT_DIR = f"/gws/nopw/j04/wiser_ewsa/mrakotomanga/Intercomparison/checkpoints/t{LEAD_TIME}_64x64"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    train_ds = ShardDataset(TRAIN_DIR)
    val_ds   = ShardDataset(VAL_DIR)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True, prefetch_factor=2)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True, prefetch_factor=2)

    model = Core2MapModel(embed_dim=128, num_heads=4, num_layers=4, lr=1e-4)
    logger = WandbLogger(project="zambia-cast", name=f"t{LEAD_TIME}-64x64-lag-scaled")

    trainer = pl.Trainer(
        max_epochs=20,
        precision="bf16-mixed",
        accelerator="gpu", devices=4, strategy="ddp",
        logger=logger, log_every_n_steps=1,
        callbacks=[
            ModelCheckpoint(
                dirpath=CHECKPOINT_DIR,
                filename="best-core2map",
                monitor="val_fss_9",
                mode="max",
                save_top_k=1,
                verbose=True
            ),
            EarlyStopping(
                monitor="val_fss_9",
                mode="max",
                patience=5,
                min_delta=0.001,
                verbose=True
            )
        ]
    )

    trainer.fit(model, train_loader, val_loader)
    print("Done.")

if __name__ == "__main__":
    main()
