import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from torchmetrics.classification import BinaryAUROC
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import numpy as np
import random

# ---------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def compute_fss(preds, targets, window=9):
    pool = nn.AvgPool2d(window, 1, window // 2)
    p = pool(preds)
    t = pool(targets)
    mse = torch.mean((p - t) ** 2)
    ref = torch.mean(p ** 2) + torch.mean(t ** 2)
    return (1 - mse / (ref + 1e-8)).clamp(0.0, 1.0)

# ---------------------------------------------------------------------
# Multi-scale Hybrid BCE + FSS loss
# ---------------------------------------------------------------------
class MultiScaleHybridLoss(nn.Module):
    def __init__(self, windows=(5, 9, 17), weights=(0.25, 0.5, 0.25),
                 pos_weight=25.0, alpha=0.7):
        super().__init__()
        assert len(windows) == len(weights)
        self.pools = nn.ModuleList([nn.AvgPool2d(k, 1, k // 2) for k in windows])
        self.weights = torch.tensor(weights, dtype=torch.float32)
        self.alpha = alpha
        self.register_buffer("pos_weight", torch.tensor(float(pos_weight), dtype=torch.float32))
        self.bce = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        fss_terms = [w * F.mse_loss(pool(probs), pool(targets))
                     for w, pool in zip(self.weights, self.pools)]
        fss_loss = torch.stack(fss_terms).sum()
        bce_loss = self.bce(logits, targets)
        return self.alpha * bce_loss + (1 - self.alpha) * fss_loss

# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------
class ShardDataset(Dataset):
    def __init__(self, shard_dir):
        files = sorted(f for f in os.listdir(shard_dir) if f.endswith(".pt"))
        if not files:
            raise RuntimeError(f"No shards found in {shard_dir}")
        X, G, Y = [], [], []
        for f in files:
            d = torch.load(os.path.join(shard_dir, f), map_location="cpu")
            X.append(d["X"].half()); G.append(d["G"].half()); Y.append(d["Y"])
        self.X, self.G, self.Y = torch.cat(X), torch.cat(G), torch.cat(Y)
        print(f"Loaded {len(self.X)} samples from {len(files)} shards ({self.X.shape[1]}x{self.X.shape[2]})")

    def __len__(self): return len(self.X)

    def __getitem__(self, i):
        x, g, y = self.X[i].float(), self.G[i].float(), self.Y[i].unsqueeze(0).float()
        lag_col = 9
        x[..., lag_col] = 2 * (x[..., lag_col] / 120.0) - 1.0
        return x, g, y

# ---------------------------------------------------------------------
# Smooth decoder (upsample + conv)
# ---------------------------------------------------------------------
class SmoothDecoder(nn.Module):
    def __init__(self, embed_dim, out_hw=(350, 370), dropout_p=0.15):
        super().__init__()
        self.out_hw = out_hw
        ch = [embed_dim, 256, 128, 64, 32]
        blocks = []
        in_c = embed_dim
        for c in ch[1:]:
            blocks += [
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(in_c, c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(c),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout_p)
            ]
            in_c = c
        self.up = nn.Sequential(*blocks)
        self.final = nn.Conv2d(ch[-1], 1, kernel_size=1)

    def forward(self, x):
        x = self.up(x)
        if x.shape[-2:] != self.out_hw:
            x = F.interpolate(x, size=self.out_hw, mode="bilinear", align_corners=False)
        return self.final(x)

# ---------------------------------------------------------------------
# Core2Map Model
# ---------------------------------------------------------------------
class Core2MapModel(pl.LightningModule):
    def __init__(self, embed_dim=128, num_heads=4, num_layers=4,
                 lr=1e-4, dropout_p=0.15, out_hw=(350, 370),
                 pos_weight=25.0, alpha=0.7,
                 logit_init=1.3, contrast_gain=1.3):
        super().__init__()
        self.save_hyperparameters()

        # Transformer encoder
        self.in_proj = nn.Sequential(nn.Linear(10, embed_dim), nn.Dropout(dropout_p))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=4 * embed_dim, dropout=dropout_p, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.global_proj = nn.Linear(4, embed_dim)
        self.map_proj = nn.Linear(embed_dim, embed_dim * 16 * 16)
        self.decoder = SmoothDecoder(embed_dim, out_hw=out_hw, dropout_p=dropout_p)

        # Loss
        self.criterion = MultiScaleHybridLoss(
            windows=(5, 9, 17), weights=(0.25, 0.5, 0.25),
            pos_weight=pos_weight, alpha=alpha
        )

        self.val_auc = BinaryAUROC()
        self.mask_col = 8
        self.logit_scale = nn.Parameter(torch.tensor(logit_init, dtype=torch.float32))
        self.contrast_gain = contrast_gain

    def forward(self, x, g):
        b, s, f = x.shape
        mask = (x[..., self.mask_col] <= 0)
        x = self.in_proj(x)
        x = self.transformer(x, src_key_padding_mask=mask)
        valid = (~mask).float().unsqueeze(-1)
        pooled = (x * valid).sum(1) / valid.sum(1).clamp_min(1.0)
        g_emb = self.global_proj(g)
        x = self.map_proj(pooled + g_emb).view(b, -1, 16, 16)
        logits = self.decoder(x)
        # Logit sharpening for contrast
        return self.contrast_gain * logits * F.softplus(self.logit_scale)

    def training_step(self, batch, _):
        x, g, y = (t.to(self.device) for t in batch)
        loss = self.criterion(self(x, g), y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, _):
        x, g, y = (t.to(self.device) for t in batch)
        preds = torch.sigmoid(self(x, g))
        for w in [3, 5, 9]:
            self.log(f"val_fss_{w}", compute_fss(preds, y, w),
                     on_epoch=True, prog_bar=(w == 9), sync_dist=True)
        self.val_auc.update(preds.flatten(), y.flatten().int())

    def on_validation_epoch_end(self):
        self.log("val_auc", self.val_auc.compute(), prog_bar=True, sync_dist=True)
        self.val_auc.reset()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)

# ---------------------------------------------------------------------
# Training entry
# ---------------------------------------------------------------------
def main():
    torch.set_float32_matmul_precision("high")

    lead_time = sys.argv[1]
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    set_seed(seed)
    pl.seed_everything(seed, workers=True)

    base_dir = f"/gws/nopw/j04/wiser_ewsa/mrakotomanga/Intercomparison/preprocessed/t{lead_time}"
    train_dir = f"{base_dir}/train_t{lead_time}"
    val_dir   = f"{base_dir}/val_t{lead_time}"

    # save in new folder to avoid overwriting previous runs
    ckpt_dir  = f"/gws/nopw/j04/wiser_ewsa/mrakotomanga/Intercomparison/checkpoints/ensemble_multiscale/sharp/t{lead_time}/seed{seed}"
    os.makedirs(ckpt_dir, exist_ok=True)

    train_ds = ShardDataset(train_dir)
    val_ds   = ShardDataset(val_dir)

    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True,
                          num_workers=4, pin_memory=True, persistent_workers=True)
    val_dl   = DataLoader(val_ds, batch_size=32, shuffle=False,
                          num_workers=4, pin_memory=True, persistent_workers=True)

    model = Core2MapModel(embed_dim=128, num_heads=4, num_layers=4,
                          dropout_p=0.15, pos_weight=25.0, alpha=0.7,
                          logit_init=1.3, contrast_gain=1.3)

    logger = WandbLogger(project="zambia-multiscale-hybrid-sharp", name=f"t{lead_time}-seed{seed}")

    trainer = pl.Trainer(
        max_epochs=25,
        accelerator="gpu", devices=4, strategy="ddp",
        precision="bf16-mixed",
        logger=logger,
        log_every_n_steps=5,
        gradient_clip_val=1.0, gradient_clip_algorithm="norm",
        callbacks=[
            ModelCheckpoint(
                dirpath=ckpt_dir, filename="best-core2map",
                monitor="val_fss_9", mode="max", save_top_k=1, verbose=True
            ),
            EarlyStopping(
                monitor="val_fss_9", mode="max",
                patience=5, min_delta=0.001, verbose=True
            )
        ]
    )

    trainer.fit(model, train_dl, val_dl)
    print(f"Training complete for seed {seed}")

if __name__ == "__main__":
    main()
