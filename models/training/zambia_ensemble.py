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
    return (1 - mse / (ref + 1e-6)).clamp(0.0, 1.0)

class FSSLoss(nn.Module):
    def __init__(self, window_size=9):
        super().__init__()
        self.pool = nn.AvgPool2d(window_size, 1, window_size // 2)
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        return F.mse_loss(self.pool(probs), self.pool(targets))

# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------
class ShardDataset(Dataset):
    def __init__(self, shard_dir):
        files = sorted(f for f in os.listdir(shard_dir) if f.endswith(".pt"))
        if not files:
            raise RuntimeError(f"No shards in {shard_dir}")
        X, G, Y = [], [], []
        for f in files:
            d = torch.load(os.path.join(shard_dir, f), map_location="cpu")
            X.append(d["X"].half()); G.append(d["G"].half()); Y.append(d["Y"])
        self.X = torch.cat(X); self.G = torch.cat(G); self.Y = torch.cat(Y)
        print(f"Loaded {len(self.X)} samples from {len(files)} shards ({self.X.shape[1]}x{self.X.shape[2]})")

    def __len__(self): return len(self.X)

    def __getitem__(self, i):
        x, g, y = self.X[i].float(), self.G[i].float(), self.Y[i].unsqueeze(0).float()
        lag_col = 9
        x[..., lag_col] = 2 * (x[..., lag_col] / 120.0) - 1.0
        return x, g, y

# ---------------------------------------------------------------------
# Decoder with dropout
# ---------------------------------------------------------------------
class SimpleDecoder(nn.Module):
    def __init__(self, embed_dim, out_hw=(350, 370), dropout_p=0.2):
        super().__init__()
        self.out_hw = out_hw
        ch = [embed_dim, 512, 256, 128, 64, 32]
        blocks = []
        for c1, c2 in zip(ch[:-1], ch[1:]):
            blocks += [
                nn.ConvTranspose2d(c1, c2, 4, 2, 1),
                nn.BatchNorm2d(c2),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout_p)
            ]
        self.up = nn.Sequential(*blocks)
        self.final = nn.Conv2d(ch[-1], 1, 1)
    def forward(self, x):
        x = self.up(x)
        if x.shape[-2:] != self.out_hw:
            x = F.interpolate(x, size=self.out_hw, mode="bilinear", align_corners=False)
        return self.final(x)

# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------
class Core2MapModel(pl.LightningModule):
    def __init__(self, embed_dim=128, num_heads=4, num_layers=4,
                 lr=1e-4, dropout_p=0.2, out_hw=(350, 370)):
        super().__init__()
        self.save_hyperparameters()
        self.in_proj = nn.Sequential(nn.Linear(10, embed_dim), nn.Dropout(dropout_p))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=4 * embed_dim,
            dropout=dropout_p, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.global_proj = nn.Linear(4, embed_dim)
        self.map_proj = nn.Linear(embed_dim, embed_dim * 16 * 16)
        self.decoder = SimpleDecoder(embed_dim, out_hw=out_hw, dropout_p=dropout_p)
        self.criterion = FSSLoss(window_size=9)
        self.val_auc = BinaryAUROC()
        self.mask_col = 8

    def forward(self, x, g):
        b, s, f = x.shape
        mask = (x[..., self.mask_col] <= 0)
        x = self.in_proj(x)
        x = self.transformer(x, src_key_padding_mask=mask)
        valid = (~mask).float().unsqueeze(-1)
        pooled = (x * valid).sum(1) / valid.sum(1).clamp_min(1.0)
        g_emb = self.global_proj(g)
        x = self.map_proj(pooled + g_emb).view(b, -1, 16, 16)
        return self.decoder(x)

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

    base_dir = f"/gws/nopw/j04/wiser_ewsa/mrakotomanga/Intercomparison/preprocessed/t{lead_time}"
    train_dir, val_dir = f"{base_dir}/train_t{lead_time}", f"{base_dir}/val_t{lead_time}"
    ckpt_dir = f"/gws/nopw/j04/wiser_ewsa/mrakotomanga/Intercomparison/checkpoints/ensemble/t{lead_time}/seed{seed}"
    os.makedirs(ckpt_dir, exist_ok=True)

    train_ds, val_ds = ShardDataset(train_dir), ShardDataset(val_dir)
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

    model = Core2MapModel(embed_dim=128, num_heads=4, num_layers=4, dropout_p=0.2)
    logger = WandbLogger(project="zambia-ensemble", name=f"t{lead_time}-seed{seed}")

    trainer = pl.Trainer(
        max_epochs=25,
        accelerator="gpu", devices=4, strategy="ddp",
        precision="bf16-mixed",
        logger=logger,
        log_every_n_steps=5,
        callbacks=[
            ModelCheckpoint(dirpath=ckpt_dir, filename="best-core2map",
                            monitor="val_fss_9", mode="max", save_top_k=1),
            EarlyStopping(monitor="val_fss_9", mode="max", patience=5, min_delta=0.001)
        ]
    )

    trainer.fit(model, train_dl, val_dl)
    print(f"Training done for seed {seed}")

if __name__ == "__main__":
    main()
