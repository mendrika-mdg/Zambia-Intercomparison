import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"

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
    # preds, targets expected in [0,1], shape [B,1,H,W]
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
# Dataset
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
        print(f"Loaded {len(self.X):,} samples from {len(files)} shards ({self.X.shape[1]}x{self.X.shape[2]})")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i].float(), self.G[i].float(), self.Y[i].unsqueeze(0).float()

# ---------------------------------------------------------------------
# Adaptive Token Grid (with bilinear splatting)
# ---------------------------------------------------------------------
class AdaptiveTokenGrid(nn.Module):
    def __init__(self, embed_dim: int, H: int = 64, W: int = 64, lat_idx: int = 0, lon_idx: int = 1, mask_idx: int = 8):
        super().__init__()
        self.D, self.H, self.W = embed_dim, H, W
        self.lat_idx, self.lon_idx, self.mask_idx = lat_idx, lon_idx, mask_idx

        # Zambia bounds
        self.domain_lat_min = -18.414806
        self.domain_lat_max = -7.9918404
        self.domain_lon_min = 21.167515
        self.domain_lon_max = 35.316326

        # StandardScaler stats for lat/lon (z-scores → degrees)
        self.mean_lat, self.scale_lat = -13.1573589, 2.86575632
        self.mean_lon, self.scale_lon = 27.7910228, 3.56468299

    def forward(self, token_feats: torch.Tensor, x_raw: torch.Tensor) -> torch.Tensor:
        B, S, D = token_feats.shape
        H, W = self.H, self.W
        device = token_feats.device

        # Undo standardisation
        lat = x_raw[..., self.lat_idx] * self.scale_lat + self.mean_lat
        lon = x_raw[..., self.lon_idx] * self.scale_lon + self.mean_lon

        # Map degrees → [0,1] using Zambia bounds; flip latitude so north is up
        lat01 = (lat - self.domain_lat_min) / (self.domain_lat_max - self.domain_lat_min)
        lon01 = (lon - self.domain_lon_min) / (self.domain_lon_max - self.domain_lon_min)
        lat01 = 1.0 - lat01.clamp(0, 1)
        lon01 = lon01.clamp(0, 1)

        # Valid mask for real cores
        valid = (x_raw[..., self.mask_idx] > 0).float()                  # [B,S]
        feats = token_feats * valid.unsqueeze(-1)                        # [B,S,D]

        # Continuous indices
        iyf = lat01 * (H - 1)
        ixf = lon01 * (W - 1)

        iy0 = iyf.floor().long().clamp(0, H - 1)
        ix0 = ixf.floor().long().clamp(0, W - 1)
        iy1 = (iy0 + 1).clamp(0, H - 1)
        ix1 = (ix0 + 1).clamp(0, W - 1)

        wy = (iyf - iy0.float()).unsqueeze(1)                            # [B,1,S]
        wx = (ixf - ix0.float()).unsqueeze(1)
        w00 = (1 - wy) * (1 - wx)
        w01 = (1 - wy) * wx
        w10 = wy * (1 - wx)
        w11 = wy * wx

        grid = torch.zeros(B, D, H * W, device=device)
        cnts = torch.zeros(B, 1, H * W, device=device)

        def add(lin, w):
            grid.scatter_add_(2, lin.unsqueeze(1).expand(-1, D, -1), feats.transpose(1, 2) * w)
            cnts.scatter_add_(2, lin.unsqueeze(1), valid.unsqueeze(1) * w)

        lin00 = iy0 * W + ix0
        lin01 = iy0 * W + ix1
        lin10 = iy1 * W + ix0
        lin11 = iy1 * W + ix1

        add(lin00, w00)
        add(lin01, w01)
        add(lin10, w10)
        add(lin11, w11)

        grid = grid / cnts.clamp_min(1.0)
        return grid.view(B, D, H, W)

# ---------------------------------------------------------------------
# Memory-efficient decoder: 64×64 latent → 350×370 output (+ coord channels)
# ---------------------------------------------------------------------
class SmoothDecoder(nn.Module):
    def __init__(self, embed_dim: int, out_hw=(350, 370)):
        super().__init__()
        self.out_hw = out_hw

        self.coord_conv = nn.Conv2d(embed_dim + 2, embed_dim, kernel_size=1)

        ch = [embed_dim, 128, 64, 32]
        blocks = []
        for c1, c2 in zip(ch[:-1], ch[1:]):
            blocks += [
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(c1, c2, kernel_size=3, padding=1),
                nn.BatchNorm2d(c2),
                nn.ReLU(inplace=True),
            ]
        self.up = nn.Sequential(*blocks)
        self.final = nn.Conv2d(ch[-1], 1, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        yy = torch.linspace(0, 1, H, device=x.device).view(1, 1, H, 1).expand(B, 1, H, W)
        xx = torch.linspace(0, 1, W, device=x.device).view(1, 1, 1, W).expand(B, 1, H, W)
        x = torch.cat([x, yy, xx], dim=1)
        x = self.coord_conv(x)
        x = self.up(x)
        x = F.interpolate(x, size=self.out_hw, mode="bilinear", align_corners=False)
        return self.final(x)

# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------
class Core2MapModel(pl.LightningModule):
    def __init__(self, embed_dim=128, num_heads=4, num_layers=4, lr=1e-4, out_hw=(350, 370)):
        super().__init__()
        self.save_hyperparameters()

        self.in_proj = nn.Linear(10, embed_dim)
        enc = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=4 * embed_dim, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc, num_layers=num_layers)

        self.global_proj = nn.Linear(4, embed_dim)
        self.gridder = AdaptiveTokenGrid(embed_dim, H=64, W=64, lat_idx=0, lon_idx=1, mask_idx=8)
        self.decoder = SmoothDecoder(embed_dim, out_hw=out_hw)

        self.criterion_small = FSSLoss(window_size=3)
        self.criterion_mid   = FSSLoss(window_size=5)
        self.criterion_big   = FSSLoss(window_size=9)
        self.val_auc = BinaryAUROC()

        self.mask_col = 8
        self.lag_col = 9

    def forward(self, x, g):
        x = x.clone()
        x[..., self.lag_col] = 2 * (x[..., self.lag_col] / 120.0) - 1.0
        mask = (x[..., self.mask_col] <= 0.0)
        tok = self.in_proj(x)
        tok = self.transformer(tok, src_key_padding_mask=mask)
        latent = self.gridder(tok, x)
        g_emb = self.global_proj(g).unsqueeze(-1).unsqueeze(-1)
        latent = latent + g_emb
        return self.decoder(latent)

    def _multiscale_fss_loss(self, logits, y):
        e = self.current_epoch
        w_small, w_mid, w_big = (0.5, 0.3, 0.2) if e < 5 else (0.3, 0.3, 0.4)
        return (w_small * self.criterion_small(logits, y) +
                w_mid   * self.criterion_mid(logits, y) +
                w_big   * self.criterion_big(logits, y))

    def training_step(self, batch, _):
        x, g, y = (t.to(self.device) for t in batch)

        # Skip degenerate batches to keep FSS gradients alive
        if (x[..., self.mask_col] > 0).sum() == 0 or y.sum() == 0:
            loss = torch.zeros((), device=self.device, requires_grad=True)
            self.log("train_loss", loss, on_step=True, on_epoch=True, sync_dist=True)
            return loss

        logits = self(x, g)
        loss = self._multiscale_fss_loss(logits, y)
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
        self.log("val_auc", self.val_auc.compute(), prog_bar=True, sync_dist=True)
        self.val_auc.reset()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=1e-5)

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    torch.set_float32_matmul_precision("high")

    LEAD_TIME = "1"
    BASE_DIR = f"/gws/nopw/j04/wiser_ewsa/mrakotomanga/Intercomparison/preprocessed/t{LEAD_TIME}"
    TRAIN_DIR = f"{BASE_DIR}/train_t{LEAD_TIME}"
    VAL_DIR   = f"{BASE_DIR}/val_t{LEAD_TIME}"
    BATCH_SIZE = 16

    CHECKPOINT_DIR = f"/gws/nopw/j04/wiser_ewsa/mrakotomanga/Intercomparison/checkpoints/t{LEAD_TIME}_grid"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    train_ds = ShardDataset(TRAIN_DIR)
    val_ds   = ShardDataset(VAL_DIR)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2, pin_memory=True, prefetch_factor=2
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=True, prefetch_factor=2
    )

    model = Core2MapModel(embed_dim=128, num_heads=4, num_layers=4, lr=1e-4)
    logger = WandbLogger(project="zambia-cast", name=f"t{LEAD_TIME}-grid64-fss-only")

    trainer = pl.Trainer(
        max_epochs=30,
        precision="bf16-mixed",
        accelerator="gpu", devices=4, strategy="ddp",
        logger=logger, log_every_n_steps=1,
        callbacks=[
            ModelCheckpoint(
                dirpath=CHECKPOINT_DIR,
                filename="best-core2map-grid",
                monitor="val_fss_9",
                mode="max",
                save_top_k=1
            ),
            EarlyStopping(
                monitor="val_fss_9",
                mode="max",
                patience=7,
                min_delta=0.0005
            )
        ]
    )

    trainer.fit(model, train_loader, val_loader)
    print("Done.")

if __name__ == "__main__":
    main()
