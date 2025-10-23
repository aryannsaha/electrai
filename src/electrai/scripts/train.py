import yaml
import argparse
from types import SimpleNamespace
from collections.abc import Callable
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataloader.registry import get_dataset
from dataloader.dataset import RhoData
from models.srgan_layernorm_pbc import GeneratorResNet


# -------------------------------
# Load YAML config
# -------------------------------
parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    type=str,
    help="Path to YAML config file"
)
args = parser.parse_args()

config_path = Path(args.config)
with open(config_path, "r") as f:
    cfg_dict = yaml.safe_load(f)

cfg = SimpleNamespace(**cfg_dict)

assert 0 < cfg.train_fraction < 1, "train_fraction must be between 0 and 1."
assert 2**cfg.n_upscale_layers == cfg.downsample_data / cfg.downsample_label


-------------------------------
Dataset / DataLoader
-------------------------------
train_sets, test_sets = get_dataset(cfg)

train_data = RhoData(*train_sets,
                     downsample_data=cfg.downsample_data,
                     downsample_label=cfg.downsample_label,
                     data_augmentation=True)

test_data = RhoData(*test_sets,
                    downsample_data=cfg.downsample_data,
                    downsample_label=cfg.downsample_label,
                    data_augmentation=False)

train_loader = DataLoader(train_data, batch_size=int(cfg.nbatch), shuffle=True)
test_loader = DataLoader(test_data, batch_size=int(cfg.nbatch), shuffle=False)

# -------------------------------
# Model / Optimizer / Scheduler
# -------------------------------
torch.cuda.empty_cache()
print(cfg.device)
print(torch.cuda.memory_allocated(cfg.device)/1e9, "GB allocated")
print(torch.cuda.memory_reserved(cfg.device)/1e9, "GB reserved")
model = GeneratorResNet(
    n_residual_blocks=int(cfg.n_residual_blocks),
    n_upscale_layers=int(cfg.n_upscale_layers),
    C=int(cfg.n_channels),
    K1=int(cfg.kernel_size1),
    K2=int(cfg.kernel_size2),
    normalize=not cfg.normalize_label
).to(cfg.device)

print("train chckpt")

optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))

# Linear + Cosine scheduler
linsch = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-5, end_factor=1, total_iters=1)
cossch = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(cfg.epochs)-1)
scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [linsch, cossch], milestones=[1])


# -------------------------------
# Loss function
# -------------------------------
class NormMAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mae = torch.nn.L1Loss(reduction='none')

    def forward(self, output, target):
        mae = self.mae(output, target)
        nelec = torch.sum(target, axis=(-3,-2,-1))
        mae = mae / nelec[...,None,None,None]
        return torch.sum(mae)

loss_fn = NormMAE()


# -------------------------------
# Training / Testing functions
# -------------------------------
def train(dataloader, model, loss_fn, optimizer, t, accum_iter=1):
    size = len(dataloader.dataset)
    model.train()
    if isinstance(loss_fn, dict):
        def loss_fn_sum(output, target):
            loss = 0
            for l, w in zip(loss_fn['loss'], loss_fn['weight']):
                if isinstance(w, Callable):
                    w = w(t)
                loss += w * l(output, target)
            return loss

    optimizer.zero_grad()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(cfg.device), y.to(cfg.device)
        pred = model(X)

        if isinstance(loss_fn, dict):
            loss = loss_fn_sum(pred, y) / accum_iter
        else:
            loss = loss_fn(pred, y) / accum_iter

        loss.backward()
        if ((batch + 1) % accum_iter == 0) or (batch + 1 == len(dataloader)):
            optimizer.step()
            optimizer.zero_grad()

        if batch % 50 == 0:
            current = batch * len(X)
            print(f"loss: {loss.item():>7e}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn, t):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    if isinstance(loss_fn, dict):
        test_loss = np.zeros(len(loss_fn['loss']))
    else:
        test_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(cfg.device), y.to(cfg.device)
            pred = model(X)
            if isinstance(loss_fn, dict):
                for i in range(len(loss_fn['loss'])):
                    test_loss[i] += loss_fn['loss'][i](pred, y).item()
            else:
                test_loss += loss_fn(pred, y).item()
    test_loss /= num_batches
    if isinstance(loss_fn, dict):
        components = test_loss.copy()
        weights = [w(t) if isinstance(w, Callable) else w for w in loss_fn['weight']]
        test_loss = np.dot(components, weights)
        print(f"Test Error:  Avg loss: {test_loss:>8f}")
        print("    Individual loss: ", *components)
    else:
        print(f"Test Error:  Avg loss: {test_loss:>7e} \n")
    return test_loss


# -------------------------------
# Checkpoint saving
# -------------------------------
def save_checkpoint(epoch, model, optimizer, scheduler, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, path)


# -------------------------------
# Training loop
# -------------------------------
prev_loss = float('inf')
for t in range(int(cfg.epochs)):
    print(f"Epoch {t+1}\n{'-'*30}")
    train(train_loader, model, loss_fn, optimizer, t, accum_iter=int(cfg.nbatch))
    test_loss = test(test_loader, model, loss_fn, t)

    if test_loss < prev_loss:
        save_checkpoint(t, model, optimizer, scheduler, f'{cfg.model_prefix}.pth')
        prev_loss = test_loss

    if t % int(cfg.save_every_epochs) == 1:
        save_checkpoint(t, model, optimizer, scheduler, f'{cfg.model_prefix}_{t}.pth')

    scheduler.step()
    print("Learning Rate: ", *scheduler.get_last_lr())

print("Training Done!")
