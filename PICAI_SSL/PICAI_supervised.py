import os
import sys
import argparse
import random
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm
import shutil

from monai.losses import DiceFocalLoss
from monai.metrics import DiceMetric
from monai.transforms import Activations, AsDiscrete

import segmentation_models_pytorch_3d as smp
from dataloaders.picaiDataset import PICAIDataset

# ------------------ Argument parser ------------------ #
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/content/drive/MyDrive/SSL/Dataset/160_160_16')
parser.add_argument('--list_path', type=str, default='/content/drive/MyDrive/SSL/Dataset/Data_split/423_pids')
parser.add_argument('--exp', type=str, default='Supervised')
parser.add_argument('--epochs', type=int, default=4)
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--save_path', type=str, default='/content/drive/MyDrive/SSL/Trained_model')
args = parser.parse_args()

# ------------------ Setup ------------------ #
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)

logging.basicConfig(filename=os.path.join(args.save_path, "train_log.txt"),
                    level=logging.INFO, format='%(asctime)s %(message)s')
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
logging.info(str(args))

# ------------------ Reproducibility ------------------ #
seed = 1337
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# ------------------ Dataset ------------------ #
train_dataset = PICAIDataset(args.root_path, args.list_path, split='train')
val_dataset = PICAIDataset(args.root_path, args.list_path, split='val')  # Add validation support

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)

# ------------------ Model ------------------ #
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = smp.Unet(
    encoder_name="resnet50",
    in_channels=3,
    strides=((2, 2, 1), (2, 2, 2), (2, 2, 2), (2, 2, 1), (2, 2, 1)),
    classes=1,
)
net = nn.DataParallel(model).to(device)

# ------------------ Loss, Optimizer, Metrics ------------------ #
loss_fn = DiceFocalLoss(sigmoid=True, to_onehot_y=False)
optimizer = optim.Adam(net.parameters(), lr=args.lr)

dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=True)
post_pred = Activations(sigmoid=True)
post_label = AsDiscrete(threshold=0.5)

# ------------------ Resume Setup ------------------ #
start_epoch = 0
best_dice = 0
checkpoint_path = os.path.join(args.save_path, "last_checkpoint.pth")

if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    best_dice = checkpoint.get('best_dice', 0)
    start_epoch = checkpoint.get('epoch', 0) + 1
    logging.info(f"Resumed training from epoch {start_epoch} with best dice {best_dice:.4f}")
else:
    logging.info(f"Training from scratch or checkpoint not found.")

# ------------------ Metrics Excel Setup ------------------ #
excel_path = os.path.join(args.save_path, 'metrics.xlsx')
if os.path.exists(excel_path):
    metrics = pd.read_excel(excel_path).to_dict('records')
else:
    metrics = []

# ------------------ Training Loop ------------------ #
for epoch in range(start_epoch, args.epochs):
    net.train()
    running_loss = 0.0

    for batch in tqdm(train_loader, ncols=80):
        images, labels = batch['image'].cuda(), batch['label'].cuda()

        # Convert shape to (B, C, D, H, W)
        if images.shape[2:] == (160, 160, 16):
            images = images.permute(0, 1, 4, 2, 3)
            labels = labels.permute(0, 1, 4, 2, 3)

        outputs = net(images)
        labels = labels.float()

        loss = loss_fn(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    dice_score = None

    # ------------------ Validation ------------------ #
    if (epoch + 1) % 2 == 0:
        net.eval()
        dice_metric.reset()

        with torch.no_grad():
            for val_batch in val_loader:
                val_images, val_labels = val_batch['image'].cuda(), val_batch['label'].cuda()
                val_images = val_images.permute(0, 1, 4, 2, 3)
                val_labels = val_labels.permute(0, 1, 4, 2, 3)

                val_outputs = net(val_images)
                val_outputs = post_pred(val_outputs)
                val_labels = val_labels.float()

                dice_metric(y_pred=post_label(val_outputs), y=post_label(val_labels))

        dice_score = dice_metric.aggregate()[0].item()
        logging.info(f"Epoch [{epoch+1}/{args.epochs}], Loss: {avg_loss:.4f}, Val Dice: {dice_score:.4f}")

        # Save best model
        best_model_path = os.path.join(args.save_path, "best_model.pth")
        if dice_score > best_dice:
            if os.path.exists(best_model_path):
                shutil.copy(best_model_path, best_model_path.replace(".pth", "_backup.pth"))
            best_dice = dice_score
            torch.save(net.state_dict(), best_model_path)
            logging.info(f"Saved new best model to {best_model_path}")
    else:
        dice_score = -1

    # ------------------ Save Metrics ------------------ #
    metrics.append({
        'epoch': epoch + 1,
        'loss': avg_loss,
        'val_dice': round(dice_score, 4) if dice_score != -1 else None
    })

    torch.save({
        'epoch': epoch,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_dice': best_dice
    }, checkpoint_path)

# ------------------ Final Excel Save ------------------ #
df_metrics = pd.DataFrame(metrics)
df_metrics.drop_duplicates(subset='epoch', keep='last', inplace=True)
df_metrics.sort_values(by='epoch', inplace=True)
df_metrics.to_excel(excel_path, index=False)
logging.info(f"Saved updated metrics to {excel_path}")
logging.info(f"Training complete. Best Dice Score: {best_dice:.4f}")
