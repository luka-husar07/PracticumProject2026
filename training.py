# %% Training Script for Slice Classification
import preprocessing
import data_loader
import model as model_module
import torch
import numpy as np
from torchvision import transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

model = model_module.SliceClassifier(num_bins=preprocessing.num_bins).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion_bin = torch.nn.CrossEntropyLoss()

def train_epoch(model, dataloader, optimizer, criterion_bin, device, lambda_pos=1.0):
    model.train()
    total_loss = 0.0

    for images, bin_labels, plane_labels in dataloader:
        images = images.to(device)
        bin_labels = bin_labels.to(device)
        plane_labels = plane_labels.to(device)

        optimizer.zero_grad()

        bin_outputs, plane_outputs = model(images)

        loss_bin = criterion_bin(bin_outputs, bin_labels)
        loss_plane = criterion_bin(plane_outputs, plane_labels)

        loss = lambda_pos * loss_bin + loss_plane
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss

def validate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    plane_correct = 0
    bin_abs_errors = []
    total = 0

    with torch.no_grad():
        for images, bin_labels, plane_labels in dataloader:
            images = images.to(device)
            bin_labels = bin_labels.to(device)
            plane_labels = plane_labels.to(device)

            bin_outputs, plane_outputs = model(images)

            loss_bin = criterion_bin(bin_outputs, bin_labels)
            loss_plane = criterion_bin(plane_outputs, plane_labels)
            loss = loss_bin + loss_plane
            total_loss += loss.item()

            plane_preds = plane_outputs.argmax(dim=1)
            plane_correct += (plane_preds == plane_labels).sum().item()

            bin_preds = bin_outputs.argmax(dim=1)
            abs_errors = torch.abs(bin_preds - bin_labels).cpu().numpy()
            bin_abs_errors.extend(abs_errors)

            total += len(bin_labels)

    return {
        "val_loss": total_loss / len(dataloader),
        "plane_acc": plane_correct / total,
        "mabd": np.mean(bin_abs_errors)
    }

# %% Main Training Loop
num_epochs = 20
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
])

train_dataset = data_loader.SliceDataset(preprocessing.df[preprocessing.df['split'] == 'train'], transform=transform)
val_dataset = data_loader.SliceDataset(preprocessing.df[preprocessing.df['split'] == 'val'], transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader, optimizer, criterion_bin, device)
    val_metrics = validate(model, val_loader, device)

    print(f"Epoch {epoch+1}/{num_epochs} \n"
          f"Train Loss: {train_loss:.4f}\n"
          f"Val Loss: {val_metrics['val_loss']:.4f}\n"
          f"Plane Acc: {val_metrics['plane_acc']:.4f}\n"
          f"MABD: {val_metrics['mabd']:.4f}")

# %%
