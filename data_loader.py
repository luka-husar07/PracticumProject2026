import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision import transforms
import nibabel as nib

class SliceDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        slice_img = np.load(row['npy_path']).astype(np.float32)

        # Normalize
        slice_img = (slice_img - slice_img.min()) / (slice_img.max() - slice_img.min() + 1e-5)

        # Convert to tensor with channel dimension (1, H, W)
        slice_tensor = torch.from_numpy(slice_img).unsqueeze(0)

        if self.transform:
            slice_tensor = self.transform(slice_tensor)

        return (
            slice_tensor,
            torch.tensor(row['bin'], dtype=torch.long),
            torch.tensor(row['plane_id'], dtype=torch.long)
        )