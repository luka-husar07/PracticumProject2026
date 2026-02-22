import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision import transforms
import nibabel as nib

class SliceDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        self.volume_cache = {}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        if row['file_path'] not in self.volume_cache:
            img = nib.load(row['file_path'])
            img = nib.as_closest_canonical(img)
            self.volume_cache[row['file_path']] = img.get_fdata()

        data = self.volume_cache[row['file_path']]
        
        # Extract the slice based on plane_id and slice_index
        if row['plane_id'] == 0:  
        # Sagittal
            slice_img = data[row['slice_index'], :, :]
        elif row['plane_id'] == 1: 
        # Coronal
            slice_img = data[:, row['slice_index'], :]
        # Axial
        else:  
            slice_img = data[:, :, row['slice_index']]
        
        # Normalize the slice
        slice_img = (slice_img - np.min(slice_img)) / (np.max(slice_img) - np.min(slice_img) + 1e-5)

        # Convert to tensor with channel dimension (1, H, W)
        slice_tensor = torch.from_numpy(slice_img.astype(np.float32)).unsqueeze(0)
        
        if self.transform:
            slice_tensor = self.transform(slice_tensor)
        
        return ( 
            slice_tensor, 
            torch.tensor(row['bin'], dtype=torch.long),
            torch.tensor(row['plane_id'], dtype=torch.long)
        )