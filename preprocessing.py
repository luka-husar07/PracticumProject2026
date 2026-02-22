#%% 
import nibabel as nib
import kagglehub
import os
import re

path = kagglehub.dataset_download("soroush361/3dbraintissuesegmentation")

all_files = []
for root, dirs, files in os.walk(path):
    for f in files:
        all_files.append(os.path.join(root, f))

print("Total files found:", len(all_files))

nii_files = [f for f in all_files if f.lower().endswith((".nii", ".nii.gz"))]

nii_path = nii_files[0]
print("Example file:", nii_path)

img = nib.load(nii_path)
print("Voxel spacing (mm):", img.header.get_zooms()[:3])
img_ras = nib.as_closest_canonical(img)
data = img_ras.get_fdata()
print("Volume shape:", data.shape)
print("Min/Max intensity:", data.min(), data.max())


# %% Defining labels and bins
plane_map = {0: "Sagittal", 1: "Coronal", 2: "Axial"}
num_bins = 5

def get_subject_id(file_path):
    base = os.path.basename(file_path)
    m = re.match(r"([a-zA-Z]+_\d+)_", base.lower())
    return m.group(1) if m else base.replace(".nii", "").replace(".gz", "")

def pos_bin(i, N, num_bins=5):
    p = i / (N-1)
    b = int(p * num_bins)
    return min(b, num_bins-1), p


# %% Building an index of all slices
import pandas as pd

rows = []

for f in nii_files:
    subject_id = get_subject_id(f)
    img = nib.load(f)
    img = nib.as_closest_canonical(img)
    data = img.get_fdata()

    # Loop over each plane
    for axis in range(3):
        N = data.shape[axis]
        plane_id = axis

        for i in range(N):
            b, p = pos_bin(i, N, num_bins)
            rows.append({
                "subject_id": subject_id,
                "file_path": f,
                "plane_id": plane_id,
                "plane_name": plane_map[plane_id],
                "slice_index": i,
                "position": p,
                "bin": b
            })

df = pd.DataFrame(rows)

print(nii_files[:5])


# %% Subject Level Split
from sklearn.model_selection import train_test_split

unique_subjects = df["subject_id"].unique()
train_subjects, temp_subjects = train_test_split(unique_subjects, test_size=0.3, random_state=42)
val_subjects, test_subjects = train_test_split(temp_subjects, test_size=0.5, random_state=42)

df["split"] = "train"
df.loc[df["subject_id"].isin(val_subjects), "split"] = "val"
df.loc[df["subject_id"].isin(test_subjects), "split"] = "test"

print("Split distribution:")
print(df["split"].value_counts())

# %%
