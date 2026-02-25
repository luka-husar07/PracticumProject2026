import nibabel as nib
import numpy as np
import os
import preprocessing as pre

output_dir = "/home/lh22i/slices"
os.makedirs(output_dir, exist_ok=True)

total = len(pre.df)
print(f"Extracting {total} slices to {output_dir}...")

for idx, row in pre.df.iterrows():
    fname = f"{row['subject_id']}_plane{row['plane_id']}_slice{row['slice_index']}.npy"
    fpath = os.path.join(output_dir, fname)

    if not os.path.exists(fpath):
        img = nib.load(row['file_path'])
        img = nib.as_closest_canonical(img)
        data = img.get_fdata().astype(np.float32)

        axis = row['plane_id']
        i = row['slice_index']
        if axis == 0:
            slc = data[i, :, :]
        elif axis == 1:
            slc = data[:, i, :]
        else:
            slc = data[:, :, i]

        np.save(fpath, slc)

    # Update the dataframe with the npy path
    pre.df.at[idx, 'npy_path'] = fpath

    if idx % 10000 == 0:
        print(f"Progress: {idx}/{total}")

pre.df.to_csv('slice_index.csv', index=False)
print("Extraction complete!")