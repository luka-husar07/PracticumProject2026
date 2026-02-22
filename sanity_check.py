# %%
import preprocessing as pre
import matplotlib.pyplot as plt

X, Y, Z = pre.data.shape

sag = pre.data[X//2, :, :]     
cor = pre.data[:, Y//2, :]      
axi = pre.data[:, :, Z//2]      

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(sag.T, cmap="gray", origin="lower"); axes[0].set_title("Sagittal"); axes[0].axis("off")
axes[1].imshow(cor.T, cmap="gray", origin="lower"); axes[1].set_title("Coronal"); axes[1].axis("off")
axes[2].imshow(axi, cmap="gray"); axes[2].set_title("Axial"); axes[2].axis("off")
plt.tight_layout(); plt.show()

# %% Testing each axis
for axis, name in [(0, "Sagittal"), (1, "Coronal"), (2, "Axial")]:
    N = pre.data.shape[axis]
    print(f"\n{name} axis:")
    N = pre.data.shape[axis]
    for i in range(0, N, N//5):
        b, p = pre.pos_bin(i, N)
        print(f"Slice {i}: p = {p:.3f}, bin = {b}")

# %%
