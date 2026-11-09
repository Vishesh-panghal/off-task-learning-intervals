import sys, os, time
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

torch.set_num_threads(16)
device = torch.device('cpu')
print(f"Using device: {device} with {torch.get_num_threads()} CPU threads", flush=True)

sys.path.insert(0, '/home/poornima/vishesh_gpu/projects/SOTLI')
from src.models.resnet_pretrained import PretrainedResNet50DualHead
from src.data.datasets import get_applied_isic_probe_loader
from src.utils.metrics import compute_effective_rank

probe_json = '/home/poornima/vishesh_gpu/projects/SOTLI/data_cache/splits/isic2019_probe_2000.json'
img_dir = '/home/poornima/vishesh_gpu/projects/SOTLI/data_cache/isic2019_224/train-image/image'
ckpt_base = '/home/poornima/vishesh_gpu/projects/SOTLI/checkpoints/applied_primary'
out_path = '/home/poornima/vishesh_gpu/projects/SOTLI/results/applied_manifold_scree_data.npz'

print("Pre-loading all 2,000 probe images into RAM...", flush=True)
t0 = time.time()
probe_loader = get_applied_isic_probe_loader(probe_json, img_dir, batch_size=128, num_workers=4)
all_imgs, all_lbls = [], []
for imgs, lbls in probe_loader:
    all_imgs.append(imgs)
    all_lbls.append(lbls)
X_probe = torch.cat(all_imgs, dim=0) # (2000, 3, 224, 224)
y_probe = torch.cat(all_lbls, dim=0).numpy()
print(f"  Probe images cached in RAM: shape {X_probe.shape} in {time.time()-t0:.1f}s", flush=True)

runs = {
    'S1_theta1': os.path.join(ckpt_base, 'seed_1001/S1/S1_seed1001_theta1.pt'),
    'S1_theta2': os.path.join(ckpt_base, 'seed_1001/S1/S1_seed1001_theta2.pt'),
    'S1_theta3': os.path.join(ckpt_base, 'seed_1001/S1/S1_seed1001_theta3.pt'),
    'C0_theta2': os.path.join(ckpt_base, 'seed_1001/C0/C0_seed1001_theta2.pt'),
}

data_dict = {'labels': y_probe}
all_feats = {}

for k, path in runs.items():
    t_ckpt = time.time()
    print(f"Processing {k} on CPU...", flush=True)
    model = PretrainedResNet50DualHead(num_classes_a=8, weights=None)
    model.init_auxiliary_head(num_classes_b=10)
    model.init_temp_head_a()
    state = torch.load(path, map_location='cpu')
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    
    feats = []
    with torch.no_grad():
        for b_start in range(0, len(X_probe), 64):
            batch = X_probe[b_start : b_start + 64]
            f = model(batch, head='features')
            feats.append(f)
    
    feats = torch.cat(feats, dim=0).float()
    feats_centered = feats - feats.mean(dim=0, keepdim=True)
    feats_np = feats_centered.numpy()
    
    # SVD
    _, s, _ = np.linalg.svd(feats_np, full_matrices=False)
    er = compute_effective_rank(feats_np)
    data_dict[f's_{k}'] = s
    data_dict[f'er_{k}'] = er
    all_feats[k] = feats_np
    print(f"  {k} completed in {time.time()-t_ckpt:.2f}s | EffRank: {er:.2f}", flush=True)

print("\nComputing Fast 2D t-SNE projections via PCA-50 pre-reduction...", flush=True)
for k, feats_np in all_feats.items():
    t_tsne = time.time()
    print(f"Running PCA-50 + t-SNE for {k}...", flush=True)
    pca_50 = PCA(n_components=50, random_state=42).fit_transform(feats_np)
    tsne = TSNE(n_components=2, perplexity=35, n_iter=1000, random_state=42, init='pca', learning_rate='auto')
    coords = tsne.fit_transform(pca_50)
    data_dict[f'tsne_{k}'] = coords
    print(f"  t-SNE {k} completed in {time.time()-t_tsne:.2f}s", flush=True)

os.makedirs(os.path.dirname(out_path), exist_ok=True)
np.savez_compressed(out_path, **data_dict)
print(f"\nSuccessfully saved all manifold and scree data to {out_path}!", flush=True)
