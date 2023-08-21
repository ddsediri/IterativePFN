import random
import torch
from torch.utils.data import Dataset
from torch.utils.data import sampler
from torch.utils.data.dataloader import default_collate
import pytorch3d.ops
from tqdm.auto import tqdm

def make_patches_for_pcl_pair(pcl_A, pcl_B, patch_size, num_patches, ratio):
    """
    Args:
        pcl_A:  The first point cloud, (N, 3).
        pcl_B:  The second point cloud, (rN, 3).
        patch_size:   Patch size M.
        num_patches:  Number of patches P.
        ratio:    Ratio r.
    Returns:
        (P, M, 3), (P, rM, 3)
    """
    N = pcl_A.size(0)
    seed_idx = torch.randperm(N)[:num_patches]   # (P, )
    seed_pnts = pcl_A[seed_idx].unsqueeze(0)   # (1, P, 3)
    _, _, pat_A = pytorch3d.ops.knn_points(seed_pnts, pcl_A.unsqueeze(0), K=patch_size, return_nn=True)
    pat_A = pat_A[0]    # (P, M, 3)
    _, idx_B, pat_B = pytorch3d.ops.knn_points(seed_pnts, pcl_B.unsqueeze(0), K=int(ratio*patch_size), return_nn=True)
    idx_B = idx_B[0]
    pat_B = pat_B[0]

    return pat_A, pat_B, seed_pnts, seed_idx
    

class PairedPatchDataset(Dataset):

    def __init__(self, datasets, split='train', patch_size=1000, num_patches=1000, patch_ratio=1.0, on_the_fly=True, transform=None):
        super().__init__()
        self.datasets = datasets
        self.split = split
        self.len_datasets = sum([len(dset) for dset in datasets])
        self.patch_ratio = patch_ratio
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.transform = transform
        self.patches = []

    def __len__(self):
        return self.len_datasets * self.num_patches

    def __getitem__(self, idx):
        pcl_dset = random.choice(self.datasets)
        pcl_data = pcl_dset[idx % len(pcl_dset)]

        if self.split == 'train':
            pat_noisy, pat_clean, seed_pts, seed_idx = make_patches_for_pcl_pair(
                    pcl_data['pcl_noisy'],
                    pcl_data['pcl_clean'],
                    patch_size=self.patch_size,
                    num_patches=1,
                    ratio=self.patch_ratio
                )

            pat_std = pcl_data['noise_std']

            data = {
                'pcl_noisy': pat_noisy[0],
                'pcl_clean': pat_clean[0],
                'seed_pnts': seed_pts[0],
                'pcl_std': pat_std
            }
        else:
            pat_noisy, pat_clean, seed_pts, seed_idx = make_patches_for_pcl_pair(
                    pcl_data['pcl_noisy'],
                    pcl_data['pcl_clean'],
                    None,
                    patch_size=self.patch_size,
                    num_patches=1,
                    ratio=self.patch_ratio
                )

            data = {
                'pcl_noisy': pat_noisy[0],
                'pcl_clean': pat_clean[0],
            }

        if self.transform is not None:
            data = self.transform(data)
        return data