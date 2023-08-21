import logging
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch3d.ops
import pytorch3d.loss.chamfer as cd_loss
import numpy as np
from .feature import FeatureExtraction
import pytorch_lightning as pl
from datasets.pcl import *
from datasets.patch import *
from utils.misc import *
from utils.transforms import *
from models.utils import chamfer_distance_unit_sphere
from models.utils import farthest_point_sampling

def get_random_indices(n, m):
    assert m < n
    return np.random.permutation(n)[:m]


class DenoiseNet(pl.LightningModule):

    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()
        self.args = args
        # geometry
        self.frame_knn = args.frame_knn # Neighbourhood side for graph convolution
        # score-matching
        self.num_modules = args.num_modules
        self.noise_decay = args.noise_decay
        self.feature_nets = nn.ModuleList()
        self.console_logger = logging.getLogger('pytorch_lightning.core')
        # networks
        input_dim = 3
        z_dim = 0 # We are not using any feature vector to provide context
        for i in range(self.num_modules):
            self.feature_nets.append(FeatureExtraction(k=self.frame_knn, input_dim=input_dim, z_dim=z_dim, embedding_dim=512, output_dim=3))
            input_dim = 3 + z_dim

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
                                        self.feature_nets.parameters(), 
                                        lr=self.args.lr, 
                                        weight_decay=self.args.weight_decay
                                    )
        scheduler = {
                        'scheduler': ReduceLROnPlateau(optimizer, patience=self.args.sched_patience, factor=self.args.sched_factor, min_lr=self.args.min_lr),
                        'interval': 'epoch',
                        'frequency': 5,
                        'monitor': 'val_loss',
                    }
        return [optimizer], [scheduler]

    def train_dataloader(self):
        # Datasets and loaders
        train_dset = PairedPatchDataset(
            datasets=[
                PointCloudDataset(
                    root=self.args.dataset_root,
                    dataset=self.args.dataset,
                    split='train',
                    resolution=resl,
                    transform=standard_train_transforms(noise_std_max=self.args.noise_max, noise_std_min=self.args.noise_min, rotate=self.args.aug_rotate)
                ) for resl in self.args.resolutions
            ],
            split='train',
            patch_size=self.args.patch_size,
            num_patches=self.args.patches_per_shape_per_epoch,
            patch_ratio=self.args.patch_ratio,
            on_the_fly=True # Currently, we only support on_the_fly=True
        )

        return DataLoader(train_dset, batch_size=self.args.train_batch_size, num_workers=4, pin_memory=True, shuffle=True)

    def val_dataloader(self):
        # Datasets and loaders
        val_dset = PointCloudDataset(
                        root=self.args.dataset_root,
                        dataset=self.args.dataset,
                        split='test',
                        resolution=self.args.resolutions[0],
                        transform=standard_train_transforms(noise_std_max=self.args.val_noise, noise_std_min=self.args.val_noise, rotate=False, scale_d=0),
                    )

        return DataLoader(val_dset, batch_size=self.args.val_batch_size, num_workers=4, pin_memory=True, shuffle=False)

    def training_step(self, train_batch, batch_idx):
        pcl_noisy = train_batch['pcl_noisy']
        pcl_clean = train_batch['pcl_clean']
        pcl_seeds = train_batch['seed_pnts']
        pcl_std = train_batch['pcl_std']

        # Forward
        if self.args.loss_type == "NN":
            loss = self.get_supervised_loss_nn(pcl_noisy=pcl_noisy, pcl_clean=pcl_clean, pcl_seeds=pcl_seeds, pcl_std=pcl_std)  
        elif self.args.loss_type == "NN_no_stitching":
            loss = self.get_supervised_loss_nn_no_weighting(pcl_noisy=pcl_noisy, pcl_clean=pcl_clean, pcl_seeds=pcl_seeds, pcl_std=pcl_std)

        # # Logging
        # self.console_logger.info('INFO: Training loss for batch idx {:d}: {:.6f}'.format(batch_idx, loss))
        self.log('loss', loss)
        return {"loss": loss, "loss_as_tensor": loss.clone().detach()} 


    def validation_step(self, val_batch, batch_idx):
        pcl_clean = val_batch['pcl_clean']
        pcl_noisy = val_batch['pcl_noisy']

        all_clean = []
        all_denoised = []
        for i, data in enumerate(pcl_noisy):
            pcl_denoised = self.patch_based_denoise(data)
            all_clean.append(pcl_clean[i].unsqueeze(0))
            all_denoised.append(pcl_denoised.unsqueeze(0))
        all_clean = torch.cat(all_clean, dim=0)
        all_denoised = torch.cat(all_denoised, dim=0)

        avg_chamfer = chamfer_distance_unit_sphere(all_denoised, all_clean, batch_reduction='mean')[0].item()
        
        return torch.tensor(avg_chamfer)

    def training_epoch_end(self, train_outputs):
        loss_all = torch.stack([train_output['loss_as_tensor'] for train_output in train_outputs], dim=0)
        loss_all = loss_all.mean()
        self.console_logger.info('INFO: Current epoch training loss: {:.6f}'.format(loss_all))
        self.log('train_epoch_loss', loss_all, sync_dist=True)
        
    def validation_epoch_end(self, val_outs):
        val_outs = torch.stack(val_outs, dim=0)
        val_loss_all = val_outs.mean()
        self.console_logger.info('INFO: Current epoch validation loss: {:.6f}'.format(val_loss_all))
        self.log('val_loss', val_loss_all, sync_dist=True)
        
    def curr_iter_add_noise(self, pcl_clean, noise_std):
        new_pcl_clean = pcl_clean + torch.randn_like(pcl_clean) * noise_std.unsqueeze(1).unsqueeze(2)
        return new_pcl_clean.float()

    def get_supervised_loss_nn(self, pcl_noisy, pcl_clean, pcl_seeds, pcl_std):
        """
        Denoising score matching.
        Args:
            pcl_noisy:  Noisy point clouds, (B, N, 3).
            pcl_clean:  Clean point clouds, (B, M, 3). Usually, M is slightly greater than N.
        """
        B, N_noisy, N_clean, d = pcl_noisy.size(0), pcl_noisy.size(1), pcl_clean.size(1), pcl_noisy.size(2)

        losses = torch.zeros(self.num_modules)

        pcl_seeds_1 = pcl_seeds.repeat(1, N_noisy, 1)

        seed_dist_sq = ((pcl_noisy - pcl_seeds_1)**2).sum(dim=-1, keepdim=True)
        max_seed_dist_sq = seed_dist_sq[:, -1, :]#torch.max(seed_dist_sq, dim=1)[0]
        seed_dist_sq = seed_dist_sq / (max_seed_dist_sq.unsqueeze(1) / 9)
        seed_weights = torch.exp(-1 * seed_dist_sq).squeeze()
        seed_weights_sum = seed_weights.sum(dim=1, keepdim=True)
        seed_weights = (seed_weights / seed_weights_sum).squeeze()

        pcl_noisy = pcl_noisy - pcl_seeds_1
        pcl_seeds_2 = pcl_seeds.repeat(1, N_clean, 1)
        pcl_clean = pcl_clean - pcl_seeds_2

        curr_std = pcl_std
        for i in range(self.num_modules):

            if i == 0:
                pcl_input = pcl_noisy
                pred_proj = None
            else:
                pcl_input = pcl_input + pred_disp

            pred_disp, pred_proj = self.feature_nets[i](pcl_input, pred_proj)

            if self.noise_decay != 1:
                prev_std = curr_std
                if i < self.num_modules - 1:
                    curr_std = curr_std / self.noise_decay
                    pcl_target_lower_noise = self.curr_iter_add_noise(pcl_clean, curr_std)
                else:
                    curr_std = 0
                    pcl_target_lower_noise = pcl_clean
            else:
                pcl_target_lower_noise = pcl_clean

            _, _, clean_pts = pytorch3d.ops.knn_points(
                pcl_input,    # (B, N, 3)
                pcl_target_lower_noise,   # (B, M, 3)
                K=1,
                return_nn=True,
            )   # (B, N, K, 3)
            clean_nbs = clean_pts.view(B, N_noisy, d)  # (B, N, 3)
            
            clean_nbs = clean_nbs - pcl_input
            dist = ((pred_disp - clean_nbs)**2).sum(dim=-1) # (B, N)
            losses[i] = (seed_weights * dist).sum(dim=-1).mean(dim=-1)
            
        return losses.sum() #, target, scores, noise_vecs

    def get_supervised_loss_nn_no_weighting(self, pcl_noisy, pcl_clean, pcl_seeds, pcl_std):
        """
        Denoising score matching.
        Args:
            pcl_noisy:  Noisy point clouds, (B, N, 3).
            pcl_clean:  Clean point clouds, (B, M, 3). Usually, M is slightly greater than N.
        """
        B, N_noisy, N_clean, d = pcl_noisy.size(0), pcl_noisy.size(1), pcl_clean.size(1), pcl_noisy.size(2)

        losses = torch.zeros(self.num_modules)

        pcl_seeds_1 = pcl_seeds.repeat(1, N_noisy, 1)

        pcl_noisy = pcl_noisy - pcl_seeds_1
        pcl_seeds_2 = pcl_seeds.repeat(1, N_clean, 1)
        pcl_clean = pcl_clean - pcl_seeds_2

        curr_std = pcl_std
        for i in range(self.num_modules):

            if i == 0:
                pcl_input = pcl_noisy
                pred_proj = None
            else:
                pcl_input = pcl_input + pred_disp

            pred_disp, pred_proj = self.feature_nets[i](pcl_input, pred_proj)

            if self.noise_decay != 1:
                prev_std = curr_std
                if i < self.num_modules - 1:
                    curr_std = curr_std / self.noise_decay
                    pcl_target_lower_noise = self.curr_iter_add_noise(pcl_clean, curr_std)
                else:
                    curr_std = 0
                    pcl_target_lower_noise = pcl_clean
            else:
                pcl_target_lower_noise = pcl_clean

            _, _, clean_pts = pytorch3d.ops.knn_points(
                pcl_input,    # (B, N, 3)
                pcl_target_lower_noise,   # (B, M, 3)
                K=1,
                return_nn=True,
            )   # (B, N, K, 3)
            clean_nbs = clean_pts.view(B, N_noisy, d)  # (B, N, 3)
            
            clean_nbs = clean_nbs - pcl_input
            dist = ((pred_disp - clean_nbs)**2).sum(dim=-1) # (B, N)
            losses[i] = dist.mean(dim=-1).mean(dim=-1)
            
        return losses.sum() #, target, scores, noise_vecs


    def patch_based_denoise(self, pcl_noisy, patch_size=1000, seed_k=5, seed_k_alpha=10, num_modules_to_use=None):
        """
        Args:
            pcl_noisy:  Input point cloud, (N, 3)
        """
        assert pcl_noisy.dim() == 2, 'The shape of input point cloud must be (N, 3).'
        N, d = pcl_noisy.size()
        pcl_noisy = pcl_noisy.unsqueeze(0)  # (1, N, 3)
        num_patches = int(seed_k * N / patch_size)
        seed_pnts, _ = farthest_point_sampling(pcl_noisy, num_patches)
        patch_dists, point_idxs_in_main_pcd, patches = pytorch3d.ops.knn_points(seed_pnts, pcl_noisy, K=patch_size, return_nn=True)
        patches = patches[0]    # (N, K, 3)

        # Patch stitching preliminaries
        seed_pnts_1 = seed_pnts.squeeze().unsqueeze(1).repeat(1, patch_size, 1)
        patches = patches - seed_pnts_1
        patch_dists, point_idxs_in_main_pcd = patch_dists[0], point_idxs_in_main_pcd[0]
        patch_dists = patch_dists / patch_dists[:, -1].unsqueeze(1).repeat(1, patch_size)

        all_dists = torch.ones(num_patches, N) / 0
        all_dists = all_dists.cuda()
        all_dists = list(all_dists)
        patch_dists, point_idxs_in_main_pcd = list(patch_dists), list(point_idxs_in_main_pcd)
     
        for all_dist, patch_id, patch_dist in zip(all_dists, point_idxs_in_main_pcd, patch_dists): 
            all_dist[patch_id] = patch_dist

        all_dists = torch.stack(all_dists,dim=0)
        weights = torch.exp(-1 * all_dists)

        best_weights, best_weights_idx = torch.max(weights, dim=0)
        patches_denoised = []

        # Denoising
        i = 0
        patch_step = int(N / (seed_k_alpha * patch_size))
        assert patch_step > 0, "Seed_k_alpha needs to be decreased to increase patch_step!"
        while i < num_patches:
            # print("Processed {:d}/{:d} patches.".format(i, num_patches))
            curr_patches = patches[i:i+patch_step]
            try:
                if num_modules_to_use is None:
                    patches_denoised_temp, _ = self.denoise_langevin_dynamics(curr_patches, num_modules_to_use=self.num_modules)
                else:
                    patches_denoised_temp, _ = self.denoise_langevin_dynamics(curr_patches, num_modules_to_use=num_modules_to_use)

            except Exception as e:
                print("="*100)
                print(e)
                print("="*100)
                print("If this is an Out Of Memory error, Seed_k_alpha might need to be increased to decrease patch_step.") 
                print("Additionally, if using multiple args.niters and a PyTorch3D ops, KNN, error arises, Seed_k might need to be increased to sample more patches for inference!")
                print("="*100)
                return
            patches_denoised.append(patches_denoised_temp)
            i += patch_step

        patches_denoised = torch.cat(patches_denoised, dim=0)
        patches_denoised = patches_denoised + seed_pnts_1
        
        # Patch stitching
        pcl_denoised = [patches_denoised[patch][point_idxs_in_main_pcd[patch] == pidx_in_main_pcd] for pidx_in_main_pcd, patch in enumerate(best_weights_idx)]

        pcl_denoised = torch.cat(pcl_denoised, dim=0)

        return pcl_denoised

    def patch_based_denoise_without_stitching(self, pcl_noisy, patch_size=1000, seed_k=5, seed_k_alpha=10, num_modules_to_use=None):
        """
        Args:
            pcl_noisy:  Input point cloud, (N, 3)
        """
        assert pcl_noisy.dim() == 2, 'The shape of input point cloud must be (N, 3).'
        N, d = pcl_noisy.size()
        pcl_noisy = pcl_noisy.unsqueeze(0)  # (1, N, 3)
        num_patches = int(seed_k * N / patch_size)
        seed_pnts, _ = farthest_point_sampling(pcl_noisy, num_patches)
        patch_dists, point_idxs_in_main_pcd, patches = pytorch3d.ops.knn_points(seed_pnts, pcl_noisy, K=patch_size, return_nn=True)
        patches = patches[0]    # (N, K, 3)

        seed_pnts_1 = seed_pnts.squeeze().unsqueeze(1).repeat(1, patch_size, 1)
        patches = patches - seed_pnts_1
        patches_denoised = []

        i = 0
        patch_step = int(N / (seed_k_alpha * patch_size))
        assert patch_step > 0, "Seed_k_alpha needs to be decreased to increase patch_step!"
        while i < num_patches:
            # print("Processed {:d}/{:d} patches.".format(i, num_patches))
            curr_patches = patches[i:i+patch_step]
            try:
                if num_modules_to_use is None:
                    patches_denoised_temp, _ = self.denoise_langevin_dynamics(curr_patches, num_modules_to_use=self.num_modules)
                else:
                    patches_denoised_temp, _ = self.denoise_langevin_dynamics(curr_patches, num_modules_to_use=num_modules_to_use)

            except Exception as e:
                print("="*100)
                print(e)
                print("="*100)
                print("If this is an Out Of Memory error, Seed_k_alpha might need to be increased to decrease patch_step.") 
                print("Additionally, if using multiple args.niters and a PyTorch3D ops, KNN, error arises, Seed_k might need to be increased to sample more patches for inference!")
                print("="*100)
                return
            patches_denoised.append(patches_denoised_temp)
            i += patch_step

        patches_denoised = torch.cat(patches_denoised, dim=0)
        patches_denoised = patches_denoised + seed_pnts_1
        
        pcl_denoised, fps_idx = farthest_point_sampling(patches_denoised.view(1, -1, d), N)
        pcl_denoised = pcl_denoised[0]

        return pcl_denoised

    def denoise_langevin_dynamics(self, pcl_noisy, num_modules_to_use):
        """
        Args:
            pcl_noisy:  Noisy point clouds, (B, N, 3).
        """
        B, N, d = pcl_noisy.size()
        pred_disps = []
        pred_projs = []
        pcl_inputs = []
        with torch.no_grad():
            # print("[INFO]: Denoising up to {} iterations".format(num_modules_to_use))
            for i in range(num_modules_to_use):
                self.feature_nets[i].eval()

                if i == 0:
                    pcl_inputs.append(pcl_noisy)
                    pred_projs.append(None)
                else:
                    pcl_inputs.append(pcl_inputs[i-1] + pred_disps[i-1])
                    pred_projs.append(pred_proj)

                # Feature extraction
                pred_points, pred_proj = self.feature_nets[i](pcl_inputs[i], pred_projs[i])  # (B, N, F)
                pred_disps.append(pred_points)
                pred_projs.append(pred_proj)
                
        return pcl_inputs[-1] + pred_disps[-1], None
