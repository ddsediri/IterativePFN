import os
import torch
import pytorch3d
import pytorch3d.loss
import numpy as np
from scipy.spatial.transform import Rotation
import pandas as pd
import point_cloud_utils as pcu
from tqdm.auto import tqdm
from models.utils import *


class BlackHole(object):
    def __setattr__(self, name, value):
        pass
    def __call__(self, *args, **kwargs):
        return self
    def __getattr__(self, name):
        return self


def load_xyz(xyz_dir):
    all_pcls = {}
    dir_list = sorted(os.listdir(xyz_dir))
    dir_list.sort()
    for fn in tqdm(dir_list, desc='Loading PCLs'):
        if fn[-3:] != 'xyz':
            continue
        name = fn[:-4]
        path = os.path.join(xyz_dir, fn)
        all_pcls[name] = torch.FloatTensor(np.loadtxt(path, dtype=np.float32))
    return all_pcls

def load_off(off_dir):
    all_meshes = {}
    dir_list = sorted(os.listdir(off_dir))
    dir_list.sort()
    for fn in tqdm(dir_list, desc='Loading meshes'):
        if fn[-3:] != 'off':
            continue
        name = fn[:-4]
        path = os.path.join(off_dir, fn)
        verts, faces = pcu.load_mesh_vf(path)
        verts = torch.FloatTensor(verts)
        faces = torch.LongTensor(faces)
        all_meshes[name] = {'verts': verts, 'faces': faces}
    return all_meshes


class Evaluator(object):

    def __init__(self, output_pcl_dir, dataset_root, dataset, summary_dir, experiment_name, device='cuda', res_gts='8192_poisson', logger=BlackHole()):
        super().__init__()
        self.output_pcl_dir = output_pcl_dir
        self.dataset_root = dataset_root
        self.dataset = dataset
        self.summary_dir = summary_dir
        self.experiment_name = experiment_name
        self.gts_pcl_dir = os.path.join(dataset_root, dataset, 'pointclouds', 'test', res_gts)
        self.gts_mesh_dir = os.path.join(dataset_root, dataset, 'meshes', 'test')
        self.res_gts = res_gts
        self.device = device
        self.logger = logger
        self.load_data()

    def load_data(self):
        self.pcls_pred = load_xyz(self.output_pcl_dir)
        self.pcls_gt = load_xyz(self.gts_pcl_dir)
        self.meshes = load_off(self.gts_mesh_dir)
        self.pcls_pred_name = list(self.pcls_pred.keys())
        self.pcls_gt_name = list(self.pcls_gt.keys())


    def run(self):
        pcls_pred, pcls_gt, pcls_pred_name, pcls_gt_name = self.pcls_pred, self.pcls_gt, self.pcls_pred_name, self.pcls_gt_name
        results = {}
        results_cd = {}
        results_p2f = {}
        for pred_name, gt_name in tqdm(zip(pcls_pred_name, pcls_gt_name), desc='Evaluate'):
            pcl_pred = pcls_pred[pred_name][:,:3].unsqueeze(0).to(self.device)
            if gt_name not in pcls_gt:
                self.logger.warning('Shape `%s` not found, ignored.' % pcls_gt_name)
                continue
            pcl_gt = pcls_gt[gt_name].unsqueeze(0).to(self.device)
            verts = self.meshes[gt_name]['verts'].to(self.device)
            faces = self.meshes[gt_name]['faces'].to(self.device)

            cd = pytorch3d.loss.chamfer_distance(pcl_pred, pcl_gt)[0].item()
            cd_sph = chamfer_distance_unit_sphere(pcl_pred, pcl_gt)[0].item()
            hd_sph = hausdorff_distance_unit_sphere(pcl_pred, pcl_gt)[0].item()

            # p2f = point_to_mesh_distance_single_unit_sphere(
            #     pcl=pcl_pred[0],
            #     verts=verts,
            #     faces=faces
            # ).sqrt().mean().item()
            if 'blensor' in self.experiment_name:
                rotmat = torch.FloatTensor(Rotation.from_euler('xyz', [-90, 0, 0], degrees=True).as_matrix()).to(pcl_pred[0])
                p2f = point_mesh_bidir_distance_single_unit_sphere(
                    pcl=pcl_pred[0].matmul(rotmat.t()),
                    verts=verts,
                    faces=faces
                ).item()
            else:
                p2f = point_mesh_bidir_distance_single_unit_sphere(
                    pcl=pcl_pred[0],
                    verts=verts,
                    faces=faces
                ).item()

            cd_sph *= 10000
            p2f *= 10000

            results[gt_name] = {
                # 'cd': cd,
                'cd_sph': cd_sph,
                'p2f': p2f,
                # 'hd_sph': hd_sph,
            }

        results = pd.DataFrame(results).transpose()
        results_cd = pd.DataFrame(results_cd)
        results_p2f = pd.DataFrame(results_p2f)
        res_mean = results.mean(axis=0)
        self.logger.info("\n" + results.to_string())
        self.logger.info("\nMean\n" + '\n'.join([
            '%s\t%.12f' % (k, v) for k, v in res_mean.items()
        ]))

#         update_summary(
#             os.path.join(self.summary_dir, 'Summary_%s.csv' % self.dataset),
#             model=self.experiment_name,
#             metrics={
#                 # 'cd(mean)': res_mean['cd'],
#                 'cd_sph(mean)': res_mean['cd_sph'],
#                 'p2f(mean)': res_mean['p2f'],
#                 # 'hd_sph(mean)': res_mean['hd_sph'],
#             }
#         )


# def update_summary(path, model, metrics):
#     if os.path.exists(path):
#         df = pd.read_csv(path, index_col=0, sep="\s*,\s*", engine='python')
#     else:
#         df = pd.DataFrame()
#     for metric, value in metrics.items():
#         setting = metric
#         if setting not in df.columns:
#             df[setting] = np.nan
#         df.loc[model, setting] = value
#     df.to_csv(path, float_format='%.12f')
#     return df
