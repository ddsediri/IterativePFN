import os
from multiprocessing import Pool

import time
import argparse
import torch
import torch.utils.tensorboard
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from datasets import *
from utils.misc import *
from utils.transforms import *
from models.denoise import *
from models.utils import chamfer_distance_unit_sphere
from Evaluate import Evaluator

def input_iter(input_dir):
    for fn in sorted(os.listdir(input_dir)):
        if fn[-3:] != 'xyz':
            continue
        pcl_noisy = torch.FloatTensor(np.loadtxt(os.path.join(input_dir, fn)))
        pcl_noisy, center, scale = NormalizeUnitSphere.normalize(pcl_noisy)
        yield {
            'pcl_noisy': pcl_noisy,
            'name': fn[:-4],
            'center': center,
            'scale': scale
        }

def main(noise):

    for resolution in args.resolutions:

        # Input/Output
        input_dir = os.path.join(args.input_root, '%s_%s_%s' % (args.dataset, resolution, noise))
        save_title = '{dataset}_Ours{modeltag}_{tag}_{res}_{noise}'.format_map({
            'dataset': args.dataset,
            'modeltag': '' if args.niters == 1 else '%dx' % args.niters,
            'tag': args.tag,
            'res': resolution,
            'noise': noise
        })
        output_dir = os.path.join(args.output_root, save_title)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)    # Output point clouds

        logger = get_logger('test_'+args.dataset+'_'+resolution+'_'+noise, output_dir)
        for k, v in vars(args).items():
            logger.info('[ARGS::%s] %s' % (k, repr(v)))

        # Model
        model = DenoiseNet.load_from_checkpoint(args.ckpt)
        model = model.to(args.device)

        # Denoise
        for data in input_iter(input_dir):
            logger.info(data['name'])
            pcl_noisy = data['pcl_noisy'].to(args.device)
            with torch.no_grad():
                model.eval()
                pcl_next = pcl_noisy
                for _ in range(args.niters):
                    if args.patch_stitching:
                        pcl_next = model.patch_based_denoise(pcl_noisy=pcl_next,
                                                            patch_size=args.patch_size, 
                                                            seed_k=args.seed_k, 
                                                            seed_k_alpha=args.seed_k_alpha, 
                                                            num_modules_to_use=args.num_modules_to_use)
                    elif not args.patch_stitching:
                        pcl_next = model.patch_based_denoise_without_stitching(pcl_noisy=pcl_next,
                                                            patch_size=args.patch_size, 
                                                            seed_k=args.seed_k, 
                                                            seed_k_alpha=args.seed_k_alpha, 
                                                            num_modules_to_use=args.num_modules_to_use)
                pcl_denoised = pcl_next.cpu()
                # Denormalize
                pcl_denoised = pcl_denoised * data['scale'] + data['center']
            
            save_path = os.path.join(output_dir, data['name'] + '.xyz')
            np.savetxt(save_path, pcl_denoised.numpy(), fmt='%.8f')

        if not args.dataset.startswith('RueMadame'): 
            # Evaluate
            evaluator = Evaluator(
                output_pcl_dir=output_dir,
                dataset_root=args.dataset_root,
                dataset='PUNet' if args.dataset.startswith('PUNet') else args.dataset,
                summary_dir=args.output_root,
                experiment_name=save_title,
                device=args.device,
                res_gts=resolution,
                logger=logger
            )
            evaluator.run()

if __name__ == '__main__':

    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default='./pretrained/denoisenet-ep-99.ckpt')
    parser.add_argument('--num_modules_to_use', type=int, default=None)
    parser.add_argument('--input_root', type=str, default='./data/examples')
    parser.add_argument('--output_root', type=str, default='./data/results')
    parser.add_argument('--dataset_root', type=str, default='./data')
    parser.add_argument('--dataset', type=str, default='PUNet')
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--resolutions', type=str_list, default=['50000_poisson']) # Set your test resolution
    parser.add_argument('--noise_lvls', type=str_list, default=['0.01']) # Set your test noise level
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=2020)

    # Filtering parameters
    parser.add_argument('--patch_size', type=int, default=1000)
    parser.add_argument('--niters', type=int, default=1)
    parser.add_argument('--denoise_knn', type=int, default=None, help='kNN size to use during testing')

    # Patch stitching params
    parser.add_argument('--patch_stitching', type=bool, default=True, help='Use patch stitching or not?')
    parser.add_argument('--seed_k', type=int, default=6) # 6 for Kinect, 6 for small PCL, 6 for RueMadame PCL
    parser.add_argument('--seed_k_alpha', type=int, default=10) # 2 for Kinect, 10 for small PCL, 20 for RueMadame PCL
    args = parser.parse_args()
    seed_all(args.seed)
    
    with Pool(len(args.noise_lvls)) as p:
        p.map(main, args.noise_lvls)