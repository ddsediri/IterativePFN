import os
import argparse
import torch
import torch.utils.tensorboard
from tqdm.auto import tqdm
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import logging
from models.denoise import DenoiseNet
from utils.misc import get_log_dir_name_tblogger, seed_all, str_list

def main(args):

    # Logging
    local_rank = os.environ.get('LOCAL_RANK', 0)

    if local_rank == 0:
        log_dir_name = get_log_dir_name_tblogger(name='D%s_' % (args.dataset))
        os.makedirs(os.path.join(args.log_root, log_dir_name))
        os.environ['LOG_DIR_NAME'] = log_dir_name
    else:
        log_dir_name = os.environ['LOG_DIR_NAME']

    log_dir = os.path.join(args.log_root, log_dir_name)

    # configure logging on module level, redirect to file
    logger = logging.getLogger('pytorch_lightning.core')
    logger.addHandler(logging.FileHandler(os.path.join(log_dir, 'run.log')))

    # Model
    logger.info('INFO: Building model...')
    model = DenoiseNet(args)

    for k, v in vars(args).items():
        logger.info('[ARGS::%s] %s' % (k, repr(v)))

    logger.info(repr(model))

    # Main loop
    try:
        logger.info('INFO: Start training...')
        seed_everything(args.seed, workers=True)
        trainer = Trainer(
            accelerator='gpu',
            devices=args.n_gpu,
            num_nodes=1,
            logger=TensorBoardLogger(args.log_root, name=log_dir_name),
            deterministic=True,
            max_epochs=100,
            check_val_every_n_epoch=args.save_interval,
            callbacks=[
                        ModelCheckpoint(
                             # monitor='val_loss',
                             every_n_epochs=args.save_interval, 
                             save_on_train_epoch_end=False,
                             save_top_k = -1,
                             dirpath=log_dir,
                             filename='denoisenet-epoch{epoch:02d}-val_loss{val_loss:.6f}',
                             auto_insert_metric_name=False
                        )
                        ],
            strategy="ddp"
        )

        trainer.fit(model)
    except KeyboardInterrupt:
        logger.info('INFO: Terminating...')
        print('Terminating...')


if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser()
    ## Dataset and loader
    parser.add_argument('--dataset_root', type=str, default='./data')
    parser.add_argument('--dataset', type=str, default='PUNet')
    parser.add_argument('--changelog', type=str, default='')
    parser.add_argument('--patches_per_shape_per_epoch', type=int, default=1000)
    parser.add_argument('--patch_ratio', type=float, default=1.2)
    parser.add_argument('--resolutions', type=str_list, default=['10000_poisson', '30000_poisson', '50000_poisson'], choices=[['x0000_poisson'], ['x0000_poisson', 'y0000_poisson'], ['x0000_poisson', 'y0000_poisson', 'z0000_poisson']])
    parser.add_argument('--noise_min', type=float, default=0.005)
    parser.add_argument('--noise_max', type=float, default=0.02)
    parser.add_argument('--train_batch_size', type=int, default=4)
    parser.add_argument('--val_batch_size', type=int, default=4)
    parser.add_argument('--n_gpu', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=12)
    parser.add_argument('--save_interval', type=int, default=5)
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--aug_rotate', type=eval, default=True, choices=[True, False])
    ## Model architecture
    parser.add_argument('--loss_type', type=str, default='NN', choices=['NN', 'NN_no_stitching'])
    ## Optimizer and scheduler
    parser.add_argument('--sched_patience', default=10, type=int)
    parser.add_argument('--sched_factor', default=0.5, type=float)
    parser.add_argument('--min_lr', default=1e-7, type=float)  
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0)
    ## Training
    parser.add_argument('--seed', type=int, default=2020)
    parser.add_argument('--log_root', type=str, default='./logs')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--val_noise', type=float, default=0.015)

    # Ablation parameters
    parser.add_argument('--patch_size', type=int, default=1000)
    parser.add_argument('--frame_knn', type=int, default=32) # Neighbourhood side for graph convolution
    parser.add_argument('--num_modules', type=int, default=4)
    parser.add_argument('--noise_decay', type=int, default=4) # Noise decay is set to 16/T where T=num_modules or set to 1 for no decay

    args = parser.parse_args()

    main(args)
