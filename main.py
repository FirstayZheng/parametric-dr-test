import argparse
import time
import os
import torch
import logging
from train import train
# from train_grad import train
# from train_loss import train

from omegaconf import OmegaConf
from test_for_linear import linear_train
# from pytorch_lightning import seed_everything
import traceback


START_TIME = time.strftime('%Y%m%d-%H_%M_%S', time.localtime(int(time.time())))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
logging.getLogger().setLevel(logging.INFO)


def main(config):
    # train(config)
    train(config)
    # linear_train(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/GlobalLoss_s_curve.yaml",
        # default="./configs/GlobalLoss_swissroll_local.yaml",
        # default="./configs/GlobalLoss_face.yaml",
        help="path to config which constructs model",
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible results)",
    )
    
    # parser.add_argument(
    #     "--dataset",
    #     type=str,
    #     default="swissroll",
    #     help="path to dataset",
    # )
    
    # argparse's priority is higher than yaml
    args = parser.parse_args()
    config = OmegaConf.load(f"{args.config}")
    config = OmegaConf.merge(config, vars(args))
    
    # generate derived params
    config.output_dir = f'{config.output_dir}/{config.dataset}-{START_TIME}'
    config.device = device
    config.plot_title = f'{config.prefix}-{config.dataset}-{config.batch_size}-{config.n_neighbors}-{config.lr}-{config.global_loss}'
    # config.data_path = f'./data/{config.prefix}/{config.dataset}.npy'
    config.data_path = f'{config.dataset_path}/{config.dataset}.npy'
    config.neighbors_cache_path = f'./data/cache/neighbors_cache_{config.dataset}.npy'
    config.pairwise_cache_path = f'./data/cache/pairwise_cache_{config.dataset}.npy'
    config.geodesic_cache_path = f'./data/cache/geodesic_cache_{config.dataset}.npy'
    
    # os.mkdir(config.output_dir)
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)
    if not os.path.exists(f'./data/cache'):
        os.makedirs(f'./data/cache')
    
    # seed_everything(config.seed)
   
    print(config)
    
    try:
        main(config)

        END_TIME = time.strftime('%Y%m%d-%H_%M_%S', time.localtime(int(time.time())))
        config.start_time = START_TIME
        config.end_time = END_TIME
        # save all params to help analyze experiment results
        with open(f'{config.output_dir}/configs.yaml', 'a') as f:
            OmegaConf.save(config=config, f=f.name)

    except:
        print(traceback.format_exc())
        os.rmdir(config.output_dir)
