import argparse
from ast import parse
import functools
import os
import pickle
import numpy as np
import torch
import torch.optim as optim
from config import config
from utils.datasets import GraphDataset
from ipdb import set_trace
from score_matching.loss import loss_fn_table_v2, loss_fn_uncond
from score_matching.sampler import ode_sampler
from score_matching.sde import (ExponentialMovingAverage, 
                                   diffusion_coeff, marginal_prob_std_v2, init_sde)
from score_nets import ArrangeScoreModelGNN, ScoreModelGNN
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
from tqdm import trange
from utils.bbox_vis import Rotated_bbox_vis, bbox_vis
from utils.misc import exists_or_mkdir


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_dataloaders(rotation, scene_name, data_root):
    dataset = GraphDataset(scene_name, data_root, rotation=rotation, base_noise_scale=config.base_noise_scale)
    print("Scene_num = ", len(dataset))
    dataloader_train = DataLoader(dataset, batch_size=config.batch_size_gf, shuffle=True, num_workers=4)
    dataloader_vis = DataLoader(dataset, batch_size=2**2, shuffle=True, num_workers=4)

    return dataloader_train, dataloader_vis
    

def get_score_network(configs, marginal_prob_std_fn, state_dim):
    if configs.env_type == 'NoTableGeo':     
        score_net = ArrangeScoreModelGNN(
            marginal_prob_std_fn,
            hidden_dim=configs.hidden_dim_gf,
            embed_dim=configs.embed_dim_gf,
            state_dim=state_dim,
        )
    elif configs.env_type == 'TableGeoCon':
        score_net = ScoreModelGNN(
            marginal_prob_std_fn, 
            num_classes=configs.num_classes, 
            hidden_dim=configs.hidden_dim_gf,
            embed_dim=configs.embed_dim_gf,
            device=device,
        )
    else:
        raise ValueError(f"Mode {configs.env_type} not recognized.")
    return score_net


def get_functions(configs):
    if configs.env_type == 'NoTableGeo': 
        loss_fn = functools.partial(loss_fn_table_v2, batch_size=configs.batch_size_gf)
    elif configs.env_type == 'TableGeoCon':
        loss_fn = functools.partial(loss_fn_uncond, num_objs=configs.num_objs)
    else:
        raise ValueError(f"Mode {configs.env_type} not recognized.")
    return loss_fn


def gf_trainer(configs, writer, rotation, scene_name, data_root):
    
    # get dataloaders
    dataloader_train, dataloader_vis = get_dataloaders(rotation, scene_name, data_root)
    # get category list
    with open(f"datasets/{scene_name}/category/category_dict.pkl", 'rb') as f:
        category_dict = pickle.load(f)

    # init SDE-related params
    sigma = configs.sigma  # @param {'type':'number'}
    marginal_prob_std_fn = functools.partial(marginal_prob_std_v2, sigma=sigma)
    diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)
    
    # init SDE config
    prior_fn, marginal_prob_fn_v2, sde_fn, sampling_eps = init_sde(config.sde_mode)

    # create models, optimizers, and loss
    state_dim = 4 if rotation==True else 2
    score = get_score_network(configs, marginal_prob_std_fn, state_dim)

    score.to(device)
    optimizer = optim.Adam(score.parameters(), lr=configs.lr, betas=(configs.beta1, 0.999))
    ema = ExponentialMovingAverage(score.parameters(), decay=config.ema_rate)
    
    # determine loss, sde-sampler and visualisation functions
    loss_fn = get_functions(configs)

    print("Starting Training Loop...")
    for epoch in trange(configs.n_epoches):
        # For each batch in the dataloader
        for i, real_data in enumerate(dataloader_train):
            cur_idx = i + epoch*len(dataloader_train)
            # calc score-matching loss
            loss = 0
            for _ in range(configs.repeat_loss):
                loss += loss_fn(score, real_data, marginal_prob_fn_v2, sde_fn)
            loss /= configs.repeat_loss
            
            optimizer.zero_grad()
            loss.backward()

            # warmup
            if config.warmup > 0:
                for g in optimizer.param_groups:
                    g['lr'] = config.lr * np.minimum(cur_idx / config.warmup, 1.0)
            
            # grad clip
            if config.grad_clip >= 0:
                torch.nn.utils.clip_grad_norm_(score.parameters(), max_norm=config.grad_clip)

            optimizer.step()
            writer.add_scalar('train/train_learning_rate', optimizer.param_groups[0]['lr'], cur_idx)
            writer.add_scalars('train/train_loss', {'current': loss}, cur_idx)

            ema.update(score.parameters())

            ''' get ema training loss '''
            if config.ema_rate > 0 and cur_idx % 5 == 0:
                ema.store(score.parameters())
                ema.copy_to(score.parameters())
                with torch.no_grad():
                    loss = 0
                    for _ in range(config.repeat_loss):
                        # calc score-matching loss
                        loss += loss_fn(score, real_data, marginal_prob_fn_v2, sde_fn, likelihood_weighting=config.likelihood_weighting)
                    loss /= config.repeat_loss
                    writer.add_scalars('train/train_loss', {'ema': loss}, cur_idx)
                ema.restore(score.parameters())

        """ start eval """
        if (epoch + 1) % configs.vis_freq == 0:
            print('loss = ', loss)
 
            ref_batch = next(iter(dataloader_vis))

            res_batch, res_video = ode_sampler(
                score,
                ref_batch,
                sde_fn,
                prior_fn,
                eps=sampling_eps,
                t0=1.0,
                batch_size=2**2,
                use_rotation=rotation,
            ) 
           
            if rotation==True: 
                # generated states visualization
                gen_bbox = Rotated_bbox_vis(res_batch, category_dict)
                writer.add_image(f'Images/gen_imgs', gen_bbox, epoch)
                # ground truth states visualization
                true_bbox = Rotated_bbox_vis(ref_batch, category_dict)
                writer.add_image(f'Images/true_imgs', true_bbox, epoch)
            else:
                # generated states visualization
                gen_bbox = bbox_vis(res_batch, category_dict)
                writer.add_image(f'Images/gen_imgs', gen_bbox, epoch)
                # ground truth states visualization
                true_bbox = bbox_vis(ref_batch, category_dict)
                writer.add_image(f'Images/true_imgs', true_bbox, epoch)
            
            # ckpt_path = os.path.join(f'./logs/{scene_name}', f'{scene_name}.pt')
            ckpt_path = f'./logs/ckpt/{scene_name}.ckpt'

            with open(ckpt_path, 'wb') as f:
                pickle.dump(score, f)
            score.to(device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rotation', default=True)
    parser.add_argument('--scene_name', default='dinning_table_vanilla')
    def get_data_root(scene_name):
        return f'./datasets/{scene_name}/refined_data'
    parser.add_argument('--data_root', default=get_data_root(parser.parse_known_args().scene_name),
                    help='The root directory of the data.')
    # load args
    args = parser.parse_args()

    tb_path = os.path.join('./logs', args.scene_name)
    exists_or_mkdir(tb_path)
    writer = SummaryWriter(tb_path)

    # Run the training pipeline
    gf_trainer(config, writer, args.rotation, args.scene_name, args.data_root)