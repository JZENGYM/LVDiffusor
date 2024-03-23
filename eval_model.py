import argparse
import functools
import os
import pickle
import cv2
import numpy as np
import torch

from config import config
from datasets import GraphDataset
from ipdb import set_trace
from PIL import Image
from score_matching.sampler import ode_sampler
from score_matching.sde import init_sde, marginal_prob_std_v2
from score_nets import ArrangeScoreModelGNN, ScoreModelGNN
from torch_geometric.loader import DataLoader
from utils.misc import exists_or_mkdir
from utils.visualisations import images_to_gif
from utils.bbox_vis import rgb_vis, Rotated_rgb_vis


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load test data
def get_dataloaders(rotation, scene_name):
    dataset= GraphDataset(scene_name, rotation=rotation, base_noise_scale=config.base_noise_scale)
    dataloader_vis = DataLoader(dataset, batch_size=2**2, shuffle=True, num_workers=4)
    return dataloader_vis

# load score model
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


def eval_model(configs, log_dir, test_dir, save_path, rotation):
    # get dataloaders
    dataloader_vis = get_dataloaders(rotation, test_dir)

    # load rgb
    rgb = cv2.imread(f'datasets/{test_dir}/rgb.png')
    with open(f'datasets/{test_dir}/masks_unsorted.pkl', 'rb') as f:
        masks = pickle.load(f)

    # load category list
    with open(f"datasets/{test_dir}/category/category_dict.pkl", 'rb') as f:
        category_dict = pickle.load(f)
    # load detection info
    with open(f'datasets/{test_dir}/data/bbox_data.pkl', 'rb') as f:
        bbox_data = pickle.load(f)

    # init SDE-related params
    sigma = configs.sigma
    marginal_prob_std_fn = functools.partial(marginal_prob_std_v2, sigma=sigma)

    # create models
    state_dim = 4 if rotation==True else 2
    score = get_score_network(configs, marginal_prob_std_fn, state_dim)

    #load pretrained model
    # with open('./logs/test/score.pt', 'rb') as f:
    #     score = pickle.load(f)
    ckpt_dir = f'./logs/{log_dir}/{log_dir}.ckpt'
    score.load_state_dict(torch.load(ckpt_dir))
    score.to(device)
    prior_fn, marginal_prob_fn_v2, sde_fn, sampling_eps = init_sde(configs.sde_mode)

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
    print("inference done!")
    # save target layout
    pos = res_batch.x.cpu().numpy()
    pos[:, :2] = (pos[:, :2] + 1.0) / 2.0

    # imgs of all samping step
    print("rendering ...")
    imgs = []
    for i in range(len(res_video)):
        if rotation==True:
            img = Rotated_rgb_vis(res_video[i], category_dict, bbox_data, rgb, masks)
        else:
            img, _ = rgb_vis(res_video[i], category_dict, bbox_data, rgb, masks)    ### TO DO: 重排initial -> target, 根据area_sort ###
        imgs.append(img)
    
    # print(f'steps = {len(imgs)}')
    print("saving videos ...")
    fps = len(imgs)//5
    test = [Image.fromarray(img[:, :, ::-1].astype(np.uint8), mode='RGB') for img in imgs]
    
    images_to_gif(os.path.join(save_path, f'{log_dir}.gif'), test, fps=200)

    cv2.imwrite(os.path.join(save_path, f'eval_initial.jpg'), imgs[0])
    cv2.imwrite(os.path.join(save_path, f'eval_ending.jpg'), imgs[-1])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', default='debugV')
    parser.add_argument('--test_dir')
    parser.add_argument('--rotation', default=True)
    # load args
    args = parser.parse_args()
    video_save_path = f'./logs/eval_video/{args.log_dir}'
    exists_or_mkdir(video_save_path)
    eval_model(config, args.log_dir, args.test_dir, video_save_path, args.rotation)
