import copy

import numpy as np
import torch
from ipdb import set_trace
from scipy import integrate
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


''' room sampler '''
# project the ori-gradient from Euler space to Riemann's
def grad_projection(scores, states_x):
    ''' project the original 4-D gradient [pos_grad(2), ori_grad(2)] to 3-D gradient [pos_grad(2), ori_grad(1)] '''
    pos_grad = scores[:, 0:2]
    ori_2d_grad = scores[:, 2:4]
    cur_n = states_x[:, 2:4]  # [sin(x), cos(x)]
    assert torch.linalg.norm(torch.sum(cur_n ** 2, dim=-1) - 1)**2 < 1e-7 # ensure cur_n are unit vectors
    cur_n = torch.cat([-cur_n[:, 0:1], cur_n[:, 1:2]], dim=-1)  # [-sin(x), cos(x)]
    ori_grad = torch.sum(torch.cat([ori_2d_grad[:, 1:2], ori_2d_grad[:, 0:1]], dim=-1) * cur_n, dim=-1, keepdim=True)
    return torch.cat([pos_grad, ori_grad], dim=-1)


def update_ori_euler(origin_ori_euler, delta_ori_riemann):
    # origin_ori_euler: [num_nodes, 2]
    # delta_ori_riemann: [mum_nodes, 1]
    cd = torch.cos(delta_ori_riemann)
    sd = torch.sin(delta_ori_riemann)
    cx = origin_ori_euler[:, 1:2]
    sx = origin_ori_euler[:, 0:1]
    updated_ori_euler = torch.cat([sx * cd + cx * sd, cx * cd - sx * sd], dim=-1)
    updated_ori_euler /= torch.sqrt(torch.sum(updated_ori_euler ** 2, dim=-1, keepdim=True))
    return updated_ori_euler

''' Sampling with rotation '''
def cond_ode_vel_sampler(
        score_fn,
        marginal_prob_std,
        diffusion_coeff,
        ref_batch,
        t0=1.,
        eps=1e-3,
        num_steps=500,
        batch_size=1,
        max_pos_vel=0.6,
        max_ori_vel=0.6,
        scale=0.04,
        device='cuda',
):
    wall_batch, obj_batch, _ = ref_batch
    wall_batch = wall_batch.to(device)
    obj_batch = obj_batch.to(device)

    # Create the latent code
    init_x = torch.randn_like(obj_batch.x, device=device) * marginal_prob_std(t0)
    

    def score_eval_wrapper(sample, time_steps):
        """A wrapper of the score-based model for use by the ODE solver."""
        with torch.no_grad():
            score = score_fn(sample, time_steps)
        return score

    def ode_func(t, x):
        """The ODE function for use by the ODE solver."""
        cur_obj_batch = copy.deepcopy(obj_batch)
        cur_obj_batch.x = torch.tensor(x.reshape(-1, 4)).to(device).float()
        time_steps = torch.ones(batch_size, device=device).unsqueeze(-1) * t
        time_steps = time_steps[obj_batch.batch]

        g = diffusion_coeff(torch.tensor(t)).cpu().numpy()
        return 0.5 * (g ** 2) * score_eval_wrapper((wall_batch, cur_obj_batch), time_steps)

    t_eval = np.linspace(t0, eps, num_steps)

    xs = []
    x = init_x
    x[:, 2:4] /= torch.sqrt(torch.sum(x[:, 2:4] ** 2, dim=-1, keepdim=True))

    # linear decay freq
    for _, t in enumerate(t_eval):
        # calc original ode_update_func
        scores = ode_func(t, x)  # [bs, 4]

        # projection
        scores_proj = grad_projection(scores, x)

        # scores -- normalise --> actions
        pos_vel = scores_proj[:, 0:2]
        pos_vel *= max_pos_vel / torch.max(torch.abs(pos_vel)) # RL action, pos_vel
        ori_vel_riemann = scores_proj[:, 2:3]
        ori_vel_riemann *= max_ori_vel / torch.max(torch.abs(ori_vel_riemann))  # RL action, ori vel

        # calc delta pos and ori
        delta_pos = pos_vel * scale
        delta_ori = ori_vel_riemann * scale

        # update pos and ori(euler)
        new_pos = x[:, 0:2] + delta_pos
        new_ori = update_ori_euler(x[:, 2:4], delta_ori)

        # update diffusion variable
        x = torch.cat([new_pos, new_ori], dim=-1)
        xs.append(x.cpu().unsqueeze(0).clone())

    return torch.cat(xs, dim=0), xs[-1][0]

''' Sampling without rotation '''
def ode_sampler(
        score_fn,
        ref_batch,
        sde_coeff,
        prior,
        t0=1.,
        batch_size=1,
        eps=1e-5,
        atol=1e-5, 
        rtol=1e-5, 
        num_steps=500, # 500
        use_rotation=True,
        denoise=True,
        device='cuda',
):
    ref_batch = ref_batch.to(device)

    if use_rotation==True:
        state_dim = 4
    else:
        state_dim = 2
        print('no rotation')
    assert ref_batch.x.shape[-1] == state_dim

    # Create the latent code
    # init_x = prior(ref_batch.x.shape).to(device)
    init_x = ref_batch.x.to(device)

    def score_eval_wrapper(sample, time_steps):
        """A wrapper of the score-based model for use by the ODE solver."""
        with torch.no_grad():
            score = score_fn(sample, time_steps)
            # print(score)
            # set_trace()
        return score.cpu().numpy().reshape(-1)

    def ode_func(t, x):
        # print(x)
        """The ODE function for use by the ODE solver."""
        cur_ref_batch = copy.deepcopy(ref_batch)
        cur_ref_batch.x = torch.tensor(x.reshape(-1, state_dim)).to(device).float()

        time_steps = torch.ones(batch_size, device=device).unsqueeze(-1) * t
        time_steps = time_steps[ref_batch.batch]

        drift, diffusion = sde_coeff(torch.tensor(t))
        drift = drift.cpu().numpy()
        diffusion = diffusion.cpu().numpy()
        # set_trace()
        # drift[-1] = 0
        # diffusion[-1] = 0
        return  drift - 0.5 * (diffusion**2) * score_eval_wrapper(cur_ref_batch, time_steps)

    # Run the black-box ODE solver. 
    t_eval = None
    if num_steps is not None:
        # num_steps, from t0 -> eps
        t_eval = np.linspace(t0, eps, num_steps)
    res = integrate.solve_ivp(
        ode_func, 
        (t0, eps), 
        init_x.reshape(-1).cpu().numpy(), 
        rtol=rtol, atol=atol, 
        method='RK45', 
        t_eval=t_eval
    )

    x = res.y[:, -1].reshape(-1, state_dim)
    
    # print(x)
    # denoise, using the predictor step in P-C sampler
    if denoise:
        # Reverse diffusion predictor for denoising
        vec_eps = torch.ones(batch_size, device=device).unsqueeze(-1) * eps
        vec_eps = vec_eps[ref_batch.batch]
        drift, diffusion = sde_coeff(vec_eps)
        diffusion = diffusion.cpu().numpy()
        # drift[-1] = 0
        # diffusion[-1] = 0

        cur_ref_batch = copy.deepcopy(ref_batch)
        cur_ref_batch.x = torch.tensor(x.reshape(-1, state_dim)).to(device).float()

        grad = score_fn(cur_ref_batch, vec_eps).cpu().detach().numpy()
        drift = drift - diffusion**2 * grad # R-SDE
        drift = drift.cpu().numpy()
        mean_x = x + drift * ((1 - eps) / (1000 if num_steps is None else num_steps))
        x = mean_x

    # print(x)
    # res_state = torch.clamp(torch.tensor(x, device=device), min=0.0, max=1.0)
    res_state = torch.clamp(torch.tensor(x, device=device), min=-1.0, max=1.0)
    if use_rotation==True:
        # set_trace()
        res_state[:, 2:4] = res_state[:, 2:4] / torch.linalg.norm(res_state[:, 2:4], dim=-1, keepdim=True)

    res_ref_batch = copy.deepcopy(ref_batch)
    res_ref_batch.x = res_state
    
    # set_trace()
    res_video = []
    for i in range(res.y.shape[1]):
        tmp = torch.clamp(torch.tensor(res.y[:, i], device=device).reshape(-1, state_dim), min=-1.0, max=1.0)
        if use_rotation==True:
            tmp[:, 2:4] /= torch.linalg.norm(tmp[:, 2:4], dim=-1, keepdim=True) # normalize --> sin^2 + cos^2 = 1
        
        # tmp = clip_state(tmp)
        res_ref_video = copy.deepcopy(ref_batch)
        res_ref_video.x = tmp
        res_video.append(res_ref_video)
    
    return res_ref_batch, res_video
