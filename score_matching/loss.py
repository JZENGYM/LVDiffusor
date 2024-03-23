import copy
import torch
from torch_scatter import scatter_sum

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# for table object rearrangement
def loss_fn_table_v2(model, x, marginal_prob_std_func, sde_fn, batch_size, eps=1e-5, likelihood_weighting=False):
    obj_batch = x
    bs = obj_batch.ptr.shape[0]
    # bs = 5
    obj_batch = obj_batch.to(device)
    random_t = torch.rand(batch_size, device=device) * (1. - eps) + eps
    # -> [bs, 1]
    random_t = random_t.unsqueeze(-1)
    # [bs, 1] -> [num_nodes, 1]
    random_t = random_t[obj_batch.batch]
    # z: [num_nodes, 3]
    z = torch.randn_like(obj_batch.x)
    # std: [num_nodes, 1]
    mu, std = marginal_prob_std_func(obj_batch.x, random_t)
    std = std.view(-1, 1) # [bs, 1]
    perturbed_obj_batch = copy.deepcopy(obj_batch)
    perturbed_obj_batch.x = mu + z * std
    output = model(perturbed_obj_batch, random_t)
    # output: [num_nodes, 3]
    
    if likelihood_weighting:
        # diffusion_coeff
        _, diffusion_coeff = sde_fn(random_t)
        sm_weights = diffusion_coeff ** 2
        tmp = torch.mean(torch.sum(((output + z / std)**2).view(bs, -1)))
        
        loss_ = sm_weights * torch.mean(torch.sum(((output + z / std)**2).view(bs, -1) * sm_weights, dim=-1))
        # set_trace()
    else:
        node_l2 = torch.sum((output * std + z) ** 2, dim=-1)
        batch_l2 = scatter_sum(node_l2, obj_batch.batch, dim=0)
        loss_ = torch.mean(batch_l2)
    return loss_



# for unconditional ball-arrangement
def loss_fn_uncond(model, x, marginal_prob_std_func, num_objs, eps=1e-5):
    x = x.to(device)
    random_t = torch.rand(x.x.shape[0]//num_objs, device=device) * (1. - eps) + eps
    # -> [bs, 1]
    random_t = random_t.unsqueeze(-1)
    z = torch.randn_like(x.x)
    # -> [bs*num_objs, 1]
    std = marginal_prob_std_func(random_t).repeat(1, num_objs).view(-1, 1)
    perturbed_x = copy.deepcopy(x)
    perturbed_x.x += z * std
    output = model(perturbed_x, random_t, num_objs)
    bs = random_t.shape[0]
    loss_ = torch.mean(torch.sum(((output * std + z)**2).view(bs, -1), dim=-1))
    return loss_

