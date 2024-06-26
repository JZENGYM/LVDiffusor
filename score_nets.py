import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import EdgeConv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""

    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


# for table objects rearrangement
class ArrangeScoreModelGNN(nn.Module):
    def __init__(self, marginal_prob_std_func, hidden_dim, embed_dim, class_num=10, state_dim=4, size_dim=2):
        super(ArrangeScoreModelGNN, self).__init__()
        self.marginal_prob_std = marginal_prob_std_func
        hidden_dim = hidden_dim
        embed_dim = embed_dim

        self.init_enc = nn.Sequential(
            nn.Linear(state_dim + size_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
       
        self.embed_sigma = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim),
                                         nn.Linear(embed_dim, embed_dim))
        self.embed_category = nn.Embedding(class_num, embed_dim)

        cond_dim = embed_dim * 2


        ''' main backbone '''
        # conv1
        self.mlp1_main = nn.Sequential(
            nn.Linear((hidden_dim + cond_dim) * 2, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.conv1_main = EdgeConv(self.mlp1_main)
        # conv2
        self.mlp2_main = nn.Sequential(
            nn.Linear(hidden_dim * 2 + cond_dim * 2, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, state_dim),
        )
        self.conv2_main = EdgeConv(self.mlp2_main)

    def forward(self, batches, t):

        obj_batch = batches
        ''' get cond feat'''
        # class_feat: -> [num_nodes, embed_dim]
        class_feat = F.relu(self.embed_category(obj_batch.category.squeeze(-1)))
        # sigma_feat: [num_nodes, embed_dim]
        sigma_feat = F.relu(self.embed_sigma(t.squeeze(-1)))
        # total_cond_feat: [num_nodes, hidden_dim*2+embed_dim*2]
        # wall_feat = self.wall_enc(wall_batch)[obj_batch.batch]
        total_cond_feat = torch.cat([class_feat, sigma_feat], dim=-1)  # remember to consider wall

        ''' get init x feat '''
        # obj_init_feat: -> [num_nodes, hidden_dim]
        obj_init_feat = self.init_enc(torch.cat([obj_batch.x, obj_batch.geo], dim=-1))

        ''' main backbone of x '''
        # start massage passing from init-feature
        x = torch.cat([obj_init_feat, total_cond_feat], dim=-1)
        # node_num = x.shape[0]
        x = F.relu(self.conv1_main(x, obj_batch.edge_index))
        x = torch.cat([x, total_cond_feat], dim=-1)
        x = self.conv2_main(x, obj_batch.edge_index)

        # normalize the output
        x = x / (self.marginal_prob_std(t) + 1e-7)
        return x



class ScoreModelGNN(nn.Module):
    def __init__(self, marginal_prob_std_func, num_classes, device, hidden_dim=64, embed_dim=32):
        super(ScoreModelGNN, self).__init__()   
        self.device = device
        self.num_classes = num_classes

        # original x
        self.init_lin = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim)
        )
        # t-feature
        self.embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=embed_dim),
            nn.ReLU(True),
            nn.Linear(embed_dim, embed_dim),
        )

        # category-feature
        self.embed_category = nn.Sequential(
            nn.Embedding(num_classes, embed_dim),
            nn.ReLU(True),
            nn.Linear(embed_dim, embed_dim)
        )
        
        init_dim = hidden_dim + embed_dim
        self.mlp1 = nn.Sequential(
            nn.Linear(init_dim*2, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.conv1 = EdgeConv(self.mlp1)
        self.mlp2 = nn.Sequential(
            nn.Linear(hidden_dim*2+embed_dim*2, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.conv2 = EdgeConv(self.mlp2)
        self.mlp3 = nn.Sequential(
            nn.Linear(hidden_dim*2+embed_dim*2, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, 2),
        )
        self.conv3 = EdgeConv(self.mlp3)
        
        self.marginal_prob_std = marginal_prob_std_func

    def forward(self, state_inp, t, num_objs):
        # t.shape == [bs, 1]
        x, edge_index, categories = state_inp.x, state_inp.edge_index, state_inp.c

        # extract initial feature
        class_feature = self.embed_category(categories)
        init_feature = torch.cat([self.init_lin(x), class_feature], dim=-1)

        # get t feature
        bs = t.shape[0]
        x_sigma = F.relu(self.embed(t.squeeze(1))).unsqueeze(1).repeat(1, num_objs, 1).view(bs*num_objs, -1)

        # start massage passing from init-feature
        x = F.relu(self.conv1(init_feature, edge_index))
        x = torch.cat([x, x_sigma], dim=-1)
        x = F.relu(self.conv2(x, edge_index))
        x = torch.cat([x, x_sigma], dim=-1)
        x = self.conv3(x, edge_index)

        # normalize the output
        x = x / (self.marginal_prob_std(t.repeat(1, num_objs).view(bs*num_objs, -1))+1e-7)
        return x
