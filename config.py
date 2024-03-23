from pickle import FALSE

import ml_collections

# def get_config():
config = ml_collections.ConfigDict()
config.env_type = 'NoTableGeo'

### Train GF ###
config.data_name = 'data'
config.sde_mode = 've'
config.likelihood_weighting = False
# config.num_classes = 3

#train
config.n_epoches = 100000
config.batch_size_gf = 16
config.repeat_loss = 10  #1000
config.lr = 2e-4
config.beta1 = 0.9
config.hidden_dim_gf = 128
config.embed_dim_gf = 64
config.sigma = 25.
# config.base_noise_scale = 0.00001
config.base_noise_scale = 0.0

config.warmup = 100
config.grad_clip = 1.
config.ema_rate = 0.999
#eval
config.vis_freq = 30
    # return config
