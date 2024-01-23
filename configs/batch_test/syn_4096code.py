_base_ = '../default.py'

expname = 'lego_prune'
basedir = './logs/nerf_synthetic'

data = dict(
    datadir='./data/nerf_synthetic/chair',
    dataset_type='blender',
    white_bkgd=True,
)



fine_train = dict(
    N_iters=20000,
    importance_step=20000,
    prune_step=20000,
)




