import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)


import argparse
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, '..')))

os.chdir("..") 



import os, sys, copy, glob, json, time, random, argparse
from shutil import copyfile
from tqdm import tqdm, trange

import mmcv
import imageio
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib import utils, dvgo, dcvgo, dmpigo
from lib.load_data import load_data

from torch_efficient_distloss import flatten_eff_distloss
import torch_scatter


import torch.optim as optim
import math



from lib.EB.EntropyBottleneck import EntropyBottleneck
from lib.RAHT.Haar3D_info_torch import haar3D, inv_haar3D


# reorder dvgo functions
from acdvgo_utils import voxel2points
from acdvgo_utils import load_acdvgo

from acdvgo_utils import create_new_model_for_acrf
from acdvgo_utils import compute_bbox_by_cam_frustrm
from acdvgo_utils import scene_rep_reconstruction
from acdvgo_utils import compute_bbox_by_coarse_geo

from acdvgo_utils import seed_everything
from acdvgo_utils import load_everything
from acdvgo_utils import render_viewpoints





def config_parser():
    '''Define command line arguments
    '''

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config',
                        help='config file path')
    parser.add_argument("--seed", type=int, default=777,
                        help='Random seed')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--no_reload_optimizer", action='store_true',
                        help='do not reload optimizer state from saved ckpt')
    parser.add_argument("--ft_path", type=str, default='',
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument("--export_bbox_and_cams_only", type=str, default='',
                        help='export scene bbox and camera poses for debugging and 3d visualization')
    parser.add_argument("--export_coarse_only", type=str, default='')

    # testing options
    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true')
    parser.add_argument("--render_train", action='store_true')
    parser.add_argument("--render_video", action='store_true')
    parser.add_argument("--render_video_flipy", action='store_true')
    parser.add_argument("--render_video_rot90", default=0, type=int)
    parser.add_argument("--render_video_factor", type=float, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')
    parser.add_argument("--dump_images", action='store_true')
    parser.add_argument("--eval_ssim", action='store_true')
    parser.add_argument("--eval_lpips_alex", action='store_true')
    parser.add_argument("--eval_lpips_vgg", action='store_true')
    # parser.add_argument("--apply_quant", default=True, type=bool)
    
    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=500,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_weights", type=int, default=100000,
                        help='frequency of weight ckpt saving')

    
    # acrf options   
    parser.add_argument("--pruning",  type=str,  default=None,
            help='pruning')    
    parser.add_argument("--importance_list", nargs="+", type=float, default=[],
            help='importance')       
    parser.add_argument("--Qstep",  type=float,  default=0.5,
            help='Qstep') 
    parser.add_argument("--Qfactor",  type=float,  default=10,
            help='Qfactor')        
    
    return parser







def compress(args, cfg, cfg_model, cfg_train, xyz_min, xyz_max, data_dict, stage, load_ckpt_path=None, ori_dir=None, compressed_dir=None):
    
    # init
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if abs(cfg_model.world_bound_scale - 1) > 1e-9:
        xyz_shift = (xyz_max - xyz_min) * (cfg_model.world_bound_scale - 1) / 2
        xyz_min -= xyz_shift
        xyz_max += xyz_shift
    HW, Ks, near, far, i_train, i_val, i_test, poses, render_poses, images = [
        data_dict[k] for k in [
            'HW', 'Ks', 'near', 'far', 'i_train', 'i_val', 'i_test', 'poses', 'render_poses', 'images'
        ]
    ]
   

    
    print(f'scene_rep_reconstruction (ac fintune): reload from {load_ckpt_path}')
    model, optimizer = create_new_model_for_acrf(cfg, cfg_model, cfg_train, xyz_min, 
                                               xyz_max, stage, load_ckpt_path, strict=False, device=device)
    # init rendering setup
    render_kwargs = {
        'near': data_dict['near'],
        'far': data_dict['far'],
        'bg': 1 if cfg.data.white_bkgd else 0,
        'rand_bkgd': cfg.data.rand_bkgd,
        'stepsize': cfg_model.stepsize,
        'inverse_y': cfg.data.inverse_y,
        'flip_x': cfg.data.flip_x,
        'flip_y': cfg.data.flip_y,
    }

    # init batch rays sampler
    def gather_training_rays():
        if data_dict['irregular_shape']:
            rgb_tr_ori = [images[i].to('cpu' if cfg.data.load2gpu_on_the_fly else device) for i in i_train]
        else:
            rgb_tr_ori = images[i_train].to('cpu' if cfg.data.load2gpu_on_the_fly else device)

        if cfg_train.ray_sampler == 'in_maskcache':
            rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz = dvgo.get_training_rays_in_maskcache_sampling(
                    rgb_tr_ori=rgb_tr_ori,
                    train_poses=poses[i_train],
                    HW=HW[i_train], Ks=Ks[i_train],
                    ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                    flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y,
                    model=model, render_kwargs=render_kwargs)
        elif cfg_train.ray_sampler == 'flatten':
            rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz = dvgo.get_training_rays_flatten(
                rgb_tr_ori=rgb_tr_ori,
                train_poses=poses[i_train],
                HW=HW[i_train], Ks=Ks[i_train], ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        else:
            rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz = dvgo.get_training_rays(
                rgb_tr=rgb_tr_ori,
                train_poses=poses[i_train],
                HW=HW[i_train], Ks=Ks[i_train], ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        index_generator = dvgo.batch_indices_generator(len(rgb_tr), cfg_train.N_rand)
        batch_index_sampler = lambda: next(index_generator)
        return rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz, batch_index_sampler

    rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz, batch_index_sampler = gather_training_rays()

    # view-count-based learning rate
    if cfg_train.pervoxel_lr:
        def per_voxel_init():
            cnt = model.voxel_count_views(
                    rays_o_tr=rays_o_tr, rays_d_tr=rays_d_tr, imsz=imsz, near=near, far=far,
                    stepsize=cfg_model.stepsize, downrate=cfg_train.pervoxel_lr_downrate,
                    irregular_shape=data_dict['irregular_shape'])
            optimizer.set_pervoxel_lr(cnt)
            model.mask_cache.mask[cnt.squeeze() <= 2] = False
        per_voxel_init()

    if cfg_train.maskout_lt_nviews > 0:
        model.update_occupancy_cache_lt_nviews(
                rays_o_tr, rays_d_tr, imsz, render_kwargs, cfg_train.maskout_lt_nviews)

    # GOGO
    torch.cuda.empty_cache()
    psnr_lst = []
    time0 = time.time()
    global_step = -1
    

    
    #=================== Initialize importance score  ====================
    
    stepsize = cfg.fine_model_and_render.stepsize
    render_viewpoints_kwargs = {
    'model': model,
    'ndc': cfg.data.ndc,
    'render_kwargs': {
        'near': data_dict['near'],
        'far': data_dict['far'],
        'bg': 1 if cfg.data.white_bkgd else 0,
        'stepsize': stepsize,
        'inverse_y': cfg.data.inverse_y,
        'flip_x': cfg.data.flip_x,
        'flip_y': cfg.data.flip_y,
        'render_depth': True,
        },
    }      
    importance_savedir = compressed_dir

    print("ACRF_F compression starts")
    time_all_begin = time.time()

    init_importance(
        render_poses=data_dict['poses'][data_dict['i_train']],
        HW=data_dict['HW'][data_dict['i_train']],
        Ks=data_dict['Ks'][data_dict['i_train']],
        savedir=importance_savedir, cfg=cfg,
        **render_viewpoints_kwargs)




    #=================== Appply voxel pruning and density quanzation  ====================
    mask_list=[]
    for importance_theta in args.importance_list:
        mask = model.init_cdf_mask(importance_theta)
        mask_list.append(mask)





    
    
    #=================== Compress   ====================
    num_cluster = len(args.importance_list)
    model.train()



    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    entropy_bottleneck_list = []
    optimizer_eb_list = []
    
    for i in range(num_cluster):
        entropy_bottleneck = EntropyBottleneck(channels=12).cuda()    
        optimizer_eb = optim.Adam(entropy_bottleneck.parameters(), lr=0.01)        
        entropy_bottleneck_list.append(entropy_bottleneck)
        optimizer_eb_list.append(optimizer_eb)


    
       
    #-------------------------# 
    # feature encoding  
    k0_grid = model.k0.grid.reshape(model.k0_dim,-1)
    k0_grid = k0_grid.T    

    pos = voxel2points(mask_list[-1].reshape(*model.world_size))

    depth = 0
    temp = pos.max()
    while temp>0:
        temp=temp//2
        depth=depth+1

    non_prune_feat = k0_grid[mask_list[-1],:].detach()


    temp=torch.zeros(mask_list[0].shape)
    for i in range(num_cluster):
        temp=temp+mask_list[i].int()
    labels_ = temp[temp!=0]-1
    labels_ = labels_.detach().cpu().numpy()

    Qstep = args.Qstep
    Qfactor = args.Qfactor

    CT_q_list = []
    for i in range(num_cluster):
        
        res = haar3D(pos.cpu().numpy()[labels_==i], non_prune_feat[labels_==i], depth)
        CT = res['CT']    
    
        CT_q = CT/(Qstep*Qfactor**(num_cluster-i-1))
        CT_q_list.append(CT_q)



    #-------------------------# 
    # feature encoding      
    num_step = 100
    for global_step in trange(1, num_step):

        bpv = 0
        for i in range(num_cluster):

            optimizer_eb_list[i].zero_grad()

            CT_q = CT_q_list[i]
            rand_idx = np.random.randint(0, CT_q.shape[0], int(CT_q.shape[0]*0.01))
            _, likelihood = entropy_bottleneck_list[i](CT_q[rand_idx], False, device)   
            bpv_i = torch.sum(torch.log(likelihood)) / -(torch.log(torch.Tensor([2.0]).to(device))) / CT_q.shape[0]        
            bpv = bpv+bpv_i

        # print(global_step, bpv)

        loss = bpv
        loss.backward()

        for i in range(num_cluster):
            optimizer_eb_list[i].step()            







  
    #-------------------------# 
    # compress 
    
    # compress CT
    strings_list=[]
    min_v_list=[]
    max_v_list=[]
    for i in range(num_cluster):
        with torch.no_grad():
            strings, min_v, max_v = entropy_bottleneck_list[i].compress(
                CT_q_list[i], device)
            strings_list.append(strings)
            min_v_list.append(min_v)
            max_v_list.append(max_v)


    time_all_end = time.time()
    time_all = time_all_end-time_all_begin
    print('time_all:', time_all_end-time_all_begin)
    print("ACRF_F compression finishes")
    
    #-------------------------# 
    # write  
    save_path=compressed_dir
       
    if os.path.exists(os.path.join(save_path, 'extreme_saving')):
        import shutil
        path = os.path.join(save_path, 'extreme_saving')
        shutil.rmtree(path)



    # save hyper, network, density
    model.ac_reformat(mask_list[-1], save_path=save_path)

    # save mask_list
    np.savez_compressed(os.path.join(save_path, 'extreme_saving/labels.npz'), labels_)
    
    for i in range(num_cluster):
        # save CT        
        CT_binname = save_path+'/extreme_saving/CT_%d.bin'%i
        with open(CT_binname, 'wb') as f:
            f.write(strings_list[i])

        shape = CT_q_list[i].shape
            
        head_binname = save_path+'/extreme_saving/head_%d.bin'%i
        with open(head_binname, 'wb') as f:
            f.write(np.array((min_v_list[i], max_v_list[i]), dtype=np.int32).tobytes())
            f.write(np.array(shape, dtype=np.int32).tobytes())
            f.write(np.array(Qstep*Qfactor**(num_cluster-i-1)*10, dtype=np.int32).tobytes()) # *10 for Qstep=0.5

        # save entropy model
        torch.save(entropy_bottleneck_list[i].state_dict(), 
                   os.path.join(save_path, 'extreme_saving/eb_%d.pth'%i))



    # time
    np.savetxt(f'{save_path}/time.txt', np.asarray([time_all]))

    
    return
    
    
    
    
    
    
        
    







def train(args, cfg, data_dict):

    # init
    print('train: start')
    eps_time = time.time()

    ori_dir=data_dict['ori_dir']
    compressed_dir=data_dict['compressed_dir']

    with open(os.path.join(compressed_dir, 'args.txt'), 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    cfg.dump(os.path.join(compressed_dir, 'config.py'))

    # coarse geometry searching (only works for inward bounded scenes)
    eps_coarse = time.time()
    xyz_min_coarse, xyz_max_coarse = compute_bbox_by_cam_frustrm(args=args, cfg=cfg, **data_dict)

    if cfg.coarse_train.N_iters > 0:
        scene_rep_reconstruction(
                args=args, cfg=cfg,
                cfg_model=cfg.coarse_model_and_render, cfg_train=cfg.coarse_train,
                xyz_min=xyz_min_coarse, xyz_max=xyz_max_coarse,
                data_dict=data_dict, stage='coarse')
        eps_coarse = time.time() - eps_coarse
        eps_time_str = f'{eps_coarse//3600:02.0f}:{eps_coarse//60%60:02.0f}:{eps_coarse%60:02.0f}'
        print('train: coarse geometry searching in', eps_time_str)
        coarse_ckpt_path = os.path.join(ori_dir, f'coarse_last.tar')
    else:
        print('train: skip coarse geometry searching')
        coarse_ckpt_path = None

    # fine detail reconstruction
    eps_fine = time.time()
    if cfg.coarse_train.N_iters == 0:
        xyz_min_fine, xyz_max_fine = xyz_min_coarse.clone(), xyz_max_coarse.clone()
    else:
        xyz_min_fine, xyz_max_fine = compute_bbox_by_coarse_geo(
                model_class=dvgo.DirectVoxGO, model_path=coarse_ckpt_path,
                thres=cfg.fine_model_and_render.bbox_thres)
    if not os.path.exists(os.path.join(ori_dir, f'fine_last.tar')):
        scene_rep_reconstruction(
                args=args, cfg=cfg,
                cfg_model=cfg.fine_model_and_render, cfg_train=cfg.fine_train,
                xyz_min=xyz_min_fine, xyz_max=xyz_max_fine,
                data_dict=data_dict, stage='fine',
                coarse_ckpt_path=coarse_ckpt_path)
        eps_fine = time.time() - eps_fine
        eps_time_str = f'{eps_fine//3600:02.0f}:{eps_fine//60%60:02.0f}:{eps_fine%60:02.0f}'
        print('train: fine detail reconstruction in', eps_time_str)

    compress(
            args=args, cfg=cfg,
            cfg_model=cfg.ac_model_and_render, cfg_train=cfg.ac_train,
            xyz_min=xyz_min_fine, xyz_max=xyz_max_fine,
            data_dict=data_dict, stage='ac',
            load_ckpt_path=os.path.join(ori_dir, f'fine_last.tar'),
            ori_dir=ori_dir,
            compressed_dir=compressed_dir,
            )
    eps_fine = time.time() - eps_fine
    eps_time_str = f'{eps_fine//3600:02.0f}:{eps_fine//60%60:02.0f}:{eps_fine%60:02.0f}'
    print('train: fine ACRF reconstruction in', eps_time_str)

    eps_time = time.time() - eps_time
    eps_time_str = f'{eps_time//3600:02.0f}:{eps_time//60%60:02.0f}:{eps_time%60:02.0f}'
    print('train: finish (eps time', eps_time_str, ')')







if __name__=='__main__':

    # load setup
    parser = config_parser()
    args = parser.parse_args()
    
 
    
    
    if args.pruning=='direct':
        from acdvgo_utils import init_importance_direct as init_importance
    elif args.pruning=='view':
        from acdvgo_utils import init_importance_view as init_importance

    cfg = mmcv.Config.fromfile(args.config)

    model_name='acrf_f_Q%0.1f'%args.Qstep
    
    ori_dir = os.path.join(cfg.basedir, 'ori', cfg.expname)
    compressed_dir = os.path.join(cfg.basedir, model_name, cfg.expname)
    if not os.path.exists(ori_dir):
        os.makedirs(ori_dir)    
    if not os.path.exists(compressed_dir):
        os.makedirs(compressed_dir)    

    args.importance_list=sorted(args.importance_list)



    # init enviroment
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    seed_everything(args)

    # load images / poses / camera settings / data split
    data_dict = load_everything(args=args, cfg=cfg)
    data_dict['ori_dir']=ori_dir
    data_dict['compressed_dir']=compressed_dir


    # train and compress
    if not args.render_only:
        train(args, cfg, data_dict)
        
        

    # load and decompress
    model_class = dvgo.DirectVoxGO
        
    model_kwargs, model_state_dict = load_acdvgo(os.path.join(compressed_dir,'extreme_saving'),device=device)
    model_kwargs['mask_cache_path'] = None
    model = model_class(**model_kwargs)
    model.eval()
    model.load_state_dict(model_state_dict, strict=False)
    
    model.to(device)
    model.mask_cache.mask[:] = True
    model.update_occupancy_cache()


    stepsize = cfg.fine_model_and_render.stepsize
    render_viewpoints_kwargs = {
        'model': model,
        'ndc': cfg.data.ndc,
        'render_kwargs': {
            'near': data_dict['near'],
            'far': data_dict['far'],
            'bg': 1 if cfg.data.white_bkgd else 0,
            'stepsize': stepsize,
            'inverse_y': cfg.data.inverse_y,
            'flip_x': cfg.data.flip_x,
            'flip_y': cfg.data.flip_y,
            'render_depth': True,
        },
    }        
        
    
    # eval
    # render trainset and eval
    if args.render_train:
        testsavedir = os.path.join(compressed_dir, f'render_train')
        os.makedirs(testsavedir, exist_ok=True)
        print('All results are dumped into', testsavedir)
        rgbs, depths, bgmaps = render_viewpoints(
                render_poses=data_dict['poses'][data_dict['i_train']],
                HW=data_dict['HW'][data_dict['i_train']],
                Ks=data_dict['Ks'][data_dict['i_train']],
                gt_imgs=[data_dict['images'][i].cpu().numpy() for i in data_dict['i_train']],
                savedir=testsavedir, dump_images=args.dump_images,
                eval_ssim=args.eval_ssim, eval_lpips_alex=args.eval_lpips_alex, eval_lpips_vgg=args.eval_lpips_vgg,
                cfg=cfg,
                **render_viewpoints_kwargs)
        imageio.mimwrite(os.path.join(testsavedir, 'video.rgb.mp4'), utils.to8b(rgbs), fps=30, quality=8)
        imageio.mimwrite(os.path.join(testsavedir, 'video.depth.mp4'), utils.to8b(1 - depths / np.max(depths)), fps=30, quality=8)

    # render testset and eval
    if args.render_test:
        
        testsavedir = os.path.join(compressed_dir, f'render_test')
        os.makedirs(testsavedir, exist_ok=True)
        print('All results are dumped into', testsavedir)
        rgbs, depths, bgmaps = render_viewpoints(
                render_poses=data_dict['poses'][data_dict['i_test']],
                HW=data_dict['HW'][data_dict['i_test']],
                Ks=data_dict['Ks'][data_dict['i_test']],
                gt_imgs=[data_dict['images'][i].cpu().numpy() for i in data_dict['i_test']],
                savedir=testsavedir, dump_images=args.dump_images,
                eval_ssim=args.eval_ssim, eval_lpips_alex=args.eval_lpips_alex, eval_lpips_vgg=args.eval_lpips_vgg,
                cfg=cfg,
                **render_viewpoints_kwargs)
        imageio.mimwrite(os.path.join(testsavedir, 'video.rgb.mp4'), utils.to8b(rgbs), fps=30, quality=8)
        imageio.mimwrite(os.path.join(testsavedir, 'video.depth.mp4'), utils.to8b(1 - depths / np.max(depths)), fps=30, quality=8)

    # render video
    if args.render_video:
        testsavedir = os.path.join(compressed_dir, f'render_video')
        os.makedirs(testsavedir, exist_ok=True)
        print('All results are dumped into', testsavedir)
        rgbs, depths, bgmaps = render_viewpoints(
                render_poses=data_dict['render_poses'],
                HW=data_dict['HW'][data_dict['i_test']][[0]].repeat(len(data_dict['render_poses']), 0),
                Ks=data_dict['Ks'][data_dict['i_test']][[0]].repeat(len(data_dict['render_poses']), 0),
                render_factor=args.render_video_factor,
                render_video_flipy=args.render_video_flipy,
                render_video_rot90=args.render_video_rot90,
                savedir=testsavedir, dump_images=args.dump_images,
                cfg=cfg,
                **render_viewpoints_kwargs)
        imageio.mimwrite(os.path.join(testsavedir, 'video.rgb.mp4'), utils.to8b(rgbs), fps=30, quality=8)
        import matplotlib.pyplot as plt
        depths_vis = depths * (1-bgmaps) + bgmaps
        dmin, dmax = np.percentile(depths_vis[bgmaps < 0.1], q=[5, 95])
        depth_vis = plt.get_cmap('rainbow')(1 - np.clip((depths_vis - dmin) / (dmax - dmin), 0, 1)).squeeze()[..., :3]
        imageio.mimwrite(os.path.join(testsavedir, 'video.depth.mp4'), utils.to8b(depth_vis), fps=30, quality=8)



    # report model size
    dir_path = os.path.join(compressed_dir,'extreme_saving')
    path_list = sorted(glob.glob(dir_path+r'/*'))
    filesize = 0
    for path in path_list:
        filesize += os.path.getsize(path)    
    print('filesize:', filesize/1024, 'KB')

    np.savetxt(f'{compressed_dir}/filesize.txt', np.asarray([filesize/1024]))

    

    print('Done')
    
    
    
    
    




