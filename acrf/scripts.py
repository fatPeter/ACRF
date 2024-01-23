import sys
import os
import argparse
from multiprocessing import Process, Queue
from typing import List, Dict
import subprocess
import mmcv

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='syn', choices=['syn', 'tnt'])

parser.add_argument('--eval', action='store_true', default=False)
parser.add_argument("--dump_images",  action='store_true', default=False)

parser.add_argument("--pruning",  type=str,  default='view')
parser.add_argument("--importance_list", nargs="+", type=float, default=[0.99, 0.999])  
parser.add_argument("--Qstep",  type=float,  default=0.5)
parser.add_argument("--Qfactor",  type=float,  default=10)
parser.add_argument("--lamda",  type=float,  default=1e-2) 

args = parser.parse_args()


if args.dataset == 'syn':
    args.configname = 'syn_4096code'
elif args.dataset == 'tnt':
    args.configname = 'tnt_4096code'

model_name='acrf_L%0.4f'%args.lamda



def run_exp(env, config, datadir, expname, basedir):
    
    cfg = mmcv.Config.fromfile(config)
    cfg.expname = expname
    cfg.data.datadir = datadir
    cfg.basedir = basedir
    
    auto_config_path = f'../configs/auto/{expname}.py'
    cfg.dump(auto_config_path)
    print('********************************************')
    
    auto_config_path = f'./configs/auto/{expname}.py'
    # --render_fine 
    base_cmd = ['python', 'run.py',  '--config', auto_config_path, 
                '--eval_ssim','--eval_lpips_vgg', '--eval_lpips_alex', 
            '--render_test',
            f'--importance_list', ' '.join([str(num) for num in args.importance_list]),
            f'--pruning {args.pruning}', 
            f'--Qstep {args.Qstep}', f'--Qfactor {args.Qfactor}',
            f'--lamda {args.lamda}', 
            ]

    if args.dump_images:
        base_cmd.append('--dump_images')

    opt_cmd = ' '.join(base_cmd)
    print(opt_cmd, "on ", env["CUDA_VISIBLE_DEVICES"])
    opt_ret = subprocess.check_output(opt_cmd, shell=True, env=env).decode(
        sys.stdout.encoding)







DatasetSetting={
    "syn": {
        "data": "./data/nerf_synthetic",
        "cfg": f"../configs/batch_test/{args.configname}.py",
        "basedir":f"./logs/{args.configname}",
        "scene_list":['chair', 'drums', 'ficus', 'hotdog', 'lego', 'materials', 'mic', 'ship'],
    },
    "tnt":{
        "data": "./data/TanksAndTemple",
        "cfg": f"../configs/batch_test/{args.configname}.py",
        "basedir":f"./logs/{args.configname}",
        "scene_list":['Barn', 'Caterpillar', 'Family', 'Ignatius', 'Truck']
    },
}






datasetting = DatasetSetting[args.dataset]
all_tasks = []

for scene in datasetting["scene_list"]:

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(0)


    task: Dict = {}
    task['datadir'] = f'{datasetting["data"]}/{scene}'
    task['expname'] = f'{args.configname}_{scene}'  
    task["config"] = datasetting['cfg']
    
    task["basedir"] = datasetting["basedir"]

    run_exp(env, **task)







class AverageMeter(object):
    def __init__(self, name=''):
        self.name=name
        self.reset()
    def reset(self):
        self.val=0
        self.sum=0
        self.avg=0
        self.count=0
    def update(self,val,n=1):
        self.val=val
        self.sum += val*n
        self.count += n
        self.avg=self.sum/self.count
    def __repr__(self) -> str:
        return f'{self.name}: average {self.count}: {self.avg}\n'

from prettytable import PrettyTable
table = PrettyTable(["Scene", "PSNR", "SSIM", "LPIPS_ALEX","LPIPS_VGG", "SIZE", "TIME"])
table.float_format = '.3'


PSNR=AverageMeter('PSNR')
SSIM=AverageMeter('SSIM')
LPIPS_A=AverageMeter('LPIPS_A')
LPIPS_V=AverageMeter('LPIPS_V')
SIZE=AverageMeter('SIZE')
TIME=AverageMeter('TIME')

for scene in datasetting["scene_list"]:
    dir_path = f'../logs/{args.configname}/{model_name}/{args.configname}_{scene}/render_test'
    
    path = dir_path+'/mean.txt'
    with open(path, 'r') as f:
        lines = f.readlines()
        psnr = float(lines[0].strip())
        ssim = float(lines[1].strip())
        lpips_a = float(lines[2].strip())
        lpips_v = float(lines[3].strip())
        PSNR.update(psnr)
        SSIM.update(ssim)
        LPIPS_A.update(lpips_a)
        LPIPS_V.update(lpips_v)
    
    path = os.path.join(dir_path, '../filesize.txt')
    with open(path, 'r') as f:
        lines = f.readlines()
        size = float(lines[0].strip())    
    path = os.path.join(dir_path, '../time.txt')
    with open(path, 'r') as f:
        lines = f.readlines()
        time = float(lines[0].strip()) 

    table.add_row([scene, psnr, ssim, lpips_a, lpips_v, size, time])
    SIZE.update(size)
    TIME.update(time)
    
        
table.add_row(['Mean', PSNR.avg, SSIM.avg, LPIPS_A.avg,LPIPS_V.avg, SIZE.avg, TIME.avg])

datasetting["basedir"] ='../'+datasetting["basedir"]
txt_file = os.path.join(datasetting["basedir"], f'merge_{model_name}.txt')
with open(txt_file, 'w') as f:
    f.writelines(table.get_string())

csv_file = os.path.join(datasetting["basedir"], f'merge_{model_name}.csv')
with open(csv_file, 'w', newline='') as f:
    f.writelines(table.get_csv_string())

print('AC-DVGO:')
print(table)




























