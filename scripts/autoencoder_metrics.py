import math
import sys

sys.path.append('./')
import os, argparse, glob, datetime, yaml
import torch
from torch.utils.data import DataLoader
import time
import numpy as np
from tqdm import tqdm
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image
from lidm.utils.misc_utils import instantiate_from_config, set_seed
from lidm.utils.boreas_utils import range2pcd_channel,parse_calib
from pathlib import Path
import metrics as metrics
import open3d as o3d

# remove annoying user warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)



def custom_to_pcd(x, config):
    x = x.squeeze().detach().cpu().numpy()
    x = (np.clip(x, -1., 1.) + 1.) / 2.
    elevation = parse_calib(os.path.join(Path(__file__).parent.parent, 'lidm', 'utils', 'calib_128.json'))
    xyz = range2pcd_channel(x, elevation, **config['data']['params']['dataset'])

    return xyz



def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        nargs="?",
        help="load from logdir or checkpoint in logdir"
    )
    
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        help="the numpy file path",
        default=1000
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        help="dataset name [nuscenes, kitti, boreas]",
        default='boreas'
    )
    return parser


def load_model_from_config(config, sd):
    model = instantiate_from_config(config)
    model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return model


def load_model(config, ckpt):
    if ckpt:
        print(f"Loading model from {ckpt}")
        pl_sd = torch.load(ckpt, map_location="cpu")
        global_step = pl_sd["global_step"]
    else:
        pl_sd = {"state_dict": None}
        global_step = None
    del config.model.params.lossconfig
    model = load_model_from_config(config.model, pl_sd["state_dict"])
    return model, global_step


def test_collate_fn(data):
    output = {}
    keys = data[0].keys()
    for k in keys:
        v = [d[k] for d in data]
        if k not in ['reproj', 'raw']:
            v = torch.from_numpy(np.stack(v, 0))
        else:
            v = [d[k] for d in data]
        output[k] = v
    return output


if __name__ == "__main__":
    sys.path.append(os.getcwd())
    command = " ".join(sys.argv)

    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    ckpt = None
    set_seed(opt.seed)

    if not os.path.exists(opt.resume):
        raise FileNotFoundError
    if os.path.isfile(opt.resume):
        try:
            logdir = '/'.join(opt.resume.split('/')[:-2])
            print(f'Logdir is {logdir}')
        except ValueError:
            paths = opt.resume.split("/")
            idx = -2  # take a guess: path/to/logdir/checkpoints/model.ckpt
            logdir = "/".join(paths[:idx])
        ckpt = opt.resume
    else:
        assert os.path.isdir(opt.resume), f"{opt.resume} is not a directory"
        logdir = opt.resume.rstrip("/")
        ckpt = os.path.join(logdir, "model.ckpt")

    base_configs = [f'{logdir}/config.yaml']
    opt.base = base_configs

    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)

    gpu = True
    eval_mode = True


    model, global_step = load_model(config, ckpt)
    print(f"global step: {global_step}")
    print(75 * "=")
    print("logging to:")

    # traverse all validation data
    log_su = 'boreas-2021-05-06-13-19'
    log_sn = 'boreas-2021-01-26-11-22'

    data_config = config['data']['params']['validation']
    data_config['params'].update({'dataset_config': config['data']['params']['dataset'],
                                    'aug_config': config['data']['params']['aug'], 'return_pcd': True, 'log' : log_su})
    dataset_su = instantiate_from_config(data_config)
    
    data_config['params'].update({'log' : log_sn})
    
    dataset_sn = instantiate_from_config(data_config)


    dataloader_su = DataLoader(dataset_su, batch_size=1, num_workers=8, shuffle=False, drop_last=False,
                            collate_fn=test_collate_fn)
    
    dataloader_sn = DataLoader(dataset_sn, batch_size=1, num_workers=8, shuffle=False, drop_last=False,
                            collate_fn=test_collate_fn)

    compare_every = 5

    with torch.no_grad():

        jsd = []
        cd = []

        
        for i, b in enumerate(tqdm(dataloader_su)):
            if i % compare_every != 0:
                continue

            log = model.log_images(b)

            inp = log['inputs']
            rec = log['reconstructions']

            pcd_inp = custom_to_pcd(inp, config)
            pcd_rec = custom_to_pcd(rec, config)

            pcd_pred = o3d.geometry.PointCloud()
            pcd_pred.points = o3d.utility.Vector3dVector(pcd_rec)
            pcd_gt = o3d.geometry.PointCloud()
            pcd_gt.points = o3d.utility.Vector3dVector(pcd_inp)

            jsd.append(metrics.compute_hist_metrics(pcd_gt, pcd_pred, bev=False))
            cd.append(metrics.compute_chamfer(pcd_gt, pcd_pred))
        


        jsd = np.mean(jsd)
        cd = np.mean(cd)
        print('SUN')
        print('JSD:', jsd)
        print('CD:', cd)
        print('--------------------------------')


        jsd = []
        cd = []

        
        for i, b in enumerate(tqdm(dataloader_sn)):
            if i % compare_every != 0:
                continue

            log = model.log_images(b)

            inp = log['inputs']
            rec = log['reconstructions']

            pcd_inp = custom_to_pcd(inp, config)
            pcd_rec = custom_to_pcd(rec, config)

            pcd_pred = o3d.geometry.PointCloud()
            pcd_pred.points = o3d.utility.Vector3dVector(pcd_rec)
            pcd_gt = o3d.geometry.PointCloud()
            pcd_gt.points = o3d.utility.Vector3dVector(pcd_inp)

            jsd.append(metrics.compute_hist_metrics(pcd_gt, pcd_pred, bev=False))
            cd.append(metrics.compute_chamfer(pcd_gt, pcd_pred))
        


        jsd = np.mean(jsd)
        cd = np.mean(cd)
        print('SNOW')
        print('JSD:', jsd)
        print('CD:', cd)
        print('--------------------------------')
