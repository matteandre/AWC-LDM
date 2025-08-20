import math
import sys

sys.path.append('./')

import os, argparse, glob, datetime, yaml
import torch
from torch.utils.data import DataLoader
import time
import numpy as np
from tqdm import tqdm
import joblib
from pathlib import Path
from omegaconf import OmegaConf
from PIL import Image
from lidm.utils.misc_utils import instantiate_from_config, set_seed
from lidm.utils.boreas_utils import range2pcd_channel,parse_calib

# remove annoying user warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


size = [128, 1024]
fov = [ 15,-25 ]
depth_range = [ 1.0,56.0 ]
depth_scale = 5.84
log_scale = True



def custom_to_pcd(x, config):
    x = x.squeeze().detach().cpu().numpy()
    x = (np.clip(x, -1., 1.) + 1.) / 2.
    elevation = parse_calib(os.path.join(Path(__file__).parent.parent, 'lidm', 'utils', 'calib_128.json'))
    xyz = range2pcd_channel(x, elevation, **config['data']['params']['dataset'])

    return xyz


def postprocessing(img_rec, img_inp):
    img_rec = img_rec.reshape(-1)
    depth_rec = np.exp2((img_rec * .5 + .5) *depth_scale).reshape(-1) -1
    
    img_inp = img_inp.reshape(-1)
    depth_inp = np.exp2((img_inp * .5 + .5) * depth_scale).reshape(-1) -1

    t = depth_rec**2.0 / 100 + 0.2
    
    delta = np.abs(depth_rec - depth_inp)
    
    for i in range(delta.shape[0]):
        if delta[i] < t[i]:
            img_rec[i] = img_inp[i]
        else:
            img_rec[i] = [img_rec[i], img_inp[i]][np.argmin(np.abs([img_rec[i], img_inp[i]]))]
    
    return img_rec.reshape(128,1024)




def run(model, dataloader, pcdlogdir, config=None, log_config={}):
    tstart = time.time()
    print(f"Running conditional sampling")
    count = 0
    for batch in tqdm(dataloader, desc="Sampling Batches (conditional)"):
        
        count +=1

        x = batch['image']
        x = x.to(model.device)
        x0 = model.encode_first_stage(x)
        cond = torch.ones(x0.shape[0],1).to(model.device) # snow
        t = torch.full((x0.shape[0],), log_config['ddim_steps'], device=model.device).long()
        x_T = model.q_sample(x0, t)
        samples = model.p_sample_loop(cond, x_T.shape, x_T=x_T, start_T=log_config['ddim_steps'])
        x_samples = model.decode_first_stage(samples, predict_cids=True)

        post = postprocessing(x_samples[0].to('cpu'), x[0].to('cpu'))

        post_xyz = custom_to_pcd(post, config)


        post_xyz.tofile(os.path.join(pcdlogdir, f'{count}.bin'))

    print(f"Sampling finished in {(time.time() - tstart) / 60.:.2f} minutes.")



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
        "-l",
        "--logdir",
        type=str,
        nargs="?",
        help="extra logdir",
        default="none"
    )
    parser.add_argument(
        "-t",
        "--number_steps",
        type=int,
        nargs="?",
        help="number of steps for ddim and fastdpm sampling",
        default= 200 #600
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
        default="boreas"
    )
    parser.add_argument(
        "--baseline",
        default=False,
        action='store_true',
        help="baseline provided by other sources (default option is not baseline)?",
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
    model = load_model_from_config(config.model, pl_sd["state_dict"])
    return model, global_step



if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    sys.path.append(os.getcwd())
    command = " ".join(sys.argv)

    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    ckpt = None
    set_seed(opt.seed)

    if not os.path.exists(opt.resume) and not os.path.exists(opt.file):
        raise FileNotFoundError
    if os.path.isfile(opt.resume):
        try:
            logdir = '/'.join(opt.resume.split('/')[:-2])
            print(f'Logdir is {logdir}')
        except ValueError:
            paths = opt.resume.split("/")
            idx = -2 # take a guess: path/to/logdir/checkpoints/model.ckpt
            logdir = "/".join(paths[:idx])
        ckpt = opt.resume


    if not opt.baseline:
        base_configs = [f'{logdir}/config.yaml']
    else:
        base_configs = [f'models/baseline/{opt.dataset}/template/config.yaml']
    opt.base = base_configs

    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)

    gpu = True
    eval_mode = True
    if opt.logdir != "none":
        locallog = logdir.split(os.sep)[-1]
        if locallog == "": locallog = logdir.split(os.sep)[-2]
        print(f"Switching logdir from '{logdir}' to '{os.path.join(opt.logdir, locallog)}'")
        logdir = os.path.join(opt.logdir, locallog)


    model, global_step = load_model(config, ckpt)
    print(f"global step: {global_step}")
    print(75 * "=")
    print("logging to:")
    logdir = os.path.join(logdir, "samples", f"{global_step:08}", now)
    pcdlogdir = os.path.join(logdir, "pcd")

    os.makedirs(pcdlogdir)
    print(logdir)
    print(75 * "=")

    # write config out
    sampling_file = os.path.join(logdir, "sampling_config.yaml")
    sampling_conf = vars(opt)

    with open(sampling_file, 'w') as f:
        yaml.dump(sampling_conf, f, default_flow_style=False)
    print(sampling_conf)

    # traverse all validation data
    data_config = config['data']['params']['validation']
    data_config['params'].update({'dataset_config': config['data']['params']['dataset'],
                                    'aug_config': config['data']['params']['aug'], 'return_pcd': False,
                                    'max_objects_per_image': 5})
    dataset = instantiate_from_config(data_config)
    
    dataloader = DataLoader(dataset, batch_size=1, num_workers=8, shuffle=False, drop_last=False)

    # settings
    log_config = {'sample': True, 'ddim_steps': opt.number_steps,
                    'quantize_denoised': False, 'inpaint': False, 'plot_progressive_rows': False,
                    'plot_diffusion_rows': False, 'dset': dataset}
    
    run(model, dataloader, pcdlogdir,
                                config=config, log_config=log_config)
