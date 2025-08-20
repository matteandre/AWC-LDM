import os
import numpy as np
import yaml
import json
import glob
from lidm.data.base import DatasetBase
from ..utils.boreas_utils import pcd2range_channel, range2pcd_channel, parse_calib
from pathlib import Path


light_snow = "boreas-2020-12-01-13-26"
hard_snow  = "boreas-2021-01-26-11-22"


def natural_keys(path):
    number = int(path.split('/')[-1][:-4])
    return number


class BoreasDataset(DatasetBase):

    def __init__(self, data_root='/home/dataset/boreas', log_conf=os.path.join(Path(__file__).parent, 'logs.yaml'), split='train', log = None, **kwargs):
        super().__init__(data_root=data_root, split=split, **kwargs)
        self.split = split
        
        if log is not None:
            self.logs = [log]
        
        else:
            print(f'log split folder: {log_conf}')
            print(f'current split: {split}')
            with open(os.path.join(log_conf),'r') as f:
                self.logs = yaml.safe_load(f)[split.upper()]
        


        self.data = []

        


        for log in self.logs:
            print(os.path.join(self.data_root, log, 'lidar', '*.bin'))
            self.data.extend(glob.glob(os.path.join(self.data_root, log, 'lidar', '*.bin')))
        
        self.data.sort(key=natural_keys)


    def __len__(self):
        return len(self.data)
    

    def __getitem__(self, idx):

        example = dict()
        
        pcd = np.fromfile(self.data[idx], dtype=np.float32)
        pcd = pcd.reshape((-1, 6))

        sweep = pcd[:, :3]
        channel = pcd[:, -2].astype(np.int32)

        if self.lidar_transform:
            sweep, _ = self.lidar_transform(sweep, None)



        proj_range, _ = pcd2range_channel(sweep, channel, self.img_size, self.fov, self.depth_range)


        proj_range, proj_mask = self.process_scan(proj_range)
        example['image'], example['mask'] = proj_range, proj_mask

        v = 0.
        if light_snow in self.data[idx]:
            v = 0.5
        elif hard_snow in self.data[idx]:
            v = 1.

        example['snow'] = v 

        
        if self.return_pcd:
            elevation = parse_calib(os.path.join(Path(__file__).parent.parent, 'utils', 'calib_128.json'))
            reproj_sweep = range2pcd_channel(proj_range[0] * .5 + .5, elevation, self.fov, self.depth_range, self.depth_scale, self.log_scale)
            example['raw'] = sweep
            example['reproj'] = reproj_sweep.astype(np.float32)

            # image degradation
            if self.degradation_transform:
                degraded_proj_range = self.degradation_transform(proj_range)
                example['degraded_image'] = degraded_proj_range


        return example


    def prepare_data(self):
        return None




