import numpy as np


def get_lidar_transform(config, split):
    transform_list = []
    if config['rotate']:
        transform_list.append(RandomRotateAligned())
    if config['flip']:
        transform_list.append(RandomFlip())
    return Compose(transform_list) if len(transform_list) > 0 and split == 'train' else None


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, pcd, pcd1=None):
        for t in self.transforms:
            pcd, pcd1 = t(pcd, pcd1)
        return pcd, pcd1


class RandomFlip(object):
    def __init__(self, p=1.):
        self.p = p

    def __call__(self, coord, coord1=None):
        if np.random.rand() < self.p:
            if np.random.rand() < 0.5:
                coord[:, 0] = -coord[:, 0]
                if coord1 is not None:
                    coord1[:, 0] = -coord1[:, 0]
            if np.random.rand() < 0.5:
                coord[:, 1] = -coord[:, 1]
                if coord1 is not None:
                    coord1[:, 1] = -coord1[:, 1]
        return coord, coord1


class RandomRotateAligned(object):
    def __init__(self, rot=np.pi / 4, p=1.):
        self.rot = rot
        self.p = p

    def __call__(self, coord, coord1=None):
        if np.random.rand() < self.p:
            angle_z = np.random.uniform(-self.rot, self.rot)
            cos_z, sin_z = np.cos(angle_z), np.sin(angle_z)
            R = np.array([[cos_z, -sin_z, 0], [sin_z, cos_z, 0], [0, 0, 1]])
            coord = np.dot(coord, R)
            if coord1 is not None:
                coord1 = np.dot(coord1, R)
        return coord, coord1


class RandomKeypointDrop(object):
    def __init__(self, num_range=(5, 60), p=.5):
        self.num_range = num_range
        self.p = p

    def __call__(self, center, category=None):
        if np.random.rand() < self.p:
            num = len(center)
            if num > self.num_range[0]:
                num_kept = np.random.randint(self.num_range[0], min(self.num_range[1], num))
                idx_kept = np.random.choice(num, num_kept, replace=False)
                center, category = center[idx_kept], category[idx_kept]
        return center, category

