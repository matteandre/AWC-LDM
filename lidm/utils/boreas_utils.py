import numpy as np
import json



def pcd2range_channel(pcd, channels, size, fov, depth_range, remission=None, labels=None, **kwargs):
    # laser parameters
    points = pcd
    channels = 127 - channels

    fov_up = fov[0] / 180.0 * np.pi  # field of view up in rad
    fov_down = fov[1] / 180.0 * np.pi  # field of view down in rad

    # get depth (distance) of all points
    depth = np.linalg.norm(points, 2, axis=1)

    # mask points out of range
    mask = np.logical_and(depth > depth_range[0], depth < depth_range[1])
    depth, points, channels = depth[mask], points[mask], channels[mask]

    # get scan components
    scan_x, scan_y, scan_z = points[:, 0], points[:, 1], points[:, 2]

    # get angles of all points
    yaw = -np.arctan2(scan_y, scan_x)

    # get projections in image coords
    proj_x = 0.5 * (yaw / np.pi + 1.0)  # in [0.0, 1.0]

    # scale to image size using angular resolution
    proj_x *= size[1]  # in [0.0, W]

    # round and clamp for use as index
    proj_x = np.maximum(0, np.minimum(size[1] - 1, np.floor(proj_x))).astype(np.int32)  # in [0,W-1]

    proj_y = channels

    # order in decreasing depth
    order = np.argsort(depth)[::-1]
    proj_x, proj_y = proj_x[order], proj_y[order]

    # project depth
    depth = depth[order]
    proj_range = np.full(size, -1, dtype=np.float32)
    proj_range[proj_y, proj_x] = depth

    # project point feature
    if remission is not None:
        remission = remission[mask][order]
        proj_feature = np.full(size, -1, dtype=np.float32)
        proj_feature[proj_y, proj_x] = remission
    elif labels is not None:
        labels = labels[mask][order]
        proj_feature = np.full(size, 0, dtype=np.float32)
        proj_feature[proj_y, proj_x] = labels
    else:
        proj_feature = None

    return proj_range, proj_feature



def parse_calib(cal_path):
    with open(cal_path, 'r') as cal:
        cal = json.load(cal)
    elevation = [e['elevation'] for e in cal]
    elevation.sort(reverse=True)
    return elevation



def range2pcd_channel(range_img, elevation, fov, depth_range, depth_scale, log_scale=True, **kwargs):
    # laser parameters
    size = range_img.shape
    fov_up = fov[0] / 180.0 * np.pi  # field of view up in rad
    fov_down = fov[1] / 180.0 * np.pi  # field of view down in rad

    # inverse transform from depth
    depth = (range_img * depth_scale).flatten()
    if log_scale:
        depth = np.exp2(depth) - 1

    scan_x, scan_y = np.meshgrid(np.arange(size[1]), np.arange(size[0]))
    scan_x = scan_x.astype(np.float64) / size[1]

    yaw = (np.pi * (scan_x * 2 - 1)).flatten()
    scan_y = scan_y.flatten()
    elevation = np.array(elevation).flatten() / 180.0 * np.pi
    pitch = elevation[scan_y]



    pcd = np.zeros((len(yaw), 3))
    pcd[:, 0] = np.cos(yaw) * np.cos(pitch) * depth
    pcd[:, 1] = -np.sin(yaw) * np.cos(pitch) * depth
    pcd[:, 2] = np.sin(pitch) * depth

    # mask out invalid points
    mask = np.logical_and(depth > depth_range[0], depth < depth_range[1])
    pcd = pcd[mask, :]


    return pcd

