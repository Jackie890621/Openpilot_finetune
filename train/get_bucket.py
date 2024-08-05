import sys
sys.path.append("/media/NVMe_SSD_980_PRO/Meng-Tai/openpilot-pipeline")
from scipy.ndimage import gaussian_filter1d
from scipy.ndimage import convolve1d
import numpy as np
from utils import load_h5_roach
import os

def get_lds_kernel_window(kernel, ks, sigma):
    assert kernel in ['gaussian', 'triang', 'laplace']
    half_ks = (ks - 1) // 2
    if kernel == 'gaussian':
        base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
        kernel_window = gaussian_filter1d(base_kernel, sigma=sigma) / max(gaussian_filter1d(base_kernel, sigma=sigma))
    elif kernel == 'triang':
        kernel_window = triang(ks)
    else:
        laplace = lambda x: np.exp(-abs(x) / sigma) / (2. * sigma)
        kernel_window = list(map(laplace, np.arange(-half_ks, half_ks + 1))) / max(map(laplace, np.arange(-half_ks, half_ks + 1)))

    return kernel_window


def get_bucket_info(max_target=201, lds=False, lds_kernel='gaussian', lds_ks=5, lds_sigma=2):
    mp4files = []
    with open('../cache/videos.txt', 'r') as f:
        mp4files = f.read().splitlines()
    num_list = range(212)
    value_dict = {x: 0 for x in range(max_target)}

    for num in num_list:
        path_to_segment = mp4files[num]
        name_mp4 = os.path.basename(path_to_segment) # xxxx.mp4
        file_name = os.path.splitext(name_mp4)[0] # xxxx
        dirname = os.path.dirname(path_to_segment)
        path_to_load = os.path.join(dirname, file_name+'.h5')
        plan_gt_h5, laneline_gt_h5, leads_gt_h5, lead_prob_gt_h5, desire_state_gt_h5 = load_h5_roach(path_to_load, dif_name=True)
        for k in range(leads_gt_h5.shape[0]):
            lead_dis = leads_gt_h5[k,0,0]
            if int(lead_dis) < max_target:
                value_dict[int(lead_dis)] += 1

    bucket_centers = np.asarray([k for k, _ in value_dict.items()])
    bucket_weights = np.asarray([v for _, v in value_dict.items()])
    if lds:
        lds_kernel_window = get_lds_kernel_window(lds_kernel, lds_ks, lds_sigma)
        print(f'Using LDS: [{lds_kernel.upper()}] ({lds_ks}/{lds_sigma})')
        bucket_weights = convolve1d(bucket_weights, weights=lds_kernel_window, mode='constant')

    bucket_centers = np.asarray([bucket_centers[k] for k, v in enumerate(bucket_weights) if v > 0])
    bucket_weights = np.asarray([bucket_weights[k] for k, v in enumerate(bucket_weights) if v > 0])
    bucket_weights = bucket_weights / bucket_weights.sum()
    return bucket_centers, bucket_weights


if __name__ == "__main__":
    bucket_centers, bucket_weights = get_bucket_info(lds=False)
    print("bucket_centers : \n",bucket_centers)
    print("bucket_weights : \n",bucket_weights)