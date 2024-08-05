import sys
sys.path.append("/home/t2-503-4090/QianXi/Openpilot_BalancedRegression_Adapter")
from utils import printf, extract_preds, load_h5_roach, FULL_FRAME_SIZE, transform_frames, RADAR_TO_CAMERA, X_IDXs
import os
from model import load_trainable_model
import gc
import numpy as np
import torch
import torch.nn.functional as F
import random
import cv2
import h5py
from tensorboardX import SummaryWriter

#https://github.com/OpenDriveLab/Openpilot-Deepdive/blob/main/utils.py#L46C72-L46C72

def get_val_metric(pred_trajectory, labels, namespace='val'):
    rtn_dict = dict()
    # print(namespace)
    # print(pred_trajectory.shape)
    # print(labels.shape)
    l2_dists = F.mse_loss(pred_trajectory, labels, reduction='none')  # B, num_pts, 2 or 3
    # print(l2_dists.shape)
    rtn_dict.update({'l2_dist': l2_dists.mean(dim=(1, 2))})

    # New Metric
    distance_splits = ((0, 10), (10, 20), (20, 30), (30, 50), (50, 1000))
    AP_thresholds = (0.5, 1, 2)
    euclidean_distances = l2_dists.sum(-1).sqrt()  # euclidean distances over the points: [B, num_pts]    
    x_distances = labels[..., 0]  # B, num_pts
    # print(namespace)
    # print(x_distances)

    for min_dst, max_dst in distance_splits:
        points_mask = (x_distances >= min_dst) & (x_distances < max_dst)  # B, num_pts,
        if points_mask.sum() == 0:
            continue  # No gt points in this range
        rtn_dict.update({'eucliden_%d_%d' % (min_dst, max_dst): euclidean_distances[points_mask]})  # [sum(mask), ]
        rtn_dict.update({'eucliden_x_%d_%d' % (min_dst, max_dst): l2_dists[..., 0][points_mask].sqrt()})  # [sum(mask), ]
        rtn_dict.update({'eucliden_y_%d_%d' % (min_dst, max_dst): l2_dists[..., 1][points_mask].sqrt()})  # [sum(mask), ]

        for AP_threshold in AP_thresholds:
            hit_mask = (euclidean_distances < AP_threshold) & points_mask
            rtn_dict.update({'AP_%d_%d_%s' % (min_dst, max_dst, AP_threshold): hit_mask[points_mask]})

    # add namespace
    if namespace is not None:
        for k in list(rtn_dict.keys()):
            rtn_dict['%s/%s' % (namespace, k)] = rtn_dict.pop(k)
    return rtn_dict


def get_val_metric_keys(namespace='val'):
    rtn_dict = dict()
    rtn_dict.update({'l2_dist': []})

    # New Metric
    distance_splits = ((0, 10), (10, 20), (20, 30), (30, 50), (50, 1000))
    AP_thresholds = (0.5, 1, 2)

    for min_dst, max_dst in distance_splits:
        rtn_dict.update({'eucliden_%d_%d' % (min_dst, max_dst): []})  # [sum(mask), ]
        rtn_dict.update({'eucliden_x_%d_%d' % (min_dst, max_dst): []})  # [sum(mask), ]
        rtn_dict.update({'eucliden_y_%d_%d' % (min_dst, max_dst): []})  # [sum(mask), ]
        for AP_threshold in AP_thresholds:
            rtn_dict.update({'AP_%d_%d_%s' % (min_dst, max_dst, AP_threshold): []})

    # add namespace
    if namespace is not None:
        for k in list(rtn_dict.keys()):
            rtn_dict['%s/%s' % (namespace, k)] = rtn_dict.pop(k)
    return rtn_dict


def get_val_leads(pred_leads, labels, namespace='lead'):
    rtn_dict = dict()

    L1_loss = F.l1_loss(pred_leads[...,:2], labels[...,:2], reduction='none')  # B, 2

    # New Metric
    distance_splits = ((0, 10), (10, 20), (20, 30), (30, 50), (50, 131))
    AP_thresholds_scale = (0.5, 1., 1.5)

    x_distances = labels[..., 0] - RADAR_TO_CAMERA # B,

    for min_dst, max_dst in distance_splits:
        points_mask = (x_distances >= min_dst) & (x_distances < max_dst)  # B,
        if points_mask.sum() == 0:
            continue  # No gt points in this range
        rtn_dict.update({'lead_x_%d_%d' % (min_dst, max_dst): L1_loss[..., 0][points_mask]})  # [sum(mask), ]
        rtn_dict.update({'lead_speed_%d_%d' % (min_dst, max_dst): L1_loss[..., 1][points_mask]})  # [sum(mask), ]

        for AP_threshold in AP_thresholds_scale:
            x_threshold = (float(AP_threshold) * torch.maximum((x_distances * 0.25), torch.tensor(5.0)))
            hit_mask_x = (L1_loss[..., 0] < x_threshold) & points_mask & (pred_leads[..., -1] > 0.5)
            speed_threshold = (float(AP_threshold) * torch.tensor(10.0))
            hit_mask_speed = (L1_loss[..., 1] < speed_threshold) & points_mask & (pred_leads[..., -1] > 0.5)
            hit_mask = hit_mask_x & hit_mask_speed
            rtn_dict.update({'AP_lead_%d_%d_%s' % (min_dst, max_dst, AP_threshold): hit_mask[points_mask]})
            rtn_dict.update({'AP_lead_x_%d_%d_%s' % (min_dst, max_dst, AP_threshold): hit_mask_x[points_mask]})
            rtn_dict.update({'AP_lead_speed_%d_%d_%s' % (min_dst, max_dst, AP_threshold): hit_mask_speed[points_mask]})

    # add namespace
    if namespace is not None:
        for k in list(rtn_dict.keys()):
            rtn_dict['%s/%s' % (namespace, k)] = rtn_dict.pop(k)
    return rtn_dict


def get_val_keys_leads(namespace='lead'):
    rtn_dict = dict()

    distance_splits = ((0, 10), (10, 20), (20, 30), (30, 50), (50, 131))
    AP_thresholds = (0.5, 1., 1.5)

    for min_dst, max_dst in distance_splits:
        rtn_dict.update({'lead_x_%d_%d' % (min_dst, max_dst): []})  # [sum(mask), ]
        rtn_dict.update({'lead_speed_%d_%d' % (min_dst, max_dst): []})  # [sum(mask), ]
        for AP_threshold in AP_thresholds:
            rtn_dict.update({'AP_lead_%d_%d_%s' % (min_dst, max_dst, AP_threshold): []})
            rtn_dict.update({'AP_lead_x_%d_%d_%s' % (min_dst, max_dst, AP_threshold): []})
            rtn_dict.update({'AP_lead_speed_%d_%d_%s' % (min_dst, max_dst, AP_threshold): []})

    # add namespace
    if namespace is not None:
        for k in list(rtn_dict.keys()):
            rtn_dict['%s/%s' % (namespace, k)] = rtn_dict.pop(k)
    return rtn_dict


def load_transformed_video(path_to_segment, seq_len=5800, read_file=False):
    
    if read_file:
        path_to_video = path_to_segment
        name_mp4 = os.path.basename(path_to_segment) # xxxx.mp4
        file_name = os.path.splitext(name_mp4)[0] # xxxx
        dirname = os.path.dirname(path_to_segment)
        # dirname = os.path.dirname(dirname)
        path_to_h5 = os.path.join(dirname, file_name+'.h5')
    elif os.path.exists(os.path.join(path_to_segment, 'viz.mp4')) and os.path.exists(os.path.join(path_to_segment, 'viz.h5')):
        path_to_video = os.path.join(path_to_segment, 'viz.mp4')
        path_to_h5 = os.path.join(path_to_segment, 'viz.h5')
    else:
        raise Exception('No viz.mp4 file found in {}'.format(path_to_segment))


    segment_video = cv2.VideoCapture(path_to_video)
    segment_h5 = h5py.File(path_to_h5, 'r')

    read_rgb_frames = np.zeros((seq_len + 1, FULL_FRAME_SIZE[1], FULL_FRAME_SIZE[0], 3), dtype=np.uint8)
    stacked_frames = np.zeros((seq_len, 12, 128, 256), dtype=np.uint8)

    ret, frame2 = segment_video.read()
    if not ret:
        print('Failed to read video from {}'.format(path_to_video))
        return None, None

    rgb_frame = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
    read_rgb_frames[0] = rgb_frame

    # start iteration from 1 because we already read 1 frame before
    for t_idx in range(1, seq_len + 1):

        ret, frame2 = segment_video.read()
        if not ret:
            print('Failed to read video from {}'.format(path_to_video))
            return None, None

        rgb_frame = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

        read_rgb_frames[t_idx] = rgb_frame

    prepared_frames = transform_frames(read_rgb_frames)

    for i in range(seq_len):
        stacked_frames[i] = np.vstack(prepared_frames[i:i+2])[None].reshape(12, 128, 256)

    frame_desire = segment_h5['desire_state'][()][:,0].astype(np.int8)

    segment_video.release()
    segment_h5.close()
    return torch.from_numpy(stacked_frames).float(), frame_desire


def visualize_predictions(model, device, path_to_segment, out_folder=None):
    
    model.eval()
    with torch.no_grad():
        saved_metric_epoch = get_val_metric_keys()
        saved_leads_AP = get_val_keys_leads()
        saved_left_lane_AP = get_val_metric_keys('left_lane')
        saved_right_lane_AP = get_val_metric_keys('right_lane')
        printf(f"===>Visualizing predictions: {path_to_segment}")

        name_mp4 = os.path.basename(path_to_segment) # xxxx.mp4
        file_name = os.path.splitext(name_mp4)[0] # xxxx
        dirname = os.path.dirname(path_to_segment)
        # dirname = os.path.dirname(dirname)
        # print(dirname)
        # dirname += '/gt'
        path_to_h5 = os.path.join(dirname, file_name+'.h5')

        plan_gt_h5, lanelines_gt_h5, leads_gt_h5, lead_prob_gt_h5, desire_state_gt_h5 = load_h5_roach(path_to_h5, dif_name=True)

        recurr_input = torch.zeros(1, 512, dtype=torch.float32, device=device, requires_grad=False)
        #desire = torch.zeros(1, 8, dtype=torch.float32, device=device)
        traffic_convention = torch.zeros(1, 2, dtype=torch.float32, device=device)
        traffic_convention[:, 1] = 1

        input_frames, frame_desire = load_transformed_video(path_to_segment, read_file=True)
        input_frames = input_frames.to(device)

        viz_prev_desire = np.int8(7)

        for t_idx in range(input_frames.shape[0]): 
            desire_cur = frame_desire[t_idx]
            viz_desire_input = torch.eye(8, dtype=torch.float32)[desire_cur:desire_cur+1] if(desire_cur != viz_prev_desire) else torch.zeros(1, 8, dtype=torch.float32)
            viz_desire_input = viz_desire_input.to(device)
            viz_prev_desire = desire_cur
            inputs =  {"input_imgs":input_frames[t_idx:t_idx+1],
                            "desire": viz_desire_input,
                            "traffic_convention": traffic_convention,
                            'initial_state': recurr_input
                            }

            outs = model(**inputs)
            recurr_input = outs[:, 5960:] # refeeding the recurrent state
            #preds = outs.detach().cpu().numpy() #(1,6472)

            val_path_prediction = outs[:, :4955].clone()
            paths = val_path_prediction.reshape(-1, 5, 991)
            path_pred_prob = paths[:, :, -1]
            best_path_idx = torch.argmax(path_pred_prob,dim=1)[0]
            best_path_pred = paths[:, best_path_idx, :-1].reshape(-1, 2, 33, 15)

            meta_pred = outs[:, 5868:5948].clone()
            desire_prob = meta_pred[:, 48:].reshape(-1, 4, 8)
            desire_prob = F.softmax(desire_prob,dim=2)
            #desire_prob = desire_prob[0] #(4,8)

            leads_pred = outs[:, 5755:5857].clone()  # -- > 102 leads
            lead_porb = outs[:, 5857:5860].clone()  # -- > 3 lead_prob
            leads = leads_pred.reshape(-1, 2, 51)
            lead_pred_prob = leads[:, :, -3] #(N,2)
            best_lead_idx = torch.argmax(lead_pred_prob, dim=1)[0]
            #best_lead_idx = int(0)
            lead_porb = torch.sigmoid(lead_porb)

            best_lead = leads[:, best_lead_idx, :-3].reshape(-1, 2, 6, 4)
            lead_car_pred = torch.cat((best_lead[:,0,0,:1], best_lead[:,0,0,2:], lead_porb[:, 0:1]),dim=1)

            fixed_distances = np.array(X_IDXs)[:,np.newaxis].reshape((1, 33, 1))
            lanelines_predictions = outs[:, 4955:5483].clone()  # -- > 528 lanelines
            lanelines_prob = outs[:, 5483:5491].clone()
            # print(lanelines_prob)
            lane_lines_deflat = lanelines_predictions.reshape((-1, 2, 264))  # (N, 2, 264)
            lane_lines_means = lane_lines_deflat[:, 0, :]  # (N, 264)
            lane_lines_means = lane_lines_means.reshape(-1, 4, 33, 2)  # (N, 4, 33, 2)

            inner_left_lane_pred = lane_lines_means[:, 1, :, :]  # (N, 33, 2)
            inner_right_lane_pred = lane_lines_means[:, 2, :, :]  # (N, 33, 2)
            calib_pts_ill_pred = np.concatenate((fixed_distances, np.array(inner_left_lane_pred.cpu())), axis=2) # (N, 33, 3)
            calib_pts_irl_pred = np.concatenate((fixed_distances, np.array(inner_right_lane_pred.cpu())), axis=2) # (N, 33, 3)
            calib_pts_ill_pred = torch.from_numpy(calib_pts_ill_pred).to(device)
            calib_pts_irl_pred = torch.from_numpy(calib_pts_irl_pred).to(device)

            plan_gt = plan_gt_h5[t_idx:t_idx+1]
            leads_gt = leads_gt_h5[t_idx:t_idx+1]
            lanelines_gt = lanelines_gt_h5[t_idx:t_idx+1]
            inner_left_lane_gt = lanelines_gt[:, 0, :, :]  # (N, 33, 2)
            inner_right_lane_gt = lanelines_gt[:, 1, :, :]  # (N, 33, 2)
            calib_pts_ill_gt = np.concatenate((fixed_distances, inner_left_lane_gt), axis=2) # (N, 33, 3)
            calib_pts_irl_gt = np.concatenate((fixed_distances, inner_right_lane_gt), axis=2) # (N, 33, 3)

            plan_gt = torch.from_numpy(plan_gt).to(device)
            leads_gt = torch.from_numpy(leads_gt).to(device)
            calib_pts_ill_gt = torch.from_numpy(calib_pts_ill_gt).to(device)
            calib_pts_irl_gt = torch.from_numpy(calib_pts_irl_gt).to(device)

            metrics = get_val_metric(best_path_pred[:,0,:,:3], plan_gt[...,:3])
            # print(metrics)
            metrics_leads = get_val_leads(lead_car_pred, leads_gt)
            metrics_left_lane = get_val_metric(calib_pts_ill_pred, calib_pts_ill_gt, 'left_lane')
            # print(metrics_left_lane)
            metrics_right_lane = get_val_metric(calib_pts_irl_pred, calib_pts_irl_gt, 'right_lane')
            
            for k, v in metrics.items():
                saved_metric_epoch[k].append(v.float().mean().item())
            for k, v in metrics_leads.items():
                saved_leads_AP[k].append(v.float().mean().item())
            for k, v in metrics_left_lane.items():
                saved_left_lane_AP[k].append(v.float().mean().item())
            for k, v in metrics_right_lane.items():
                saved_right_lane_AP[k].append(v.float().mean().item())

        metric_single = torch.zeros((len(saved_metric_epoch), ), dtype=torch.float32, device='cuda')
        counter_single = torch.zeros((len(saved_metric_epoch), ), dtype=torch.int32, device='cuda')

        for i, k in enumerate(sorted(saved_metric_epoch.keys())):
            metric_single[i] = np.mean(saved_metric_epoch[k])
            counter_single[i] = len(saved_metric_epoch[k])
            print(f'{k}:{metric_single[i]}')
            print(f'counter_single[{i}]:{counter_single[i]}')
            writer.add_scalar(k, metric_single[i], int(file_name))

        metric_lead_single = torch.zeros((len(saved_leads_AP), ), dtype=torch.float32, device='cuda')
        counter_lead_single = torch.zeros((len(saved_leads_AP), ), dtype=torch.int32, device='cuda')

        for i, k in enumerate(sorted(saved_leads_AP.keys())):
            metric_lead_single[i] = np.mean(saved_leads_AP[k])
            counter_lead_single[i] = len(saved_leads_AP[k])
            print(f'{k}:{metric_lead_single[i]}')
            print(f'counter_lead_single[{i}]:{counter_lead_single[i]}')
            writer.add_scalar(k, metric_lead_single[i], int(file_name))

        metric_laneline_single = torch.zeros((len(saved_left_lane_AP), ), dtype=torch.float32, device='cuda')
        counter_laneline_single = torch.zeros((len(saved_left_lane_AP), ), dtype=torch.int32, device='cuda')

        for i, (k1, k2) in enumerate(zip(sorted(saved_left_lane_AP.keys()), sorted(saved_right_lane_AP.keys()))):
            metric_laneline_single[i] = (np.mean(saved_left_lane_AP[k1]) + np.mean(saved_right_lane_AP[k2])) / 2
            counter_laneline_single[i] = len(saved_left_lane_AP[k1])
            print(f'{k1}:{metric_laneline_single[i]}')
            print(f'counter_laneline_single[{i}]:{counter_laneline_single[i]}')
            writer.add_scalar(k1, metric_laneline_single[i], int(file_name))
        
        return metric_single, counter_single, metric_lead_single, counter_lead_single, metric_laneline_single, counter_laneline_single
        

cuda = torch.cuda.is_available()
if cuda:
    #import torch.backends.cudnn as cudnn
    #cudnn.benchmark = True
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("=>Using '{}' for computation.".format(device))

mp4files_path = "/home/t2-503-4090/QianXi/training_data/openpilot_camera_laneline"

path_list = []
for path in os.listdir(mp4files_path):
    if path.endswith('mp4'):
        path_list.append(os.path.join(mp4files_path, path))

rng = np.random.default_rng(42)
rng.shuffle(path_list)
path_list = path_list[198:] #val split

pth_path = "/home/t2-503-4090/QianXi/Openpilot_BalancedRegression_Adapter/train/nets/703_3phase_comma2k19/0703_laneline.pth" 
path_to_supercombo = '../common/models/supercombo.onnx'
pathplan_layer_names = [
                         "Conv_785", "Conv_797", "Conv_809", "Conv_832", 
                         "Conv_844", "Conv_856", "Conv_868", "Conv_891",
                        ] 
comma_model = load_trainable_model(path_to_supercombo, trainable_layers=pathplan_layer_names)
comma_model.load_state_dict(torch.load(pth_path))
comma_model.eval()
comma_model = comma_model.to(device)
log_path = "./log_tensorboard/laneline_0703_3phase_comma2k19"
os.makedirs(log_path, exist_ok=True)

metric_list = []
counter_list = []
met_lead_list = []
cont_lead_list = []
met_laneline_list = []
cont_laneline_list = []

writer = SummaryWriter(log_dir=log_path)
for path in path_list:
    metric_single, counter_single, metric_lead_single, counter_lead_single, metric_laneline_single, counter_laneline_single = visualize_predictions(comma_model, device, path)
    metric_list.append(metric_single.cpu().detach().numpy())
    counter_list.append(counter_single.cpu().detach().numpy())
    met_lead_list.append(metric_lead_single.cpu().detach().numpy())
    cont_lead_list.append(counter_lead_single.cpu().detach().numpy())
    met_laneline_list.append(metric_laneline_single.cpu().detach().numpy())
    cont_laneline_list.append(counter_laneline_single.cpu().detach().numpy())
writer.close()

metric_list = np.array(metric_list)
counter_list = np.array(counter_list)
met_lead_list = np.array(met_lead_list)
cont_lead_list = np.array(cont_lead_list)
met_laneline_list = np.array(met_laneline_list)
cont_laneline_list = np.array(cont_laneline_list)

metric_list[np.isnan(metric_list)] = 0
counter_list[np.isnan(counter_list)] = 0
met_lead_list[np.isnan(met_lead_list)] = 0
cont_lead_list[np.isnan(cont_lead_list)] = 0
met_laneline_list[np.isnan(met_laneline_list)] = 0
cont_laneline_list[np.isnan(cont_laneline_list)] = 0

_metric_epoch = get_val_metric_keys()
_leads_AP = get_val_keys_leads()
_lanelines_AP = get_val_metric_keys('laneline')
list_total_AP = [1,4,7,10,13]
total_cnt = 0
total_pos = 0.0

with open(log_path+"/mean_AP.txt",'w') as ftxt :
    for i, k in enumerate(sorted(_metric_epoch.keys())):
        metric_mean = (metric_list[:,i]*counter_list[:,i]).sum(0) / counter_list[:,i].sum(0)
        ftxt.write(f"{k} : {metric_mean}\n")
    ftxt.write(f"-------------------------------------------------\n")
    
    for i, k in enumerate(sorted(_leads_AP.keys())):
        metric_lead_mean = (met_lead_list[:,i]*cont_lead_list[:,i]).sum(0) / cont_lead_list[:,i].sum(0)
        ftxt.write(f"{k} : {metric_lead_mean}\n")
        if i in list_total_AP:
            total_pos += (met_lead_list[:,i]*cont_lead_list[:,i]).sum(0)
            total_cnt += cont_lead_list[:,i].sum(0)

    ftxt.write(f"total_AP : {total_pos / total_cnt}\n")
    ftxt.write(f"-------------------------------------------------\n")
    
    total_cnt = 0
    total_pos = 0.0
    for i, k in enumerate(sorted(_lanelines_AP.keys())):
        metric_laneline_mean = (met_laneline_list[:,i]*cont_laneline_list[:,i]).sum(0) / cont_laneline_list[:,i].sum(0)
        ftxt.write(f"{k} : {metric_laneline_mean}\n")
    #     if i in list_total_AP:
    #         total_pos += (met_laneline_list[:,i]*cont_laneline_list[:,i]).sum(0)
    #         total_cnt += cont_laneline_list[:,i].sum(0)
    
    # ftxt.write(f"total_AP : {total_pos / total_cnt}\n")