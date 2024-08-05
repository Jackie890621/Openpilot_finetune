import sys
######################################################################################
sys.path.append("/home/t2-503-4090/QianXi/Openpilot_BalancedRegression_Adapter")
######################################################################################
from utils import Calibration, draw_path, printf, extract_preds, load_h5_roach, extract_gt_roach
import os
from model import load_trainable_model
import gc
import numpy as np
import torch
import random
from dataloader import load_transformed_video
import cv2
import h5py
import matplotlib.pyplot as plt

PLOT_IMG_WIDTH, PLOT_IMG_HEIGHT = 1440, 960

def visualization(lanelines, roadedges, calib_path, leadcar, desire, im_rgb,frame_num):

    rpy_calib = [0, 0, 0]
    calibration_pred = Calibration(rpy_calib, plot_img_width=PLOT_IMG_WIDTH, plot_img_height=PLOT_IMG_HEIGHT)
    laneline_colors = [(255, 0, 0), (0, 255, 0), (255, 0, 255), (0, 255, 255)]
    vis_image = draw_path(lanelines, roadedges, calib_path[0, :, :3], leadcar, desire, im_rgb, calibration_pred, laneline_colors,frame_n=frame_num)

    return vis_image

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def visualize_predictions(model, device, path_to_segment, out_folder):

    model.eval()
    with torch.no_grad():
        printf(f"===>Visualizing predictions: {path_to_segment}")
        name_mp4 = os.path.basename(path_to_segment) # xxxx.mp4
        file_name = os.path.splitext(name_mp4)[0] # xxxx
        dirname = os.path.dirname(path_to_segment)
        path_to_load = os.path.join(dirname, file_name+'.h5')
        #name4load = "gt_roach_" + file_name + ".h5" # gt_roach_xxxx.h5
        #path_to_load = os.path.join(out_folder, name4load)
        f_lead = open(os.path.join(out_folder,file_name+".txt"),"w")

        recurr_input = torch.zeros(1, 512, dtype=torch.float32, device=device, requires_grad=False)
        #desire = torch.zeros(1, 8, dtype=torch.float32, device=device)
        traffic_convention = torch.zeros(1, 2, dtype=torch.float32, device=device)
        traffic_convention[:, 1] = 1
        viz_prev_desire = np.int8(7)

        input_frames, rgb_frames, frame_desire = load_transformed_video(path_to_segment, plot_img_width=PLOT_IMG_WIDTH, plot_img_height=PLOT_IMG_HEIGHT, read_file=True)
        input_frames = input_frames.to(device)

        video_array_pred = np.zeros((rgb_frames.shape[0],rgb_frames.shape[1],rgb_frames.shape[2], rgb_frames.shape[3]), dtype=np.uint8)
        video_array_gt = np.zeros((rgb_frames.shape[0],rgb_frames.shape[1],rgb_frames.shape[2], rgb_frames.shape[3]), dtype=np.uint8)

        plan_gt_h5, laneline_gt_h5, leads_gt_h5, lead_prob_gt_h5, desire_state_gt_h5 = load_h5_roach(path_to_load, dif_name=True)

        print("before loop")
        for t_idx in range(rgb_frames.shape[0]): 
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
            preds = outs.detach().cpu().numpy() #(1,6472)

            lanelines, road_edges, best_path, best_lead, desire_state = extract_preds(preds)[0]
            im_rgb = rgb_frames[t_idx].copy()
            #vis_image = visualization(lanelines,road_edges,best_path, best_lead, desire_state, im_rgb,frame_num=t_idx)
            vis_image = visualization(lanelines,None,best_path, best_lead, desire_state, im_rgb,frame_num=t_idx)
            video_array_pred[t_idx:t_idx+1,:,:,:] = vis_image
            
            f_lead.write(f"---------------------{t_idx}--------------------------\n")
            ######################################################################################
            leads_pred = preds[:, 5755:5857]  # -- > 102 leads
            lead_porb = preds[:, 5857:5860]  # -- > 3 lead_prob
            ######################################################################################
            leads = leads_pred.reshape(-1, 2, 51)
            lead_1 = leads[:, 0, :-3].reshape(-1, 2, 6, 4)
            lead_2 = leads[:, 1, :-3].reshape(-1, 2, 6, 4)
            lead_1_Std = np.exp(lead_1[0,1,0,:])
            lead_2_Std = np.exp(lead_2[0,1,0,:])
            lead_pred_prob = leads[:, :, -3:-1] #(N,2,2)
            lead_pred_prob = sigmoid(lead_pred_prob)
            lead_porb = sigmoid(lead_porb)
            f_lead.write(f"lead_GT : x({leads_gt_h5[t_idx,0,0]}), speed({leads_gt_h5[t_idx,0,1]}), accel({leads_gt_h5[t_idx,0,2]})\n")
            f_lead.write(f"lead_1 : x({lead_1[0,0,0,0]}), speed({lead_1[0,0,0,2]}), accel({lead_1[0,0,0,3]})\n")
            f_lead.write(f"lead_1Std : x({lead_1_Std[0]}), speed({lead_1_Std[2]}), accel({lead_1_Std[3]})\n")
            f_lead.write(f"lead_2 : x({lead_2[0,0,0,0]}), speed({lead_2[0,0,0,2]}), accel({lead_2[0,0,0,3]})\n")
            f_lead.write(f"lead_2Std : x({lead_2_Std[0]}), speed({lead_2_Std[2]}), accel({lead_2_Std[3]})\n")
            f_lead.write(f"choice : 0-0({lead_pred_prob[0,0,0]}), 0-1({lead_pred_prob[0,0,1]})\n")
            f_lead.write(f"choice : 1-0({lead_pred_prob[0,1,0]}), 1-1({lead_pred_prob[0,1,1]})\n")
            f_lead.write(f"prob: {lead_porb[0,0]}, {lead_porb[0,1]}, {lead_porb[0,2]}\n")

        # for k in range(plan_gt_h5.shape[0]):
        #     laneline_h5, roadedg_h5, path_h5, leadcar_h5, desire_h5 = extract_gt_roach(laneline_gt_h5[k:k+1], plan_gt_h5[k:k+1], leads_gt_h5[k:k+1], lead_prob_gt_h5[k:k+1], desire_state_gt_h5[k:k+1])[0]
        #     image_rgb_gt = rgb_frames[k]

        #     image_gt = visualization(laneline_h5, None, path_h5, leadcar_h5, desire_h5, image_rgb_gt,frame_num=k)
        #     video_array_gt[k:k+1,:,:,:] = image_gt

        video_array_pred = video_array_pred[..., ::-1]  #RGB 2 BGR
        # video_array_gt = video_array_gt[..., ::-1]
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # gt_video_name = os.path.join(out_folder, ('laneline_gt_' + file_name + '.mp4'))
        pred_video_name = os.path.join(out_folder, ('laneline_pred_' + file_name + '.mp4'))
        # out_gt_video = cv2.VideoWriter(gt_video_name, fourcc, 20.0, (PLOT_IMG_WIDTH, PLOT_IMG_HEIGHT))
        out_pred_video = cv2.VideoWriter(pred_video_name, fourcc, 20.0, (PLOT_IMG_WIDTH, PLOT_IMG_HEIGHT))

        for video_idx in range(rgb_frames.shape[0]):
            # out_gt_video.write(video_array_gt[video_idx])
            out_pred_video.write(video_array_pred[video_idx])

        # out_gt_video.release()
        out_pred_video.release()

        del video_array_pred
        del video_array_gt

        gc.collect()
        
        f_lead.close()


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


cuda = torch.cuda.is_available()
if cuda:
    #import torch.backends.cudnn as cudnn
    #cudnn.benchmark = True
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
# device = torch.device("cpu")
print("=>Using '{}' for computation.".format(device))

path_folder = "./vis_result_test/laneline_supercombo"
os.makedirs(path_folder, exist_ok=True)
''' #################create mp4 split##################
h5files = []
with open('../cache/roach_h5.txt', 'r') as f:
    h5files = f.read().splitlines()
'''
mp4files = []
with open('../cache/viz_videos.txt', 'r') as f:
    mp4files = f.read().splitlines()

mp4files = ['/home/t2-503-4090/QianXi/training_data/openpilot_camera_laneline/0239.mp4']

num_list = list(range(len(mp4files)))
#rng = np.random.default_rng(42)
#rng.shuffle(num_list)
#num_list = num_list[:199]
#num_list = num_list[199:]


pth_path = "/home/t2-503-4090/QianXi/Openpilot_BalancedRegression_Adapter/train/nets/619_3phase/0619_laneline.pth" 
pathplan_layer_names = [
                        "Conv_785", "Conv_797", "Conv_809", "Conv_832", 
                        "Conv_844", "Conv_856", "Conv_868", "Conv_891",
                        ]
path_to_supercombo = '../common/models/supercombo.onnx'
comma_model = load_trainable_model(path_to_supercombo, trainable_layers=pathplan_layer_names)
# comma_model.load_state_dict(torch.load(pth_path))
comma_model.eval()

comma_model = comma_model.to(device)
print("w/o minus radar to camera 1.52")

'''##############lead distribution#################
max_target = 131
distribution = np.zeros(max_target)
tol_data = 0
'''

'''##############count desire state#################
tol_none = 0
tol_turnL = 0
tol_turnR = 0
tol_changeL = 0
tol_changeR = 0
'''
for num in num_list:
    visualize_predictions(comma_model, device, mp4files[num], path_folder)

''' ##############lead distribution#################
    path_to_segment = mp4files[num]
    name_mp4 = os.path.basename(path_to_segment) # xxxx.mp4
    file_name = os.path.splitext(name_mp4)[0] # xxxx
    dirname = os.path.dirname(path_to_segment)
    path_to_load = os.path.join(dirname, file_name+'.h5')
    plan_gt_h5, laneline_gt_h5, leads_gt_h5, lead_prob_gt_h5, desire_state_gt_h5 = load_h5_roach(path_to_load, dif_name=True)
    for k in range(leads_gt_h5.shape[0]):
        lead_dis = leads_gt_h5[k,0,0]
        if int(lead_dis) < max_target:
            distribution[int(lead_dis)] += 1
            tol_data += 1
plt.rcParams.update({'font.size': 22})
plt.bar(range(131),distribution/tol_data)
plt.title("Label distribution of lead distance")
plt.ylabel("(%)")
plt.xlabel("Distance of lead (meter)")
plt.show()
'''

''' #################create mp4 split##################
    path_to_segment = h5files[num]
    name_h5 = os.path.basename(path_to_segment)
    file_name = os.path.splitext(name_h5)[0] # xxxx
    segment_data = h5py.File(path_to_segment, 'r')

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_name = os.path.join('/media/NVMe_SSD_980_PRO/Meng-Tai/roach-dataset/cc_roach_split', (file_name + '.mp4'))
    out_video = cv2.VideoWriter(video_name, fourcc, 20.0, (900, 256))
    for mp4idx in range(5801):
        a_group_key = 'step_' + str(mp4idx)
        frame2 = segment_data[a_group_key]['obs']['central_rgb']['data'][()]
        frame2 = frame2[..., ::-1]
        out_video.write(frame2)
    out_video.release()
    segment_data.close()
'''

''' ##############count desire state#################
    sum_none = 0
    sum_turnL = 0
    sum_turnR = 0
    sum_changeL = 0
    sum_changeR = 0
    path_to_segment = h5files[num]
    name_h5 = os.path.basename(path_to_segment)
    name4load = "gt_roach_" + name_h5 # gt_roach_xxxx.h5
    path_to_load = os.path.join(path_folder, name4load)
    plan_gt_h5, laneline_gt_h5, leads_gt_h5, lead_prob_gt_h5, desire_state_gt_h5 = load_h5_roach(path_to_load, dif_name=True)
    for k in range(desire_state_gt_h5.shape[0]):
        des_st = desire_state_gt_h5[k,0]
        if des_st == 0:
            sum_none += 1
        elif des_st == 1:
            sum_turnL += 1
        elif des_st == 2 :
            sum_turnR += 1
        elif des_st == 3 :
            sum_changeL += 1
        elif des_st == 4 :
            sum_changeR += 1
    print(f'{name_h5} - none : {sum_none}, turnL : {sum_turnL}, turnR : {sum_turnR}, changeL : {sum_changeL}, changeR : {sum_changeR}')
    tol_none += sum_none
    tol_turnL += sum_turnL
    tol_turnR += sum_turnR
    tol_changeL += sum_changeL
    tol_changeR += sum_changeR
print(f'Toltal - none : {tol_none}, turnL : {tol_turnL}, turnR : {tol_turnR}, changeL : {tol_changeL}, changeR : {tol_changeR}')
#Toltal - none : 1088298, turnL : 58170, turnR : 66619, changeL : 8171, changeR : 8342
#0.885   0.047   0.054   0.0066  0.0068
'''
