import numpy as np
import math
import h5py
from scipy.interpolate import interp1d
import cv2
import os
import sys

def carla_get_matrix(location, rotation):
    yaw = rotation[2];
    cy = math.cos(math.radians(yaw));
    sy = math.sin(math.radians(yaw));

    roll = rotation[0];
    cr = math.cos(math.radians(roll));
    sr = math.sin(math.radians(roll));

    pitch = rotation[1];
    cp = math.cos(math.radians(pitch));
    sp = math.sin(math.radians(pitch));

    transform = np.array([
        [cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr, location[0]],
        [cp * sy, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr, location[1]],
        [sp, -cp * sr, cp * cr, location[2]],
        [0.0, 0.0, 0.0, 1.0]])
    return transform

def get_rotation_rate(rotation):
    t_anchors = np.array(
        (0.        ,  0.00976562,  0.0390625 ,  0.08789062,  0.15625   ,
         0.24414062,  0.3515625 ,  0.47851562,  0.625     ,  0.79101562,
         0.9765625 ,  1.18164062,  1.40625   ,  1.65039062,  1.9140625 ,
         2.19726562,  2.5       ,  2.82226562,  3.1640625 ,  3.52539062,
         3.90625   ,  4.30664062,  4.7265625 ,  5.16601562,  5.625     ,
         6.10351562,  6.6015625 ,  7.11914062,  7.65625   ,  8.21289062,
         8.7890625 ,  9.38476562, 9.9)
    )
    rotation_rate = np.zeros((199,3))
    for i in range(len(rotation)-1):
        rotation_rate[i] = (rotation[i+1] - rotation[i]) * 20.0
    time_rotation_rate = np.linspace(0, 9.9, num=199)
    fs = [interp1d(time_rotation_rate, rotation_rate[:, j]) for j in range(3)]
    interp_rr = [fs[j](t_anchors)[:, None] for j in range(3)]
    interp_rr = np.concatenate(interp_rr, axis=1)

    return interp_rr

def create_gt_roach(path_of_h5, path_of_output):
    t_anchors = np.array(
        (0.        ,  0.00976562,  0.0390625 ,  0.08789062,  0.15625   ,
         0.24414062,  0.3515625 ,  0.47851562,  0.625     ,  0.79101562,
         0.9765625 ,  1.18164062,  1.40625   ,  1.65039062,  1.9140625 ,
         2.19726562,  2.5       ,  2.82226562,  3.1640625 ,  3.52539062,
         3.90625   ,  4.30664062,  4.7265625 ,  5.16601562,  5.625     ,
         6.10351562,  6.6015625 ,  7.11914062,  7.65625   ,  8.21289062,
         8.7890625 ,  9.38476562, 9.9)
    )
    time_i = np.linspace(0, 9.95, num=200)
    segment_data = h5py.File(path_of_h5, 'r')
    n_tol = 5800 #len(keys)=6000 , (6000-200)
    #frame2 = segment_data['step_0']['obs']['central_rgb']['data'][()]

    plan_10s = np.zeros((200,12))

    for i in range(200):
        a_group_key = 'step_' + str(i)
        frame_loc = segment_data[a_group_key]['obs']['central_rgb']['location'][()]
        frame_loc[2] = -frame_loc[2] # 'z' up to down
        frame_rot = segment_data[a_group_key]['obs']['central_rgb']['rotation'][()]
        frame_rot[2] = -frame_rot[2]
        frame_vel_f = segment_data[a_group_key]['obs']['speed']['forward_speed'][()]
        frame_vel_rd = segment_data[a_group_key]['obs']['speed']['right_up_speed'][()]
        frame_vel_rd[1] = -frame_vel_rd[1] # 'z' up to down
        frame_accel = segment_data[a_group_key]['obs']['speed']['forward_right_up_accel'][()]
        frame_accel[2] = -frame_accel[2] # 'z' up to down

        plan_10s[i] = np.hstack((frame_loc, frame_vel_f, frame_vel_rd, frame_accel, frame_rot))

    gt_plan_seq = np.zeros((n_tol, 33, 15), dtype=np.float32)
    gt_lanelines_lr_seq = np.zeros((n_tol, 2, 33, 2), dtype=np.float32)
    gt_leads_seq = np.zeros((n_tol, 1, 3), dtype=np.float32)
    gt_leads_prob = np.zeros((n_tol, 3), dtype=np.float32)
    gt_desire_seq = np.zeros((n_tol, 4), dtype=np.float32)

    for n_idx in range(1,n_tol+1):

        curr_idx = n_idx
        a_group_key = 'step_' + str(curr_idx)
        group_key_10s = 'step_' + str(curr_idx+199)

        #frame2 = segment_data[a_group_key]['obs']['central_rgb']['data'][()]
        #yuv_frame2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2YUV_I420)
        #frame_seq[t_idx] = frame2
        plan_10s[:-1] = plan_10s[1:]
        frame_0_loc = plan_10s[0,:3]
        frame_0_rot = plan_10s[0,9:12]
        trans_matrix = carla_get_matrix(frame_0_loc, frame_0_rot)
        
        if group_key_10s not in segment_data:
            break
        
        frame_loc = segment_data[group_key_10s]['obs']['central_rgb']['location'][()]
        frame_loc[2] = -frame_loc[2] # 'z' up to down
        frame_rot = segment_data[group_key_10s]['obs']['central_rgb']['rotation'][()]
        frame_rot[2] = -frame_rot[2]
        frame_vel_f = segment_data[group_key_10s]['obs']['speed']['forward_speed'][()]
        frame_vel_rd = segment_data[group_key_10s]['obs']['speed']['right_up_speed'][()]
        frame_vel_rd[1] = -frame_vel_rd[1] # 'z' up to down
        frame_accel = segment_data[group_key_10s]['obs']['speed']['forward_right_up_accel'][()]
        frame_accel[2] = -frame_accel[2] # 'z' up to down

        plan_10s[-1] = np.hstack((frame_loc, frame_vel_f, frame_vel_rd, frame_accel, frame_rot))

        relative_loc = plan_10s[:,:3] - frame_0_loc
        relative_loc = np.hstack((relative_loc, np.zeros((200, 1), dtype=np.float32)))
        relative_loc = relative_loc.T

        transformed_coords = np.dot(trans_matrix, relative_loc)

        final_loc = transformed_coords[:3, :].T

        relative_rot = plan_10s[:,9:12] - frame_0_rot
        relative_rot = np.radians(relative_rot)
        interp_to_op = np.hstack((final_loc, plan_10s[:,3:9], relative_rot))

        fs = [interp1d(time_i, interp_to_op[:, j]) for j in range(12)]
        interp_positions = [fs[j](t_anchors)[:, None] for j in range(12)]
        interp_positions = np.concatenate(interp_positions, axis=1)

        rotation_rate = get_rotation_rate(relative_rot)
        gt_plan_seq[n_idx-1] = np.hstack((interp_positions, rotation_rate))

        _,left_line,right_line,_ = segment_data[a_group_key]['obs']['central_rgb']['lanelines'][()]
        left_line[:,2] = -left_line[:,2] # 'z' up to down
        right_line[:,2] = -right_line[:,2] # 'z' up to down
        relative_left = (left_line - frame_0_loc)
        relative_right = (right_line - frame_0_loc)
        relative_left = np.hstack((relative_left, np.zeros((33, 1), dtype=np.float32)))
        relative_right = np.hstack((relative_right, np.zeros((33, 1), dtype=np.float32)))
        transformed_left = np.dot(trans_matrix, relative_left.T)
        transformed_right = np.dot(trans_matrix, relative_right.T)
        left_line = transformed_left[1:3, :].T
        right_line = transformed_right[1:3, :].T

        left_line = left_line[np.newaxis, :]
        right_line = right_line[np.newaxis, :]
        gt_lanelines_lr_seq[n_idx-1] = np.concatenate((left_line,right_line), axis=0)

        lead_dis = segment_data[a_group_key]['obs']['central_rgb']['lead_distance'][()]
        lead_speed = segment_data[a_group_key]['obs']['central_rgb']['lead_speed'][()]
        lead_accel = segment_data[a_group_key]['obs']['central_rgb']['lead_accel'][()]
        if lead_dis >0 :
            lead_dis = lead_dis + 1.52
            gt_leads_prob[n_idx-1] = np.array(([1.,1.,1.]),dtype=np.float32)
        else :
            lead_dis = 255.0
            lead_speed = 10.0
            lead_accel = 0.0
            gt_leads_prob[n_idx-1] = np.array(([0.,0.,0.]),dtype=np.float32)
        gt_leads_seq[n_idx-1] = np.hstack((lead_dis, lead_speed, lead_accel))
        for desire_idx in range(4):
            desire_group_key = 'step_' + str(curr_idx + desire_idx*40) # 0,2,4,6s
            desire_num = segment_data[desire_group_key]['obs']['gnss']['command'][()][0]
            if desire_num >= 5:
                desire_num -= 2
            elif desire_num == 1 or desire_num == 2:
                desire_num = desire_num
            else :
                desire_num = 0
            gt_desire_seq[n_idx-1, desire_idx] = desire_num

    segment_data.close()
    with h5py.File(path_of_output, 'w') as h5file_object:
        h5file_object.create_dataset("plans", data=np.stack(gt_plan_seq))
        #h5file_object.create_dataset("plans_prob", data=np.stack(plans_prob)) 
        h5file_object.create_dataset("lanelines", data=np.stack(gt_lanelines_lr_seq))
        #h5file_object.create_dataset("laneline_probs", data=np.stack(laneline_probs))
        #h5file_object.create_dataset("road_edges", data=np.stack(road_edges))
        #h5file_object.create_dataset("road_edge_stds", data=np.stack(road_edge_stds))
        h5file_object.create_dataset("leads", data=np.stack(gt_leads_seq))
        h5file_object.create_dataset("lead_prob", data=np.stack(gt_leads_prob))
        h5file_object.create_dataset("desire_state", data=np.stack(gt_desire_seq))
        
        
def create_mp4(path_of_h5, path_of_output):
    segment_data = h5py.File(path_of_h5, 'r')

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(path_of_output, fourcc, 20.0, (1440, 960))
    flag = False
    for mp4idx in range(5801):  #len(keys)=2000 , 2000-200+1 = 1801
        a_group_key = 'step_' + str(mp4idx)
        if a_group_key not in segment_data:
            flag = True
            break
        frame2 = segment_data[a_group_key]['obs']['central_rgb']['data'][()]
        frame2 = frame2[..., ::-1]
        out_video.write(frame2)

    out_video.release()
    segment_data.close()
    if flag:
        os.remove(path_of_output)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python create_dataset.py <dataset_directory")
    else:
        dataset_dir = sys.argv[1]
        
        gt_dir = os.path.join(dataset_dir, "gt")
        video_dir = os.path.join(dataset_dir, "video")
        os.makedirs(gt_dir, exist_ok=True)
        os.makedirs(video_dir, exist_ok=True)

        for file_name in sorted(os.listdir(dataset_dir)):
            file_path = os.path.join(dataset_dir, file_name)

            print("creating", file_name)
            create_gt_roach(file_path, gt_dir + '/' + file_name[:4] + '.h5')
            create_mp4(file_path, video_dir  + '/' + file_name[:4] + '.mp4')
            print(file_name, "success!")
        
