from common.transformations.camera import normalize, get_view_frame_from_calib_frame
from common.transformations.model import medmodel_intrinsics
import common.transformations.orientation as orient
import numpy as np
import math
import os
import cv2
import glob
import h5py
import argparse
from scipy.interpolate import interp1d

PATH_TO_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')
FULL_FRAME_SIZE = (1440, 960)
W, H = FULL_FRAME_SIZE[0], FULL_FRAME_SIZE[1]
FOV = 40.0
eon_focal_length = FOCAL = W / (2 * math.tan(FOV * math.pi / 360))

# aka 'K' aka camera_frame_from_view_frame
eon_intrinsics = np.array([
    [FOCAL,   0.,   W/2.],
    [0.,  FOCAL,  H/2.],
    [0.,    0.,     1.]])

X_IDXs = [
        0.,   0.1875,   0.75,   1.6875,   3.,   4.6875,
        6.75,   9.1875,  12.,  15.1875,  18.75,  22.6875,
        27.,  31.6875,  36.75,  42.1875,  48.,  54.1875,
        60.75,  67.6875,  75.,  82.6875,  90.75,  99.1875,
        108., 117.1875, 126.75, 136.6875, 147., 157.6875,
        168.75, 180.1875, 192.]

RADAR_TO_CAMERA = 1.52

def printf(*args, **kwargs):
    print(flush=True, *args, **kwargs)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    exp_x = np.exp(x)
    f_x = exp_x / np.sum(exp_x)
    return f_x

def dir_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path")

def get_segment_dirs(base_dir, video_names=['video.hevc', 'fcamera.hevc']):
    '''Get paths to all segments.'''

    paths_to_videos = []
    for video_name in video_names:
        paths = sorted(glob.glob(base_dir + f'/**/{video_name}', recursive=True))
        paths_to_videos += paths
    return sorted(list(set([os.path.dirname(f) for f in paths_to_videos])))

def load_h5(seg_path):

    file_path = os.path.join(seg_path, 'gt_distill.h5')
    print(os.path.exists(file_path))
    file = h5py.File(file_path,'r')

    plan = file['plans'][...]
    plan_prob = file['plans_prob'][...]
    lanelines = file['lanelines'][...]
    lanelines_prob = file['laneline_probs'][...]
    road_edg = file['road_edges'][...]
    road_edg_std = file['road_edge_stds'][...]

    file.close()
    
    return plan, plan_prob, lanelines, lanelines_prob, road_edg, road_edg_std

def extract_gt(plan_gt, plan_prob_gt, lanelines_gt, lanelines_prob_gt, road_edg_gt, road_edg_std_gt, best_plan_only=True):
    # plan
    plans = plan_gt # (N, 5, 2, 33, 15)
    best_plan_idx = np.argmax(plan_prob_gt, axis=1)[0]  # (N,)
    best_plan = plans[:, best_plan_idx, ...]  # (N, 2, 33, 15)

    # lane lines
    outer_left_lane = lanelines_gt[:, 0, :, :]  # (N, 33, 2)
    inner_left_lane = lanelines_gt[:, 1, :, :]  # (N, 33, 2)
    inner_right_lane = lanelines_gt[:, 2, :, :]  # (N, 33, 2)
    outer_right_lane = lanelines_gt[:, 3, :, :]  # (N, 33, 2)

    # lane lines probs
    outer_left_prob = lanelines_prob_gt[:, 0]  # (N,)
    inner_left_prob = lanelines_prob_gt[:, 1]  # (N,)
    inner_right_prob = lanelines_prob_gt[:, 2]  # (N,)
    outer_right_prob = lanelines_prob_gt[:, 3]  # (N,)

    # road edges
    left_edge = road_edg_gt[:, 0, :, :]  # (N, 33, 2)
    right_edge = road_edg_gt[:, 1, :, :]
    left_edge_std = road_edg_std_gt[:, 0, :, :]  # (N, 33, 2)
    right_edge_std = road_edg_std_gt[:, 1, :, :]

    batch_size = best_plan.shape[0]
    
    result_batch = []
    
    # each element of the output list is a tuple of predictions at respective sample_idx
    for i in range(batch_size):
        lanelines = [outer_left_lane[i], inner_left_lane[i], inner_right_lane[i], outer_right_lane[i]]
        lanelines_probs = [outer_left_prob[i], inner_left_prob[i], inner_right_prob[i], outer_right_prob[i]]
        road_edges = [left_edge[i], right_edge[i]]
        road_edges_probs = [left_edge_std[i], right_edge_std[i]]

        if best_plan_only:
            plan = best_plan[i]

        result_batch.append(((lanelines, lanelines_probs), (road_edges, road_edges_probs), plan))

    return result_batch

def extract_preds(outputs, best_plan_only=True):
    # N is batch_size
    ######################################################################################
    plan_start_idx = 0
    plan_end_idx = 4955

    lanes_start_idx = plan_end_idx
    lanes_end_idx = lanes_start_idx + 528

    lane_lines_prob_start_idx = lanes_end_idx
    lane_lines_prob_end_idx = lane_lines_prob_start_idx + 8

    road_start_idx = lane_lines_prob_end_idx
    road_end_idx = road_start_idx + 264
    ######################################################################################

    # plan
    plan = outputs[:, plan_start_idx:plan_end_idx]  # (N, 4955)
    plans = plan.reshape((-1, 5, 991))  # (N, 5, 991)
    plan_probs = plans[:, :, -1]  # (N, 5)
    plans = plans[:, :, :-1].reshape(-1, 5, 2, 33, 15)  # (N, 5, 2, 33, 15)
    best_plan_idx = np.argmax(plan_probs, axis=1)[0]  # (N,)
    best_plan = plans[:, best_plan_idx, ...]  # (N, 2, 33, 15)

    # lane lines
    lane_lines = outputs[:, lanes_start_idx:lanes_end_idx]  # (N, 528)
    lane_lines_deflat = lane_lines.reshape((-1, 2, 264))  # (N, 2, 264)
    lane_lines_means = lane_lines_deflat[:, 0, :]  # (N, 264)
    lane_lines_means = lane_lines_means.reshape(-1, 4, 33, 2)  # (N, 4, 33, 2)

    outer_left_lane = lane_lines_means[:, 0, :, :]  # (N, 33, 2)
    inner_left_lane = lane_lines_means[:, 1, :, :]  # (N, 33, 2)
    inner_right_lane = lane_lines_means[:, 2, :, :]  # (N, 33, 2)
    outer_right_lane = lane_lines_means[:, 3, :, :]  # (N, 33, 2)

    # lane lines probs
    lane_lines_probs = outputs[:, lane_lines_prob_start_idx:lane_lines_prob_end_idx]  # (N, 8)
    lane_lines_probs = lane_lines_probs.reshape((-1, 4, 2))  # (N, 4, 2)
    lane_lines_probs = sigmoid(lane_lines_probs[:, :, 1])  # (N, 4), 0th is deprecated

    outer_left_prob = lane_lines_probs[:, 0]  # (N,)
    inner_left_prob = lane_lines_probs[:, 1]  # (N,)
    inner_right_prob = lane_lines_probs[:, 2]  # (N,)
    outer_right_prob = lane_lines_probs[:, 3]  # (N,)

    # road edges
    road_edges = outputs[:, road_start_idx:road_end_idx]
    road_edges_deflat = road_edges.reshape((-1, 2, 132))  # (N, 2, 132)
    road_edge_means = road_edges_deflat[:, 0, :].reshape(-1, 2, 33, 2)  # (N, 2, 33, 2)
    road_edge_stds = road_edges_deflat[:, 1, :].reshape(-1, 2, 33, 2)  # (N, 2, 33, 2)

    left_edge = road_edge_means[:, 0, :, :]  # (N, 33, 2)
    right_edge = road_edge_means[:, 1, :, :]
    left_edge_std = road_edge_stds[:, 0, :, :]  # (N, 33, 2)
    right_edge_std = road_edge_stds[:, 1, :, :]

    #leads
    leads_pred = outputs[:, 5755:5857]  # -- > 102 leads
    lead_porb = outputs[:, 5857:5860]  # -- > 3 lead_prob
    leads = leads_pred.reshape(-1, 2, 51)
    lead_pred_prob = leads[:, :, -3] #(N,2)
    best_lead_idx = np.argmax(lead_pred_prob, axis=1)[0]
    lead_porb = sigmoid(lead_porb)
    best_lead = leads[:, best_lead_idx, :-3].reshape(-1, 2, 6, 4)
    lead_car = np.hstack((best_lead[0,0,0,0], best_lead[0,0,0,2:], lead_porb[0, 0]))

    #desire_state
    desire_state_pred = outputs[:, 5860:5868]
    desire_state = softmax(desire_state_pred[0])

    batch_size = best_plan.shape[0]

    result_batch = []

    for i in range(batch_size):
        lanelines = [outer_left_lane[i], inner_left_lane[i], inner_right_lane[i], outer_right_lane[i]]
        lanelines_probs = [outer_left_prob[i], inner_left_prob[i], inner_right_prob[i], outer_right_prob[i]]
        road_edges = [left_edge[i], right_edge[i]]
        road_edges_probs = [left_edge_std[i], right_edge_std[i]]

        if best_plan_only:
            plan = best_plan[i]
        else:
            plan = (plans[i], plan_probs[i])

        result_batch.append(((lanelines, lanelines_probs), (road_edges, road_edges_probs), plan, lead_car, desire_state))

    return result_batch

def transform_img(base_img,
                  augment_trans=np.array([0, 0, 0]),
                  augment_eulers=np.array([0, 0, 0]),
                  from_intr=eon_intrinsics,
                  to_intr=medmodel_intrinsics,
                  output_size=None,
                  pretransform=None,
                  top_hacks=False,
                  yuv=False,
                  alpha=1.0,
                  beta=0,
                  blur=0):
    # import cv2  # pylint: disable=import-error
    cv2.setNumThreads(1)

    #if yuv:
    #    base_img = cv2.cvtColor(base_img, cv2.COLOR_YUV2RGB_I420)

    size = base_img.shape[:2]
    if not output_size:
        output_size = size[::-1]

    cy = from_intr[1, 2]

    def get_M(h=1.22):
        quadrangle = np.array([[0, cy + 20],
                            [size[1]-1, cy + 20],
                            [0, size[0]-1],
                            [size[1]-1, size[0]-1]], dtype=np.float32)
        quadrangle_norm = np.hstack((normalize(quadrangle, intrinsics=from_intr), np.ones((4, 1))))
        quadrangle_world = np.column_stack((h*quadrangle_norm[:, 0]/quadrangle_norm[:, 1],
                                            h*np.ones(4),
                                            h/quadrangle_norm[:, 1]))
        rot = orient.rot_from_euler(augment_eulers)
        to_extrinsics = np.hstack((rot.T, -augment_trans[:, None]))
        to_KE = to_intr.dot(to_extrinsics)
        warped_quadrangle_full = np.einsum('jk,ik->ij', to_KE, np.hstack((quadrangle_world, np.ones((4, 1)))))
        warped_quadrangle = np.column_stack((warped_quadrangle_full[:, 0]/warped_quadrangle_full[:, 2],
                                            warped_quadrangle_full[:, 1]/warped_quadrangle_full[:, 2])).astype(np.float32)
        M = cv2.getPerspectiveTransform(quadrangle, warped_quadrangle.astype(np.float32))
        return M

    M = get_M()
    if pretransform is not None:
        M = M.dot(pretransform)
    augmented_rgb = cv2.warpPerspective(base_img, M, output_size, borderMode=cv2.BORDER_REPLICATE)

    if top_hacks:
        cyy = int(math.ceil(to_intr[1, 2]))
        M = get_M(1000)
        if pretransform is not None:
            M = M.dot(pretransform)
        augmented_rgb[:cyy] = cv2.warpPerspective(base_img, M, (output_size[0], cyy), borderMode=cv2.BORDER_REPLICATE)

    # brightness and contrast augment
    # augmented_rgb = np.clip((float(alpha)*augmented_rgb + beta), 0, 255).astype(np.uint8)

    # print('after clip:', augmented_rgb.shape, augmented_rgb.dtype)
    # gaussian blur
    if blur > 0:
        augmented_rgb = cv2.GaussianBlur(augmented_rgb, (blur*2+1, blur*2+1), cv2.BORDER_DEFAULT)

    if yuv:
        # print(augmented_rgb.shape)
        augmented_img = cv2.cvtColor(augmented_rgb, cv2.COLOR_RGB2YUV_I420)
        # print(augmented_img.shape)
    else:
        augmented_img = augmented_rgb

    # cv2.imwrite('test.jpg', augmented_img)
    return augmented_img

def reshape_yuv(frames):
    H = (frames.shape[1]*2)//3
    W = frames.shape[2]
    in_img1 = np.zeros((frames.shape[0], 6, H//2, W//2), dtype=np.uint8)

    in_img1[:, 0] = frames[:, 0:H:2, 0::2]
    in_img1[:, 1] = frames[:, 1:H:2, 0::2]
    in_img1[:, 2] = frames[:, 0:H:2, 1::2]
    in_img1[:, 3] = frames[:, 1:H:2, 1::2]
    in_img1[:, 4] = frames[:, H:H+H//4].reshape((-1, H//2, W//2))
    in_img1[:, 5] = frames[:, H+H//4:H+H//2].reshape((-1, H//2, W//2))
    return in_img1

def inverse_reshape_yuv(in_img1):
    # Determine the original H and W
    H = in_img1.shape[2] * 2
    W = in_img1.shape[3] * 2
    
    # Create an empty array for the original frames
    frames = np.zeros((in_img1.shape[0], H + H // 2, W), dtype=np.uint8)
    
    # Reverse the operations
    frames[:, 0:H:2, 0::2] = in_img1[:, 0]
    frames[:, 1:H:2, 0::2] = in_img1[:, 1]
    frames[:, 0:H:2, 1::2] = in_img1[:, 2]
    frames[:, 1:H:2, 1::2] = in_img1[:, 3]
    
    frames[:, H:H+H//4] = in_img1[:, 4].reshape((-1, H//4, W))
    frames[:, H+H//4:H+H//2] = in_img1[:, 5].reshape((-1, H//4, W))
    
    return frames

def load_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    yuv_frames = []
    index = 0
    while cap.isOpened():
        index += 1
        ret, frame = cap.read()
        if not ret:
            break

        yuv_frames.append(bgr_to_yuv(frame))
        if index == 20:
            return yuv_frames

    return yuv_frames

def load_calibration(segment_path):
    logs_file = os.path.join(segment_path, 'rlog.bz2')
    lr = LogReader(logs_file)
    liveCalibration = [m.liveCalibration for m in lr if m.which() == 'liveCalibration']  # probably not 1200, but 240
    return liveCalibration

def bgr_to_yuv(img_bgr):
    img_yuv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YUV_I420)
    assert img_yuv.shape == ((874*3//2, 1164))
    return img_yuv

def bgr_to_rgb(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def yuv_to_rgb(yuv):
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB_I420)

def rgb_to_yuv(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2YUV_I420)

def transform_frames(frames):
    imgs_med_model = np.zeros((len(frames), 384, 512), dtype=np.uint8)
    for i, img in enumerate(frames):
        imgs_med_model[i] = transform_img(img, 
                                          from_intr=eon_intrinsics,
                                          to_intr=medmodel_intrinsics, 
                                          yuv=True,
                                          output_size=(512, 256))
        
    # print(imgs_med_model.shape)
    reshaped = reshape_yuv(imgs_med_model)
    # print(reshaped.shape)
    # inverse_reshaped = inverse_reshape_yuv(reshaped)
    # print(inverse_reshaped.shape)

    return reshaped

def transform_frames_test(frames):
    imgs_med_model = np.zeros((len(frames), 256, 512, 3), dtype=np.uint8)
    for i, img in enumerate(frames):
        imgs_med_model[i] = transform_img(img, 
                                          from_intr=eon_intrinsics,
                                          to_intr=medmodel_intrinsics, 
                                          yuv=False,
                                          output_size=(512, 256))

    return imgs_med_model

class Calibration:
    def __init__(self, rpy, intrinsic=eon_intrinsics, plot_img_width=640, plot_img_height=480):
        self.intrinsic = intrinsic
        self.extrinsics_matrix = get_view_frame_from_calib_frame(rpy[0], rpy[1], rpy[2], 0)[:, :3]
        self.plot_img_width = plot_img_width
        self.plot_img_height = plot_img_height
        self.zoom = W / plot_img_width
        self.CALIB_BB_TO_FULL = np.asarray([
            [self.zoom, 0., 0.],
            [0., self.zoom, 0.],
            [0., 0., 1.]])

    def car_space_to_ff(self, x, y, z):
        car_space_projective = np.column_stack((x, y, z)).T
        ep = self.extrinsics_matrix.dot(car_space_projective)
        kep = self.intrinsic.dot(ep)
        # TODO: fix numerical instability (add 1e-16)
        # UPD: this turned out to slow things down a lot. How do we do it then?
        return (kep[:-1, :] / kep[-1, :]).T

    def car_space_to_bb(self, x, y, z):
        pts = self.car_space_to_ff(x, y, z)
        return pts / self.zoom

def project_path(path, calibration, z_off):
    '''Projects paths from calibration space (model input/output) to image space.'''

    x = path[:, 0]
    y = path[:, 1]
    z = path[:, 2] + z_off
    pts = calibration.car_space_to_bb(x, y, z)
    pts[pts < 0] = np.nan
    valid = np.isfinite(pts).all(axis=1)
    pts = pts[valid].astype(int)

    return pts

def create_image_canvas(img_rgb, zoom_matrix, plot_img_height, plot_img_width):
    '''Transform with a correct warp/zoom transformation.'''
    img_plot = np.zeros((plot_img_height, plot_img_width, 3), dtype='uint8')
    cv2.warpAffine(img_rgb, zoom_matrix[:2], (img_plot.shape[1], img_plot.shape[0]), dst=img_plot, flags=cv2.WARP_INVERSE_MAP)
    return img_plot


def draw_path(lane_lines, road_edges, path_plan, lead_car, desire, img_plot, calibration, lane_line_color_list, width=1, height=1.22, fill_color=(128, 0, 255), line_color=(0, 255, 0), frame_n=None):
    
    '''Draw model predictions on an image.'''    

    overlay = img_plot.copy()
    alpha = 0.4
    fixed_distances = np.array(X_IDXs)[:,np.newaxis]
    img_h, img_w, _ = img_plot.shape

    # lane_lines are sequentially parsed ::--> means--> std's
    if lane_lines is not None:
        (oll, ill, irl, orl), (oll_prob, ill_prob, irl_prob, orl_prob) = lane_lines

        calib_pts_oll = np.hstack((fixed_distances, oll)) # (33, 3)
        calib_pts_ill = np.hstack((fixed_distances, ill)) # (33, 3)
        calib_pts_irl = np.hstack((fixed_distances, irl)) # (33, 3)
        calib_pts_orl = np.hstack((fixed_distances, orl)) # (33, 3)

        img_pts_oll = project_path(calib_pts_oll, calibration, z_off=0).reshape(-1,1,2)
        img_pts_ill = project_path(calib_pts_ill, calibration, z_off=0).reshape(-1,1,2)
        img_pts_irl = project_path(calib_pts_irl, calibration, z_off=0).reshape(-1,1,2)
        img_pts_orl = project_path(calib_pts_orl, calibration, z_off=0).reshape(-1,1,2)

        lane_lines_with_probs = [(img_pts_oll, oll_prob), (img_pts_ill, ill_prob), (img_pts_irl, irl_prob), (img_pts_orl, orl_prob)]

         # plot lanelines
        for i, (line_pts, prob) in enumerate(lane_lines_with_probs):
            line_overlay = img_plot.copy()
            cv2.polylines(line_overlay,[line_pts],False,lane_line_color_list[i],thickness=2)
            img_plot = cv2.addWeighted(line_overlay, prob, img_plot, 1 - prob, 0)

    # road edges
    if road_edges is not None:
        (left_road_edge, right_road_edge), _ = road_edges

        calib_pts_ledg = np.hstack((fixed_distances, left_road_edge))
        calib_pts_redg = np.hstack((fixed_distances, right_road_edge))
        
        img_pts_ledg = project_path(calib_pts_ledg, calibration, z_off=0).reshape(-1,1,2)
        img_pts_redg = project_path(calib_pts_redg, calibration, z_off=0).reshape(-1,1,2)

        # plot road_edges
        cv2.polylines(overlay,[img_pts_ledg],False,(255,128,0),thickness=1)
        cv2.polylines(overlay,[img_pts_redg],False,(255,234,0),thickness=1)
    
    # path plan
    if path_plan is not None:

        path_plan_l = path_plan - np.array([0, width, 0])
        path_plan_r = path_plan + np.array([0, width, 0])
    
        img_pts_l = project_path(path_plan_l, calibration, z_off=height)
        img_pts_r = project_path(path_plan_r, calibration, z_off=height)

        for i in range(1, len(img_pts_l)):
            if i >= len(img_pts_r): break

            u1, v1, u2, v2 = np.append(img_pts_l[i-1], img_pts_r[i-1])
            u3, v3, u4, v4 = np.append(img_pts_l[i], img_pts_r[i])
            pts = np.array([[u1, v1], [u2, v2], [u4, v4], [u3, v3]], np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(overlay, [pts], fill_color)
            cv2.polylines(overlay, [pts], True, line_color)

    if lead_car is not None:

        #const_speedBuff = 10.
        #const_leadBuff = 40.
        d_rel = lead_car[0] #- RADAR_TO_CAMERA
        v_rel = lead_car[1]
        lead_prob = lead_car[-1]

        '''
        fillAlpha = 0
        if d_rel < const_leadBuff:
            fillAlpha = 255 * (1.0 - (d_rel / const_leadBuff))
            if v_rel < 0:
                fillAlpha += 255 * (-1 * (v_rel / const_speedBuff))
            fillAlpha = int(min(fillAlpha, 255))
        '''
        leadcar_xyz = np.array([[d_rel, 0, 0]])
        img_pts_lead = project_path(leadcar_xyz, calibration, z_off=height)
        

        if len(img_pts_lead)>0 :
            sz = np.clip((25 * 30) / (d_rel / 3 + 30), 15.0, 30.0) * 2.35
            x = np.clip(img_pts_lead[0,0], 0, img_w - sz / 2)
            y = np.min([img_h - sz * .6, img_pts_lead[0,1]])

            g_xo = sz / 5
            g_yo = sz / 10

            glow = np.array([(x + (sz * 1.35) + g_xo, y + sz + g_yo), (x, y - g_yo), (x - (sz * 1.35) - g_xo, y + sz + g_yo)], dtype=np.int32)
            if lead_prob >= 0.5 :
                cv2.fillPoly(img_plot, [glow], (218, 202, 37))
        cv2.rectangle(img_plot, (0, 0), (240, 72), (0,0,0), -1)
        cv2.putText(img_plot, f'lead_prob : {lead_prob:.2f}', (0, 17), cv2.FONT_HERSHEY_PLAIN,1.5, (0, 255, 0), 2, cv2.LINE_AA)   
        cv2.putText(img_plot, f'lead_dis : {d_rel:.2f}', (0, 36), cv2.FONT_HERSHEY_PLAIN,1.5, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(img_plot, f'lead_speed : {v_rel:.2f}', (0, 54), cv2.FONT_HERSHEY_PLAIN,1.5, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(img_plot, f'lead_accel : {lead_car[2]:.2f}', (0, 72), cv2.FONT_HERSHEY_PLAIN,1.5, (0, 255, 0), 2, cv2.LINE_AA)
        
        # cv2.rectangle(img_plot, (0, 100), (480, 70), (0,0,0), -1)
        cv2.putText(img_plot, f'left_prob : {ill_prob:.2f}', (0, 135), cv2.FONT_HERSHEY_PLAIN,3, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(img_plot, f'right_prob : {irl_prob:.2f}', (0, 170), cv2.FONT_HERSHEY_PLAIN,3, (0, 255, 0), 2, cv2.LINE_AA)
        # chevron
        #chevron = np.array([(x + (sz * 1.25), y + sz), (x, y), (x - (sz * 1.25), y + sz)], dtype=np.int32)
        #cv2.fillPoly(img, [chevron], (0, 0, 255, fillAlpha))

    if desire is not None:
        cv2.rectangle(img_plot, (img_w-190, 0), (img_w, 93), (0,0,0), -1)
        cv2.putText(img_plot, f'none : {desire[0]:.2f}', (img_w-190, 17), cv2.FONT_HERSHEY_PLAIN,1.5, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(img_plot, f'turn_L : {desire[1]:.2f}', (img_w-190, 36), cv2.FONT_HERSHEY_PLAIN,1.5, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(img_plot, f'turn_R : {desire[2]:.2f}', (img_w-190, 54), cv2.FONT_HERSHEY_PLAIN,1.5, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(img_plot, f'changeL : {desire[3]:.2f}', (img_w-190, 72), cv2.FONT_HERSHEY_PLAIN,1.5, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(img_plot, f'changeR : {desire[4]:.2f}', (img_w-190, 90), cv2.FONT_HERSHEY_PLAIN,1.5, (0, 255, 0), 2, cv2.LINE_AA)
    # drawing the plots on original iamge
    cv2.rectangle(img_plot, (0, img_h-18), (165, img_h-1), (0,0,0), -1)
    cv2.putText(img_plot, f'frame : {frame_n}', (0, img_h-1), cv2.FONT_HERSHEY_PLAIN,1.5, (0, 255, 0), 2, cv2.LINE_AA)
    img_plot = cv2.addWeighted(overlay, alpha, img_plot, 1 - alpha, 0)

    return img_plot

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

def create_gt_distill_roach(path_of_h5, path_of_output):
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
            lead_dis = lead_dis + RADAR_TO_CAMERA
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

def load_h5_roach(seg_path, dif_name=False):

    if dif_name :
        file_path = seg_path
    else :
        file_path = os.path.join(seg_path, 'viz.h5')

    file = h5py.File(file_path,'r')

    plan = file['plans'][()]
    lanelines = file['lanelines'][()]
    leads = file['leads'][()]
    lead_prob = file['lead_prob'][()]
    desire_state = file['desire_state'][()]

    file.close()
    
    return plan, lanelines, leads, lead_prob, desire_state

def extract_gt_roach(lanelines_gt, plan_gt, leads_gt, lead_prob_gt, desire_state_gt):
    
    # plan_gt (N, 33, 15)
    batch_size = plan_gt.shape[0]
    plan = np.zeros((batch_size,2,33,15),dtype=np.float32)
    plan[:,0,:,:] = plan_gt

    # lane lines (N, 2, 33, 2)
    outer_left_lane = np.zeros((batch_size,33,2),dtype=np.float32)  # (N, 33, 2)
    inner_left_lane = lanelines_gt[:, 0, :, :]  # (N, 33, 2)
    inner_right_lane = lanelines_gt[:, 1, :, :]  # (N, 33, 2)
    outer_right_lane = np.zeros((batch_size,33,2),dtype=np.float32)  # (N, 33, 2)

    # lane lines probs
    outer_left_prob = np.zeros((batch_size),dtype=np.float32)  # (N,)
    inner_left_prob = np.ones((batch_size),dtype=np.float32)  # (N,)
    inner_right_prob = np.ones((batch_size),dtype=np.float32)  # (N,)
    outer_right_prob = np.zeros((batch_size),dtype=np.float32)  # (N,)

    ## road edges
    left_edge = np.zeros((batch_size,33,2),dtype=np.float32)  # (N, 33, 2)
    right_edge = np.zeros((batch_size,33,2),dtype=np.float32)
    left_edge_std = np.zeros((batch_size,33,2),dtype=np.float32)  # (N, 33, 2)
    right_edge_std = np.zeros((batch_size,33,2),dtype=np.float32)

    #leads_gt (N, 1, 3)
    #lead_prob_gt (N, 3)
    lead_car = np.hstack((leads_gt[0,0,:], lead_prob_gt[0,0]))

    desire_state_idx = desire_state_gt[0,0].astype(int)
    desire_state = np.zeros((8),dtype=np.float32)
    desire_state[desire_state_idx] = 1.0
    #batch_size = best_plan.shape[0]
    
    result_batch = []
    
    # each element of the output list is a tuple of predictions at respective sample_idx
    for i in range(batch_size):
        lanelines = [outer_left_lane[i], inner_left_lane[i], inner_right_lane[i], outer_right_lane[i]]
        lanelines_probs = [outer_left_prob[i], inner_left_prob[i], inner_right_prob[i], outer_right_prob[i]]
        road_edges = [left_edge[i], right_edge[i]]
        road_edges_probs = [left_edge_std[i], right_edge_std[i]]

        result_batch.append(((lanelines, lanelines_probs),(road_edges, road_edges_probs), plan[0], lead_car, desire_state))

    return result_batch


def read_video_file(filename: str):
    # Load the video
    cap = cv2.VideoCapture(filename)

    while True:
        ret, frame = cap.read()
        if not ret:
            break 
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        frame = transform_img(frame, 
                                          from_intr=eon_intrinsics,
                                          to_intr=medmodel_intrinsics, 
                                          yuv=False,
                                          output_size=(512, 256))
        cv2.imwrite('1440.png', frame)
        break

read_video_file('/media/t2-503-4090/4TB_SSD/QianXi_data/roach_data/openpilot_camera_laneline/video/0163.mp4')
