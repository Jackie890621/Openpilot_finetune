import warnings

warnings.filterwarnings("ignore", category=UserWarning, message='Length of IterableDataset')
warnings.filterwarnings("ignore", category=UserWarning,
                        message='The given NumPy array is not writeable, and PyTorch does not support non-writeable tensors')
warnings.filterwarnings("ignore", category=UserWarning, message='Using experimental implementation that allows \'batch_size > 1\'')

import argparse
import time
import numpy as np
from tqdm import tqdm
import torch
import torch.optim as topt
from torch.nn import CrossEntropyLoss, BCELoss
import torch.nn.functional as F
from dataloader_comma2k19 import CommaDataset, BatchDataLoader, BackgroundGenerator, load_transformed_video, configure_worker
from torch.utils.data import DataLoader
import wandb
from timing import Timing, MultiTiming, pprint_stats
from utils_comma2k19 import Calibration, draw_path, printf, extract_preds, dir_path, create_gt_distill_roach, load_h5_roach, extract_gt_roach, inverse_reshape_yuv, yuv_to_rgb #extract_gt, load_h5
import os
from model import load_trainable_model
import gc
import sys
import dotenv
import shutil
import math
from get_bucket import get_bucket_info
import joblib
from collections import defaultdict
import cv2

dotenv.load_dotenv()

def pprint_seconds(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{int(hours):1d}h {int(minutes):1d}min {int(seconds):1d}s"


# visualizing the model predictions
def visualization(lanelines, roadedges, calib_path, leadcar, desire, im_rgb):
    plot_img_height, plot_img_width = 480, 640

    rpy_calib = [0, 0, 0]
    calibration_pred = Calibration(rpy_calib, plot_img_width=plot_img_width, plot_img_height=plot_img_height)
    laneline_colors = [(255, 0, 0), (0, 255, 0), (255, 0, 255), (0, 255, 255)]
    vis_image = draw_path(lanelines, roadedges, calib_path[0, :, :3], leadcar, desire, im_rgb, calibration_pred, laneline_colors)

    return vis_image

def mean_std(array, eps=1e-10):
    mean = array[:, 0, :, :]
    std = array[:, 1, :, :]
    # These are the original openpilot_pipeline code, 
    # I think these aren't correct.
    # Instead, set lower bound for std.
    #std = torch.exp(std)
    #std = torch.add(std, eps)

    return mean, std


def path_laplacian_nll_loss(mean_true, mean_pred, sigma, sigma_clamp: float = 1e-3, loss_clamp: float = 1000.):
    err = torch.abs(mean_true - mean_pred) #+ 0.36788
    sigma = torch.clamp(sigma, min=sigma_clamp) # lower bound
    #sigma = torch.max(sigma, torch.log(1e-6 + err/loss_clamp))
    nll = err * torch.exp(-sigma) + sigma
    return nll.sum(dim=(1,2))

def lanelines_loss(lanelines_pred, lanelines_gt, device): # not correct do not use (wrong reshape)

    # lane lines gt
    outer_left_lane = lanelines_gt[:, 0, :, :]
    inner_left_lane = lanelines_gt[:, 1, :, :]  # (N, 33, 2)
    inner_right_lane = lanelines_gt[:, 2, :, :]  # (N, 33, 2)
    outer_right_lane = lanelines_gt[:, 3, :, :]

    # lane lines pred
    lane_lines_deflat = lanelines_pred.reshape((-1, 2, 264))  # (N, 2, 264)
    lane_lines_means = lane_lines_deflat[:, 0, :]  # (N, 264)
    lane_lines_means = lane_lines_means.reshape(-1, 4, 33, 2)  # (N, 4, 33, 2)

    outer_left_lane_pred = lane_lines_means[:, 0, :, :]
    inner_left_lane_pred = lane_lines_means[:, 1, :, :]  # (N, 33, 2)
    inner_right_lane_pred = lane_lines_means[:, 2, :, :]  # (N, 33, 2)
    outer_right_lane_pred = lane_lines_means[:, 3, :, :]

    lane_lines_stds = lane_lines_deflat[:, 1, :]  # (N, 264)
    lane_lines_stds = lane_lines_stds.reshape(-1, 4, 33, 2)  # (N, 4, 33, 2)

    outer_left_lane_std = lane_lines_stds[:, 0, :, :]
    inner_left_lane_std = lane_lines_stds[:, 1, :, :]  # (N, 33, 2)
    inner_right_lane_std = lane_lines_stds[:, 2, :, :]  # (N, 33, 2)
    outer_right_lane_std = lane_lines_stds[:, 3, :, :]

    lanelines_ol_loss = path_laplacian_nll_loss(outer_left_lane, outer_left_lane_pred, outer_left_lane_std)
    lanelines_il_loss = path_laplacian_nll_loss(inner_left_lane, inner_left_lane_pred, inner_left_lane_std)
    lanelines_ir_loss = path_laplacian_nll_loss(inner_right_lane, inner_right_lane_pred, inner_right_lane_std)
    lanelines_or_loss = path_laplacian_nll_loss(outer_right_lane, outer_right_lane_pred, outer_right_lane_std)

    # MHP loss
    lanelines_head_loss = torch.stack([lanelines_ol_loss, lanelines_il_loss, lanelines_ir_loss, lanelines_or_loss]).T

    lanelines_head_loss = lanelines_head_loss.sum(dim=1).mean()

    return lanelines_head_loss

def train(run, model, train_loader, val_loader, optimizer, scheduler, recurr_warmup, epoch, 
          log_frequency_steps, train_segment_for_viz, val_segment_for_viz, batch_size):

    recurr_input = torch.zeros(batch_size, 512, dtype=torch.float32, device=device, requires_grad=True)
    desire = torch.zeros(batch_size, 8, dtype=torch.float32, device=device)
    traffic_convention = torch.zeros(batch_size, 2, dtype=torch.float32, device=device)
    traffic_convention[:, 1] = 1
    model.train()

    train_loss_accum = 0.0
    segments_finished = True

    start_point = time.time()
    timings = dict()
    multitimings = MultiTiming(timings)
    multitimings.start('batch_load')

    for tr_it, batch in enumerate(train_loader):
        batch_load_time = multitimings.end('batch_load')

        should_log_train = (tr_it+1) % log_frequency_steps == 0
        should_run_valid = (tr_it+1) % val_frequency_steps == 0
        
        printf()
        printf(f"> Got new batch: {batch_load_time:.2f}s - training iteration i am in ", tr_it)
        multitimings.start('train_batch')

        should_backprop = (not recurr_warmup) or (recurr_warmup and not segments_finished)

        stacked_frames, gt_plans, gt_lanelines, segments_finished = batch
        segments_finished = torch.all(segments_finished)

        loss, recurr_input = train_batch(run, model, optimizer, stacked_frames, gt_plans, gt_lanelines,
                        desire, traffic_convention, recurr_input, device, timings, should_backprop=should_backprop)

        train_batch_time = multitimings.end('train_batch')
        fps = batch_size * seq_len / train_batch_time
        printf(f"> Batch trained: {train_batch_time:.2f}s (FPS={fps:.2f}).")

        if segments_finished:
            # reset the hidden state for new segments
            printf('Resetting hidden state.')
            recurr_input = recurr_input.zero_().detach()

        train_loss_accum += loss

        if should_run_valid:
            #with Timing(timings, 'visualize_preds'):
            #    visualize_predictions(model, device, train_segment_for_viz, val_segment_for_viz)
            with Timing(timings, 'validate'):
                val_loss = validate(model, val_loader, batch_size, device)

            scheduler.step(val_loss.item())

            checkpoint_save_file = 'commaitr' + date_it + str(val_loss) + '_' + str(epoch+1) + ".pth"
            checkpoint_save_path = os.path.join(checkpoints_dir, checkpoint_save_file)
            torch.save(model.state_dict(), checkpoint_save_path)
            model.train()

            wandb.log({
                'validation_loss': val_loss,
            }, commit=False)

        if should_log_train:
            timings['forward_pass']['time'] *= seq_len
            timings['lanelines_loss']['time'] *= seq_len
        
            running_loss = train_loss_accum.item() / log_frequency_steps

            printf()
            printf(f'Epoch {epoch+1}/{epochs}. Done {tr_it+1} steps of ~{train_loader_len}. Running loss: {running_loss:.4f}')
            pprint_stats(timings)
            printf()

            wandb.log({
                'epoch': epoch,
                'train_tol_loss': running_loss,
                'lr': scheduler.optimizer.param_groups[0]['lr'],
                **{f'time_{k}': v['time'] / v['count'] for k, v in timings.items()}
            }, commit=True)

            timings = dict()
            train_loss_accum = 0.0

        multitimings.start('batch_load')

    printf()
    printf(f"Epoch {epoch+1} done! Took {pprint_seconds(time.time() - start_point)}")
    printf()


def visualize_predictions(model, device, train_segment_for_viz, val_segment_for_viz):
    segments_for_viz = [train_segment_for_viz, val_segment_for_viz]

    model.eval()
    with torch.no_grad():

        for i in range(len(segments_for_viz)):

            path_to_segment = segments_for_viz[i]
            printf(f"===>Visualizing predictions: {path_to_segment}")

            recurr_input = torch.zeros(1, 512, dtype=torch.float32, device=device, requires_grad=False)
            #desire = torch.zeros(1, 8, dtype=torch.float32, device=device)
            traffic_convention = torch.zeros(1, 2, dtype=torch.float32, device=device)
            traffic_convention[:, 1] = 1

            input_frames, rgb_frames, frame_desire = load_transformed_video(path_to_segment)
            input_frames = input_frames.to(device)

            video_array_pred = np.zeros((rgb_frames.shape[0],rgb_frames.shape[1],rgb_frames.shape[2], rgb_frames.shape[3]), dtype=np.uint8)
            viz_prev_desire = np.int8(7)

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
                im_rgb = rgb_frames[t_idx] 
                vis_image = visualization(lanelines,road_edges,best_path, best_lead, desire_state, im_rgb)
                video_array_pred[t_idx:t_idx+1,:,:,:] = vis_image

            video_array_gt = np.zeros((rgb_frames.shape[0],rgb_frames.shape[1],rgb_frames.shape[2], rgb_frames.shape[3]), dtype=np.uint8)

            if os.path.exists(os.path.join(path_to_segment, 'viz.h5')):
                printf('viz.h5 exists')
                plan_gt_h5, laneline_gt_h5, leads_gt_h5, lead_prob_gt_h5, desire_state_gt_h5 = load_h5_roach(path_to_segment)

            else :
                raise Exception('No viz.h5 file found in {}'.format(path_to_segment))

            #plan_gt_h5, plan_prob_gt_h5, laneline_gt_h5, laneline_prob_gt_h5, road_edg_gt_h5, road_edgstd_gt_h5 = load_h5(path_to_segment)

            for k in range(plan_gt_h5.shape[0]):

                #lane_h5, roadedg_h5, path_h5 = extract_gt(plan_gt_h5[k:k+1], plan_prob_gt_h5[k:k+1], laneline_gt_h5[k:k+1], laneline_prob_gt_h5[k:k+1], road_edg_gt_h5[k:k+1], road_edgstd_gt_h5[k:k+1])[0]
                lane_h5, roadedg_h5, path_h5, leadcar_h5, desire_h5 = extract_gt_roach(plan_gt_h5[k:k+1], laneline_gt_h5[k:k+1], leads_gt_h5[k:k+1], lead_prob_gt_h5[k:k+1], desire_state_gt_h5[k:k+1])[0]
                image_rgb_gt = rgb_frames[k]

                image_gt = visualization(None, None, path_h5, leadcar_h5, desire_h5, image_rgb_gt)
                video_array_gt[k:k+1,:,:,:] = image_gt


            video_array_pred = video_array_pred.transpose(0,3,1,2)
            video_array_gt = video_array_gt.transpose(0,3,1,2)
                
            if i == 0:
                video_pred_log_title = "train_pred_video"
                video_gt_log_title = "train_gt_video"
            else:
                video_pred_log_title = "validation_pred_video"
                video_gt_log_title = "validation_gt_video"

            wandb.log({video_pred_log_title: wandb.Video(video_array_pred, fps = 20, format= 'mp4')}, commit=False)
            wandb.log({video_gt_log_title: wandb.Video(video_array_gt, fps = 20, format= 'mp4')}, commit=False)

            del video_array_pred
            del video_array_gt

            gc.collect()


def validate(model, data_loader, batch_size, device):

    model.eval()
    
    # dice.eval()
    # saving memory by not accumulating activations
    with torch.no_grad():

        val_loss = 0.0

        printf(">>>>>validating<<<<<<<")
        val_itr = None
        recurr_input = torch.zeros(batch_size, 512, dtype=torch.float32, device=device, requires_grad=False)
        for val_itr, val_batch in enumerate(data_loader):

            val_stacked_frames, val_plans, val_lanelines, segments_finished = val_batch
            segments_finished = torch.all(segments_finished)

            batch_loss, recurr_input = validate_batch(model, val_stacked_frames, val_plans, val_lanelines, recurr_input, device)
            val_loss += batch_loss

            if segments_finished:
                # reset the hidden state for new segments
                printf('Resetting hidden state.')
                recurr_input.zero_()

            if (val_itr+1) % 10 == 0:
                running_loss = val_loss.item() / (val_itr+1)  # average over entire validation set, no reset as in train
                printf(f'[Validation] Done {val_itr+1} steps of ~{val_loader_len}. Running loss: {running_loss:.4f}')

        if val_itr is not None:
            val_avg_loss = val_loss/(val_itr+1)
            printf(f"Validation Loss: {val_avg_loss:.4f}")

        return val_avg_loss


def train_batch(run, model, optimizer, stacked_frames, gt_plans, gt_lanelines, gt_desire, traffic_convention, recurr_input, device, timings, should_backprop=True, awl=None):
    seq_len = stacked_frames.shape[1]

    with Timing(timings, 'inputs_to_gpu'):
        stacked_frames = stacked_frames.to(device).float()  # -- (batch_size, seq_len, 12, 128, 256)
        gt_lanelines = gt_lanelines.to(device) # -- (batch_size,seq_len,2,33,2)

    optimizer.zero_grad(set_to_none=True)

    batch_laneline_loss = 0.0

    for i in range(seq_len):

        inputs_to_pretained_model = {"input_imgs": stacked_frames[:, i, :, :, :],
                                     "desire": gt_desire,
                                     "traffic_convention": traffic_convention,
                                     'initial_state': recurr_input.clone()  # TODO: why are we cloning recurr_input in 3 places (here, line 428 and line 439?
                                     }

        with Timing(timings, 'forward_pass'):
            outputs = model(**inputs_to_pretained_model)  # -- > [32,6472]

        lanelines_predictions = outputs[:, 4955:5483].clone()  # -- > 528 lanelines
        recurr_out = outputs[:, 5960:].clone()  # -- > [32,512] important to refeed state of GRU

        with Timing(timings, 'lanelines_loss'):
            step_lanelines_loss = lanelines_loss(lanelines_predictions, gt_lanelines[:, i, :, :, :], device)

        if i == seq_len - 1:
            # final hidden state in sequence, no need to backpropagate it through time
            pass
        else:
            recurr_input = recurr_out.clone()

        batch_laneline_loss += step_lanelines_loss

    batch_laneline_loss /= seq_len

    if should_backprop:
        with Timing(timings, 'backward_pass'):
            batch_laneline_loss.backward(retain_graph=True)

    with Timing(timings, 'clip_gradients'):
        torch.nn.utils.clip_grad_norm_(model.parameters(), run.config.grad_clip)

    with Timing(timings, 'optimize_step'):
        optimizer.step()

    loss = batch_laneline_loss.detach()
    return loss, recurr_out.detach()


def validate_batch(model, val_stacked_frames, val_plans, val_lanelines, recurr_input, device):
    batch_size = val_stacked_frames.shape[0]
    seq_len = val_stacked_frames.shape[1]

    desire = torch.zeros(batch_size, 8, dtype=torch.float32, device=device)
    traffic_convention = torch.zeros(batch_size, 2, dtype=torch.float32, device=device)
    traffic_convention[:, 1] = 1

    val_input = val_stacked_frames.float().to(device)
    val_lanelines = val_lanelines.to(device)

    val_batch_loss = 0.0

    for i in range(seq_len):
        val_inputs_to_pretained_model = {"input_imgs": val_input[:, i, :, :, :],
                                         "desire": desire,
                                         "traffic_convention": traffic_convention,
                                         "initial_state": recurr_input}

        val_outputs = model(**val_inputs_to_pretained_model)  # --> [32,6472]
        recurr_input = val_outputs[:, 5960:].clone()  # --> [32,512] important to refeed state of GRU
        val_lanelines_predictions = val_outputs[:, 4955:5483].clone()  # -- > 528 lanelines
    
        single_lanelines_loss = lanelines_loss(val_lanelines_predictions, val_lanelines[:, i, :, :, :], device)
    
    #val_batch_loss = val_batch_loss / seq_len / batch_size
    val_batch_loss = single_lanelines_loss / seq_len
    return val_batch_loss.detach(), recurr_input


if __name__ == "__main__":

    cuda = torch.cuda.is_available()
    if cuda:
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("=>Using '{}' for computation.".format(device))


    # NOTE: important for data loader
    torch.multiprocessing.set_start_method('spawn')
    torch.autograd.set_detect_anomaly(False)

    print("=>Initializing CLI args")
    # CLI parser
    parser = argparse.ArgumentParser(description='Args for comma supercombo train pipeline')
    parser.add_argument("--batch_size", type=int, default=28, help="batch size")
    parser.add_argument("--date_it", type=str, required=True, help="run date/name")  # "16Jan_1_seg"
    parser.add_argument("--epochs", type=int, default=15, help="number of epochs")
    parser.add_argument("--grad_clip", type=float, default=torch.inf, help="gradient clip norm")
    parser.add_argument("--l2_lambda", type=float, default=1e-4, help="weight decay rate")
    parser.add_argument("--log_frequency", type=int, default=100, help="log to wandb every this many steps")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--lrs_factor", type=float, default=0.75, help="lrs factor")
    parser.add_argument("--lrs_min", type=float, default=1e-6, help="lrs min")
    parser.add_argument("--lrs_patience", type=int, default=3, help="lrs patience")
    parser.add_argument("--lrs_thresh", type=float, default=1e-4, help="lrs threshold")
    parser.add_argument("--mhp_loss", dest='distill', help="use Laplacian MHP loss instead of distillation", action='store_false')  # "16Jan_1_seg"
    parser.add_argument("--no_recurr_warmup", dest='recurr_warmup', action='store_false')
    parser.add_argument("--no_wandb", dest="no_wandb", action="store_true", help="disable wandb")
    parser.add_argument("--recordings_basedir", type=dir_path, default="/gpfs/space/projects/Bolt/comma_recordings", help="path to base directory with recordings")
    parser.add_argument("--recurr_warmup", dest='recurr_warmup', action='store_true')
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--seq_len", type=int, default=100, help="sequence length")
    parser.add_argument("--split", type=float, default=0.94, help="train/val split")
    parser.add_argument("--val_frequency", type=int, default=400, help="run validation every this many steps")
    parser.set_defaults(recurr_warmup=True)
    parser.set_defaults(distill=True)
    args = parser.parse_args()
 
    # for reproducibility
    torch.manual_seed(args.seed)

    date_it = args.date_it
    train_run_name = date_it
    comma_recordings_basedir = args.recordings_basedir
    path_to_supercombo = '../common/models/supercombo.onnx'

    checkpoints_dir = './nets/701_2phase_comma2k19'
    result_model_dir = './nets/701_2phase_comma2k19'
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(result_model_dir, exist_ok=True)

    # Hyperparams
    batch_size = num_workers = args.batch_size  # MUST BE batch_size == num_workers
    assert batch_size == num_workers, 'Batch size must be equal to number of workers'

    epochs = args.epochs
    l2_lambda = args.l2_lambda
    log_frequency_steps = args.log_frequency
    lr = args.lr
    lrs_cd = 0
    lrs_factor = args.lrs_factor
    lrs_min = args.lrs_min
    lrs_patience = args.lrs_patience
    lrs_thresh = args.lrs_thresh
    prefetch_factor = 2
    recurr_warmup = args.recurr_warmup
    seq_len = args.seq_len
    train_val_split = args.split
    val_frequency_steps = args.val_frequency
    
    pathplan_layer_names = [ "Conv_785", "Conv_797", "Conv_809", "Conv_832", #RepAdapter
                             "Conv_844", "Conv_856", "Conv_868", "Conv_891", #RepAdapter
                            #  "Gemm_959", "Gemm_981","Gemm_983","Gemm_1036", #plan
                            #  "Gemm_961", "Gemm_986","Gemm_988","Gemm_1037", #leads
                            #  "Gemm_963", "Gemm_991","Gemm_993","Gemm_1038", #lead_prob
                             "Gemm_969", "Gemm_1006","Gemm_1008","Gemm_1041", #outer_left_lane
                             "Gemm_971", "Gemm_1011","Gemm_1013","Gemm_1042", #left_lane
                             "Gemm_973", "Gemm_1016","Gemm_1018","Gemm_1043", #right_lane
                             "Gemm_975", "Gemm_1021","Gemm_1023","Gemm_1044", #outer_right_lane
                            #  "Gemm_979", "Gemm_1031","Gemm_1033","Gemm_1046", #desire_state
                            #  "Gemm_912", "Gemm_921","Gemm_923","Gemm_932", #desire_prob
                             ]

    # wandb init
    run = wandb.init(entity="jackie890621", project="openpilot-pipeline-Repadapter", name=train_run_name, mode='offline' if args.no_wandb else 'online')
    #run = wandb.init(entity="gary1111255", project="openpilot-pipeline-Repadapter", name=train_run_name, mode='offline' if args.no_wandb else 'online')
    # Load data and split in test and train
    printf("=>Loading data")
    printf("=>Preparing the dataloader")
    printf(f"=>Batch size is {batch_size}")

    train_dataset = CommaDataset(comma_recordings_basedir, batch_size=batch_size, train_split=train_val_split, seq_len=seq_len,
                                 shuffle=True, seed=42)
    #train_segment_for_viz = os.path.dirname(train_dataset.hevc_file_paths[train_dataset.segment_indices[0]]) # '/home/nikita/data/2021-09-14--09-19-21/2'
    train_segment_for_viz = (train_dataset.mp4_file_paths[train_dataset.segment_indices[0]])
    train_loader = DataLoader(train_dataset, batch_size=None, num_workers=num_workers, shuffle=False, prefetch_factor=prefetch_factor,
                              persistent_workers=True, collate_fn=None, worker_init_fn=configure_worker)
    train_loader = BatchDataLoader(train_loader, batch_size=batch_size)
    train_loader_len = len(train_loader)
    train_loader = BackgroundGenerator(train_loader)

    val_dataset = CommaDataset(comma_recordings_basedir, batch_size=batch_size, train_split=train_val_split, seq_len=seq_len,
                               validation=True, shuffle=True, seed=42)
    #val_segment_for_viz = os.path.dirname(val_dataset.hevc_file_paths[val_dataset.segment_indices[0]]) # '/home/nikita/data/2021-09-19--10-22-59/18'
    val_segment_for_viz = (val_dataset.mp4_file_paths[val_dataset.segment_indices[0]])
    val_loader = DataLoader(val_dataset, batch_size=None, num_workers=num_workers, shuffle=False, prefetch_factor=prefetch_factor,
                            persistent_workers=True, collate_fn=None, worker_init_fn=configure_worker)
    val_loader = BatchDataLoader(val_loader, batch_size=batch_size)
    val_loader_len = len(val_loader)
    val_loader = BackgroundGenerator(val_loader)

    printf('Train visualization segment:', train_segment_for_viz)
    printf('Validation visualization segment:', val_segment_for_viz)

    os.makedirs('tmp/train_segment_for_viz', exist_ok=True)
    os.makedirs('tmp/val_segment_for_viz', exist_ok=True)
    with open('tmp/train_segment_for_viz/num.txt','w') as f_n:
        f_n.write(train_segment_for_viz)
    with open('tmp/val_segment_for_viz/num.txt','w') as f_n:
        f_n.write(val_segment_for_viz)
    shutil.copy2(train_segment_for_viz, 'tmp/train_segment_for_viz/viz.mp4')
    shutil.copy2(val_segment_for_viz, 'tmp/val_segment_for_viz/viz.mp4')
    train_segment_for_viz = (train_dataset.gt_file_paths[train_dataset.segment_indices[0]])
    val_segment_for_viz = (val_dataset.gt_file_paths[val_dataset.segment_indices[0]])
    shutil.copy2(train_segment_for_viz, 'tmp/train_segment_for_viz/viz.h5')
    shutil.copy2(val_segment_for_viz, 'tmp/val_segment_for_viz/viz.h5')
    train_segment_for_viz = 'tmp/train_segment_for_viz'
    val_segment_for_viz = 'tmp/val_segment_for_viz'


    printf('Batches in train_loader:', train_loader_len)
    printf('Batches in val_loader:', val_loader_len)

    printf("=>Loading the model")

    pth_path = "/home/t2-503-4090/QianXi/Openpilot_BalancedRegression_Adapter/train/nets/model_itr/adapter_1928x1208.pth"
    comma_model = load_trainable_model(path_to_supercombo, trainable_layers=pathplan_layer_names)
    with open("./model_freeze.txt",'w') as freeze_txt:
        for name, param in comma_model.named_parameters():
            freeze_txt.write(f'{name} : {param.requires_grad}\n')
            #print(f'{name} : {param.requires_grad}')

    comma_model.load_state_dict(torch.load(pth_path))
    comma_model = comma_model.to(device)

    wandb.watch(comma_model) # Log the gradients  

    param_group = comma_model.parameters()
    optimizer = topt.Adam([{'params': param_group}], lr, weight_decay=l2_lambda)
    scheduler = topt.lr_scheduler.ReduceLROnPlateau(optimizer, factor=lrs_factor, patience=lrs_patience,
                                                    threshold=lrs_thresh, verbose=True, min_lr=lrs_min,
                                                    cooldown=lrs_cd)

    with run:
        printf("=>Run parameters: \n")
        for arg in vars(args):
            wandb.config.update({arg: getattr(args, arg)})
            printf(arg, getattr(args, arg))
        printf()

        printf("=====>Starting to train")
        with torch.autograd.profiler.profile(enabled=False):
            with torch.autograd.profiler.emit_nvtx(enabled=False, record_shapes=False):
                for epoch in tqdm(range(epochs)):
                    train(run, comma_model, train_loader, val_loader, optimizer, scheduler,
                        recurr_warmup, epoch, log_frequency_steps,
                        train_segment_for_viz, val_segment_for_viz, batch_size)

        result_model_save_path = os.path.join(result_model_dir, train_run_name + '.pth')
        torch.save(comma_model.state_dict(), result_model_save_path)
        printf("Saved trained model")
        printf("training_finished")

    sys.exit(0)