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
from dataloader import CommaDataset, BatchDataLoader, BackgroundGenerator, load_transformed_video, configure_worker
from torch.utils.data import DataLoader
import wandb
from timing import Timing, MultiTiming, pprint_stats
from utils import Calibration, draw_path, printf, extract_preds, dir_path, create_gt_distill_roach, load_h5_roach, extract_gt_roach, inverse_reshape_yuv, yuv_to_rgb #extract_gt, load_h5
import os
from model import load_trainable_model
import gc
import sys
import dotenv
import shutil
import math
from AutomaticWeightedLoss import AutomaticWeightedLoss
from get_bucket import get_bucket_info
import joblib
from collections import defaultdict
import cv2

dotenv.load_dotenv()

check_loss = defaultdict(list)

# class DiceBCELoss(torch.nn.Module):
#     def __init__(self, weight=None, size_average=True):
#         super(DiceBCELoss, self).__init__()

#     def forward(self, inputs, targets, smooth=1):
        
#         #comment out if your model contains a sigmoid or equivalent activation layer
#         inputs = F.sigmoid(inputs)       
        
#         #flatten label and prediction tensors
#         inputs = inputs.reshape(-1)
#         targets = targets.reshape(-1)
        
#         intersection = (inputs * targets).sum()                            
#         dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
#         BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
#         Dice_BCE = BCE + dice_loss
        
#         return Dice_BCE

# class DiceLoss(torch.nn.Module):
#     def __init__(self, weight=None, size_average=True):
#         super(DiceLoss, self).__init__()

#     def forward(self, inputs, targets, smooth=1):
        
#         #comment out if your model contains a sigmoid or equivalent activation layer
#         inputs = F.sigmoid(inputs)       
        
#         #flatten label and prediction tensors
#         inputs = inputs.reshape(-1)
#         targets = targets.reshape(-1)
        
#         intersection = (inputs * targets).sum()                            
#         dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
#         return 1 - dice

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

class GAILoss(torch.nn.Module):
    def __init__(self, gmm):
        super(GAILoss, self).__init__()
        self.gmm = joblib.load(gmm)
        self.gmm = {k: torch.tensor(self.gmm[k]).cuda() for k in self.gmm}

    def forward(self, pred, target, noise_scale):
        loss = gai_loss(pred, target, self.gmm, noise_scale)
        return loss

def gai_loss(pred, target, gmm, noise_scale):
    gmm = {k: gmm[k].reshape(1, -1).expand(pred.shape[0], -1) for k in gmm}
    noise_scale =  torch.clamp(noise_scale, min=1e-3) # lower bound
    mse_term = F.l1_loss(pred, target, reduction='none') / noise_scale + noise_scale.log()
    sum_scale = (gmm['variances'] / 2).sqrt() + noise_scale
    balancing_term = - sum_scale.log() - abs(pred - gmm['means']) / sum_scale + gmm['weights'].log()
    balancing_term = torch.logsumexp(balancing_term, dim=-1, keepdim=True)

    loss = mse_term + balancing_term

    loss = loss * noise_scale.detach()

    return loss.squeeze()

def path_kl_div_loss(mean1, mean2, std1, std2):
    """
    scratch :Laplace or gaussian likelihood 
    model distillation: gaussian or laplace, KL divergence
    """
    d1 = torch.distributions.laplace.Laplace(mean1, std1)
    d2 = torch.distributions.laplace.Laplace(mean2, std2)
    loss = torch.distributions.kl.kl_divergence(d1, d2).sum(dim=2).sum(dim=1).mean(dim=0)
    return loss

def plan_distill_loss(plan_pred, plan_gt, plan_prob_gt, device):

    paths = plan_pred.reshape(-1, 5, 991)
    path1_pred = paths[:, 0, :-1].reshape(-1, 2, 33, 15)
    path2_pred = paths[:, 1, :-1].reshape(-1, 2, 33, 15)
    path3_pred = paths[:, 2, :-1].reshape(-1, 2, 33, 15)
    path4_pred = paths[:, 3, :-1].reshape(-1, 2, 33, 15)
    path5_pred = paths[:, 4, :-1].reshape(-1, 2, 33, 15)
    path_pred_prob = paths[:, :, -1]

    path1_gt = plan_gt[:, 0, :, :, :]
    path2_gt = plan_gt[:, 1, :, :, :]
    path3_gt = plan_gt[:, 2, :, :, :]
    path4_gt = plan_gt[:, 3, :, :, :]
    path5_gt = plan_gt[:, 4, :, :, :]

    mean_pred_path1, std_pred_path1 = mean_std(path1_pred)
    mean_gt_path1, std_gt_path1 = mean_std(path1_gt)

    mean_pred_path2, std_pred_path2 = mean_std(path2_pred)
    mean_gt_path2, std_gt_path2 = mean_std(path2_gt)

    mean_pred_path3, std_pred_path3 = mean_std(path3_pred)
    mean_gt_path3, std_gt_path3 = mean_std(path3_gt)

    mean_pred_path4, std_pred_path4 = mean_std(path4_pred)
    mean_gt_path4, std_gt_path4 = mean_std(path4_gt)

    mean_pred_path5, std_pred_path5 = mean_std(path5_pred)
    mean_gt_path5, std_gt_path5 = mean_std(path5_gt)

    path1_loss = path_kl_div_loss(mean_pred_path1, mean_gt_path1, std_pred_path1, std_gt_path1)
    path2_loss = path_kl_div_loss(mean_pred_path2, mean_gt_path2, std_pred_path2, std_gt_path2)
    path3_loss = path_kl_div_loss(mean_pred_path3, mean_gt_path3, std_pred_path3, std_gt_path3)
    path4_loss = path_kl_div_loss(mean_pred_path4, mean_gt_path4, std_pred_path4, std_gt_path4)
    path5_loss = path_kl_div_loss(mean_pred_path5, mean_gt_path5, std_pred_path5, std_gt_path5)

    path_pred_prob_d = torch.distributions.categorical.Categorical(logits=path_pred_prob)
    path_gt_prob_d = torch.distributions.categorical.Categorical(logits=plan_prob_gt)
    path_prob_loss = torch.distributions.kl.kl_divergence(path_pred_prob_d, path_gt_prob_d).mean(dim=0)

    plan_loss = path1_loss + path2_loss + path3_loss + path4_loss + path5_loss + path_prob_loss

    return plan_loss

# TODO: vectorize for speedup?
def plan_mhp_loss(plan_pred, plan_gt, device):

    batch_size = plan_pred.shape[0]

    paths = plan_pred.reshape(-1, 5, 991)
    path1_pred = paths[:, 0, :-1].reshape(-1, 2, 33, 15)
    path2_pred = paths[:, 1, :-1].reshape(-1, 2, 33, 15)
    path3_pred = paths[:, 2, :-1].reshape(-1, 2, 33, 15)
    path4_pred = paths[:, 3, :-1].reshape(-1, 2, 33, 15)
    path5_pred = paths[:, 4, :-1].reshape(-1, 2, 33, 15)
    path_pred_prob = paths[:, :, -1]

    mean_pred_path1, std_pred_path1 = mean_std(path1_pred)
    mean_pred_path2, std_pred_path2 = mean_std(path2_pred)
    mean_pred_path3, std_pred_path3 = mean_std(path3_pred)
    mean_pred_path4, std_pred_path4 = mean_std(path4_pred)
    mean_pred_path5, std_pred_path5 = mean_std(path5_pred)

    path1_loss = path_laplacian_nll_loss(plan_gt, mean_pred_path1, std_pred_path1)
    path2_loss = path_laplacian_nll_loss(plan_gt, mean_pred_path2, std_pred_path2)
    path3_loss = path_laplacian_nll_loss(plan_gt, mean_pred_path3, std_pred_path3)
    path4_loss = path_laplacian_nll_loss(plan_gt, mean_pred_path4, std_pred_path4)
    path5_loss = path_laplacian_nll_loss(plan_gt, mean_pred_path5, std_pred_path5)

    # MHP loss
    path_head_loss = torch.stack([path1_loss, path2_loss, path3_loss, path4_loss, path5_loss]).T

    idx = torch.argmin(path_head_loss, dim=1)
    best_path_mask = torch.zeros((batch_size, 5), device=device)
    mask = torch.full((batch_size, 5), 1e-6, device=device)
    best_path_mask[torch.arange(idx.shape[0]), idx] = 1
    mask[torch.arange(idx.shape[0]), idx] = 1

    path_perhead_loss = torch.mul(path_head_loss, mask)
    path_perhead_loss = path_perhead_loss.sum(dim=1).mean()

    cross_entropy_loss = CrossEntropyLoss(reduction='mean')
    path_prob_loss = cross_entropy_loss(path_pred_prob, best_path_mask)

    plan_loss = path_perhead_loss + path_prob_loss
    return plan_loss, path_perhead_loss, path_prob_loss

def lanelines_loss(lanelines_pred, lanelines_gt, device, dice):

    # lane lines gt
    inner_left_lane = lanelines_gt[:, 0, :, :]  # (N, 33, 2)
    inner_right_lane = lanelines_gt[:, 1, :, :]  # (N, 33, 2)

    # lane lines pred
    lane_lines_deflat = lanelines_pred.reshape((-1, 2, 264))  # (N, 2, 264)
    lane_lines_means = lane_lines_deflat[:, 0, :]  # (N, 264)
    lane_lines_means = lane_lines_means.reshape(-1, 4, 33, 2)  # (N, 4, 33, 2)

    inner_left_lane_pred = lane_lines_means[:, 1, :, :]  # (N, 33, 2)
    inner_right_lane_pred = lane_lines_means[:, 2, :, :]  # (N, 33, 2)
    lane_lines_stds = lane_lines_deflat[:, 1, :]  # (N, 264)
    lane_lines_stds = lane_lines_stds.reshape(-1, 4, 33, 2)  # (N, 4, 33, 2)

    inner_left_lane_std = lane_lines_stds[:, 1, :, :]  # (N, 33, 2)
    inner_right_lane_std = lane_lines_stds[:, 2, :, :]  # (N, 33, 2)

    lanelines_l_loss = path_laplacian_nll_loss(inner_left_lane, inner_left_lane_pred, inner_left_lane_std)
    lanelines_r_loss = path_laplacian_nll_loss(inner_right_lane, inner_right_lane_pred, inner_right_lane_std)

    # MHP loss
    lanelines_head_loss = torch.stack([lanelines_l_loss, lanelines_r_loss]).T

    lanelines_head_loss = lanelines_head_loss.sum(dim=1).mean()

    return lanelines_head_loss

def leads_loss(leads_pred, lead_prob_pred, leads_gt, leads_prob_gt, device, gai_1, gai_2):

    batch_size = leads_pred.shape[0]

    leads = leads_pred.reshape(-1, 2, 51)
    lead1_pred = leads[:, 0, :-3].reshape(-1, 2, 6, 4)
    lead2_pred = leads[:, 1, :-3].reshape(-1, 2, 6, 4)
    lead_pred_prob_s0 = leads[:, :, -3]
    lead_pred_prob_s1 = leads[:, :, -2]
    lead_prob_pred = torch.sigmoid(lead_prob_pred)

    mean_pred_lead1, std_pred_lead1 = mean_std(lead1_pred)
    mean_pred_lead2, std_pred_lead2 = mean_std(lead2_pred)

    lead1_gai_loss_s0 = gai_1(mean_pred_lead1[:,0,:1], leads_gt[:,0,:1], std_pred_lead1[:,0,:1]) #lead distance
    lead2_gai_loss_s0 = gai_2(mean_pred_lead2[:,0,:1], leads_gt[:,0,:1], std_pred_lead2[:,0,:1])

    lead1_loss_s0 = path_laplacian_nll_loss(leads_gt[:,:,1:], mean_pred_lead1[:,:1,2:], std_pred_lead1[:,:1,2:]) #speed, accel
    lead2_loss_s0 = path_laplacian_nll_loss(leads_gt[:,:,1:], mean_pred_lead2[:,:1,2:], std_pred_lead2[:,:1,2:])

    lead1_loss_s0 += lead1_gai_loss_s0
    lead2_loss_s0 += lead2_gai_loss_s0

    mean_pred_lead1_s1 = torch.cat((mean_pred_lead1[:,1:2,:1],mean_pred_lead1[:,1:2,2:]),dim=2)  # instead of y position
    mean_pred_lead2_s1 = torch.cat((mean_pred_lead2[:,1:2,:1],mean_pred_lead2[:,1:2,2:]),dim=2)
    std_pred_lead1_s1 = torch.cat((std_pred_lead1[:,1:2,:1],std_pred_lead1[:,1:2,2:]),dim=2)
    std_pred_lead2_s1 = torch.cat((std_pred_lead2[:,1:2,:1],std_pred_lead2[:,1:2,2:]),dim=2)

    lead1_loss_s1 = path_laplacian_nll_loss(leads_gt, mean_pred_lead1_s1, std_pred_lead1_s1)
    lead2_loss_s1 = path_laplacian_nll_loss(leads_gt, mean_pred_lead2_s1, std_pred_lead2_s1)

    # MHP loss
    lead_head_loss_s0 = torch.stack([lead1_loss_s0, lead2_loss_s0]).T
    lead_head_loss_s1 = torch.stack([lead1_loss_s1, lead2_loss_s1]).T

    idx_s0 = torch.argmin(lead_head_loss_s0, dim=1)
    idx_s1 = torch.argmin(lead_head_loss_s1, dim=1)
    far_lead_idx = leads_gt[:,0,0] >= 131.0 # treat lead_dis >= 131 as no lead
    batch_num_lead = float((~far_lead_idx).sum().item())
    batch_num_lead = 1.0 if (batch_num_lead < 1.0) else batch_num_lead
    wo_lead_idx = leads_prob_gt[:,0] == 0

    best_lead_mask_s0 = torch.zeros((batch_size, 2), device=device)
    best_lead_mask_s1 = torch.zeros((batch_size, 2), device=device)
    mask_s0 = torch.full((batch_size, 2), 1.0, device=device)
    mask_s1 = torch.full((batch_size, 2), 1.0, device=device)
    best_lead_mask_s0[torch.arange(idx_s0.shape[0]), idx_s0] = 1
    best_lead_mask_s0[far_lead_idx] = 0.5
    best_lead_mask_s1[torch.arange(idx_s1.shape[0]), idx_s1] = 1
    best_lead_mask_s1[far_lead_idx] = 0.5
    mask_s0[far_lead_idx] = 1e-6
    mask_s1[far_lead_idx] = 1e-6

    lead_perhead_loss_s0 = torch.mul(lead_head_loss_s0, mask_s0)
    lead_perhead_loss_s0 = lead_perhead_loss_s0.sum() / batch_num_lead
    lead_perhead_loss_s1 = torch.mul(lead_head_loss_s1, mask_s1)
    lead_perhead_loss_s1 = lead_perhead_loss_s1.sum() / batch_num_lead
    lead_perhead_loss = lead_perhead_loss_s0 + lead_perhead_loss_s1

    cross_entropy_loss = CrossEntropyLoss(reduction='none')

    lead_pred_prob_loss_s0 = cross_entropy_loss(lead_pred_prob_s0, best_lead_mask_s0)
    lead_pred_prob_loss_s0 = torch.where(far_lead_idx , lead_pred_prob_loss_s0 * 1e-6, lead_pred_prob_loss_s0)
    lead_pred_prob_loss_s0 = lead_pred_prob_loss_s0.sum() / batch_num_lead
    lead_pred_prob_loss_s1 = cross_entropy_loss(lead_pred_prob_s1, best_lead_mask_s1)
    lead_pred_prob_loss_s1 = torch.where(far_lead_idx , lead_pred_prob_loss_s1 * 1e-6, lead_pred_prob_loss_s1)
    lead_pred_prob_loss_s1 = lead_pred_prob_loss_s1.sum() / batch_num_lead
    lead_pred_prob_loss = lead_pred_prob_loss_s0 + lead_pred_prob_loss_s1

    bce_loss = BCELoss(reduction='none')
    lead_prob_loss = bce_loss(lead_prob_pred, leads_prob_gt).sum(dim=1)
    lead_prob_loss = lead_prob_loss.mean()

    lead_loss = lead_perhead_loss + lead_pred_prob_loss + lead_prob_loss
    return lead_loss, lead_perhead_loss, lead_pred_prob_loss, lead_prob_loss

def desire_loss(meta_pred, desire_state_pred, desire_gt, device):
    desire_prob = meta_pred[:, 48:].reshape(-1, 4, 8)
    desire_gt = desire_gt.long()
    desire_gt = desire_gt.to(device)

    desire_prob_loss = 0.0
    cross_entropy_loss = CrossEntropyLoss(reduction='none')
    for time_i in range(4):
        single_loss = cross_entropy_loss(desire_prob[:, time_i], desire_gt[:, time_i])
        single_loss = torch.where(desire_gt[:, time_i] == 0, single_loss * 0.115, single_loss) # balance loss 
        desire_prob_loss += single_loss.mean()

    desire_state_loss = cross_entropy_loss(desire_state_pred, desire_gt[:,0])
    desire_state_loss = torch.where(desire_gt[:, 0] == 0, desire_state_loss * 0.115, desire_state_loss) # balance loss 
    desire_state_loss = desire_state_loss.mean()

    desire_tol_loss = desire_prob_loss + desire_state_loss
    return desire_tol_loss, desire_prob_loss, desire_state_loss

def train(run, model, train_loader, val_loader, optimizer, scheduler, recurr_warmup, epoch, 
          log_frequency_steps, train_segment_for_viz, val_segment_for_viz, batch_size, awl, gai_1, gai_2, dice=None):

    recurr_input = torch.zeros(batch_size, 512, dtype=torch.float32, device=device, requires_grad=True)
    desire = torch.zeros(batch_size, 8, dtype=torch.float32, device=device)
    traffic_convention = torch.zeros(batch_size, 2, dtype=torch.float32, device=device)
    traffic_convention[:, 1] = 1
    model.train()
    awl.train()
    gai_1.train()
    gai_2.train()
    # dice.train()

    train_loss_accum = 0.0
    log_loss1 = 0.0
    log_loss2 = 0.0
    log_loss3 = 0.0
    log_loss4 = 0.0
    log_loss5 = 0.0
    log_loss6 = 0.0
    log_loss7 = 0.0
    log_loss8 = 0.0
    log_awl_loss = 0.0
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

        stacked_frames, gt_plans, gt_lanelines, gt_leads, gt_leads_prob, gt_desire, segments_finished = batch
        segments_finished = torch.all(segments_finished)

        loss, loss1, loss2, loss3, loss4, loss5, loss6, loss7, loss8, recurr_input, awl_loss = train_batch(run, model, optimizer, stacked_frames, gt_plans, gt_lanelines,
                        gt_leads, gt_leads_prob, gt_desire, traffic_convention, recurr_input, device, timings, should_backprop=should_backprop,
                        awl=awl, gai_1=gai_1, gai_2=gai_2, dice=dice)

        train_batch_time = multitimings.end('train_batch')
        fps = batch_size * seq_len / train_batch_time
        printf(f"> Batch trained: {train_batch_time:.2f}s (FPS={fps:.2f}).")

        if segments_finished:
            # reset the hidden state for new segments
            printf('Resetting hidden state.')
            recurr_input = recurr_input.zero_().detach()

        train_loss_accum += loss
        log_loss1 += loss1
        log_loss2 += loss2
        log_loss3 += loss3
        log_loss4 += loss4
        log_loss5 += loss5
        log_loss6 += loss6
        log_loss7 += loss7
        log_loss8 += loss8
        log_awl_loss += awl_loss

        if should_run_valid:
            #with Timing(timings, 'visualize_preds'):
            #    visualize_predictions(model, device, train_segment_for_viz, val_segment_for_viz)
            with Timing(timings, 'validate'):
                val_loss = validate(model, val_loader, batch_size, device, awl, gai_1, gai_2, dice)

            scheduler.step(val_loss.item())

            checkpoint_save_file = 'commaitr' + date_it + str(val_loss) + '_' + str(epoch+1) + ".pth"
            checkpoint_save_path = os.path.join(checkpoints_dir, checkpoint_save_file)
            torch.save(model.state_dict(), checkpoint_save_path)
            model.train()
            awl.train()
            gai_1.train()
            gai_2.train()

            wandb.log({
                'validation_loss': val_loss,
            }, commit=False)

        if should_log_train:
            timings['forward_pass']['time'] *= seq_len
            timings['path_plan_loss']['time'] *= seq_len
            timings['lanelines_loss']['time'] *= seq_len
            timings['leads_loss']['time'] *= seq_len
            timings['desire_loss']['time'] *= seq_len
            
            running_loss = train_loss_accum.item() / log_frequency_steps
            plan_loss = log_loss1.item() / log_frequency_steps
            plan_pb_loss = log_loss2.item() / log_frequency_steps
            leads_loss = log_loss3.item() / log_frequency_steps
            leads_pb_loss = log_loss4.item() / log_frequency_steps
            pb_lead_loss = log_loss5.item() / log_frequency_steps
            desire_pb_loss = log_loss6.item() / log_frequency_steps
            desire_st_loss = log_loss7.item() / log_frequency_steps
            laneline_loss = log_loss8.item() / log_frequency_steps
            awl_loss_wandb = log_awl_loss.item() / log_frequency_steps

            printf()
            printf(f'Epoch {epoch+1}/{epochs}. Done {tr_it+1} steps of ~{train_loader_len}. Running loss: {running_loss:.4f}')
            pprint_stats(timings)
            printf()

            wandb.log({
                'epoch': epoch,
                'train_tol_loss': running_loss,
                'train_plan_loss': plan_loss,
                'train_plan_pb_loss': plan_pb_loss,
                'train_leads_loss': leads_loss,
                'train_leads_pb_loss': leads_pb_loss,
                'train_pb_lead_loss': pb_lead_loss,
                'train_desire_pb_loss': desire_pb_loss,
                'train_desire_st_loss': desire_st_loss,
                'train_laneline_loss': laneline_loss, 
                'train_awl_loss' : awl_loss_wandb,
                'lr': scheduler.optimizer.param_groups[0]['lr'],
                **{f'time_{k}': v['time'] / v['count'] for k, v in timings.items()}
            }, commit=True)

            timings = dict()
            train_loss_accum = 0.0
            log_loss1 = 0.0
            log_loss2 = 0.0
            log_loss3 = 0.0
            log_loss4 = 0.0
            log_loss5 = 0.0
            log_loss6 = 0.0
            log_loss7 = 0.0
            log_loss8 = 0.0
            log_awl_loss = 0.0

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

def validate(model, data_loader, batch_size, device, awl, gai_1, gai_2, dice=None):

    model.eval()
    awl.eval()
    gai_1.eval()
    gai_2.eval()
    # dice.eval()
    # saving memory by not accumulating activations
    with torch.no_grad():

        val_loss = 0.0

        printf(">>>>>validating<<<<<<<")
        val_itr = None
        recurr_input = torch.zeros(batch_size, 512, dtype=torch.float32, device=device, requires_grad=False)
        for val_itr, val_batch in enumerate(data_loader):

            val_stacked_frames, val_plans, val_lanelines, val_leads, val_leads_prob, val_desire, segments_finished = val_batch
            segments_finished = torch.all(segments_finished)

            batch_loss, recurr_input = validate_batch(model, val_stacked_frames, val_plans, val_lanelines, val_leads, val_leads_prob, val_desire, recurr_input, device, awl, gai_1, gai_2, dice=None)
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

def train_batch(run, model, optimizer, stacked_frames, gt_plans, gt_lanelines, gt_leads, gt_leads_prob, gt_desire, traffic_convention, recurr_input, device, timings, should_backprop=True, awl=None, gai_1=None, gai_2=None, dice=None):
    batch_size_empirical = stacked_frames.shape[0]
    seq_len = stacked_frames.shape[1]

    with Timing(timings, 'inputs_to_gpu'):
        stacked_frames = stacked_frames.to(device).float()  # -- (batch_size, seq_len, 12, 128, 256)
        gt_plans = gt_plans.to(device)  # -- (batch_size,seq_len,5,2,33,15) -> new (batch_size,seq_len,33,15)
        #gt_plans_probs = gt_plans_probs.to(device)  # -- (batch_size,seq_len,5,1)
        gt_lanelines = gt_lanelines.to(device) # -- (batch_size,seq_len,2,33,2)
        gt_leads = gt_leads.to(device) # -- (batch_size,seq_len,1,3)
        gt_leads_prob = gt_leads_prob.to(device)# -- (batch_size,seq_len,3)
        #gt_desire # -- (batch_size,seq_len,4)

    optimizer.zero_grad(set_to_none=True)

    #batch_loss = 0.0
    batch_plan_loss = 0.0
    batch_plan_pb_loss = 0.0
    batch_leads_loss = 0.0
    batch_leads_pb_loss = 0.0
    batch_pb_lead_loss = 0.0
    batch_desire_pb_loss = 0.0
    batch_desire_st_loss = 0.0
    batch_awl_loss = 0.0
    batch_laneline_loss = 0.0
    prev_desire = torch.ones((gt_desire.shape[0]), dtype=torch.float32) * 7

    for i in range(seq_len):
        # time.sleep(1)
        desire_mask = (gt_desire[:,i,0] != prev_desire)
        desire_idx = torch.nonzero(desire_mask).squeeze()
        one_hot_desire = torch.eye(8, dtype=torch.float32)[gt_desire[:,i,0].long()]
        desire_input = torch.zeros((gt_desire.shape[0], 8), dtype=torch.float32)
        desire_input[desire_idx] = one_hot_desire[desire_idx].squeeze()
        prev_desire = gt_desire[:,i,0]
        desire_input = desire_input.to(device)

        inputs_to_pretained_model = {"input_imgs": stacked_frames[:, i, :, :, :],
                                     "desire": desire_input,
                                     "traffic_convention": traffic_convention,
                                     'initial_state': recurr_input.clone()  # TODO: why are we cloning recurr_input in 3 places (here, line 428 and line 439?
                                     }

        with Timing(timings, 'forward_pass'):
            outputs = model(**inputs_to_pretained_model)  # -- > [32,6472]

        plan_predictions = outputs[:, :4955].clone()  # -- > [32,4955]
        lanelines_predictions = outputs[:, 4955:5483].clone()  # -- > 528 lanelines
        #outputs[:, 5483:5491].clone()  # -- > 8 lanelines_prob
        #outputs[:, 5491:5755].clone()  # -- > 264 road-edges
        leads_predictions = outputs[:, 5755:5857].clone()  # -- > 102 leads
        lead_porb_predictions = outputs[:, 5857:5860].clone()  # -- > 3 lead_prob
        desire_state_predictions = outputs[:, 5860:5868].clone()  # -- > 8 desire_state
        meta_predictions = outputs[:, 5868:5948].clone()  # -- > 80 meta
        #outputs[:, 5948:5960].clone()  # -- > 12 pose
        recurr_out = outputs[:, 5960:].clone()  # -- > [32,512] important to refeed state of GRU

        with Timing(timings, 'path_plan_loss'):
            loss_func = plan_distill_loss if run.config.distill else plan_mhp_loss
            _, sg_plan_loss, sg_plan_pb_loss = loss_func(plan_predictions, gt_plans[:, i, :, :], device)

        with Timing(timings, 'lanelines_loss'):
            step_lanelines_loss = lanelines_loss(lanelines_predictions, gt_lanelines[:, i, :, :, :], device, dice=None)

        with Timing(timings, 'leads_loss'):
            _, sg_leads_loss, sg_leads_pb_loss, sg_pb_lead_loss = leads_loss(leads_predictions,lead_porb_predictions, gt_leads[:, i, :, :], gt_leads_prob[:,i,:], device, gai_1, gai_2)

        with Timing(timings, 'desire_loss'):
            _, sg_desire_pb_loss, sg_desire_st_loss = desire_loss(meta_predictions, desire_state_predictions, gt_desire[:,i,:], device)

        if i == seq_len - 1:
            # final hidden state in sequence, no need to backpropagate it through time
            pass
        else:
            recurr_input = recurr_out.clone()

        #batch_loss += (single_plan_loss + step_leads_loss + step_desire_loss)
        batch_plan_loss += sg_plan_loss
        batch_plan_pb_loss += sg_plan_pb_loss
        batch_leads_loss += sg_leads_loss
        batch_leads_pb_loss += sg_leads_pb_loss
        batch_pb_lead_loss += sg_pb_lead_loss
        batch_desire_pb_loss += sg_desire_pb_loss
        batch_desire_st_loss += sg_desire_st_loss
        batch_laneline_loss += step_lanelines_loss
        batch_awl_loss += awl(sg_plan_pb_loss, sg_leads_pb_loss, sg_pb_lead_loss, sg_desire_pb_loss, sg_desire_st_loss)

    #complete_batch_loss = batch_loss / seq_len / batch_size_empirical  # mean of losses over batches of sequences
    batch_plan_loss /= seq_len
    batch_plan_pb_loss /= seq_len
    batch_leads_loss /= seq_len
    batch_leads_pb_loss /= seq_len
    batch_pb_lead_loss /= seq_len
    batch_desire_pb_loss /= seq_len
    batch_desire_st_loss /= seq_len
    batch_laneline_loss /= seq_len
    batch_awl_loss /= seq_len

    if awl is not None :
        complete_batch_loss = batch_plan_loss + batch_leads_loss + batch_laneline_loss + batch_awl_loss
        # complete_batch_loss = batch_awl_loss
    else :
        complete_batch_loss = batch_plan_loss + batch_plan_pb_loss + batch_leads_loss + batch_leads_pb_loss + batch_pb_lead_loss + batch_desire_pb_loss + batch_desire_st_loss + batch_laneline_loss

    if should_backprop:
        with Timing(timings, 'backward_pass'):
            complete_batch_loss.backward(retain_graph=True)

    with Timing(timings, 'clip_gradients'):
        torch.nn.utils.clip_grad_norm_(model.parameters(), run.config.grad_clip)

    with Timing(timings, 'optimize_step'):
        optimizer.step()

    loss = complete_batch_loss.detach()  # loss for one iteration
    loss1 = batch_plan_loss.detach()
    loss2 = batch_plan_pb_loss.detach()
    loss3 = batch_leads_loss.detach()
    loss4 = batch_leads_pb_loss.detach()
    loss5 = batch_pb_lead_loss.detach()
    loss6 = batch_desire_pb_loss.detach()
    loss7 = batch_desire_st_loss.detach()
    loss8 = batch_laneline_loss.detach()
    awl_loss = batch_awl_loss.detach()

    return loss, loss1, loss2, loss3, loss4, loss5, loss6, loss7, loss8, recurr_out.detach(), awl_loss

def validate_batch(model, val_stacked_frames, val_plans, val_lanelines, val_leads, val_leads_prob, val_desire, recurr_input, device, awl, gai_1, gai_2, dice=None):
    batch_size = val_stacked_frames.shape[0]
    seq_len = val_stacked_frames.shape[1]

    #desire = torch.zeros(batch_size, 8, dtype=torch.float32, device=device)
    traffic_convention = torch.zeros(batch_size, 2, dtype=torch.float32, device=device)
    traffic_convention[:, 1] = 1

    val_input = val_stacked_frames.float().to(device)
    val_label_path = val_plans.to(device)
    val_lanelines = val_lanelines.to(device)
    val_leads = val_leads.to(device)
    val_leads_prob = val_leads_prob.to(device)
    #val_label_path_prob = val_plans_probs.to(device)
    prev_desire = torch.ones((val_desire.shape[0]), dtype=torch.float32) * 7

    val_batch_loss = 0.0

    for i in range(seq_len):
        desire_mask = (val_desire[:,i,0] != prev_desire)
        desire_idx = torch.nonzero(desire_mask).squeeze()
        one_hot_desire = torch.eye(8, dtype=torch.float32)[val_desire[:,i,0].long()]
        val_desire_input = torch.zeros((val_desire.shape[0], 8), dtype=torch.float32)
        val_desire_input[desire_idx] = one_hot_desire[desire_idx].squeeze()
        prev_desire = val_desire[:,i,0]
        val_desire_input = val_desire_input.to(device)
        val_inputs_to_pretained_model = {"input_imgs": val_input[:, i, :, :, :],
                                         "desire": val_desire_input,
                                         "traffic_convention": traffic_convention,
                                         "initial_state": recurr_input}

        val_outputs = model(**val_inputs_to_pretained_model)  # --> [32,6472]
        recurr_input = val_outputs[:, 5960:].clone()  # --> [32,512] important to refeed state of GRU
        val_path_prediction = val_outputs[:, :4955].clone()  # --> [32,4955]
        val_lanelines_predictions = val_outputs[:, 4955:5483].clone()  # -- > 528 lanelines
        #val_outputs[:, 5483:5491].clone()  # -- > 8 lanelines_prob
        #val_outputs[:, 5491:5755].clone()  # -- > 264 road-edges
        val_leads_predictions = val_outputs[:, 5755:5857].clone()  # -- > 102 leads
        val_lead_porb_predictions = val_outputs[:, 5857:5860].clone()  # -- > 3 lead_prob
        val_desire_state_predictions = val_outputs[:, 5860:5868].clone()  # -- > 8 desire_state
        val_meta_predictions = val_outputs[:, 5868:5948].clone()  # -- > 80 meta

        loss_func = plan_distill_loss if run.config.distill else plan_mhp_loss
        #single_val_loss,_,_ = loss_func(val_path_prediction, val_label_path[:, i, :, :], device)
        _, sg_plan_loss, sg_plan_pb_loss = loss_func(val_path_prediction, val_label_path[:, i, :, :], device)
        single_lanelines_loss = lanelines_loss(val_lanelines_predictions, val_lanelines[:, i, :, :, :], device, dice)
        #single_leads_loss,_,_,_ = leads_loss(val_leads_predictions, val_lead_porb_predictions, val_leads[:, i, :, :], val_leads_prob[:,i,:], device)
        _, sg_leads_loss, sg_leads_pb_loss, sg_pb_lead_loss = leads_loss(val_leads_predictions, val_lead_porb_predictions, val_leads[:, i, :, :], val_leads_prob[:,i,:], device, gai_1, gai_2)
        #single_desire_loss,_,_ = desire_loss(val_meta_predictions, val_desire_state_predictions, val_desire[:,i,:], device)
        _, sg_desire_pb_loss, sg_desire_st_loss = desire_loss(val_meta_predictions, val_desire_state_predictions, val_desire[:,i,:], device)
        #val_batch_loss += (single_val_loss + single_leads_loss + single_desire_loss)
        val_batch_loss += sg_plan_loss +_sg_leads_loss + single_lanelines_loss + awl(sg_plan_pb_loss, sg_leads_pb_loss, sg_pb_lead_loss, sg_desire_pb_loss, sg_desire_st_loss)

    #val_batch_loss = val_batch_loss / seq_len / batch_size
    val_batch_loss = val_batch_loss / seq_len
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

    # chackpoints directory and final model directory
    checkpoints_dir = './nets/first_phase_checkpoint'
    result_model_dir = './nets/first_phase'
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
    
    pathplan_layer_names = ["Conv_785", "Conv_797", "Conv_809", "Conv_832", #RepAdapter
                            "Conv_844", "Conv_856", "Conv_868", "Conv_891", #RepAdapter
                            "Gemm_959", "Gemm_981","Gemm_983","Gemm_1036", #plan
                            "Gemm_961", "Gemm_986","Gemm_988","Gemm_1037", #leads
                            "Gemm_963", "Gemm_991","Gemm_993","Gemm_1038", #lead_prob
                            "Gemm_969", "Gemm_1006","Gemm_1008","Gemm_1041", #outer_left_lane
                            "Gemm_971", "Gemm_1011","Gemm_1013","Gemm_1042", #left_lane
                            "Gemm_973", "Gemm_1016","Gemm_1018","Gemm_1043", #right_lane
                            "Gemm_975", "Gemm_1021","Gemm_1023","Gemm_1044", #outer_right_lane
                            "Gemm_979", "Gemm_1031","Gemm_1033","Gemm_1046", #desire_state
                            "Gemm_912", "Gemm_921","Gemm_923","Gemm_932", #desire_prob
                            ]

    # wandb init
    run = wandb.init(entity="jackie890621", project="openpilot-pipeline-Repadapter", name=train_run_name, mode='offline' if args.no_wandb else 'online')
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

    comma_model = load_trainable_model(path_to_supercombo, trainable_layers=pathplan_layer_names)
    with open("./model_freeze.txt",'w') as freeze_txt:
        for name, param in comma_model.named_parameters():
            freeze_txt.write(f'{name} : {param.requires_grad}\n')
            #print(f'{name} : {param.requires_grad}')

    # weight input for phase training
    # pth_path = "/home/t2-503-4090/QianXi/Openpilot_BalancedRegression_Adapter/train/nets/701_2phase_comma2k19/0701_laneline.pth"
    # comma_model.load_state_dict(torch.load(pth_path))
    comma_model = comma_model.to(device)
    awl = AutomaticWeightedLoss(num=5)
    awl = awl.to(device)

    gai_1 = GAILoss("gmm_131.pkl")
    gai_1 = gai_1.to(device)

    gai_2 = GAILoss("gmm_131.pkl")
    gai_2 = gai_2.to(device)

    # dice = DiceBCELoss()

    wandb.watch(comma_model) # Log the gradients  

    param_group = comma_model.parameters()
    optimizer = topt.Adam([{'params': param_group},{'params': awl.parameters()},{'params': gai_1.parameters()},{'params': gai_2.parameters()}], lr, weight_decay=l2_lambda)
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
                        train_segment_for_viz, val_segment_for_viz, batch_size, awl, gai_1, gai_2)

        result_model_save_path = os.path.join(result_model_dir, train_run_name + '.pth')
        torch.save(comma_model.state_dict(), result_model_save_path)
        printf("Saved trained model")
        printf("training_finished")

    sys.exit(0)