import warnings
import joblib

warnings.filterwarnings("ignore", category=UserWarning, message='Length of IterableDataset')
warnings.filterwarnings("ignore", category=UserWarning,
                        message='The given NumPy array is not writeable, and PyTorch does not support non-writeable tensors')
warnings.filterwarnings("ignore", category=UserWarning, message='Using experimental implementation that allows \'batch_size > 1\'')

import sys
import numpy as np
from tqdm import tqdm
import h5py
import glob
from torch.utils.data import IterableDataset, DataLoader
import os
import cv2
import math
import torch
import time
from torch import multiprocessing

import subprocess

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # noqa
from utils_comma2k19 import bgr_to_yuv, transform_frames, printf, FULL_FRAME_SIZE, create_image_canvas, PATH_TO_CACHE, RADAR_TO_CAMERA  # noqa


MIN_SEGMENT_LENGTH = 1190

path_to_videos_cache = os.path.join(PATH_TO_CACHE, 'videos.txt')
path_to_plans_cache = os.path.join(PATH_TO_CACHE, 'plans.txt')

def load_transformed_video_org(path_to_segment, plot_img_width=640, plot_img_height=480, seq_len=1190):
    
    if os.path.exists(os.path.join(path_to_segment, 'video.hevc')):
        path_to_video = os.path.join(path_to_segment, 'video.hevc')
    elif os.path.exists(os.path.join(path_to_segment, 'fcamera.hevc')):
        path_to_video = os.path.join(path_to_segment, 'fcamera.hevc')
    else:
        raise Exception('No video file found in {}'.format(path_to_segment))

    zoom = FULL_FRAME_SIZE[0] / plot_img_width
    CALIB_BB_TO_FULL = np.asarray([
        [zoom, 0., 0.],
        [0., zoom, 0.],
        [0., 0., 1.]])

    segment_video = cv2.VideoCapture(path_to_video)

    rgb_frames = np.zeros((seq_len, plot_img_height, plot_img_width, 3), dtype=np.uint8)
    #yuv_frames = np.zeros((seq_len + 1, FULL_FRAME_SIZE[1]*3//2, FULL_FRAME_SIZE[0]), dtype=np.uint8)
    read_rgb_frames = np.zeros((seq_len + 1, FULL_FRAME_SIZE[1], FULL_FRAME_SIZE[0], 3), dtype=np.uint8)
    stacked_frames = np.zeros((seq_len, 12, 128, 256), dtype=np.uint8)

    ret, frame2 = segment_video.read()
    frame2 = frame2[270:-270,:]
    if not ret:
        print('Failed to read video from {}'.format(path_to_video))
        return None, None

    rgb_frame =  cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
    read_rgb_frames[0] = rgb_frame

    # start iteration from 1 because we already read 1 frame before
    for t_idx in range(1, seq_len + 1):

        ret, frame2 = segment_video.read()
        frame2 = frame2[270:-270,:]
        if not ret:
            print('Failed to read video from {}'.format(path_to_video))
            return None, None

        rgb_frame =  cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

        rgb_frames[t_idx-1] = create_image_canvas(rgb_frame, CALIB_BB_TO_FULL, plot_img_height, plot_img_width)
        read_rgb_frames[t_idx] = rgb_frame

    prepared_frames = transform_frames(read_rgb_frames)

    for i in range(seq_len):
        stacked_frames[i] = np.vstack(prepared_frames[i:i+2])[None].reshape(12, 128, 256)

    segment_video.release()
    return torch.from_numpy(stacked_frames).float(), rgb_frames

def load_transformed_video(path_to_segment, plot_img_width=640, plot_img_height=480, seq_len=5800, read_file=False):
    
    if read_file:
        path_to_video = path_to_segment
        name_mp4 = os.path.basename(path_to_segment) # xxxx.mp4
        file_name = os.path.splitext(name_mp4)[0] # xxxx
        dirname = os.path.dirname(path_to_segment)
        path_to_h5 = os.path.join(dirname, file_name+'.h5')
    elif os.path.exists(os.path.join(path_to_segment, 'viz.mp4')) and os.path.exists(os.path.join(path_to_segment, 'viz.h5')):
        path_to_video = os.path.join(path_to_segment, 'viz.mp4')
        path_to_h5 = os.path.join(path_to_segment, 'viz.h5')
    else:
        raise Exception('No viz.mp4 file found in {}'.format(path_to_segment))


    zoom = FULL_FRAME_SIZE[0] / plot_img_width
    CALIB_BB_TO_FULL = np.asarray([
        [zoom, 0., 0.],
        [0., zoom, 0.],
        [0., 0., 1.]])

    segment_video = cv2.VideoCapture(path_to_video)
    segment_h5 = h5py.File(path_to_h5, 'r')

    rgb_frames = np.zeros((seq_len, plot_img_height, plot_img_width, 3), dtype=np.uint8)
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

        rgb_frames[t_idx-1] = create_image_canvas(rgb_frame, CALIB_BB_TO_FULL, plot_img_height, plot_img_width)
        read_rgb_frames[t_idx] = rgb_frame

    prepared_frames = transform_frames(read_rgb_frames)

    for i in range(seq_len):
        stacked_frames[i] = np.vstack(prepared_frames[i:i+2])[None].reshape(12, 128, 256)

    frame_desire = segment_h5['desire_state'][()][:,0].astype(np.int8)

    segment_video.release()
    segment_h5.close()
    return torch.from_numpy(stacked_frames).float(), rgb_frames, frame_desire


def parse_affinity(cmd_output):
    ''' 
    Extracts the list of CPU ids from the `taskset -cp <pid>` command.
    example input : b"pid 38293's current affinity list: 0-3,96-99,108\n" 
    example output: [0,1,2,3,96,97,98,99,108]
    '''
    ranges_str = cmd_output.decode('utf8').split(': ')[-1].rstrip('\n').split(',')

    list_of_cpus = []
    for rng in ranges_str:
        is_range = '-' in rng

        if is_range:
            start, end = rng.split('-')
            rng_cpus = range(int(start), int(end)+1) # include end
            list_of_cpus += rng_cpus
        else:
            list_of_cpus.append(int(rng))

    return list_of_cpus


def set_affinity(pid, cpu_list):
    cmd = ['taskset', '-pc', ','.join(map(str, cpu_list)), str(pid)]
    subprocess.check_output(cmd, shell=False)


def get_affinity(pid):
    cmd = ['taskset', '-pc',  str(pid)]
    output = subprocess.check_output(cmd, shell=False)
    return parse_affinity(output)


def configure_worker(worker_id):
    '''
    Configures the worker to use the correct affinity.
    '''
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:
        worker_id = 0
        num_workers = 1
    else:
        worker_id = worker_info.id
        num_workers = worker_info.num_workers

    worker_pid = os.getpid()
    dataset = worker_info.dataset
    
    avail_cpus = get_affinity(worker_pid)
    validation_term = 1 if dataset.validation else 0  # support two loaders at a time
    offset = len(avail_cpus) - (num_workers * 2) # keep the first few cpus free (it seemed they were faster, important for BackgroundGenerator)
    cpu_idx = max(offset + num_workers * validation_term + worker_id, 0)

    # force the process to only use 1 core instead of all
    set_affinity(worker_pid, [avail_cpus[cpu_idx]])


class CommaDataset(IterableDataset):

    def __init__(self, recordings_basedir, batch_size, train_split=0.8, seq_len=32, validation=False, shuffle=False, seed=42):
        super(CommaDataset, self).__init__()
        """
        Dataloader for Comma model train. pipeline
        Summary:
            This dataloader can be used for intial testing and for proper training
            Images are converted into YUV 4:2:0 channels and brought to a calib frame of reff
            as used in the official comma pipeline.
        Args: ------------------
        """
        self.batch_size = batch_size
        self.recordings_basedir = recordings_basedir
        self.validation = validation
        self.train_split = train_split
        self.shuffle = shuffle
        self.seq_len = seq_len
        self.seed = seed

        if self.recordings_basedir is None or not os.path.exists(self.recordings_basedir):
            raise TypeError("recordings path is wrong")

        self.mp4_file_paths, self.gt_file_paths = self.get_paths_roach_split(self.recordings_basedir)
        n_segments = len(self.mp4_file_paths)
        printf("Total # segments", n_segments)

        full_index = list(range(n_segments))

        # shuffle full train+val together *once*
        if self.shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(full_index)

        split_idx = int(np.round(n_segments * self.train_split))

        if not self.validation:
            self.segment_indices = full_index[:split_idx]
        else:
            self.segment_indices = full_index[split_idx:]

        printf("Subset # segments:", len(self.segment_indices))

    # NOTE: this is a rough estimate (less or equal to the true value). Do NOT rely on this number.
    def __len__(self):
        batches_per_segment = MIN_SEGMENT_LENGTH // self.seq_len
        return len(self.segment_indices) * batches_per_segment // self.batch_size

    def __iter__(self):
        # shuffle data subset after each epoch
        if self.shuffle and not self.validation:
            rng = np.random.default_rng(self.seed)
            rng.shuffle(self.segment_indices)

        # support single & multi-processing
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        for segment_vidx in range(worker_id, len(self.segment_indices), num_workers):

            # retrieve true index of segment [0, 2331] using virtual index [0, 2331 * train_split]
            segment_idx = self.segment_indices[segment_vidx]

            segment_video = cv2.VideoCapture(self.mp4_file_paths[segment_idx])
            segment_gts = h5py.File(self.gt_file_paths[segment_idx], 'r')
            segment_length = segment_gts['plans'].shape[0]
            n_seqs = math.floor(segment_length / self.seq_len)

            _, frame2 = segment_video.read()  # initialize last frame
            frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
            
            # TODO: (for further optimization) check if model can handle images pre-processed through (shorter) path2.
            # path1 and path2 look identical when saved to a PNG, but have some 
            # structured differences (to see, print the flattened difference between the two)

            # path1: bgr -> yuv -> rgb
            # path2: bgr -> rgb
            # path1 = yuv_to_rgb(bgr_to_yuv(frame2))
            # path2 = bgr_to_rgb(frame2)
            # printf('diff between paths:', list((path1 - path2).flatten()))

            for sequence_idx in range(n_seqs): # with seq_len=200 this should run even fewer iterations than usual, so wtf?

                segment_finished = sequence_idx == n_seqs-1

                frame_seq = np.zeros((self.seq_len + 1, 874, 1164, 3), dtype=np.uint8)
                frame_seq[0] = frame2

                # start iteration from 1 because we already read 1 frame before
                for t_idx in range(1, self.seq_len + 1):
                    _, frame2 = segment_video.read()
                    frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
                    frame_seq[t_idx] = frame2

                prepared_frames = transform_frames(frame_seq)

                stacked_frame_seq = np.zeros((self.seq_len, 12, 128, 256), dtype=np.uint8)
                for i in range(self.seq_len):
                    stacked_frame_seq[i] = np.vstack(prepared_frames[i:i+2])[None].reshape(12, 128, 256)

                # shift slice by +1 to skip the 1st step which didn't see 2 stacked frames yet
                abs_t_indices = slice(sequence_idx*self.seq_len, (sequence_idx+1)*self.seq_len)
                gt_plan_seq = segment_gts['plans'][abs_t_indices]
                gt_lanelines_lr_seq = segment_gts['lanelines'][abs_t_indices]

                yield stacked_frame_seq, gt_plan_seq, gt_lanelines_lr_seq, segment_finished, worker_id

            segment_gts.close()
            segment_video.release()

    def get_segment_dirs(self, base_dir, gt_file_name='gt_distill.h5'):
        '''Get paths to segments that have ground truths.'''

        if os.path.exists(segments_cache := os.path.join(PATH_TO_CACHE, 'segments.txt')):
            with open(segments_cache, 'r') as f:
                segment_dirs = [line.strip() for line in f.readlines()]
        else:
            gt_files = sorted(glob.glob(base_dir + f'/**/{gt_file_name}', recursive=True))
            segment_dirs = sorted(list(set([os.path.dirname(f) for f in gt_files])))

        return segment_dirs

    def get_paths(self, base_dir, min_segment_len=1190):
        '''Get paths to videos and ground truths. Cache them for future reuse.'''

        os.makedirs(PATH_TO_CACHE, exist_ok=True)

        if os.path.exists(path_to_videos_cache) and os.path.exists(path_to_plans_cache):
            printf('Using cached paths to videos and GTs...')
            video_paths = []
            gt_paths = []
            with open(path_to_videos_cache, 'r') as f:
                video_paths = f.read().splitlines()

            with open(path_to_plans_cache, 'r') as f:
                gt_paths = f.read().splitlines()
                
        else:
            printf('Resolving paths to videos and GTs...')
            segment_dirs = self.get_segment_dirs(base_dir)

            # prevent duplicate writes
            with open(path_to_videos_cache, 'w'): pass
            with open(path_to_plans_cache, 'w'): pass

            gt_filename = 'gt_distill.h5'
            video_filenames = ['fcamera.hevc', 'video.hevc']

            video_paths = []
            gt_paths = []

            for segment_dir in tqdm(segment_dirs):

                gt_file_path = os.path.join(segment_dir, gt_filename)

                if not os.path.exists(gt_file_path):
                    printf(f'WARNING: not found {gt_filename} file in segment: {segment_dir}')

                    continue
               
                try:
                    gt_plan = h5py.File(gt_file_path, 'r')['plans']

                    if gt_plan.shape[0] >= min_segment_len:  # keep segments that have >= 1190 samples

                        video_files = os.listdir(segment_dir)
                        video_files = [file for file in video_files if file in video_filenames]

                        found_one_video = 0 <= len(video_files) <= 1

                        if found_one_video:
                            with open(path_to_videos_cache, 'a') as video_paths_f:
                                video_path = os.path.join(segment_dir, video_files[0])
                                video_paths.append(video_path)
                                video_paths_f.write(video_path + '\n')  # cache it

                            with open(path_to_plans_cache, 'a') as gt_paths_f:
                                gt_paths.append(gt_file_path)
                                gt_paths_f.write(gt_file_path + '\n')  # cache it
                        else:
                            printf(f'WARNING: found {len(video_files)} in segment: {segment_dir}')
                except :
                    continue

        return video_paths, gt_paths

    def get_paths_roach_split(self, base_dir):

        os.makedirs(PATH_TO_CACHE, exist_ok=True)
        if os.path.exists(path_to_videos_cache) and os.path.exists(path_to_plans_cache):
            printf('Using cached paths to videos and GTs(roach)...')
            video_paths = []
            gt_paths = []
            with open(path_to_videos_cache, 'r') as f:
                video_paths = f.read().splitlines()

            with open(path_to_plans_cache, 'r') as f:
                gt_paths = f.read().splitlines()

        else :
            printf('Resolving paths to videos and GTs(roach)...')
            video_paths = sorted(glob.glob(base_dir+'/*.mp4'))
            gt_paths = sorted(glob.glob(base_dir+'/*.h5'))

            with open(path_to_videos_cache, 'w') as video_paths_f:
                for file_path in video_paths:
                    video_paths_f.write(file_path + '\n')

            with open(path_to_plans_cache, 'w') as gt_paths_f:
                for file_path in gt_paths:
                    gt_paths_f.write(file_path + '\n')
            

        return video_paths, gt_paths

class BatchDataLoader:
    '''Assumes batch_size == num_workers to ensure same ordering of segments in each batch'''

    def __init__(self, loader, batch_size):
        self.loader = loader
        self.batch_size = batch_size

    def __iter__(self):
        bs = self.batch_size
        batch = [None] * bs
        current_bs = 0
        workers_seen = set()
        for d in self.loader:
            worker_id = d[-1]

            # this means there're fewer segments left than the size of the batch — drop the last ones
            if worker_id in workers_seen:
                printf(f'WARNING: sequence from worker:{worker_id} already seen in this batch. Dropping segments.')
                return  # FIXME: maybe pad the missing sequences with zeros and yield?

            batch[worker_id] = d
            current_bs += 1
            workers_seen.add(worker_id)

            if current_bs == bs:
                collated_batch = self.collate_fn(batch)
                yield collated_batch
                batch = [None] * bs
                current_bs = 0
                workers_seen = set()

    def collate_fn(self, batch):

        stacked_frames = torch.stack([item[0] for item in batch])
        gt_plan = torch.stack([item[1] for item in batch])
        gt_lanelines = torch.stack([item[2] for item in batch])
        segment_finished = torch.tensor([item[3] for item in batch])

        return stacked_frames, gt_plan, gt_lanelines, segment_finished

    def __len__(self):
        return len(self.loader)


class BackgroundGenerator(multiprocessing.Process):
    def __init__(self, generator):
        super(BackgroundGenerator, self).__init__() 
        # TODO: use prefetch factor instead of harcoded value
        self.queue = torch.multiprocessing.Queue(2)
        self.generator = generator
        self.start()

    def run(self):
        while True:
            for item in self.generator:

                # do not start (blocking) insertion into a full queue, just wait and then retry
                # this way we do not block the consumer process, allowing instant batch fetching for the model
                while self.queue.full():
                    time.sleep(2)

                self.queue.put(item)
            self.queue.put(None)

    def __iter__(self):
        try:
            next_item = self.queue.get()
            while next_item is not None:
                yield next_item
                next_item = self.queue.get()
        except (ConnectionResetError, ConnectionRefusedError) as err:
            printf('[BackgroundGenerator] Error:', err)
            self.shutdown()
            raise StopIteration

"""
Loader for visualization
"""

def prepare_frames(frame1, frame2):
    yuv_frame1 = bgr_to_yuv(frame1)
    yuv_frame2 = bgr_to_yuv(frame2)
    list_yuv_frame = [yuv_frame1, yuv_frame2]

    prepared_frames = transform_frames(list_yuv_frame)
    # print(prepared_frames[0].shape)
    stack_frames = np.zeros((1,12,128,256))
    stack_frames = (np.vstack((prepared_frames[0], prepared_frames[1]))).reshape(1,12,128,256)
    # print(stack_frames.shape)

    return stack_frames 
