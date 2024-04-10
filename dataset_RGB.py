import os
import numpy as np
from torch.utils.data import Dataset
import torch
from PIL import Image
import torchvision.transforms.functional as TF
from pdb import set_trace as stx
import random
import os, sys, math, random, glob, cv2, h5py, logging, random
import utils
import torch.utils.data as data

from torch.utils.data.sampler import Sampler
from torch.utils.data import DataLoader



def binary_events_to_voxel_grid(events, num_bins, width, height):
    """
    Build a voxel grid with bilinear interpolation in the time domain from a set of events.

    :param events: a [N x 4] NumPy array containing one event per row in the form: [timestamp, x, y, polarity]
    :param num_bins: number of bins in the temporal axis of the voxel grid
    :param width, height: dimensions of the voxel grid
    """

    assert(events.shape[1] == 4)
    assert(num_bins > 0)
    assert(width > 0)
    assert(height > 0)

    voxel_grid = np.zeros((num_bins, height, width), np.float32).ravel()

    # normalize the event timestamps so that they lie between 0 and num_bins
    last_stamp = events[-1, 0]
    first_stamp = events[0, 0]
    deltaT = last_stamp - first_stamp

    if deltaT == 0:
        deltaT = 1.0
    # print('tis',events[:, 0])

    events[:, 0] = (num_bins - 1) * (events[:, 0] - first_stamp) / deltaT
    ts = events[:, 0]
    xs = events[:, 1].astype(np.int)
    ys = events[:, 2].astype(np.int)
    pols = events[:, 3]

    pols[pols == 0] = -1  # polarity should be +1 / -1,这里没有问题

    tis = ts.astype(np.int)
    # print('tis',tis)
    dts = ts - tis

    vals_left = pols * (1.0 - dts)

    vals_right = pols * dts


    valid_indices = tis < num_bins
    # print('valid_indices',valid_indices)
    np.add.at(voxel_grid, xs[valid_indices] + ys[valid_indices] * width
              + tis[valid_indices] * width * height, vals_left[valid_indices])

    valid_indices = (tis + 1) < num_bins
    # print('valid_indices',valid_indices)
    np.add.at(voxel_grid, xs[valid_indices] + ys[valid_indices] * width
              + (tis[valid_indices] + 1) * width * height, vals_right[valid_indices])

    voxel_grid = np.reshape(voxel_grid, (num_bins, height, width))
    # print('voxel_grid',voxel_grid)

    return voxel_grid


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif'])

class DataLoaderTrain(Dataset):
    def __init__(self, rgb_dir, args):
        super(DataLoaderTrain, self).__init__()

        self.inp_filenames=[]
        self.args=args
        train_file_dirs=rgb_dir
        for file_item in train_file_dirs:
            print(file_item)
            with h5py.File(file_item, 'r') as f:
                blur_image_name_list = sorted(list(f['blur_images'].keys()))
                sharp_image_name_list = sorted(list(f['sharp_images'].keys()))
                event_image_name_list = sorted(list(f['event_frames'].keys()))
                img_num = len(blur_image_name_list)
                for i in range(img_num):
                    blur_img = np.asarray(f['blur_images'][blur_image_name_list[i]]).transpose([2, 0, 1])
                    sharp_img = np.asarray(f['sharp_images'][sharp_image_name_list[i]]).transpose([2, 0, 1])
                    event_frame = np.asarray(f['event_frames'][event_image_name_list[i]])

                    # input_img, input_event, target = utils.image_proess(blur_img, event_frame, sharp_img,
                    #                                                     self.args.TRAINING.TRAIN_PS, self.args)
                    data = (blur_img, event_frame, sharp_img)
                    self.inp_filenames.append(data)

        self.sizex       = len(self.inp_filenames)  # get the size of target


    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex
        data=self.inp_filenames[index_]
        input_img, input_event, target = utils.image_proess(data[0], data[1], data[2],
                                                            self.args.TRAINING.TRAIN_PS, self.args)
        data = (input_img, input_event, target)
        return data




class DataLoaderVal(Dataset):
    def __init__(self,rgb_dir, args):
        # super(DataLoaderTrain, self).__init__()
        self.args=args
        self.seqs = rgb_dir
        self.seqs_info = {}
        self.length = 0
        for i in range(len(self.seqs)):
            seq_info = {}
            seq_info['seq'] = self.seqs[i]
            f = h5py.File(self.seqs[i], 'r')
            length_temp=len(list(f['blur_images'].keys()))
            seq_info['length'] = length_temp
            self.length += length_temp
            self.seqs_info[i] = seq_info
        self.seqs_info['length'] = self.length
        self.seqs_info['num'] = len(self.seqs)
        print(self.seqs_info)

        self.DVS_stream_height = 720
        self.DVS_stream_width = 1280

    def __len__(self):
        return self.seqs_info['length'] - (self.args.unrolling_len - 1) * self.seqs_info['num']

    def __getitem__(self, idx):
        ori_idx = idx
        seq_idx, frame_idx = 0, 0
        blur_imgs, sharp_imgs, event_imgs = list(), list(), list()

        for i in range(self.seqs_info['num']):
            seq_length = self.seqs_info[i]['length'] - self.args.unrolling_len+1
            if idx - seq_length < 0:
                seq_idx = i
                frame_idx = idx
                break
            else:
                idx -= seq_length
        for i in range(self.args.unrolling_len):
            f = h5py.File(self.seqs[seq_idx], 'r')
            blur_image_name_list = sorted(list(f['blur_images'].keys()))
            sharp_image_name_list = sorted(list(f['sharp_images'].keys()))
            event_image_name_list = sorted(list(f['event_frames'].keys()))
            try:
                blur_img = np.asarray(f['blur_images'][blur_image_name_list[frame_idx + i]]).transpose([2, 0, 1])

                sharp_img = np.asarray(f['sharp_images'][sharp_image_name_list[frame_idx + i]]).transpose([2, 0, 1])
                event_frame = np.asarray(f['event_frames'][event_image_name_list[frame_idx + i]])

                blur_img = torch.from_numpy(blur_img)
                sharp_img = torch.from_numpy(sharp_img)
                event_frame = torch.from_numpy(event_frame)

                blur_imgs.append(blur_img)
                sharp_imgs.append(sharp_img)
                event_imgs.append(event_frame)
            except TypeError as err:
                print('Handling run-time error:', err)
                print('failed case: idx {}, seq_idx {}, frame_idx {}'.format(ori_idx, seq_idx, frame_idx))

        data = (blur_imgs, event_imgs, sharp_imgs)
        return data


class SubsetSequentialSampler(Sampler):

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)



def create_data_loader(data_set, opts, mode='train'):

    total_samples = opts.train_iters * opts.OPTIM.BATCH_SIZE


    num_epochs = int(math.ceil(float(total_samples) / len(data_set)))

    indices = np.random.permutation(len(data_set))
    indices = np.tile(indices, num_epochs)
    indices = indices[:total_samples]

    ### generate data sampler and loader
    sampler = SubsetSequentialSampler(indices)
    data_loader = DataLoader(dataset=data_set, num_workers=4,
                             batch_size=opts.OPTIM.BATCH_SIZE, sampler=sampler, pin_memory=True,shuffle=False,drop_last=False)

    return data_loader


class DataLoaderTrain_npz(data.Dataset):


    def __init__(self, rgb_dir, args):
        super(DataLoaderTrain_npz, self).__init__()
        self.args = args
        self.blur_img_path = os.path.join(rgb_dir, 'blur')
        print(self.blur_img_path)
        self.event_img_path = os.path.join(rgb_dir, 'event')
        self.shapr_img_path = os.path.join(rgb_dir, 'gt')

        inp_files_dirs = sorted(os.listdir(self.blur_img_path))

        self.sequences_list = inp_files_dirs
        logging.info('[%s] Total %d event sequences:' %
                     (self.__class__.__name__, len(self.sequences_list)))

        print(len(self.sequences_list))

    def __len__(self):
        return len(self.sequences_list)

    def __getitem__(self, index):
        file_item = self.sequences_list[index]

        blur_image_name_list = sorted(glob.glob(os.path.join(self.blur_img_path, file_item, '*.png')))

        sharp_image_name_list = sorted(glob.glob(os.path.join(self.shapr_img_path, file_item, '*.png')))
        event_image_name_list = sorted(glob.glob(os.path.join(self.event_img_path, file_item, '*.npz')))
        blur_num_images = len(blur_image_name_list)
        self.DVS_stream_height = 720
        self.DVS_stream_width = 1280

        # print(blur_num_images)
        start_index = random.randint(0, blur_num_images - 1)  #
        frame_index=start_index
        blur_img = cv2.imread(blur_image_name_list[frame_index])
        blur_img = np.float32(blur_img) / 255.0

        sharp_img = cv2.imread(sharp_image_name_list[frame_index])
        sharp_img = np.float32(sharp_img) / 255.0

        event = np.load(event_image_name_list[frame_index])
        if len(event['t'])==0:
            event_div_tensor = np.zeros((self.args.num_bins, self.DVS_stream_height, self.DVS_stream_width))
        else:
            event_window = np.stack((event['t'],event['x'],event['y'],event['p']),axis=1)
            event_div_tensor = binary_events_to_voxel_grid(event_window,
                                             num_bins=self.args.num_bins,
                                             width=self.DVS_stream_width,
                                             height=self.DVS_stream_height)


        event_frame = np.float32(event_div_tensor)

        event_frame = event_frame
        blur_img = blur_img.transpose([2, 0, 1])
        sharp_img = sharp_img.transpose([2, 0, 1])

        blur_img, event_frame, sharp_img = utils.image_proess(blur_img, event_frame, sharp_img,
                                                              self.args.TRAINING.TRAIN_PS, self.args)

        data=(blur_img, event_frame, sharp_img)

        return data

class DataLoaderTrain_npy(data.Dataset):

    def __init__(self, rgb_dir, args):
        super(DataLoaderTrain_npy, self).__init__()
        self.args = args
        self.blur_img_path = os.path.join(rgb_dir, 'blur')
        print(self.blur_img_path)
        self.event_img_path = os.path.join(rgb_dir, 'event')
        self.shapr_img_path = os.path.join(rgb_dir, 'gt')

        inp_files_dirs = sorted(os.listdir(self.blur_img_path))


        self.sequences_list = inp_files_dirs
        logging.info('[%s] Total %d event sequences:' %
                     (self.__class__.__name__, len(self.sequences_list)))

        print(len(self.sequences_list))

    def __len__(self):
        return len(self.sequences_list)

    def __getitem__(self, index):
        file_item = self.sequences_list[index]

        blur_image_name_list = sorted(glob.glob(os.path.join(self.blur_img_path, file_item, '*.png')))

        sharp_image_name_list = sorted(glob.glob(os.path.join(self.shapr_img_path, file_item, '*.png')))
        event_image_name_list = sorted(glob.glob(os.path.join(self.event_img_path, file_item, '*.npy')))
        blur_num_images = len(blur_image_name_list)
        DVS_stream_height = 720
        DVS_stream_width = 1280

        # print(blur_num_images)
        start_index = random.randint(0, blur_num_images - 1)  #
        frame_index=start_index
        blur_imgs, sharp_imgs, event_imgs = list(), list(), list()
        blur_img = cv2.imread(blur_image_name_list[frame_index])
        blur_img = np.float32(blur_img) / 255.0

        sharp_img = cv2.imread(sharp_image_name_list[frame_index])
        sharp_img = np.float32(sharp_img) / 255.0

        event = np.load(event_image_name_list[frame_index])
        event_frame = np.float32(event)

        event_frame = event_frame
        blur_img = blur_img.transpose([2, 0, 1])
        sharp_img = sharp_img.transpose([2, 0, 1])

        blur_img, event_frame, sharp_img = utils.image_proess(blur_img, event_frame, sharp_img,
                                                              self.args.TRAINING.TRAIN_PS, self.args)

        data=(blur_img, event_frame, sharp_img)

        return data

class DataLoaderVal_npz(Dataset):
    def __init__(self,rgb_dir, args):
        # super(DataLoaderTrain, self).__init__()
        self.rgb_dir=rgb_dir
        blur_img_path=os.path.join(rgb_dir, 'blur')
        event_img_path=os.path.join(rgb_dir, 'event')
        shapr_img_path=os.path.join(rgb_dir, 'gt')

        inp_files_dirs = sorted(os.listdir(blur_img_path))


        self.DVS_stream_height = 720
        self.DVS_stream_width = 1280

        self.args=args

        self.seqs = inp_files_dirs
        print('self.seqs',self.seqs)
        self.seqs_info = {}
        self.length = 0
        for i in range(len(self.seqs)):
            seq_info = {}
            seq_info['seq'] = self.seqs[i]
            blur_img_lists=sorted(glob.glob(os.path.join(blur_img_path,self.seqs[i], '*.png')))
            event_lists=sorted(glob.glob(os.path.join(event_img_path,self.seqs[i], '*.npz')))
            gt_img_lists=sorted(glob.glob(os.path.join(shapr_img_path,self.seqs[i], '*.png')))
            seq_info['blur'] = blur_img_lists
            seq_info['event'] = event_lists
            seq_info['gt'] = gt_img_lists
            length_temp=len(blur_img_lists)
            seq_info['length'] = length_temp
            self.length += length_temp
            self.seqs_info[i] = seq_info
        self.seqs_info['length'] = self.length
        self.seqs_info['num'] = len(self.seqs)

    def __len__(self):
        return self.seqs_info['length']
    def __getitem__(self, idx):
        ori_idx = idx
        seq_idx, frame_idx = 0, 0

        for i in range(self.seqs_info['num']):
            seq_length = self.seqs_info[i]['length'] - self.args.unrolling_len+1
            if idx - seq_length < 0:
                seq_idx = i
                frame_idx = idx
                break
            else:
                idx -= seq_length

        blur_img = cv2.imread(self.seqs_info[seq_idx]['blur'][frame_idx])
        blur_img = np.float32(blur_img) / 255.0
        blur_img = blur_img.transpose([2, 0, 1])

        event = np.load(self.seqs_info[seq_idx]['event'][frame_idx])
        if len(event['t'])==0:
            event_div_tensor = np.zeros((self.args.num_bins, self.DVS_stream_height, self.DVS_stream_width))
        else:
            event_window = np.stack((event['t'],event['x'],event['y'],event['p']),axis=1)
            event_div_tensor = binary_events_to_voxel_grid(event_window,
                                             num_bins=self.args.num_bins,
                                             width=self.DVS_stream_width,
                                             height=self.DVS_stream_height)

        event_frame = np.float32(event_div_tensor)

        sharp_img = cv2.imread(self.seqs_info[seq_idx]['gt'][frame_idx])
        sharp_img = np.float32(sharp_img) / 255.0

        sharp_img = sharp_img.transpose([2, 0, 1])

        blur_img = torch.from_numpy(blur_img)
        sharp_img = torch.from_numpy(sharp_img)
        event_frame = torch.from_numpy(event_frame)
        data=(blur_img,event_frame,sharp_img)

        return data


class DataLoaderVal_npy(Dataset):
    def __init__(self,rgb_dir, args):
        # super(DataLoaderTrain, self).__init__()
        self.rgb_dir=rgb_dir
        blur_img_path=os.path.join(rgb_dir, 'blur')
        event_img_path=os.path.join(rgb_dir, 'event')
        shapr_img_path=os.path.join(rgb_dir, 'gt')

        inp_files_dirs = sorted(os.listdir(blur_img_path))
        event_files_dirs = sorted(os.listdir(event_img_path))

        tar_files_dirs = sorted(os.listdir(shapr_img_path))

        self.DVS_stream_height = 720
        self.DVS_stream_width = 1280

        self.args=args

        self.seqs = inp_files_dirs
        print('self.seqs',self.seqs)
        self.seqs_info = {}
        self.length = 0
        for i in range(len(self.seqs)):
            seq_info = {}
            seq_info['seq'] = self.seqs[i]
            blur_img_lists=sorted(glob.glob(os.path.join(blur_img_path,self.seqs[i], '*.png')))
            event_lists=sorted(glob.glob(os.path.join(event_img_path,self.seqs[i], '*.npy')))
            gt_img_lists=sorted(glob.glob(os.path.join(shapr_img_path,self.seqs[i], '*.png')))
            seq_info['blur'] = blur_img_lists
            seq_info['event'] = event_lists
            seq_info['gt'] = gt_img_lists
            length_temp=len(blur_img_lists)
            seq_info['length'] = length_temp
            self.length += length_temp
            self.seqs_info[i] = seq_info
        self.seqs_info['length'] = self.length
        self.seqs_info['num'] = len(self.seqs)
        # print(self.seqs_info)

        # self.DVS_stream_height = 720
        # self.DVS_stream_width = 1280

    def __len__(self):
        return self.seqs_info['length']
    def __getitem__(self, idx):
        ori_idx = idx
        seq_idx, frame_idx = 0, 0
        blur_imgs, sharp_imgs, event_imgs = list(), list(), list()

        for i in range(self.seqs_info['num']):
            seq_length = self.seqs_info[i]['length'] - self.args.unrolling_len+1
            if idx - seq_length < 0:
                seq_idx = i
                frame_idx = idx
                break
            else:
                idx -= seq_length

        blur_img = cv2.imread(self.seqs_info[seq_idx]['blur'][frame_idx])
        blur_img = np.float32(blur_img) / 255.0
        blur_img = blur_img.transpose([2, 0, 1])

        # print(dvs_h,dvs_w)
        event = np.load(self.seqs_info[seq_idx]['event'][frame_idx])

        event_frame = np.float32(event)

        sharp_img = cv2.imread(self.seqs_info[seq_idx]['gt'][frame_idx])
        sharp_img = np.float32(sharp_img) / 255.0

        sharp_img = sharp_img.transpose([2, 0, 1])

        blur_img = torch.from_numpy(blur_img)
        sharp_img = torch.from_numpy(sharp_img)
        event_frame = torch.from_numpy(event_frame)
        data=(blur_img,event_frame,sharp_img)

        return data


class DataLoaderTest_npy(Dataset):
    def __init__(self,rgb_dir, file_dir,args):
        self.rgb_dir=rgb_dir
        blur_img_path=os.path.join(rgb_dir, 'blur')
        event_img_path=os.path.join(rgb_dir, 'event')
        shapr_img_path=os.path.join(rgb_dir, 'gt')

        inp_files_dirs = [file_dir]


        self.DVS_stream_height = 720
        self.DVS_stream_width = 1280

        self.args=args

        self.seqs = inp_files_dirs
        print('self.seqs',self.seqs)
        self.seqs_info = {}
        self.length = 0
        for i in range(len(self.seqs)):
            seq_info = {}
            seq_info['seq'] = self.seqs[i]
            blur_img_lists=sorted(glob.glob(os.path.join(blur_img_path,self.seqs[i], '*.png')))
            event_lists=sorted(glob.glob(os.path.join(event_img_path,self.seqs[i], '*.npy')))
            gt_img_lists=sorted(glob.glob(os.path.join(shapr_img_path,self.seqs[i], '*.png')))
            seq_info['blur'] = blur_img_lists
            seq_info['event'] = event_lists
            seq_info['gt'] = gt_img_lists
                # sorted(os.listdir(os.path.join(rgb_dir, 'blur',self.seqs[i],'images')))
            length_temp=len(blur_img_lists)
            seq_info['length'] = length_temp
            self.length += length_temp
            self.seqs_info[i] = seq_info
        self.seqs_info['length'] = self.length
        self.seqs_info['num'] = len(self.seqs)
        # print(self.seqs_info)

        # self.DVS_stream_height = 720
        # self.DVS_stream_width = 1280

    def __len__(self):
        return self.seqs_info['length']

    def __getitem__(self, idx):
        ori_idx = idx
        seq_idx, frame_idx = 0, 0
        blur_imgs, sharp_imgs, event_imgs = list(), list(), list()

        for i in range(self.seqs_info['num']):
            seq_length = self.seqs_info[i]['length'] - self.args.unrolling_len+1
            if idx - seq_length < 0:
                seq_idx = i
                frame_idx = idx
                break
            else:
                idx -= seq_length


        blur_img=cv2.imread(self.seqs_info[seq_idx]['blur'][frame_idx+i])
        blur_img = np.float32(blur_img) / 255.0
        blur_img=blur_img.transpose([2, 0, 1])

        # print(dvs_h,dvs_w)
        event=np.load(self.seqs_info[seq_idx]['event'][frame_idx+i])

        event_frame=np.float32(event)

        sharp_img=cv2.imread(self.seqs_info[seq_idx]['gt'][frame_idx+i])
        sharp_img = np.float32(sharp_img) / 255.0

        sharp_img=sharp_img.transpose([2, 0, 1])

        blur_img = torch.from_numpy(blur_img)
        sharp_img = torch.from_numpy(sharp_img)
        event_frame = torch.from_numpy(event_frame)
        print(event_frame.size())

        data=(blur_img,event_frame,sharp_img)

        return data



class DataLoaderTest_npz(Dataset):
    def __init__(self,rgb_dir, file_dir,args):
        self.rgb_dir=rgb_dir
        blur_img_path=os.path.join(rgb_dir, 'blur')
        event_img_path=os.path.join(rgb_dir, 'event')
        shapr_img_path=os.path.join(rgb_dir, 'gt')

        inp_files_dirs = [file_dir]


        self.DVS_stream_height = 720
        self.DVS_stream_width = 1280

        self.args=args

        self.seqs = inp_files_dirs
        print('self.seqs',self.seqs)
        self.seqs_info = {}
        self.length = 0
        for i in range(len(self.seqs)):
            seq_info = {}
            seq_info['seq'] = self.seqs[i]
            blur_img_lists=sorted(glob.glob(os.path.join(blur_img_path,self.seqs[i], '*.png')))
            event_lists=sorted(glob.glob(os.path.join(event_img_path,self.seqs[i], '*.npz')))
            gt_img_lists=sorted(glob.glob(os.path.join(shapr_img_path,self.seqs[i], '*.png')))
            seq_info['blur'] = blur_img_lists
            seq_info['event'] = event_lists
            seq_info['gt'] = gt_img_lists
                # sorted(os.listdir(os.path.join(rgb_dir, 'blur',self.seqs[i],'images')))
            length_temp=len(blur_img_lists)
            seq_info['length'] = length_temp
            self.length += length_temp
            self.seqs_info[i] = seq_info
        self.seqs_info['length'] = self.length
        self.seqs_info['num'] = len(self.seqs)

    def __len__(self):
        return self.seqs_info['length']

    def __getitem__(self, idx):
        seq_idx, frame_idx = 0, 0

        for i in range(self.seqs_info['num']):
            seq_length = self.seqs_info[i]['length']
            if idx - seq_length < 0:
                seq_idx = i
                frame_idx = idx
                break
            else:
                idx -= seq_length


        blur_img=cv2.imread(self.seqs_info[seq_idx]['blur'][frame_idx+i])
        blur_img = np.float32(blur_img) / 255.0
        blur_img=blur_img.transpose([2, 0, 1])

        # print(dvs_h,dvs_w)
        event=np.load(self.seqs_info[seq_idx]['event'][frame_idx+i])
        if len(event['t'])==0:
            event_div_tensor = np.zeros((self.args.num_bins, self.DVS_stream_height, self.DVS_stream_width))
        else:
            event_window = np.stack((event['t'],event['x'],event['y'],event['p']),axis=1)
            event_div_tensor = binary_events_to_voxel_grid(event_window,
                                             num_bins=self.args.num_bins,
                                             width=self.DVS_stream_width,
                                             height=self.DVS_stream_height)
        event_frame=np.float32(event_div_tensor)

        sharp_img=cv2.imread(self.seqs_info[seq_idx]['gt'][frame_idx+i])
        sharp_img = np.float32(sharp_img) / 255.0

        sharp_img=sharp_img.transpose([2, 0, 1])

        blur_img = torch.from_numpy(blur_img)
        sharp_img = torch.from_numpy(sharp_img)
        event_frame = torch.from_numpy(event_frame)

        data = (blur_img, event_frame,sharp_img)
        return data

