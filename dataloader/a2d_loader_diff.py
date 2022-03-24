# import cv2
import random
from PIL import Image

from collections import Iterable

import numpy as np
import os.path as osp
import h5py
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as data
from .word_utils import Corpus
import scipy.io as scio
import glob

class DatasetNotFoundError(Exception):
    pass


class ResizeAnnotation:
    """Resize the largest of the sides of the annotation to a given size"""

    def __init__(self, size):
        if not isinstance(size, (int, Iterable)):
            raise TypeError("Got inappropriate size arg: {}".format(size))

        self.size = size

    def __call__(self, img):
        im_h, im_w = img.shape[-2:]
        scale_h, scale_w = self.size / im_h, self.size / im_w
        resized_h = int(np.round(im_h * scale_h))
        resized_w = int(np.round(im_w * scale_w))
        out = (
            F.interpolate(
                Variable(img).unsqueeze(0).unsqueeze(0),
                size=(resized_h, resized_w),
                mode="bilinear",
                align_corners=True,
            )
            .squeeze()
            .data
        )
        return out


class ReferDataset(data.Dataset):
    def __init__(
        self,
        data_root,
        dataset="a2d",
        transform=None,
        transform_orig=None,
        transform_flow=None,
        eval_ann_transform=None,
        split="train",
        max_query_len=20,
        glove_path="path_to_glove",
        image_dim=320,
        interval=2,
        save_result=False
    ):
        self.images = [] # []
        # ann_file = osp.join(data_root, dataset, f'{split}.txt') # './data/a2d/train.txt'
        if split == 'train':
            if dataset in ['a2d', 'jhmdb']:
                ann_file = osp.join(data_root, dataset, f'{split}.txt') # './data/a2d/train.txt'
            elif dataset == 'ytvos':
                ann_file = osp.join(data_root, dataset, f'{split}.txt')
            else:
                ann_file = osp.join(data_root, dataset,
                                   f'17/DAVIS/ImageSets/2017/{split}_17_full.txt')  # './data/a2d/train.txt'
        else:
            if dataset in ['a2d', 'jhmdb']:
                ann_file = osp.join(data_root, dataset, f'{split}.txt') # './data/a2d/train.txt'
            elif dataset == 'ytvos':
                if not save_result:
                    ann_file = osp.join(data_root, dataset, f'val.txt')
                else:
                    ann_file = osp.join(data_root, dataset, f'valid_new_pure.txt')  # for save
            else:
                ann_file = osp.join(data_root, dataset,
                                    f'17/DAVIS/ImageSets/2017/{split}_17_full.txt')  # './data/a2d/train.txt'
        with open(ann_file, 'r') as f:
            self.images = f.readlines()
        self.save_result = save_result
        self.transform_orig = transform_orig
        self.transform_flow = transform_flow
        self.data_root = data_root # "./data"
        self.dataset = dataset # "a2d"
        self.query_len = max_query_len # 25
        self.corpus = Corpus(glove_path)
        self.transform = transform
        self.eval_ann_transform = eval_ann_transform
        self.split = split # "train"
        if dataset == 'a2d':
            self.im_dir = osp.join(data_root, dataset, 'images') # './data/a2d/images'
            self.mask_dir = osp.join(data_root, dataset, 'masks') # './data/a2d/masks'
        elif dataset == 'jhmdb':
            self.im_dir = osp.join(data_root, dataset, 'Rename_Images')
            self.mask_dir = osp.join(data_root, dataset, 'puppet_mask')
        elif dataset == 'ytvos':
            self.im_dir = osp.join(data_root, dataset, split, 'JPEGImages')  # './data/a2d/images'
            self.mask_dir = osp.join(data_root, dataset, split, 'Annotations')  # './data/a2d/masks'
        elif dataset == 'davis':
            self.im_dir = osp.join(data_root, dataset, '17/DAVIS', 'JPEGImages/Full-Resolution')  # './data/a2d/images'
            self.mask_dir = osp.join(data_root, dataset, '17/DAVIS',
                                     'Annotations/Full-Resolution')  # './data/a2d/masks'
        else:
            raise RuntimeError(f'{dataset} does not exist!!!')
        self.image_dim = image_dim
        self.interval = interval

    def pull_item(self, idx):
        if self.dataset == 'ytvos':
            if self.split == 'val':
                if not self.save_result:
                    img_file, ori_h, ori_w, ins_id, phrase = self.images[idx].strip().split('~')
                else:
                    img_file, ori_h, ori_w, ins_id, phrase = self.images[idx].strip().split('#')  ## for save
            else:
                img_file, ori_h, ori_w, ins_id, phrase = self.images[idx].strip().split('#')
        elif self.dataset in ['a2d', 'jhmdb']:
            img_file, ori_h, ori_w, ins_id, phrase, cate = self.images[idx].strip().split(',')
        else:
            img_file, ori_h, ori_w, ins_id, phrase = self.images[idx].strip().split(',')
        video_name, video_num = img_file.split('/')
        if self.dataset in ['a2d', 'ytvos', 'davis']:
            img_path = osp.join(self.im_dir, f'{img_file}.jpg')
        elif self.dataset == 'jhmdb':
            img_path = glob.glob(osp.join(self.im_dir, '*', f'{img_file}.png'))[0]
        else:
            raise RuntimeError(f'{self.dataset} dost not exist!!!')
        img = Image.open(img_path).convert("RGB")

        if self.dataset == 'a2d':
            mask_path = osp.join(self.mask_dir, f'{img_file}.h5')
            with h5py.File(mask_path, 'r') as f:
                instances = f['instance'][()]
                idx = np.where(instances==int(ins_id))[0][0]
                if instances.shape[0] == 1:
                    mask = f['reMask'][()].transpose(1, 0)
                else:
                    mask = f['reMask'][()][idx].transpose(1, 0)
        elif self.dataset == 'jhmdb':
            mask_path = glob.glob(osp.join(self.mask_dir, '*', video_name, 'puppet_mask.mat'))[0]
            masks = scio.loadmat(mask_path)['part_mask']
            mask = masks[:, :, int(video_num) - 1]
        elif self.dataset == 'ytvos':
            mask_path = osp.join(self.mask_dir, f'{img_file}.png')
            mask = Image.open(mask_path)
            mask = np.array(mask)
            mask = (mask == int(ins_id)).astype(np.uint8)
        elif self.dataset == 'davis':
            mask_path = osp.join(self.mask_dir, f'{img_file}.png')
            mask = Image.open(mask_path)
            mask = np.array(mask)
            mask = (mask == int(ins_id)).astype(np.uint8)
        else:
            raise RuntimeError(f'{self.dataset} dost not exist!!!')

        frame_ind = img_path.split('/')[-1][:-4]
        frame_ind = int(frame_ind)
        if self.dataset == 'a2d':
            if osp.exists(osp.join(self.im_dir, img_file.split('/')[0], f'{frame_ind-self.interval:05d}.jpg')):
                pre_frame_ind = frame_ind - self.interval
            else:
                pre_frame_ind = frame_ind - 6

        elif self.dataset == 'jhmdb':
            pre_frame_ind = frame_ind - 5
        elif self.dataset == 'davis':
            if frame_ind >= 2:
                pre_frame_ind = frame_ind - 2
            else:
                pre_frame_ind = frame_ind + 2
        else:
            if osp.exists(osp.join(self.im_dir, img_file.split('/')[0], f'{frame_ind-5:05d}.jpg')):
                pre_frame_ind = frame_ind - 5
            else:
                frames = sorted(glob.glob(osp.join(self.im_dir, img_file.split('/')[0], '*')))[0]
                pre_frame_ind = int(frames.split('/')[-1][:-4])
        if self.dataset in ['a2d', 'ytvos', 'davis']:
            flow_path = osp.join(self.im_dir, img_file.split('/')[0], f'{pre_frame_ind:05d}.jpg')
        elif self.dataset == 'jhmdb':
            flow_path = glob.glob(osp.join(self.im_dir, '*', img_file.split('/')[0], f'{pre_frame_ind:05d}.png'))[0]
        else:
            raise RuntimeError(f'{self.dataset} dost not exist!!!')

        flow = Image.open(flow_path).convert('RGB')
        return img, flow, img_path, mask, phrase, int(ori_h), int(ori_w), ins_id

    def tokenize_phrase(self, phrase):
        return self.corpus.tokenize(phrase, self.query_len)

    def untokenize_word_vector(self, words):
        return self.corpus.dictionary[words]

    def generate_random_phrase(self):
        data_len = self.__len__()
        random_idx = random.choice(range(data_len))
        random_phrase = self.images[random_idx][-1]
        return random_phrase

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        orig_img, orig_flow, img_path, orig_mask, orig_phrase, ori_h, ori_w, ins_id = self.pull_item(idx)
        # orig_img: (426, 320) RGB
        # orig_mask: (320, 426)
        
        img = self.transform(orig_img)
        flow = self.transform_flow(orig_flow)
        flow = torch.abs(img - flow)
    
        orig_img = self.transform_orig(orig_img)
        orig_flow = self.transform_orig(orig_flow)
        orig_flow = torch.abs(orig_img - orig_flow)

        # [3, 320, 320]
        eval_mask = self.eval_ann_transform(torch.from_numpy(orig_mask).float())
        eval_mask[eval_mask > 0] = 1
        groundable = 1
        phrase, phrase_mask = self.tokenize_phrase(orig_phrase)
        batch = {
            "orig_img": orig_img,
            'orig_flow': orig_flow,
            "image": img,
            'flow': flow,
            "phrase": phrase,
            "phrase_mask": phrase_mask,
            "index": idx,
            "groundable": groundable,
            "img_path": img_path,
            ## "orig_image": resize(np.array(orig_img), (576, 576), anti_aliasing=True),
            "orig_phrase": orig_phrase,
            "orig_size": np.array([ori_h, ori_w]),
            "orig_mask": eval_mask,
            'ins_id': ins_id
        }
        return batch
