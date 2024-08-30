
import pytorch_lightning as pl
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from kornia.filters import gaussian_blur2d
from torch.utils.data import DataLoader
from sorcery import dict_of
import os
from scipy import io
import random
from src.dataset.sequence import seq


def _blur_down(img, scale=0.25, ksize=(3, 3), sigma=(1.5, 1.5)):
    blur = gaussian_blur2d(img, ksize, sigma)
    return F.interpolate(blur, scale_factor=scale, mode="bicubic", align_corners=True)


def _get_global_pos(x, size, down_factor):
    N = 1
    L = size ** 2
    row = torch.arange(0, size * down_factor, down_factor, device=x.device).unsqueeze(-1)
    col = size * down_factor * down_factor * torch.arange(size, device=x.device)
    corner = (row + col).T  # top left
    offset = torch.randint(down_factor, (N, size, size, 2), device=x.device)  # offset to right and bottom
    sample_idxs = corner + offset[..., 0] * size * down_factor + offset[..., 1]
    global_sampled_pos = sample_idxs.reshape(N, -1)
    # generate mask
    mask_shape = N, (size * down_factor) ** 2
    mask = torch.zeros(mask_shape, dtype=torch.int64, device=x.device)
    mask.scatter_(1, global_sampled_pos, 1)
    mask = mask.reshape(N, -1)
    # replace non masked area with index
    nonzero_indices = torch.nonzero(mask)
    indices = torch.arange(1, L + 1, dtype=torch.int64, device=x.device).unsqueeze(0).expand(N, -1).reshape(-1)
    ids_restore = mask.index_put((nonzero_indices[:, 0], nonzero_indices[:, 1]), indices)
    return global_sampled_pos, mask, ids_restore  # B HW , B H'W'


def _random_mask(x, size=64, down_factor=4):
    # x bchw
    pos, mask, ids_restore = _get_global_pos(x, size, down_factor)
    x = x.flatten(2).transpose(1, 2)
    x = torch.gather(x, dim=1, index=pos.unsqueeze(-1).repeat(1, 1, x.shape[2]))
    x = x.transpose(2, 1).view(1, x.shape[2], size, size)
    return x, pos


def _random_mask_by_pos(x, size=64, pos=None):
    # x bchw
    x = x.flatten(2).transpose(1, 2)
    x = torch.gather(x, dim=1, index=pos.unsqueeze(-1).repeat(1, 1, x.shape[2]))
    x = x.transpose(2, 1).view(1, x.shape[2], size, size)
    return x


class DatasetRandomMask(Dataset):
    def __init__(self, data_dir, split="train", full_test=False, seed=42):
        self.data_dir = data_dir
        self.data_dir = os.path.expanduser(data_dir)
        self.split = split
        self.full_test = full_test
        random_seed = seed
        train_ratio = 0.6
        val_ratio = 0.1

        mat_dir = os.path.join(self.data_dir, "MS_256")
        mat_files = sorted(os.listdir(mat_dir))
        num_files = len(mat_files)

        random.seed(random_seed)
        np.random.seed(random_seed)

        if 'ikonos' in data_dir.lower():
            random_sequence = seq['ik']
        elif 'quickbird' in data_dir.lower():
            random_sequence = seq['qb']
        elif 'gaofen-1' in data_dir.lower():
            random_sequence = seq['gf1']
        elif 'worldview-2' in data_dir.lower():
            random_sequence = seq['wv2']
        elif 'worldview-3' in data_dir.lower():
            random_sequence = seq['wv3']
        elif 'worldview-4' in data_dir.lower():
            random_sequence = seq['wv4']
        else:
            raise RuntimeError("Found no training sequence!")

        train_idx = int(num_files * train_ratio)
        val_idx = train_idx + int(num_files * val_ratio)

        self.train_mat_files_files = [mat_files[idx] for idx in random_sequence[:train_idx]]
        self.val_mat_files_files = [mat_files[idx] for idx in random_sequence[train_idx:val_idx]]
        self.test_mat_files_files = [mat_files[idx] for idx in random_sequence[val_idx:]]

        # print(data_dir, ":", self.split, ":")
        # print("train:")
        # print(self.train_mat_files_files)
        # print("val:")
        # print(self.val_mat_files_files)
        # print("test:")
        # print(self.test_mat_files_files)

        if self.split == "train":
            self.mat_files = self.train_mat_files_files
        elif self.split == "test":
            self.mat_files = self.test_mat_files_files
        elif self.split == "val":
            self.mat_files = self.val_mat_files_files
        else:
            raise RuntimeError("Wrong split.")

        print(data_dir, ":", self.split, ":", self.mat_files[:10])

        if "gaofen" in data_dir.lower():
            self.max_val = 1023
        else:
            self.max_val = 2047

    def __len__(self):
        return len(self.mat_files)

    def __getitem__(self, idx):

        # load mat file
        mat_ms = io.loadmat(os.path.join(self.data_dir, "MS_256", self.mat_files[idx]))
        mat_pan = io.loadmat(os.path.join(self.data_dir, "PAN_1024", self.mat_files[idx]))

        # fix key error
        key_ms = "imgMS" if "imgMS" in mat_ms.keys() else "I_MS"
        key_pan = "imgPAN" if "imgPAN" in mat_pan.keys() else "I_PAN"
        if "imgPAN" not in mat_pan.keys() and "I_PAN" not in mat_pan.keys():
            key_pan = "block"
        ms = torch.from_numpy((mat_ms[key_ms] / self.max_val).astype(np.float32)).permute(2, 0, 1)
        if ms.shape[0] == 8:  # only use 4 bands to test generalization
            ms = ms[[1, 2, 4, 6], ...]
        pan = torch.from_numpy((mat_pan[key_pan] / self.max_val).astype(np.float32)).unsqueeze(0)

        gt = ms

        # prepare data
        if self.split == "train" or self.split == "val" or not self.full_test:
            ms_down = _blur_down(ms.unsqueeze(0)).squeeze(0)
            pan_down = _blur_down(pan.unsqueeze(0)).squeeze(0)
            masked_ms, pos = _random_mask(ms.unsqueeze(0), size=ms.shape[-1] // 4)
            masked_ms = masked_ms.squeeze(0)
            masked_pan = _random_mask_by_pos(pan_down.unsqueeze(0), pan_down.shape[-1] // 4, pos).squeeze(0)
            pan = pan_down
            up_ms = F.interpolate(ms_down.unsqueeze(0), scale_factor=4, mode="bicubic", align_corners=True).squeeze(0)
        else:  # full
            up_ms = F.interpolate(ms.unsqueeze(0), (pan.shape[-2], pan.shape[-1]), mode="bicubic", align_corners=True).squeeze(0)
            masked_pan = _blur_down(pan.unsqueeze(0)).squeeze(0)
            masked_ms = ms

        inp_dict = dict_of(masked_ms, masked_pan, pan, up_ms, gt)
        return inp_dict


class plNBUDataset(pl.LightningDataModule):
    def __init__(self, data_dir_train, data_dirs_test, batch_size, num_workers=4, pin_memory=True, seed=42):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.dataset_train = DatasetRandomMask(data_dir_train, split="train", seed=seed)
        self.dataset_val = DatasetRandomMask(data_dir_train, split="val", seed=seed)
        self.dataset_test_full = DatasetRandomMask(data_dirs_test, split="test", full_test=True, seed=seed)
        self.dataset_test = DatasetRandomMask(data_dirs_test, split="test", full_test=False, seed=seed)

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset_val,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )

    def test_dataloader(self):
        return [DataLoader(
            self.dataset_test,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        ),
            DataLoader(
            self.dataset_test_full,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )]
