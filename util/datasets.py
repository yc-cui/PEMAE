import torch
from torch.utils.data import Dataset
import h5py
from kornia.filters import gaussian_blur2d
import torch.nn.functional as F


def _high_pass_filter(img, ksize=(5,5)):
    blur = gaussian_blur2d(img, (3, 3), (1.5, 1.5))
    high_pass_filtered = img - blur
    return high_pass_filtered

def slide(x, k):
    B, C, H, W = x.shape
    patches = x.unfold(2, k, k).unfold(3, k, k).permute(0, 2, 3, 1, 4, 5).contiguous().view(-1, C, k, k)
    return patches


class TrainDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        
        with h5py.File(data_dir, "r") as file:

            ms = file["ms"][:]
            pan = file["pan"][:]
            gt = file["gt"][:]

        
        self.ms = torch.from_numpy(ms)
        self.pan = torch.from_numpy(pan)
        self.gt = torch.from_numpy(gt)
        self.hp_pan = _high_pass_filter(self.pan)
        self.hp_ms = _high_pass_filter(self.ms)
        
        print("training ms: ", self.ms.shape)
        print("training pan: ", self.pan.shape)
        print("training gt: ", self.gt.shape)
        
    def __len__(self):
        return len(self.ms)

    def __getitem__(self, idx):
        
        ms = self.ms[idx]
        pan = self.pan[idx]
        gt = self.gt[idx]
        
        ms = ms / 2047.
        pan = pan / 2047.
        gt = gt / 2047.

        up_ms = F.interpolate(ms.unsqueeze(0), (pan.shape[-2], pan.shape[-1]), mode="bicubic", align_corners=True).squeeze(0)
        hp_pan = self.hp_pan[idx]
        hp_ms = self.hp_ms[idx]

        return ms, pan, hp_pan, hp_ms, up_ms, gt


class ValidDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        
        with h5py.File(data_dir, "r") as file:

            ms = file["ms"][:]
            pan = file["pan"][:]
            gt = file["gt"][:]

        self.ms = slide(torch.from_numpy(ms), 16)
        self.pan = slide(torch.from_numpy(pan), 64)
        self.gt = slide(torch.from_numpy(gt), 64)
        self.hp_pan = _high_pass_filter(self.pan)
        self.hp_ms = _high_pass_filter(self.ms)

        print("testing ms: ", self.ms.shape)
        print("testing pan: ", self.pan.shape)

    def __len__(self):
        return len(self.ms)
    

    def __getitem__(self, idx):
        
        ms = self.ms[idx]
        pan = self.pan[idx]
        gt = self.gt[idx]
        
        ms = ms / 2047.
        pan = pan / 2047.
        gt = gt / 2047.

        up_ms = F.interpolate(ms.unsqueeze(0), (pan.shape[-2], pan.shape[-1]), mode="bicubic", align_corners=True).squeeze(0)
        hp_pan = self.hp_pan[idx]
        hp_ms = self.hp_ms[idx]

        return ms, pan, hp_pan, hp_ms, up_ms, gt


class OriTestDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        
        with h5py.File(data_dir, "r") as file:

            ms = file["ms"][:]
            pan = file["pan"][:]

        self.ms = slide(torch.from_numpy(ms), 16)
        self.pan = slide(torch.from_numpy(pan), 64)
        self.hp_pan = _high_pass_filter(self.pan)
        self.hp_ms = _high_pass_filter(self.ms)

        print("testing ms: ", self.ms.shape)
        print("testing pan: ", self.pan.shape)

    def __len__(self):
        return len(self.ms)

    def __getitem__(self, idx):
        
        ms = self.ms[idx]
        pan = self.pan[idx]
        
        ms = ms / 2047.
        pan = pan / 2047.

        up_ms = F.interpolate(ms.unsqueeze(0), (pan.shape[-2], pan.shape[-1]), mode="bicubic", align_corners=True).squeeze(0)
        hp_pan = self.hp_pan[idx]
        hp_ms = self.hp_ms[idx]

        return ms, pan, hp_pan, hp_ms, up_ms
    