import json
from pathlib import Path

import numpy as np
import torch
from einops import rearrange
from skimage import io, transform
from torch.utils.data import Dataset
from torchvision import transforms


class SmokeRingTrain(Dataset):
    def __init__(self, root, toTensor, normalizeImage, normalizeParams):
        self.root = root
        self.transform = transfromForSmokeRing(
            toTensor, normalizeImage, normalizeParams
        )
        self.data = list(Path(root, "train", "img").glob("*.bmp"))
        self.params = json.loads(Path(root, "train", "params.json").read_text())

    def __getitem__(self, index):
        image = io.imread(str(self.data[index]))
        file_name = self.data[index].name
        params = self.params[file_name]
        sample = {"image": image, "params": params}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.data)


class SmokeRingVal(Dataset):
    def __init__(self, root, toTensor, normalizeImage, normalizeParams):
        self.root = root
        self.transform = transfromForSmokeRing(
            toTensor, normalizeImage, normalizeParams
        )
        self.data = list(Path(root, "test", "img").glob("*.bmp"))
        self.params = json.loads(Path(root, "test", "params.json").read_text())

    def __getitem__(self, index):
        image = io.imread(str(self.data[index]))
        file_name = self.data[index].name
        params = self.params[file_name]
        sample = {"image": image, "params": params}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.data)


class ToTensor(object):
    def __call__(self, sample):
        image = sample["image"].astype(np.float32)
        params = np.array(sample["params"], dtype=(np.float32))

        image = rearrange(image, "h w c -> c h w")
        return {
            "image": torch.from_numpy(image),
            "params": torch.from_numpy(params),
        }


class NormalizeImage(object):
    def __call__(self, sample):
        image = sample["image"]
        image = image.astype(np.float32) / 255
        return {
            "image": image,
            "params": sample["params"],
        }


class NormalizeParams(object):
    def __call__(self, sample):
        params = sample["params"]

        # parama正規化
        return {
            "image": sample["image"],
            "params": params,
        }


def transfromForSmokeRing(toTensor, normalizeImage, normalizeParams):
    trlist = []
    if normalizeImage:
        trlist.append(NormalizeImage())
    if normalizeParams:
        trlist.append(NormalizeParams())
    if toTensor:
        trlist.append(ToTensor())
    return transforms.Compose(trlist)
