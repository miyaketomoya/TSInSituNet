import json
from pathlib import Path

import numpy as np
import torch
from einops import rearrange
from skimage import io, transform
from torch.utils.data import Dataset
from torchvision import transforms


class SmokeRingTrain(Dataset):
    def __init__(self, root, toTensor, normalizeImage):
        self.root = root
        self.transform = transfromForSmokeRing(toTensor, normalizeImage)
        self.imglist, self.cond, self.params,self.steps = setup(Path(root, "train", "params.json"))

    def __getitem__(self, index):

        file_name = self.imglist[index]
        after_path = Path(self.root, "train/img", file_name)
        after = io.imread(after_path)
        params = self.params[file_name]
        before_name = self.cond[file_name]["target"]
        before_path = Path(self.root, "train/img", before_name)
        before = io.imread(before_path)
        sample = {"image": before, "params": params, "target": after}
        # print(target_path,image_path)

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.imglist)


class SmokeRingVal(Dataset):
    def __init__(self, root, toTensor, normalizeImage):
        self.root = root
        self.transform = transfromForSmokeRing(toTensor, normalizeImage)
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
        target = sample["target"].astype(np.float32)
        # params = np.array(sample["params"],dtype=(np.float32))
        params = sample["params"]

        for key, values in params.items():
            params[key] = torch.from_numpy(np.array(values, dtype=(np.float32)))

        image = rearrange(image, "h w c -> c h w")
        target = rearrange(target, "h w c -> c h w")
        return {
            "image": torch.from_numpy(image),
            "params": params,
            "target": torch.from_numpy(target),
        }


class NormalizeImage(object):
    def __call__(self, sample):
        image = sample["image"]
        image = image.astype(np.float32) / 255

        target = sample["target"]
        target = target.astype(np.float32) / 255
        return {
            "image": image,
            "params": sample["params"],
            "target": target,
        }


# class NormalizeParams(object):
#     def __call__(self, sample):
#         params = sample["params"]

#         #parama正規化
#         return {"image": sample["image"],
#                 "params": params,
#                 "target": sample["target"]
#                 }


def transfromForSmokeRing(toTensor, normalizeImage):
    trlist = []
    if normalizeImage:
        trlist.append(NormalizeImage())
    # if normalizeParams:
    #     trlist.append(NormalizeParams())
    if toTensor:
        trlist.append(ToTensor())
    return transforms.Compose(trlist)


def setup(fname):
    with open(fname, "r", encoding="utf-8") as file:
        data = json.load(file)

    name_img = data["name_img"]
    steps = data["steps"]
    maxstep = data["maxstep"]
    maxmin = data["maxmin"]
    params = data["params"]

    imglist = []
    # nextlist = {}
    beforelist = {}
    # dellist = []
    for key, value in params.items():

        parts = key.split(":")
        timestep = parts[0]
        
        # if int(timestep) == 0:
        #     #初期値のtimestepは真っ白な画面から生成したい。 firstとなのつくkey -> 初期条件の画像へ
        #     # nextlist[str(-int(steps))+":".join(parts)] = {"target": ":".join(parts)}
        #     beforelist[key] = {"target": str(-int(steps))+":".join(parts[1:])}
        #     imglist.append(str(-int(steps))+":".join(parts))
        
        # if int(timestep) + steps > maxstep:
        #     dellist.append(key)
        #     continue
        # else:
        
        if int(timestep) == 0:
            #time stepが0の時は全ての画像が真っ白な画面が前情報
            before = str(-int(steps))+".bmp"
        else:
            parts[0] = str(int(int(timestep) - steps))
            # next = ":".join(parts)
            before = ":".join(parts)

        imglist.append(key)
        beforelist[key] = {"target": before}
        # nextlist[key] = {"target": next}

    # print(beforelist)
    params = normalizeparam(maxmin, params)

    return imglist, beforelist, params,steps


def normalizeparam(maxmin, params):
    for filename, param in params.items():
        for key, value in param.items():
            if key == "timestep":
                continue
            else:
                targets = value
                normlist = maxmin[key]
                new = []
                for target, (mi, ma) in zip(targets, normlist):
                    mid = (mi + ma) * 0.5
                    wari = (ma - mi) * 0.5
                    if wari == 0:
                        wari = 1
                    target = (target - mid) / wari
                    new.append(target)

                params[filename][key] = new
    return params


# root = "/data/data2/tomoya/dataSet"
# dataset = SmokeRingTrain(root=root,toTensor=True,normalizeImage=True)
# # データセットの長さをテスト
# print("Dataset length:", len(dataset))
# for i in range(len(dataset)):
#     print(dataset[i])

# # # 最初の要素を取得してテスト
# # sample = dataset[0]
# # print("Sample keys:", sample.keys())
# # print("Image shape:", sample["image"].shape)
# # print("Params:", sample["params"])
# # print("Target:", sample["target"].shape)
