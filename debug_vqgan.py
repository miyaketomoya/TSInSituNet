from torch.utils.data import Dataset, DataLoader
import torch
from VQGAN.model.vqgan import VQModel
from data.smokeRing2 import SmokeRingTrain
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
import numpy as np


# 構成パラメータを設定
ddconfig = {
    "double_z": False,
    "z_channels": 256,
    "resolution": 512,
    "in_channels": 3,
    "out_ch": 3,
    "ch": 64,
    "ch_mult": [1, 2, 4, 8, 8, 8],
    "num_res_blocks": 1,
    "attn_resolutions": [64],
    "dropout": 0.0
}

lossconfig = {
    "target": "VQGAN.modules.losses.vqperceptual.DummyLoss"
}

# VQModelのインスタンス化
vq_model = VQModel(
    ddconfig=ddconfig,
    lossconfig=lossconfig,
    n_embed=16384,
    embed_dim=256,
    image_key="image",
    ckpt_path="/home/tomoyam/Study/logs/2024-05-07T18-40-13_VQGANSmokeRing/checkpoints/epoch=000099.ckpt",# 実際のチェックポイントパスに置き換えてください
    ignore_keys=["loss"]
)

# モデルに重みがロードされ、使用準備が整いました

# データセットのインスタンス化
root_path = "/data/data2/tomoya/dataSet"
train_dataset = SmokeRingTrain(root=root_path, toTensor=True, normalizeImage=True)

# データローダの作成
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=1)

# データローダーからデータを取得し、モデルを通して処理
for data in train_loader:
    images = data['image']
    outputs, _ = vq_model(images)  # outputs はデコードされた画像、_ は埋め込み損失を無視


    # 結果と元の画像を並べる
    combined = torch.cat((images, outputs), dim=0)
    grid = make_grid(combined, nrow=1)  # 6 はバッチサイズに合わせて調整
    
    # 結果を表示（オプショナル）
    plt.figure(figsize=(12, 8))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis('off')
    plt.show()
    
    # 結果をファイルに保存
    save_image(grid, 'combined_results.png')
    
    break  # デモンストレーションのため、最初のバッチのみ処理