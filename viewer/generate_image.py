import yaml
from pathlib import Path
from viewerUtil import input_to_tensor_SmokeLing
from omegaconf import OmegaConf
import torch

import sys
sys.path.append("../")
from util import instantiate_from_config


def setup_model(logger_dir):
    config_path = Path(logger_dir) / "configs"
    model_path = Path(logger_dir) / "checkpoints" / "last.ckpt"
    # *project.yamlにマッチするファイルを検索
    yaml_files = list(config_path.glob("*project.yaml"))
    if not yaml_files:
        raise FileNotFoundError("No project.yaml file found.")
    # 最初のyamlファイルを読み込む
    config = OmegaConf.load(yaml_files[0])
    OmegaConf.set_struct(config, False)  # 設定を変更可能にする
    config.model.params.ckpt_path = model_path  # 新しいパスを設定
    
    model = instantiate_from_config(config.model)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    model = model.to(device)
    return model,device


def generate_img(model,device,viscosity, thermal_diffusivity, timestep, selected_x, selected_y, selected_z,interval):
    #初期条件の処理が必要 cがsimとviewに関連すること
    c = {}
    # パラメータをFloat型に変換
    viscosity = (viscosity - 0.0275) / 0.0075
    thermal_diffusivity = thermal_diffusivity - 0.03
    selected_x = selected_x/3.0
    selected_y = selected_y/3.0
    selected_z = selected_z/3.0
    
    img_list = []
    code_block_list = []
    timestep_list = []
    #初期の写真を取得、それをコードブロックに
    
    
    
    c["simparam"] = input_to_tensor_SmokeLing([viscosity,thermal_diffusivity]).to(device)
    c["viewparam"] = input_to_tensor_SmokeLing([selected_x, selected_y, selected_z]).to(device)
    code_block =  torch.zeros(1, 16*16, dtype=torch.int).to(device)
    timestep_list.append(0)
    
    #cの整理する
    defconfig = {"z_indices":code_block,"c":c}
    
    for i in range(timestep):
        print("generating ..",i)
        defconfig, code_block, img = model.forward_from_codeblock(**defconfig)
        timestep_list.append((i+1)*interval)
        img_list.append(img)
        code_block_list.append(code_block)
        
    return img_list,code_block_list,timestep_list


