import argparse, os, sys, datetime, glob, importlib
from omegaconf import OmegaConf
import numpy as np
from PIL import Image
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from util import *



if __name__ == "__main__":
    #VQGANのモデルをロード
    
    #Transformerを作成
    
    #Data - VQENCODE - latentMAP(学習対象データ)& param
    
    #task1,parameter予測  input = param, output = latentMAPで学習
    #task2,timestep予測   input = preStepMap, output = nextStepMap　
    