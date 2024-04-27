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
    sys.path.append(os.getcwd())
    torch.set_float32_matmul_precision('medium')
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    #perserからoptを作成
    #-baseでmodel,dataのconfigを指定
    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    
    #opt.name, opt.resumeからディレクトリパスを作成
    #./logs/(指定した名前)/checkpoints/
    #./logs/(指定した名前)/configs/
    opt,nowname,logdir,ckptdir,cfgdir = get_storedir(opt,now)
    
    # Create logdirs and save configs
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(ckptdir, exist_ok=True)
    os.makedirs(cfgdir, exist_ok=True)
    
    try:
        #全てのconfigを作成
        _cfg = get_train_config(opt,nowname,logdir,ckptdir,cfgdir,now,unknown)
        
        #min_epochなど、pytorch_lightningを宣言するために必要なパラメータ
        lightning_config = _cfg["lightning_config"]
        
        #model,dataを作成するための引数
        config = _cfg["config"]
        
        #callbackを作成するための引数
        callbacks_cfg = _cfg["callbacks_cfg"]
        
        #trainerのloggerの作成option
        logger_cfg = _cfg["logger_cfg"]
        
        #trainer作成時に渡す引数
        trainer_opt = _cfg["trainer_opt"]
        
        #cpu使うならTrue gpuならFalse
        cpu = _cfg["cpu"]
        
        seed_everything(opt.seed)
        
        #modelの作成
        model = instantiate_from_config(config.model)
        #dataloaderの作成
        data = instantiate_from_config(config.data)
        data.prepare_data()
        data.setup()
        
        #ナニコレ → これ実装するにはget_train_configにmodel作成を組み込まなければいけないので一旦スルー
        # if hasattr(model, "monitor"):
        #     print(f"Monitoring {model.monitor} as checkpoint metric.")
        #     default_modelckpt_cfg["params"]["monitor"] = model.monitor
        #     default_modelckpt_cfg["params"]["save_top_k"] = 3
        
        # 設定の確認
        bs, base_lr = config.data.params.batch_size, config.model.base_learning_rate
        if not cpu:
            ngpu = lightning_config.trainer.devices
        else:
            ngpu = 1
            
        #accumulate_grad_batchsの設定
        accumulate_grad_batches = lightning_config.trainer.get("accumulate_grad_batches", 1)
        print(f"accumulate_grad_batches = {accumulate_grad_batches}")
        lightning_config.trainer.accumulate_grad_batches = accumulate_grad_batches
        
        #modelの学習率の設定 accumulate_grad_batches * ngpu * batch_size * base_lerning_late
        model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
        print("Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus) * {} (batchsize) * {:.2e} (base_lr)".format(
            model.learning_rate, accumulate_grad_batches, ngpu, bs, base_lr))
        

        #インスタンスを全て作成しtrainerの作成
        trainer_kwargs = dict()
        trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)
        trainer_kwargs["callbacks"] = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]
        # print(trainer_kwargs)

        trainer = Trainer(**trainer_opt, **trainer_kwargs)
    
        # allow checkpointing via USR1
        def melk(*args, **kwargs):
            # run all checkpoint hooks
            if trainer.global_rank == 0:
                print("Summoning checkpoint.")
                ckpt_path = os.path.join(ckptdir, "last.ckpt")
                trainer.save_checkpoint(ckpt_path)

        def divein(*args, **kwargs):
            if trainer.global_rank == 0:
                import pdb; pdb.set_trace()

        import signal
        signal.signal(signal.SIGUSR1, melk)
        signal.signal(signal.SIGUSR2, divein)
        
        # run
        if opt.train:
            try:
                trainer.fit(model, data)
            except Exception:
                melk()
                raise
        if not opt.no_test and not trainer.interrupted:
            trainer.test(model, data)
            
    except Exception:
        if opt.debug and trainer.global_rank==0:
            try:
                import pdb as debugger
            except ImportError:
                import pdb as debugger
            debugger.post_mortem()
        raise
    finally:
        # move newly created debug project to debug_runs
        if opt.debug and not opt.resume and trainer.global_rank==0:
            dst, name = os.path.split(logdir)
            dst = os.path.join(dst, "debug_runs", name)
            os.makedirs(os.path.split(dst)[0], exist_ok=True)
            os.rename(logdir, dst)
            