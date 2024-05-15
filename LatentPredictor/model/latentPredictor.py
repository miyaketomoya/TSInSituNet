import os, math
import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from util import instantiate_from_config


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class Net2NetTransformer(pl.LightningModule):
    def __init__(self,
                 transformer_config,
                 first_stage_config,
                 permuter_config=None,
                 ckpt_path=None,
                 ignore_keys=[],
                 ):
        super().__init__()
        self.init_first_stage_from_ckpt(first_stage_config)
        if permuter_config is None:
            permuter_config = {"target": "LatentPredictor.transformer.permuter.Identity"}
        self.permuter = instantiate_from_config(config=permuter_config)
        self.transformer = instantiate_from_config(config=transformer_config)

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        
        print(self.permuter)
        print(self.transformer)

    #ckpt_patがロードするモデルのpath　関係ないデータを削除？
    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        for k in sd.keys():
            for ik in ignore_keys:
                if k.startswith(ik):
                    self.print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def init_first_stage_from_ckpt(self, config):
        model = instantiate_from_config(config)
        model = model.eval()
        model.train = disabled_train
        self.first_stage_model = model

    def forward(self,x,c,temperature=1.0,top_k=1,base_num=1):
        """
        input
        x:入力画像(batch,3,512,512)
        c:入力画像の条件()
        
        output
        output:予想後のコードブロック
        """
        
        #taegetとxのコードブロックを取得
        _, z_indices = self.encode_to_z(x)
        
        if base_num == 1:
            base_num = z_indices.shape[1]
        input = z_indices
    
        for k in range(base_num):
            logits = self.transformer(input,c_embeddings=c)            
            #最後のidのところだけ撮って、確率の高いIDだけ残す
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                logits = self.top_k_logits(logits, top_k)
            # apply softmax to convert to probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution or take the most likely
            _, ix = torch.topk(probs, k=1, dim=-1)
            # append to the sequence and continue

            input = torch.cat((input, ix), dim=1)

        return input[:,base_num:]
    
    def train_forward(self, x, c, t,temperature=1.0,top_k=1):
        """
        input
        x:入力画像(batch,3,512,512)
        c:入力画像の条件()
        t:予測してほしいタイムステップ後の画像
        
        output
        output:予想後のコードブロック
        target:正解
        """
        
        #taegetとxのコードブロックを取得
        _, t_indices = self.encode_to_z(t)
        _, z_indices = self.encode_to_z(x)
        ix_num = t_indices.shape[1]
        base_num = z_indices.shape[1]
        batch_size = z_indices.shape[0]  # バッチサイズをz_indicesから取得
        device = z_indices.device  # デバイス情報をz_indicesから取得
        
        input = torch.cat((z_indices,t_indices), dim=1)
    
        # 空のTensorを作成。最初は列が0なので、次元を追加しないといけない
        output = torch.empty((batch_size, 0), dtype=torch.long, device=device)
        
        loss = 0
        for k in range(ix_num):
            logits = self.transformer(input[:,:base_num+k],c_embeddings=c)
            loss += F.cross_entropy(logits[:, -1, :], input[:,base_num+k])

            # print(k,logits[:, -1, :].shape)
            # print(k,input[:,base_num+k])
            # print(k)
            
            #最後のidのところだけ撮って、確率の高いIDだけ残す
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                logits = self.top_k_logits(logits, top_k)
            # apply softmax to convert to probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution or take the most likely
            _, ix = torch.topk(probs, k=1, dim=-1)
            # append to the sequence and continue
            output = torch.cat((output, ix), dim=1)
        
        # print(output) 
        # print(target)
        return output,loss
    

    def top_k_logits(self, logits, k):
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[..., [-1]]] = -float('Inf')
        return out

    @torch.no_grad()
    def encode_to_z(self, x):
        """
        input
        x:image_data(b,3,512,512)
        
        output
        quant_z: code_img
        indices: code_seq (permute)
        """
        
        quant_z, _, info = self.first_stage_model.encode(x)
        indices = info[2].view(quant_z.shape[0], -1)
        indices = self.permuter(indices)
        return quant_z, indices

    @torch.no_grad()
    def decode_to_img(self, index, zshape):
        """
        input
        index:flatten codeblock
        
        output
        x:image
        """

        index = self.permuter(index, reverse=True)
        bhwc = (zshape[0],zshape[2],zshape[3],zshape[1])
        quant_z = self.first_stage_model.quantize.get_codebook_entry(
            index.reshape(-1), shape=bhwc)
        x = self.first_stage_model.decode(quant_z)
        return x

    @torch.no_grad()
    def log_images(self, batch, temperature=None, top_k=None, callback=None, lr_interface=False, **kwargs):
        log = dict()

        N = 4
        x, c, t= self.get_xct(batch, N)
        x = x.to(device=self.device)
        if isinstance(c, dict):
            c = {key: value.to(device=self.device) for key, value in c.items()}
        t = t.to(device=self.device)

        
        # quant_z, z_indices = self.encode_to_z(x)
        
        #index_sample->t_sample_det:生成したコードブロックの復元,
        #target->t_rec:正解のコードブロックの復元
        
        quant_z, z_indices = self.encode_to_z(x)
        quant_t, t_indices = self.encode_to_z(t)
        
        index_sample,loss = self.train_forward(x,c,t)
        t_sample_det = self.decode_to_img(index_sample, quant_t.shape)
        
        input_code_block = self.code_to_img(z_indices,quant_z.shape)
        target_code_block_ans = self.code_to_img(t_indices, quant_t.shape)
        target_code_block_predict = self.code_to_img(index_sample, quant_t.shape)

        log["input"] = x
        log["input_code_block"] = input_code_block
        # log["reconstruct_input_image"] = self.decode_to_img(z_indices,quant_z.shape)
        
        log["target"] = t
        log["target_code_block_ans"] = target_code_block_ans
        # log["reconstruct_target_image_ans"] = self.decode_to_img(t_indices, quant_t.shape)
         
        log["target_code_block_predict"] = target_code_block_predict
        log["reconstruct_target_image_predict"] = self.decode_to_img(index_sample, quant_t.shape)
        
        log["code_block_diff"] = torch.abs(target_code_block_ans-target_code_block_predict)
        
        return log
    
    def code_to_img(self,index,zshape):
        index = self.permuter(index, reverse=True)
        index = index/self.transformer.config.vocab_size
        img_index = index.view(zshape[0],zshape[2],zshape[3])
        return img_index
        

    def get_input(self, key, batch):
        x = batch[key]
        # if len(x.shape) == 3:
        #     x = x[..., None]
        # if len(x.shape) == 4:
        #     x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        # if x.dtype == torch.double:
        #     x = x.float()
        return x

    def get_xct(self, batch, N=None):
        x = self.get_input("image", batch)
        c = self.get_input("params", batch)
        t = self.get_input("target",batch)
        if N is not None:
            x = x[:N]
            for key in c.keys():
                c[key] = c[key][:N]
            t = t[:N]
        return x, c, t

    def shared_step(self, batch, batch_idx):
        x, c, t = self.get_xct(batch)
        logits,loss = self.train_forward(x, c, t)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        self.log("train/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        self.log("val/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        """
        Following minGPT:
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.transformer.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.transformer.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.01},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=self.learning_rate, betas=(0.9, 0.95))
        return optimizer
