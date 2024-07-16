import torch
from PIL import Image
import io
import base64

def load_xyz(file_path):
    with open(file_path, "r")  as f:
        lines = f.readlines()[1:]
    x = []
    y = []
    z = []
    for line in lines:
        x.append(float(line.split(",")[1]))
        y.append(float(line.split(",")[2]))
        z.append(float(line.split(",")[3]))
    return x, y, z

def input_to_tensor_SmokeLing(params):
    return torch.tensor([params])

def tensor_to_base64(img_tensor):
    # バッチ次元を削除 (1, 3, 512, 512) -> (3, 512, 512)
    img_tensor = img_tensor.squeeze(0)
    # チャネル次元を最後に移動 (3, 512, 512) -> (512, 512, 3)
    img_tensor = img_tensor.permute(1, 2, 0)
    # GPU上のテンソルをCPUに移動
    img_tensor = img_tensor.cpu()
    # テンソルを [0, 1] から [0, 255] にスケーリングし、uint8にキャスト
    img_tensor = (img_tensor * 255).byte()
    # NumPy配列に変換
    img_array = img_tensor.numpy()
    # PIL Imageオブジェクトに変換
    img = Image.fromarray(img_array)
    # バイト列に変換
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    # Base64エンコード
    img_base64 = base64.b64encode(buffered.getvalue()).decode()
    return img_base64