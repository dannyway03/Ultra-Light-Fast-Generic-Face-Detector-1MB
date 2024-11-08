"""
This code is used to convert the pytorch model into an onnx format model.
"""
import sys

import torch.onnx

from vision.ssd.config.fd_config import define_img_size

input_img_size = 640  # define input size ,default optional(128/160/320/480/640/1280)
define_img_size(input_img_size)
from vision.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd
from vision.ssd.mb_tiny_fd import create_mb_tiny_fd

# net_type = "slim"  # inference faster,lower precision
net_type = "slim"  # inference lower,higher precision

label_path = "models/voc-model-labels.txt"
class_names = [name.strip() for name in open(label_path).readlines()]
num_classes = len(class_names)

if net_type == 'slim':
    # model_path = "models/pretrained/version-slim-320.pth"
    model_path = "models/pretrained/version-slim-640.pth"
    net = create_mb_tiny_fd(len(class_names), is_test=True, device="cpu")
elif net_type == 'RFB':
    # model_path = "models/pretrained/version-RFB-320.pth"
    model_path = "models/pretrained/version-RFB-640.pth"
    net = create_Mb_Tiny_RFB_fd(len(class_names), is_test=True, device="cpu")

else:
    print("unsupport network type.")
    sys.exit(1)
net.load(model_path)
net.eval()
net.to("cpu")

model_name = model_path.split("/")[-1].split(".")[0]
model_path = f"models/onnx/{model_name}.onnx"

# dummy_input = torch.randn(1, 3, 240, 320).to("cuda")
dummy_input = torch.randn(1, 3, 480, 640).to("cpu") #if input size is 640*480
torch.onnx.export(net, dummy_input, model_path, verbose=False, input_names=['input'], output_names=['scores', 'boxes'])
