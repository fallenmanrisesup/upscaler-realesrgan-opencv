import os
import torch
from upscaler.torchimpl import TorchImpl
import openvino as ov

weight_name = 'RealESRGAN_x2'
scale_factor=4

device = torch.device('cpu')
model = TorchImpl(device, scale_factor=scale_factor)
model.load_weights(f"./models/{weight_name}_plus/{weight_name}.pth", download=True)

dummy_input = torch.randn(1, 3, 224, 224)

torch.onnx.export(model.rootModel(),
                  dummy_input,
                  f"./models/{weight_name}_plus/{weight_name}_plus.onnx",
                  export_params=True,
                  opset_version=12,
                  do_constant_folding=True,
                  input_names=['input'],
                  output_names=['output'],
                  dynamic_axes={
                      "input": {2: "height", 3: "width"},  # Динамические размеры для входа
                      "output": {2: "height", 3: "width"}  # Динамические размеры для выхода
                  })
onnx_path = f"./models/{weight_name}_plus/{weight_name}_plus.onnx"
ov_model = ov.convert_model(onnx_path)
ov.save_model(ov_model, onnx_path.replace('onnx', 'xml'))

