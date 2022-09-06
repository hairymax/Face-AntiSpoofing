import torch
import onnx
import onnxsim

from src.config import PretrainedConfig
from src.antispoof_pretrained import AntiSpoofPretrained
import os

import argparse

def check_onnx_model(model):
    try:
        onnx.checker.check_model(model)
    except onnx.checker.ValidationError as e:
        print('ONNX model is invalid:', e)
    else:
        print('ONNX model is valid!')

if __name__ == "__main__":
    # parsing arguments
    p = argparse.ArgumentParser(description="Convert model weights from .pth to .onnx")
    p.add_argument("model_path", type=str, 
                   help="Path to .pth model weights")
    p.add_argument("num_classes", type=int, default=2,
                   help="Number of classes that model is trained to predict")
    p.add_argument("--onnx_model_path", type=str, default=None,
                   help="Path to save converted .onnx model weights")
    p.add_argument("--print_summary", type=bool, default=False,
                   help="Whether to print the model information (torchsummary is needed)")
    args = p.parse_args()
    
    assert os.path.isfile(args.model_path), 'Model {} not found!'.format(args.model_path)
    # 'saved_models/AntiSpoofing_print-replay_128.pth'
    cnf = PretrainedConfig(args.model_path, num_classes=args.num_classes)
    
    model = AntiSpoofPretrained(cnf).model
    print(args.model_path, 'loaded successfully')
        
    if args.print_summary:
        from torchsummary import summary
        summary(model)
        
    onnx_model_path = args.onnx_model_path
    if onnx_model_path is None:
        onnx_model_path = cnf.model_path.replace('.pth','.onnx')
    # Save onnx model
    model.eval()
    dummy_input = torch.randn(1, 3, cnf.input_size, cnf.input_size).to(cnf.device)
    torch.onnx.export(model, 
                      dummy_input,
                      onnx_model_path,
                      #verbose=False,
                      input_names=['input'],
                      output_names=['output'],
                      export_params=True,
                     )
    # Load onnx model
    onnx_model = onnx.load(onnx_model_path)  
    print('\nCheck exported model')
    check_onnx_model(onnx_model)
    # Simplify the model    
    onnx_model, check = onnxsim.simplify(onnx_model,
                        #dynamic_input_shape=True,
                        #input_shapes={'input': list(dummy_input.shape)} 
                        )
    print('\nCheck simplified model')
    assert check, "Simplified ONNX model could not be validated"
    check_onnx_model(onnx_model)
    # Save simplified model   
    onnx.save(onnx_model, onnx_model_path)
    
    print('\nIR version:', onnx_model.ir_version)
    print('ONNX model exported to:', onnx_model_path)