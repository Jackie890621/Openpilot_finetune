import warnings

warnings.filterwarnings("ignore", category=UserWarning,
                        message='The given NumPy array is not writeable, and PyTorch does not support non-writeable tensors')
warnings.filterwarnings("ignore", category=UserWarning, message='Using experimental implementation that allows \'batch_size > 1\'')
warnings.filterwarnings("ignore", message='Converting a tensor to a Python boolean might cause the trace to be incorrect')
warnings.filterwarnings("ignore", message='Iterating over a tensor might cause the trace to be incorrect')

import onnx
import onnxruntime as rt
import sys
import os
import torch
import numpy as np

from model import load_trainable_model, ORIGINAL_MODEL


def save_torch_to_onnx(path_to_model, trainable_layers):
    torch_model = load_trainable_model(ORIGINAL_MODEL, trainable_layers=trainable_layers)
    torch_model.load_state_dict(torch.load(path_to_model))
    torch_model.eval()

    inputs = {
        'input_imgs': torch.randn((1, 12, 128, 256), dtype=torch.float32, requires_grad=True),
        'desire': torch.zeros((1, 8), dtype=torch.float32, requires_grad=True),
        'traffic_convention': torch.tensor([0, 1], dtype=torch.float32).reshape(1, 2).requires_grad_(),
        'initial_state': torch.zeros((1, 512), dtype=torch.float32, requires_grad=True)
    }

    torch_out = torch_model(**inputs)

    torch.onnx.export(torch_model, 
                      (inputs['input_imgs'], inputs['desire'], inputs['traffic_convention'], inputs['initial_state'], {}), 
                      path_to_output, 
                      export_params=True, 
                      opset_version=9, 
                      do_constant_folding=False,
                      input_names=['input_imgs', 'desire', 'traffic_convention', 'initial_state'], 
                      output_names=['outputs'],
                      dynamic_axes={'input_imgs': {0: 'batch_size'}, 
                                    'desire': {0: 'batch_size'},
                                    'traffic_convention': {0: 'batch_size'},
                                    'initial_state': {0: 'batch_size'},
                                    'outputs': {0: 'batch_size'}})

    return inputs, torch_out


def verify_exported_model(path_exported_model, torch_ins, torch_out):
    onnx_model = onnx.load(path_exported_model)
    onnx.checker.check_model(onnx_model)

    ort_session = rt.InferenceSession(path_to_output, providers=['CPUExecutionProvider'])

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    # compute ONNX Runtime output prediction
    ort_inputs = {k: to_numpy(v) for k, v in torch_ins.items()}
    ort_outs = ort_session.run(None, ort_inputs)

    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)


if __name__ == '__main__':
    path_to_model = sys.argv[1]
    path_to_output = sys.argv[2] if len(sys.argv) > 2 else '.'.join(path_to_model.split('.')[:-1]) + '.onnx'

    pathplan_layer_names  = [ 
                             "Conv_785", "Conv_797", "Conv_809", "Conv_832", #RepAdapter
                             "Conv_844", "Conv_856", "Conv_868", "Conv_891", #RepAdapter
                             ]

    # pathplan_layer_names = [ "Conv_785", "Conv_797", "Conv_809", "Conv_832", #RepAdapter
    #                          "Conv_844", "Conv_856", "Conv_868", "Conv_891", #RepAdapter
    #                          "Gemm_959", "Gemm_981","Gemm_983","Gemm_1036", #plan
    #                          "Gemm_961", "Gemm_986","Gemm_988","Gemm_1037", #leads
    #                          "Gemm_963", "Gemm_991","Gemm_993","Gemm_1038", #lead_prob
    #                          "Gemm_965", "Gemm_996","Gemm_998","Gemm_1039", #lead_prob
    #                          "Gemm_979", "Gemm_1031","Gemm_1033","Gemm_1046", #desire_state
    #                          "Gemm_912", "Gemm_921","Gemm_923","Gemm_932", #desire_prob
    #                          ]

    # convert
    print('Saving ONNX model to {}...'.format(path_to_output))
    torch_ins, torch_out = save_torch_to_onnx(path_to_model, pathplan_layer_names)

    # verify
    verify_exported_model(path_to_output, torch_ins, torch_out)

    os._exit(0)