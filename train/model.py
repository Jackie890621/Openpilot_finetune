import onnx
from onnx2pytorch import ConvertModel
import torch
import onnxruntime as rt
import os
import torch.nn as nn

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORIGINAL_MODEL = os.path.join(parent_dir, 'common/models/supercombo.onnx')


def reinitialize_weights(layer_weight):
    torch.nn.init.xavier_uniform_(layer_weight)


def load_trainable_model(path_to_supercombo, trainable_layers=[]):

    onnx_model = onnx.load(path_to_supercombo)
    model = ConvertModel(onnx_model, experimental=True)  # pretrained_model

    # enable batch_size > 1 for onnx2pytorch
    model.Constant_1047.constant[0] = -1
    model.Constant_1049.constant[0] = -1
    model.Constant_1051.constant[0] = -1
    model.Constant_1053.constant[0] = -1
    model.Constant_1057.constant[0] = -1
    model.Constant_1059.constant[0] = -1

    # ensure immutability https://github.com/ToriML/onnx2pytorch/pull/38
    model.Elu_907.inplace = False

    # reinitialize trainable layers
    for layer_name, layer in model.named_children():
        # TODO: support layers other than Linear?
        if isinstance(layer, torch.nn.Linear) and layer_name in trainable_layers:
            reinitialize_weights(layer.weight)
            layer.bias.data.fill_(0.01)
        elif isinstance(layer, torch.nn.Conv2d) and layer_name in trainable_layers:
            repada = RepAdapter2D(in_features=layer.weight.shape[0])
            for param in layer.parameters():
                param.requires_grad = False
            layer = nn.Sequential(repada,
                                  layer)
            model._modules[layer_name] = layer

    # freeze other layers
    for name, param in model.named_parameters():
        name_layer = name.split(".")[0]
        if name_layer not in trainable_layers:
            param.requires_grad = False

    return model


def load_inference_model(path_to_model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if path_to_model.endswith('.onnx'):
        onnx_graph = onnx.load(path_to_model)
        output_names = [node.name for node in onnx_graph.graph.output]
        model = rt.InferenceSession(path_to_model, providers=['CPUExecutionProvider'])

        def run_model(inputs):
            outs =  model.run(output_names, inputs)[0]
            recurrent_state = outs[:, -512:]
            return outs, recurrent_state


    elif path_to_model.endswith('.pth'):

        model = load_trainable_model(ORIGINAL_MODEL)
        model.load_state_dict(torch.load(path_to_model))
        model.eval()
        model = model.to(device)

        def run_model(inputs):
            with torch.no_grad():
                inputs = {k: torch.from_numpy(v).to(device) for k, v in inputs.items()}
                outs = model(**inputs)
                recurrent_state = outs[:, -512:]
                return outs.cpu().numpy(), recurrent_state

    return model, run_model

class RepAdapter2D(nn.Module):
    """ Pytorch Implemention of RepAdapter for 2d tensor"""

    def __init__(
            self,
            in_features=768,
            hidden_dim=8,
            groups=2,
            scale=1,
            bias_train = False
    ):
        super().__init__()
        self.conv_A = nn.Conv2d(in_features, hidden_dim, 1, groups=1, bias=bias_train)
        self.conv_B = nn.Conv2d(hidden_dim, in_features, 1, groups=groups, bias=bias_train)
        self.dropout = nn.Dropout(0.1)
        self.groups = groups
        self.scale = scale

        nn.init.xavier_uniform_(self.conv_A.weight)
        nn.init.zeros_(self.conv_B.weight)
        if bias_train:
            nn.init.zeros_(self.conv_A.bias)
            nn.init.zeros_(self.conv_B.bias)

    def forward(self, x):
        x = self.conv_B(self.dropout(self.conv_A(x))) * self.scale + x
        return x

if __name__ == "__main__":
    '''pathplan_layer_names  = ["Gemm_959", "Gemm_981","Gemm_983","Gemm_1036", #plan
                             "Gemm_971","Gemm_1011","Gemm_1013","Gemm_1042", #left laneline
                             "Gemm_973","Gemm_1016","Gemm_1018","Gemm_1043", #right laneline
                             "Gemm_961", "Gemm_986","Gemm_988","Gemm_1037", #leads
                             "Gemm_963", "Gemm_991","Gemm_993","Gemm_1038", #lead_prob
                             "Gemm_979", "Gemm_1031","Gemm_1033","Gemm_1046", #desire_state
                             "Gemm_912", "Gemm_921","Gemm_923","Gemm_932", #desire_prob
                             "Conv_645", "Conv_668", "Conv_680", "Conv_703", 
                             "Conv_715", "Conv_738", "Conv_750", "Conv_762",
                             "Conv_785", "Conv_797", "Conv_809", "Conv_832", 
                             "Conv_844", "Conv_856", "Conv_868", "Conv_891",
                             ]
    '''
    pathplan_layer_names = [ #"Conv_785", "Conv_797", "Conv_809", "Conv_832", 
                             #"Conv_844", "Conv_856", "Conv_868", "Conv_891",
                             "Gemm_959", "Gemm_981","Gemm_983","Gemm_1036", #plan
                             "Gemm_961", "Gemm_986","Gemm_988","Gemm_1037", #leads
                             "Gemm_963", "Gemm_991","Gemm_993","Gemm_1038", #lead_prob
                             "Gemm_979", "Gemm_1031","Gemm_1033","Gemm_1046", #desire_state
                             "Gemm_912", "Gemm_921","Gemm_923","Gemm_932", #desire_prob
                             ]
    path_to_supercombo = '../common/models/supercombo.onnx'
    model = load_trainable_model(path_to_supercombo, pathplan_layer_names)

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print("pytorch_total_params : ", pytorch_total_params)
    pytorch_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("pytorch_train_params : ", pytorch_train_params)
    #for name, param in model.named_parameters():
    #    print(f'{name} : {param.requires_grad}')
