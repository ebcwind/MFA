import einops
import torch
import torch.nn.functional as F
import torchvision
from torch import nn

from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List
from dat.util.misc import NestedTensor
from .position_encoding import build_position_encoding

class LayerNormProxy(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = einops.rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        return einops.rearrange(x, 'b h w c -> b c h w')


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n, eps=1e-5):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))
        self.eps = eps

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = self.eps
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):
    def __init__(self, backbone: nn.Module, train_backbone: bool, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            # return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
            return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
            self.strides = [8, 16, 32]
            self.num_channels = [512, 1024, 2048]
        else:
            return_layers = {'layer4': "0"}
            self.strides = [32]
            self.num_channels = [2048]
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.patch_proj = nn.Sequential(
            LayerNormProxy(3),
            nn.Conv2d(3, 3, 7, 7, 0),
            nn.ReLU(inplace=True)
            )
        self.mask_pool = nn.MaxPool2d(7, 7, 0)
        # self.normal = LayerNormProxy(3)
        # self.cnn = nn.Conv2d(1, 3, 1, 1)

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(self.patch_proj(tensor_list.tensors))
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = self.mask_pool(tensor_list.mask.float()).to(torch.bool)
            # m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        norm_layer = FrozenBatchNorm2d
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            weights=None, norm_layer=norm_layer)
        backbone.load_state_dict(torch.load('resnet50.pth'))
        # assert name not in ('resnet18', 'resnet34'), "number of channels are hard coded"
        super().__init__(backbone, train_backbone, return_interm_layers)
        if dilation:
            self.strides[-1] = self.strides[-1] // 2


class Backbone2(nn.Module):
    def __init__(self, position_embedding):
        super().__init__()
        input_cnn = []
        for i in range(3):
            input_cnn.append(nn.Sequential(
                # nn.Conv2d(3, 128, 1, 1),
                # nn.Conv2d(128, 256, (7, 2), (7, 2)),
                # nn.MaxPool2d((7, 2), (7, 2)),
                # nn.ReLU(inplace=True),
                nn.Conv2d(1, 64, (1, (i+1)*2+1)),
                nn.ReLU(inplace=True),
                # nn.Conv2d(64, 128, 2, 2),
                # nn.ReLU(inplace=True),
                nn.MaxPool2d((1, 2))
                # nn.Conv2d(256, 512, ((i+1)*2+1), 64),
                # nn.ReLU(inplace=True),
                # nn.MaxPool2d(((i + 1) * 2 + 1), 1),
                )
            )
        self.input_proj = nn.ModuleList(input_cnn)
        self.pos_emb = position_embedding
        # self.normal = LayerNormProxy(3)
        self.strides = [2, 2, 2]
        self.num_channels = [64, 64, 64]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight, gain=1)
                nn.init.constant_(m.bias, 0)
        # for proj in self.input_proj:
        #     nn.init.xavier_uniform_(proj[0].weight, gain=1)
        #     nn.init.constant_(proj[0].bias, 0)

    def forward(self, tensor_list: NestedTensor):
        N, H, W = tensor_list.tensors.shape
        layer_normal = nn.LayerNorm([H, W]).cuda()
        feature = layer_normal(tensor_list.tensors)[:, None, :, :]
        out: List[NestedTensor] = []
        pos = []
        for i in range(3):
            x = self.input_proj[i](feature)
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out.append(NestedTensor(x, mask))
        for x in out:
            pos.append(self.pos_emb(x).to(x.tensors.dtype))
        return out, pos


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self.strides = backbone.strides
        self.num_channels = backbone.num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in sorted(xs.items()):
            out.append(x)

        # position encoding
        for x in out:
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks or (args.num_feature_levels > 1)
    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    model = Joiner(backbone, position_embedding)
    # model = Backbone2(position_embedding)

    return model
