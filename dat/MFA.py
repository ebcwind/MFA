import torch
import torch.nn.functional as F
from torch import nn
import math
from CBAM import CBAMLayer
from dat.util.misc import NestedTensor
import copy


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class MFA(nn.Module):
    def __init__(self, backbone, transformer, num_classes, num_queries, num_feature_levels,
                 aux_loss=True):
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.num_feature_levels = num_feature_levels
        self.query_embed = nn.Embedding(num_queries, hidden_dim*2)
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])
        self.backbone = backbone
        self.aux_loss = aux_loss
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value

        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        num_pred = transformer.decoder.num_layers
        self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
        self.transformer.decoder.bbox_embed = None
        self.fn = nn.Sequential(
            nn.Linear(2 * num_queries, 64),
            nn.ReLU()
        )
        self.cbam = CBAMLayer(6)
        self.net_process = nn.Conv1d(1, 3, 1, stride=1)
        self.fcn = nn.Sequential(
            nn.Conv2d(6, 2, kernel_size=(64, 1)),
            nn.Sigmoid()
        )

    def forward(self, samples: NestedTensor, net):
        features, pos = self.backbone(samples)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        query_embeds = self.query_embed.weight
        hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact = self.transformer(srcs, masks, pos, query_embeds)

        outputs_classes = []
        for lvl in range(hs.shape[0]):
            outputs_class = self.class_embed[lvl](hs[lvl])
            outputs_classes.append(outputs_class)
        outputs_class = torch.stack(outputs_classes)  # (num_feature_levels,N,num_query,2)
        # out = outputs_class[-1]  # (N,num_query,2)
        # out = out.sigmoid()
        # prob = self.fn(out.transpose(2, 1)).flatten(1)
        out = outputs_class.transpose(0, 1).flatten(2)
        mixture_feature = torch.cat([self.fn(out), self.net_process(net[:, None, :])], dim=1)
        mixture_feature = self.cbam(mixture_feature[..., None])
        prob = self.fcn(mixture_feature).flatten(1)

        return prob
