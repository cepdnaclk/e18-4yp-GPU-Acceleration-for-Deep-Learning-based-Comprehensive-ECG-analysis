import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from fastai.layers import *
from fastai.core import *

##############################################################################################################################################
# utility functions


class AdaptiveConcatPool1d(nn.Module):
    "Layer that concats `AdaptiveAvgPool1d` and `AdaptiveMaxPool1d`."

    def __init__(self, sz: Optional[int] = None):
        "Output will be 2*sz or 2 if sz is None"
        super().__init__()
        sz = sz or 1  # before this sz is None...
        self.ap, self.mp = nn.AdaptiveAvgPool1d(sz), nn.AdaptiveMaxPool1d(sz)

    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 1)


def create_head1d(nf: int, nc: int, lin_ftrs: Optional[Collection[int]] = None, ps: Floats = 0.5, bn_final: bool = False, bn: bool = True, act="relu", concat_pooling=True):
    "Model head that takes `nf` features, runs through `lin_ftrs`, and about `nc` classes; added bn and act here"
    lin_ftrs = [2 * nf if concat_pooling else nf, nc] if lin_ftrs is None else [2 * nf if concat_pooling else nf] + lin_ftrs + [nc]  # was [nf, 512,nc]
    ps = listify(ps)
    if len(ps) == 1:
        ps = [ps[0] / 2] * (len(lin_ftrs) - 2) + ps
    actns = [nn.ReLU(inplace=True) if act == "relu" else nn.ELU(inplace=True)] * (len(lin_ftrs) - 2) + [None]
    layers = [AdaptiveConcatPool1d() if concat_pooling else nn.MaxPool1d(2), Flatten()]
    for ni, no, p, actn in zip(lin_ftrs[:-1], lin_ftrs[1:], ps, actns):
        layers += bn_drop_lin(ni, no, bn, p, actn)
    if bn_final:
        layers.append(nn.BatchNorm1d(lin_ftrs[-1], momentum=0.01))
    return nn.Sequential(*layers)


##############################################################################################################################################


def conv(in_planes, out_planes, kernel_size=3, stride=1):
    "convolution with padding"
    return nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2, bias=False)


def noop(x):
    return x


class InceptionBlock1d(nn.Module):
    def __init__(self, ni, nb_filters, kss, stride=1, act="linear", bottleneck_size=32):
        super().__init__()
        self.bottleneck = conv(ni, bottleneck_size, 1, stride) if (bottleneck_size > 0) else noop

        self.convs = nn.ModuleList([conv(bottleneck_size if (bottleneck_size > 0) else ni, nb_filters, ks) for ks in kss])
        self.conv_bottle = nn.Sequential(nn.MaxPool1d(3, stride, padding=1), conv(ni, nb_filters, 1))
        self.bn_relu = nn.Sequential(nn.BatchNorm1d((len(kss) + 1) * nb_filters), nn.ReLU())

    def forward(self, x):
        # print("block in",x.size())
        bottled = self.bottleneck(x)
        out = self.bn_relu(torch.cat([c(bottled) for c in self.convs] + [self.conv_bottle(x)], dim=1))
        return out


class Shortcut1d(nn.Module):
    def __init__(self, ni, nf):
        super().__init__()
        self.act_fn = nn.ReLU(True)
        self.conv = conv(ni, nf, 1)
        self.bn = nn.BatchNorm1d(nf)

    def forward(self, inp, out):
        return self.act_fn(out + self.bn(self.conv(inp)))


class InceptionBackbone(nn.Module):
    def __init__(self, input_channels, kss, depth, bottleneck_size, nb_filters, use_residual):
        super().__init__()

        self.depth = depth  # under normal init it will be 6
        assert (depth % 3) == 0
        self.use_residual = use_residual  # usual nam true

        n_ks = len(kss) + 1  # 3 + 1 = 4
        self.im = nn.ModuleList([InceptionBlock1d(input_channels if d == 0 else n_ks * nb_filters, nb_filters=nb_filters, kss=kss, bottleneck_size=bottleneck_size) for d in range(depth)])
        self.sk = nn.ModuleList([Shortcut1d(input_channels if d == 0 else n_ks * nb_filters, n_ks * nb_filters) for d in range(depth // 3)])

    def forward(self, x):

        input_res = x
        for d in range(self.depth):
            x = self.im[d](x)
            if self.use_residual and d % 3 == 2:
                x = (self.sk[d // 3])(input_res, x)
                input_res = x.clone()
        return x


class Inception1d(nn.Module):
    """inception time architecture"""

    def __init__(self, num_classes=5, input_channels=8, kernel_size=40, depth=6, bottleneck_size=32, nb_filters=32, use_residual=True, lin_ftrs_head=None, ps_head=0.5, bn_final_head=False, bn_head=True, act_head="relu", concat_pooling=True):
        super().__init__()
        assert kernel_size >= 40
        kernel_size = [k - 1 if k % 2 == 0 else k for k in [kernel_size, kernel_size // 2, kernel_size // 4]]  # is 39,19,9
        layers = [InceptionBackbone(input_channels=input_channels, kss=kernel_size, depth=depth, bottleneck_size=bottleneck_size, nb_filters=nb_filters, use_residual=use_residual)]

        n_ks = len(kernel_size) + 1  # value is 4

        # head
        head = create_head1d(n_ks * nb_filters, nc=num_classes, lin_ftrs=lin_ftrs_head, ps=ps_head, bn_final=bn_final_head, bn=bn_head, act=act_head, concat_pooling=concat_pooling)
        layers.append(head)

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Inception1dCombined(nn.Module):
    """inception time architecture"""

    def __init__(self, model1, model2, model3, model4):
        super().__init__()
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3
        self.model4 = model4
        
        self.last_inception = Inception1d(num_classes=5, input_channels=4, use_residual=True, ps_head=0.5, lin_ftrs_head=[128], kernel_size=40).to("cuda:0")

        self.layers = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 5),
            # nn.Softmax()
        )

    def forward(self, x):
        model1_output = self.model1(x)
        model2_output = self.model2(x)
        model3_output = self.model3(x)
        model4_output = self.model4(x)

        # combined_output = torch.cat((model1_output, model2_output, model3_output, model4_output), dim=1)
        
        # stach the output from the models and create a matrix with dimension 4
        combined_output = torch.stack((model1_output, model2_output, model3_output, model4_output), dim=1)
        
        # x = self.layers(combined_output)
        x = self.last_inception(combined_output)
        
        return x
