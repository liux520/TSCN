import torch
import torch.nn as nn
import math
import warnings
from itertools import repeat
import collections.abc
import torch.nn.functional as F
import torchvision.io
from demo.Transformer_Block import constant_init, xavier_init, normal_init, trunc_normal_init, drop_path, DropPath, \
    to_2tuple
from typing import Type


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in_ = nn.Conv2d(dim, hidden_features * 2, 1, bias=bias)
        self.dwconv_ = nn.Conv2d(hidden_features * 2, hidden_features * 2, 3, 1, 1, groups=hidden_features * 2,
                                 bias=bias)
        self.project_out_ = nn.Conv2d(hidden_features, dim, 1, bias=bias)
        self.attn = BiAttn(dim)  # eca(dim)

    def forward(self, x):
        x = self.project_in_(x)
        x1, x2 = self.dwconv_(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out_(x)
        x = self.attn(x)
        return x


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self,
                 normalized_shape,
                 eps=1e-6,
                 data_format="channels_first"):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        assert self.data_format in ["channels_last", "channels_first"]
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class DWConv(nn.Module):
    def __init__(self, dim=64):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x


class BiAttn(nn.Module):
    def __init__(self, in_channels, act_ratio=0.25, act_fn=nn.GELU, gate_fn=nn.Sigmoid):
        super(BiAttn, self).__init__()
        reduce_channels = int(in_channels * act_ratio)
        self.norm = LayerNorm(in_channels, data_format='channels_first')  # nn.LayerNorm(in_channels)
        self.global_reduce = nn.Conv2d(in_channels, reduce_channels, 1)
        self.local_reduce = nn.Conv2d(in_channels, reduce_channels, 1)
        self.act_fn = act_fn()
        self.channel_select = nn.Conv2d(reduce_channels, in_channels, 1)
        self.spatial_select = nn.Conv2d(reduce_channels * 2, 1, 1)
        self.gate_fn = gate_fn()

    def forward(self, x):
        ori_x = x
        x = self.norm(x)
        x_global = F.adaptive_avg_pool2d(x, 1)  # x.mean(1, keepdim=True)
        x_global = self.act_fn(self.global_reduce(x_global))  # [b, 16, 1, 1]
        x_local = self.act_fn(self.local_reduce(x))  # [b, 16, h, w]

        c_attn = self.channel_select(x_global)  # [b, 64, 1, 1]
        c_attn = self.gate_fn(c_attn)  # [b, 64, 1, 1]
        s_attn = self.spatial_select(torch.cat([x_local, x_global.expand(-1, -1, x_local.shape[2], x_local.shape[3])],
                                               dim=1))  # .expand(-1, x.shape[1], -1)
        s_attn = self.gate_fn(s_attn)  # [B, N, 1]

        attn = c_attn * s_attn  # [B, N, C]
        return ori_x * attn


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.attn = BiAttn(out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)

        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.attn(x)
        x = self.drop(x)

        return x


class LKA(nn.Module):
    def __init__(self, dim):
        super(LKA, self).__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)

        return u * attn


class LKA_D(nn.Module):
    def __init__(self, dim, chunk=4):
        super(LKA_D, self).__init__()
        self.chunk = chunk
        c = dim // chunk
        self.LKA7 = nn.Sequential(
            nn.Conv2d(c, c, 7, 1, 7 // 2, groups=c),
            nn.Conv2d(c, c, 9, stride=1, padding=(9 // 2) * 4, groups=c, dilation=4),
        )
        self.LKA5 = nn.Sequential(
            nn.Conv2d(c, c, 5, 1, 5 // 2, groups=c),
            nn.Conv2d(c, c, 7, stride=1, padding=(7 // 2) * 3, groups=c, dilation=3),
        )
        self.LKA3 = nn.Sequential(
            nn.Conv2d(c, c, 3, 1, 1, groups=c),
            nn.Conv2d(c, c, 5, stride=1, padding=(5 // 2) * 2, groups=c, dilation=2),
        )

        self.conv1 = nn.Conv2d(c, c, 1, 1, 0)
        self.conv2 = nn.Conv2d(c, c, 1, 1, 0)
        self.conv3 = nn.Conv2d(c, c, 1, 1, 0)

        # self.act = nn.GELU()

    def forward(self, x):
        xs = torch.chunk(x, self.chunk, dim=1)
        ys = []
        for s in range(self.chunk):
            if s == 0:
                ys.append(xs[s])
            elif s == 1:
                ys.append((self.conv1((self.LKA3(xs[s] + ys[-1])))) * (xs[s] + ys[-1]))
            elif s == 2:
                ys.append((self.conv2((self.LKA5(xs[s] + ys[-1])))) * (xs[s] + ys[-1]))
            elif s == 3:
                ys.append((self.conv3((self.LKA7(xs[s] + ys[-1])))) * (xs[s] + ys[-1]))
        out = torch.cat(ys, 1)
        return out * x


class Attention_lka(nn.Module):
    def __init__(self, d_model):
        super(Attention_lka, self).__init__()

        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = LKA_D(d_model)  # LKA(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

        self.pixel_norm = nn.LayerNorm(d_model)
        # self.shuffle = Shuffle(d_model)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        # x = self.shuffle(x)
        x = x + shorcut

        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x = self.pixel_norm(x)
        x = x.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)
        return x


class AttentionModule(nn.Module):
    def __init__(self, dim):
        super(AttentionModule, self).__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv0_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)

        self.conv1_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)

        self.conv2_1 = nn.Conv2d(dim, dim, (1, 21), padding=(0, 10), groups=dim)
        self.conv2_2 = nn.Conv2d(dim, dim, (21, 1), padding=(10, 0), groups=dim)
        self.conv3 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)

        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)

        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)

        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)
        attn = attn + attn_0 + attn_1 + attn_2

        attn = self.conv3(attn)

        return attn * u


class SpatialAttention(nn.Module):
    def __init__(self, d_model):
        super(SpatialAttention, self).__init__()
        self.d_model = d_model
        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = AttentionModule(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x


class Block(nn.Module):
    def __init__(self,
                 dim: int,
                 mlp_ratio: float = 2.66,
                 drop: float = 0.,
                 drop_path: float = 0.,
                 act_layer: Type[nn.Module] = nn.GELU,
                 norm: Type[nn.Module] = nn.BatchNorm2d):
        super(Block, self).__init__()
        self.norm1 = norm(dim)
        self.attn = Attention_lka(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm(dim)
        self.mlp = FeedForward(dim, mlp_ratio, True)
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        return x


class TSCN(nn.Module):
    def __init__(self,
                 embed_dims: int = 64,
                 scale: int = 2,
                 depths: int = 1,
                 mlp_ratios: float = 2.66,
                 drop_rate: float = 0.,
                 drop_path_rate: float = 0.1,
                 num_stages: int = 16,
                 normc: Type[nn.Module] = nn.BatchNorm2d):
        super(TSCN, self).__init__()
        depths_ = [depths] * num_stages

        self.depths = depths
        self.num_stages = num_stages

        # stochastic depth decay rule  sum(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths_))]
        cur = 0

        self.head = nn.Sequential(
            nn.Conv2d(3, embed_dims, 3, 1, 1)
        )
        self.tail = nn.Sequential(
            nn.Conv2d(embed_dims, 3 * scale * scale, 3, 1, 1),  # nn.Conv2d(embed_dims, 3 * scale * scale, 3, 1, 1),
            nn.PixelShuffle(scale)
        )

        for i in range(self.num_stages):
            block = nn.ModuleList([Block(dim=embed_dims, mlp_ratio=mlp_ratios,
                                         drop=drop_rate, drop_path=0,  # dpr[cur + j],
                                         norm=normc)
                                   for j in range(depths_[i])])
            norm = nn.LayerNorm(embed_dims)
            cur += depths_[i]

            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

    def _grad(self):
        for name, param in self.named_parameters():
            # print(name, param.shape)
            if name.split(".")[0] == 'tail_':
                param.requires_grad = True
            else:
                param.requires_grad = False

        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_init(m, std=.02, bias=0.)
            elif isinstance(m, nn.LayerNorm):
                constant_init(m, val=1.0, bias=0.)
            elif isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                normal_init(m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)

    def forward(self, x):
        x = self.head(x)
        idn = x

        for i in range(self.num_stages):
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            for blk in block:
                x = blk(x)
            x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
            x = norm(x)
            x = x.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)

        output = idn + x  # outs[0] + outs[-1]
        output = self.tail(output)
        return output


def get_TSCN(scale):
    model = TSCN(scale=scale, num_stages=12, mlp_ratios=2.66)
    return model


if __name__ == '__main__':
    from demo.ntire.model_summary import get_model_flops, get_model_activation

    scale = 4
    model = get_TSCN(scale).cuda()  # TSCN(scale=scale, num_stages=16, mlp_ratios=4)
    h, w = 720 // scale, 1280 // scale
    x = torch.randn((1, 3, h, w)).cuda()
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name, param.shape)

    with torch.no_grad():
        out = model(x)
        print(out.shape)

        input_dim = (3, h, w)  # set the input dimension
        # activations, num_conv = get_model_activation(model, input_dim)
        # activations = activations / 10 ** 6
        # print("{:>16s} : {:<.4f} [M]".format("#Activations", activations))
        # print("{:>16s} : {:<d}".format("#Conv2d", num_conv))

        flops = get_model_flops(model, input_dim, False)
        flops = flops / 10 ** 9
        print("{:>16s} : {:<.4f} [G]".format("FLOPs", flops))

        num_parameters = sum(map(lambda x: x.numel(), model.parameters()))
        num_parameters = num_parameters / 10 ** 6
        print("{:>16s} : {:<.4f} [M]".format("#Params", num_parameters))