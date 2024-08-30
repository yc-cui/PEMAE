import os
from einops import rearrange
from torch.jit import Final
import torch.nn.functional as F
import torch
import torch.nn as nn
from timm.models.layers import DropPath
from mmcv.ops import ModulatedDeformConv2dPack as Conv2d


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = Conv2d(in_features, hidden_features, kernel_size=1)
        self.act = act_layer()
        self.fc2 = Conv2d(hidden_features, out_features, kernel_size=1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, relu_slope=0.2, use_HIN=True):
        super(ResidualBlock, self).__init__()
        self.identity = Conv2d(in_channels, in_channels, 1, 1, 0)

        self.conv_1 = Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)
        if use_HIN:
            self.norm = nn.InstanceNorm2d(in_channels // 2, affine=True)
        self.use_HIN = use_HIN

    def forward(self, x):
        resi = self.relu_1(self.conv_1(x))
        out_1, out_2 = torch.chunk(resi, 2, dim=1)
        resi = torch.cat([self.norm(out_1), out_2], dim=1)
        resi = self.relu_2(self.conv_2(resi))
        return x + resi


class PatchEmbed(nn.Module):
    def __init__(self, img_size, ms_chans, embed_dim, patch_size=4):
        super().__init__()

        self.proj = nn.Sequential(
            Conv2d(ms_chans, embed_dim, kernel_size=3, stride=1, padding=1),
            ResidualBlock(in_channels=embed_dim),
            ResidualBlock(in_channels=embed_dim),
            ResidualBlock(in_channels=embed_dim),
        )

    def forward(self, x):
        x = self.proj(x)

        return x

# ++++++++++++++++++++++++++++++++++++++


class Attention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            dim,
            num_heads=8,
            qk_norm=False,
            attn_drop=0.,
            norm_layer=nn.LayerNorm,
            focus=6,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.proj_in_q = Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.proj_in_k = Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.proj_in_v = Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)

        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_x = Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.proj_p = Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)

        self.softmax = nn.Softmax(dim=-1)
        self.focusing_factor = focus

        self.kernel_function = nn.LeakyReLU(negative_slope=0.1)

        kernel_size = 3
        self.dwc = Conv2d(
            in_channels=self.head_dim,
            out_channels=self.head_dim,
            kernel_size=kernel_size,
            groups=self.head_dim,
            padding=kernel_size // 2)

    def forward(self, x, p):
        '''
        :param x: [batch_size, c, h, w]
        :param p: [batch_size, c, h, w]
        :return:
        '''

        assert x.shape == p.shape, f'shape mismatch, got ms of shape {x.shape}, pan of shape {p.shape}'
        b, c, h, w = x.shape

        q = self.proj_in_q(x)   # [batch_size, c, h, w] = [3, 512, 512, 512]
        k = self.proj_in_k(p)   # [batch_size, c, h, w] = [3, 512, 512, 512]
        v = self.proj_in_v(p)   # [batch_size, c, h, w] = [3, 512, 512, 512]

        q = rearrange(q, 'b c h w -> b (h w) c')   # [batch_size, h*w, c] = [3, 262144, 512]
        k = rearrange(k, 'b c h w -> b (h w) c')   # [batch_size, h*w, c] = [3, 262144, 512]
        v = rearrange(v, 'b c h w -> b (h w) c')   # [batch_size, h*w, c] = [3, 262144, 512]

        focusing_factor = self.focusing_factor

        q = self.kernel_function(q)
        k = self.kernel_function(k)
        q_norm = q.norm(dim=-1, keepdim=True)
        k_norm = k.norm(dim=-1, keepdim=True)

        if float(focusing_factor) <= 6:
            q = q ** focusing_factor
            k = k ** focusing_factor
        else:
            q = (q / q.max(dim=-1, keepdim=True)[0]) ** focusing_factor
            k = (k / k.max(dim=-1, keepdim=True)[0]) ** focusing_factor

        q = (q / q.norm(dim=-1, keepdim=True)) * q_norm
        k = (k / k.norm(dim=-1, keepdim=True)) * k_norm
        q = q / q.norm(dim=-1, keepdim=True)
        k = k / k.norm(dim=-1, keepdim=True)
        v = v / v.norm(dim=-1, keepdim=True)
        q, k, v = (rearrange(x, "b n (h c) -> (b h) n c", h=self.num_heads) for x in [q, k, v])
        i, j, c, d = q.shape[-2], k.shape[-2], k.shape[-1], v.shape[-1]

        z = 1 / (torch.einsum("b i c, b c -> b i", q, k.sum(dim=1)) + 1e-6)
        if i * j * (c + d) > c * d * (i + j):
            kv = torch.einsum("b j c, b j d -> b c d", k, v)
            x = torch.einsum("b i c, b c d, b i -> b i d", q, kv, z)
        else:
            qk = torch.einsum("b i c, b j c -> b i j", q, k)
            x = torch.einsum("b i j, b j d, b i -> b i d", qk, v, z)

        num = int(v.shape[1] ** 0.5)
        feature_map = rearrange(v, "b (w h) c -> b c w h", w=num, h=num)
        feature_map = rearrange(self.dwc(feature_map), "b c w h -> b (w h) c")

        x = x + feature_map

        x = rearrange(x, "(b h) n c -> b n (h c)", h=self.num_heads)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)   # [batch_size, c, h, w]
        feature_map = rearrange(feature_map, "(b h) n c -> b n (h c)", h=self.num_heads)
        feature_map = rearrange(feature_map, 'b (h w) c -> b c h w', h=h, w=w)   # [batch_size, c, h, w]

        x = self.proj_x(x)
        p = self.proj_p(feature_map)
        return x, p


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class LayerNorm2d(nn.LayerNorm):
    """LayerNorm on channels for 2d images.
    Args:
        num_channels (int): The number of channels of the input tensor.
        eps (float): a value added to the denominator for numerical stability.
            Defaults to 1e-5.
        elementwise_affine (bool): a boolean value that when set to ``True``,
            this module has learnable per-element affine parameters initialized
            to ones (for weights) and zeros (for biases). Defaults to True.
    """

    def __init__(self, num_channels: int, **kwargs) -> None:
        super().__init__(num_channels, **kwargs)
        self.num_channels = self.normalized_shape[0]

    def forward(self, x):
        assert x.dim() == 4, 'LayerNorm2d only supports inputs with shape ' \
            f'(N, C, H, W), but got tensor with shape {x.shape}'
        return F.layer_norm(
            x.permute(0, 2, 3, 1), self.normalized_shape, self.weight,
            self.bias, self.eps).permute(0, 3, 1, 2)


class Block(nn.Module):

    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.,
            qk_norm=False,
            proj_drop=0.,
            attn_drop=0.,
            init_values=None,
            drop_path=0.1,
            act_layer=nn.GELU,
            norm_layer=LayerNorm2d,
            mlp_layer=Mlp,
            focus=6,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            norm_layer=norm_layer,
            focus=focus
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm3 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, p):
        at, p = self.attn(self.norm1(x), self.norm2(p))
        x = x + self.drop_path1(self.ls1(at))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm3(x))))
        return x, p


class Refine(nn.Module):
    def __init__(self, n_feat, out_channel):
        super(Refine, self).__init__()

        self.conv_in = Conv2d(n_feat, n_feat, 3, stride=1, padding=1)
        self.process = nn.Sequential(
            CALayer(n_feat, 4))
        self.conv_last = Conv2d(in_channels=n_feat, out_channels=out_channel, kernel_size=3, stride=1, padding=1)

    def forward(self, x):

        out = self.conv_in(x)
        out = self.process(out)
        out = self.conv_last(out)

        return out


class CALayer(nn.Module):
    def __init__(self, channel, reduction):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.process = nn.Sequential(
            Conv2d(channel, channel, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            Conv2d(channel, channel, 3, stride=1, padding=1)
        )

    def forward(self, x):
        y = self.process(x)
        y = self.avg_pool(y)
        z = self.conv_du(y)
        return z * y + x


def check_and_make(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


def regularize_inputs(*args):
    output = []
    for v in args:
        output.append(torch.clip(v, 0., 1.))
    return tuple(output)
