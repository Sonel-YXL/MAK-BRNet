import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Sequence


class GSiLU(nn.Module):
    """Global Sigmoid-Gated Linear Unit"""
    def __init__(self):
        super().__init__()
        self.adpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        return x * torch.sigmoid(self.adpool(x))


class CAA(nn.Module):
    """Context Anchor Attention"""
    def __init__(
            self,
            channels: int,
            h_kernel_size: int = 11,
            v_kernel_size: int = 11,
            norm_cfg: Optional[dict] = None,
            act_cfg: Optional[dict] = None,
            init_cfg: Optional[dict] = None,
    ):
        super().__init__()
        self.avg_pool = nn.AvgPool2d(7, 1, 3)
        self.conv1 = nn.Conv2d(channels, channels, 1, 1, 0)
        self.h_conv = nn.Conv2d(channels, channels, (1, h_kernel_size), 1, (0, h_kernel_size // 2), groups=channels)
        self.v_conv = nn.Conv2d(channels, channels, (v_kernel_size, 1), 1, (v_kernel_size // 2, 0), groups=channels)
        self.conv2 = nn.Conv2d(channels, channels, 1, 1, 0)
        self.act = nn.Sigmoid()

    def forward(self, x):
        attn_factor = self.act(self.conv2(self.v_conv(self.h_conv(self.conv1(self.avg_pool(x))))))
        return attn_factor


class ConvFFN(nn.Module):
    """Multi-layer perceptron implemented with ConvModule"""
    def __init__(
            self,
            in_channels: int,
            out_channels: Optional[int] = None,
            hidden_channels_scale: float = 4.0,
            hidden_kernel_size: int = 3,
            dropout_rate: float = 0.,
            add_identity: bool = True,
            norm_cfg: Optional[dict] = None,
            act_cfg: Optional[dict] = None,
            init_cfg: Optional[dict] = None,
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = int(in_channels * hidden_channels_scale)

        self.ffn_layers = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=hidden_kernel_size, stride=1,
                      padding=hidden_kernel_size // 2, groups=hidden_channels),
            GSiLU(),
            nn.Dropout(dropout_rate),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
        )
        self.add_identity = add_identity

    def forward(self, x):
        x = x + self.ffn_layers(x) if self.add_identity else self.ffn_layers(x)
        return x


class Stem(nn.Module):
    """Stem layer"""
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            expansion: float = 1.0,
            norm_cfg: Optional[dict] = None,
            act_cfg: Optional[dict] = None,
            init_cfg: Optional[dict] = None,
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)

        self.down_conv = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=2, padding=1)
        self.conv1 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        return self.conv2(self.conv1(self.down_conv(x)))


class DownSamplingLayer(nn.Module):
    """Down sampling layer"""
    def __init__(
            self,
            in_channels: int,
            out_channels: Optional[int] = None,
            norm_cfg: Optional[dict] = None,
            act_cfg: Optional[dict] = None,
            init_cfg: Optional[dict] = None,
    ):
        super().__init__()
        out_channels = out_channels or (in_channels * 2)

        self.down_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.down_conv(x)


class InceptionBottleneck(nn.Module):
    """Bottleneck with Inception module"""
    def __init__(
            self,
            in_channels: int,
            out_channels: Optional[int] = None,
            kernel_sizes: Sequence[int] = (3, 5, 7, 9, 11),
            dilations: Sequence[int] = (1, 1, 1, 1, 1),
            expansion: float = 1.0,
            add_identity: bool = True,
            with_caa: bool = True,
            caa_kernel_size: int = 11,
            norm_cfg: Optional[dict] = None,
            act_cfg: Optional[dict] = None,
            init_cfg: Optional[dict] = None,
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = int(out_channels * expansion)

        self.pre_conv = nn.Conv2d(in_channels, hidden_channels, 1, 1, 0)
        self.dw_conv = nn.Conv2d(hidden_channels, hidden_channels, kernel_sizes[0], 1,
                                 kernel_sizes[0] // 2, dilations[0], groups=hidden_channels)
        self.dw_conv1 = nn.Conv2d(hidden_channels, hidden_channels, kernel_sizes[1], 1,
                                  kernel_sizes[1] // 2, dilations[1], groups=hidden_channels)
        self.dw_conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_sizes[2], 1,
                                  kernel_sizes[2] // 2, dilations[2], groups=hidden_channels)
        self.dw_conv3 = nn.Conv2d(hidden_channels, hidden_channels, kernel_sizes[3], 1,
                                  kernel_sizes[3] // 2, dilations[3], groups=hidden_channels)
        self.dw_conv4 = nn.Conv2d(hidden_channels, hidden_channels, kernel_sizes[4], 1,
                                  kernel_sizes[4] // 2, dilations[4], groups=hidden_channels)
        self.pw_conv = nn.Conv2d(hidden_channels, hidden_channels, 1, 1, 0)

        if with_caa:
            self.caa_factor = CAA(hidden_channels, caa_kernel_size, caa_kernel_size)
        else:
            self.caa_factor = None

        self.add_identity = add_identity and in_channels == out_channels
        self.post_conv = nn.Conv2d(hidden_channels, out_channels, 1, 1, 0)

    def forward(self, x):
        x = self.pre_conv(x)
        y = x
        x = self.dw_conv(x)
        x = x + self.dw_conv1(x) + self.dw_conv2(x) + self.dw_conv3(x) + self.dw_conv4(x)
        x = self.pw_conv(x)
        if self.caa_factor is not None:
            y = self.caa_factor(y)
        if self.add_identity:
            y = x * y
            x = x + y
        else:
            x = x * y
        x = self.post_conv(x)
        return x


class PKIBlock(nn.Module):
    """Poly Kernel Inception Block"""
    def __init__(
            self,
            in_channels: int,
            out_channels: Optional[int] = None,
            kernel_sizes: Sequence[int] = (3, 5, 7, 9, 11),
            dilations: Sequence[int] = (1, 1, 1, 1, 1),
            with_caa: bool = True,
            caa_kernel_size: int = 11,
            expansion: float = 1.0,
            ffn_scale: float = 4.0,
            ffn_kernel_size: int = 3,
            dropout_rate: float = 0.,
            drop_path_rate: float = 0.,
            layer_scale: Optional[float] = 1.0,
            add_identity: bool = True,
            norm_cfg: Optional[dict] = None,
            act_cfg: Optional[dict] = None,
            init_cfg: Optional[dict] = None,
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = int(out_channels * expansion)

        self.norm1 = nn.BatchNorm2d(in_channels)
        self.norm2 = nn.BatchNorm2d(hidden_channels)

        self.block = InceptionBottleneck(in_channels, hidden_channels, kernel_sizes, dilations,
                                         expansion=1.0, add_identity=True,
                                         with_caa=with_caa, caa_kernel_size=caa_kernel_size)
        self.ffn = ConvFFN(hidden_channels, out_channels, ffn_scale, ffn_kernel_size, dropout_rate, add_identity=False)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

        self.layer_scale = layer_scale
        if self.layer_scale:
            self.gamma1 = nn.Parameter(layer_scale * torch.ones(hidden_channels), requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale * torch.ones(out_channels), requires_grad=True)
        self.add_identity = add_identity and in_channels == out_channels

    def forward(self, x):
        if self.layer_scale:
            if self.add_identity:
                x = x + self.drop_path(self.gamma1.unsqueeze(-1).unsqueeze(-1) * self.block(self.norm1(x)))
                x = x + self.drop_path(self.gamma2.unsqueeze(-1).unsqueeze(-1) * self.ffn(self.norm2(x)))
            else:
                x = self.drop_path(self.gamma1.unsqueeze(-1).unsqueeze(-1) * self.block(self.norm1(x)))
                x = self.drop_path(self.gamma2.unsqueeze(-1).unsqueeze(-1) * self.ffn(self.norm2(x)))
        else:
            if self.add_identity:
                x = x + self.drop_path(self.block(self.norm1(x)))
                x = x + self.drop_path(self.ffn(self.norm2(x)))
            else:
                x = self.drop_path(self.block(self.norm1(x)))
                x = self.drop_path(self.ffn(self.norm2(x)))
        return x


class PKIStage(nn.Module):
    """Poly Kernel Inception Stage"""
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            num_blocks: int,
            kernel_sizes: Sequence[int] = (3, 5, 7, 9, 11),
            dilations: Sequence[int] = (1, 1, 1, 1, 1),
            expansion: float = 0.5,
            ffn_scale: float = 4.0,
            ffn_kernel_size: int = 3,
            dropout_rate: float = 0.,
            drop_path_rate: Union[float, list] = 0.,
            layer_scale: Optional[float] = 1.0,
            shortcut_with_ffn: bool = True,
            shortcut_ffn_scale: float = 4.0,
            shortcut_ffn_kernel_size: int = 5,
            add_identity: bool = True,
            with_caa: bool = True,
            caa_kernel_size: int = 11,
            norm_cfg: Optional[dict] = None,
            act_cfg: Optional[dict] = None,
            init_cfg: Optional[dict] = None,
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)

        self.downsample = DownSamplingLayer(in_channels, out_channels)
        self.conv1 = nn.Conv2d(out_channels, 2 * hidden_channels, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(2 * hidden_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)

        self.ffn = ConvFFN(hidden_channels, hidden_channels, shortcut_ffn_scale, shortcut_ffn_kernel_size, 0.,
                           add_identity=True) if shortcut_with_ffn else None

        self.blocks = nn.ModuleList([
            PKIBlock(hidden_channels, hidden_channels, kernel_sizes, dilations, with_caa,
                     caa_kernel_size+2*i, 1.0, ffn_scale, ffn_kernel_size, dropout_rate,
                     drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate,
                     layer_scale, add_identity) for i in range(num_blocks)
        ])

    def forward(self, x):
        x = self.downsample(x)
        x, y = list(self.conv1(x).chunk(2, 1))
        if self.ffn is not None:
            x = self.ffn(x)

        z = [x]
        t = torch.zeros(y.shape, device=y.device)
        for block in self.blocks:
            t = t + block(y)
        z.append(t)
        z = torch.cat(z, dim=1)
        z = self.conv2(z)
        z = self.conv3(z)
        return z


class DropPath(nn.Module):
    """DropPath"""
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class PKINet(nn.Module):
    """Poly Kernel Inception Network"""
    arch_settings = {
        'T': [[16, 32, 4, (3, 5, 7, 9, 11), (1, 1, 1, 1, 1), 0.5, 4.0, 3, 0.1, 1.0, True, 8.0, 5, True, True, 11],
              [32, 64, 14, (3, 5, 7, 9, 11), (1, 1, 1, 1, 1), 0.5, 4.0, 3, 0.1, 1.0, True, 8.0, 7, True, True, 11],
              [64, 128, 22, (3, 5, 7, 9, 11), (1, 1, 1, 1, 1), 0.5, 4.0, 3, 0.1, 1.0, True, 4.0, 9, True, True, 11],
              [128, 256, 4, (3, 5, 7, 9, 11), (1, 1, 1, 1, 1), 0.5, 4.0, 3, 0.1, 1.0, True, 4.0, 11, True, True, 11]],

        'S': [[32, 64, 4, (3, 5, 7, 9, 11), (1, 1, 1, 1, 1), 0.5, 4.0, 3, 0.1, 1.0, True, 8.0, 5, True, True, 11],
              [64, 128, 12, (3, 5, 7, 9, 11), (1, 1, 1, 1, 1), 0.5, 4.0, 3, 0.1, 1.0, True, 8.0, 7, True, True, 11],
              [128, 256, 20, (3, 5, 7, 9, 11), (1, 1, 1, 1, 1), 0.5, 4.0, 3, 0.1, 1.0, True, 4.0, 9, True, True, 11],
              [256, 512, 4, (3, 5, 7, 9, 11), (1, 1, 1, 1, 1), 0.5, 4.0, 3, 0.1, 1.0, True, 4.0, 11, True, True, 11]],

        'B': [[40, 80, 6, (3, 5, 7, 9, 11), (1, 1, 1, 1, 1), 0.5, 4.0, 3, 0.1, 1.0, True, 8.0, 5, True, True, 11],
              [80, 160, 16, (3, 5, 7, 9, 11), (1, 1, 1, 1, 1), 0.5, 4.0, 3, 0.1, 1.0, True, 8.0, 7, True, True, 11],
              [160, 320, 24, (3, 5, 7, 9, 11), (1, 1, 1, 1, 1), 0.5, 4.0, 3, 0.1, 1.0, True, 4.0, 9, True, True, 11],
              [320, 640, 6, (3, 5, 7, 9, 11), (1, 1, 1, 1, 1), 0.5, 4.0, 3, 0.1, 1.0, True, 4.0, 11, True, True, 11]],
    }

    def __init__(
            self,
            arch: str = 'S',
            out_indices: Sequence[int] = (2, 3, 4),
            drop_path_rate: float = 0.1,
            frozen_stages: int = -1,
            norm_eval: bool = False,
            arch_setting: Optional[Sequence[list]] = None,
            init_cfg: Optional[dict] = None,
    ):
        super().__init__()
        arch_setting = arch_setting or self.arch_settings[arch]

        assert set(out_indices).issubset(i for i in range(len(arch_setting) + 1))
        if frozen_stages not in range(-1, len(arch_setting) + 1):
            raise ValueError(f'frozen_stages must be in range(-1, len(arch_setting) + 1). But received {frozen_stages}')

        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval
        self.stages = nn.ModuleList()

        self.stem = Stem(3, arch_setting[0][0], expansion=1.0)
        self.stages.append(self.stem)

        depths = [x[2] for x in arch_setting]
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        for i, (in_channels, out_channels, num_blocks, kernel_sizes, dilations, expansion, ffn_scale, ffn_kernel_size,
                dropout_rate, layer_scale, shortcut_with_ffn, shortcut_ffn_scale, shortcut_ffn_kernel_size,
                add_identity, with_caa, caa_kernel_size) in enumerate(arch_setting):
            stage = PKIStage(in_channels, out_channels, num_blocks, kernel_sizes, dilations, expansion,
                             ffn_scale, ffn_kernel_size, dropout_rate, dpr[sum(depths[:i]):sum(depths[:i + 1])],
                             layer_scale, shortcut_with_ffn, shortcut_ffn_scale, shortcut_ffn_kernel_size,
                             add_identity, with_caa, caa_kernel_size)
            self.stages.append(stage)

    def forward(self, x):
        outs = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5), mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def train(self, mode=True):
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for i in range(self.frozen_stages):
                m = self.stages[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False


def load_pretrained_model(model, pretrained_path):
    """
    加载预训练模型权重到指定模型。

    参数:
        model (nn.Module): 目标模型。
        pretrained_path (str): 预训练模型权重文件路径。
    """
    if not pretrained_path:
        raise ValueError("预训练模型路径不能为空！")

    
    pretrained_dict = torch.load(pretrained_path, map_location=torch.device('cpu'))

    
    model_dict = model.state_dict()

    
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    print(f"预训练模型已成功加载：{pretrained_path}")
    return model


if __name__ == "__main__":
    
    model = PKINet(arch='T', out_indices=(1, 2, 3))

    
    pretrained_path = "/home/sonel/code/Sonel_code/OOD_SOTA/Models/Debug_model/pkinet_t_pretrain.pth"
    model = load_pretrained_model(model, pretrained_path)

    
    model.eval()
    input_tensor = torch.randn(1, 3, 224, 224)
    outputs = model(input_tensor)

    for idx, output in enumerate(outputs):
        print(f"Output {idx} shape: {output.shape}")