from torch import nn
import math
from vit_pytorch import ViT
from einops import rearrange
import torch

#################################### MobileNetV2 ####################################
def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv3d(inp, oup, kernel_size=3, stride=stride, padding=(1, 1, 1), bias=False),
        nn.BatchNorm3d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv3d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm3d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == (1, 1, 1) and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv3d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm3d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv3d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm3d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv3d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm3d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv3d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm3d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv3d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm3d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, width_mult=1., last_channel=128):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 16
        last_channel = last_channel
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, (1, 1, 1)],
            [6, 24, 2, (2, 2, 2)],
            [6, 32, 3, (2, 2, 2)],
            [6, 64, 4, (2, 2, 2)],
            [6, 80, 3, (1, 1, 1)],
            [6, 160, 3, (2, 2, 2)],
            [6, 256, 1, (1, 1, 1)],
        ]

        # building first layer
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(1, input_channel, (2, 2, 2))]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else (1, 1, 1)
                self.features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1x1_bn(input_channel, self.last_channel))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


#################################### sNet ####################################
class sNet(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, dim // 8, kernel_size=(3, 3, 3), padding=1),
            nn.GroupNorm(16, dim // 8),
            nn.ReLU(),
            nn.Conv3d(dim // 8, dim // 8, kernel_size=(3, 3, 3), padding=1),
            nn.GroupNorm(16, dim // 8),
            nn.ReLU(),
            nn.MaxPool3d(2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(dim // 8, dim // 4, kernel_size=(3, 3, 3), padding=1),
            nn.GroupNorm(16, dim // 4),
            nn.ReLU(),
            nn.Conv3d(dim // 4, dim // 4, kernel_size=(3, 3, 3), padding=1),
            nn.GroupNorm(16, dim // 4),
            nn.ReLU(),
            nn.MaxPool3d(2, stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(dim // 4, dim // 2, kernel_size=(3, 3, 3), padding=1),
            nn.GroupNorm(16, dim // 2),
            nn.ReLU(),
            nn.Conv3d(dim // 2, dim // 2, kernel_size=(3, 3, 3), padding=1),
            nn.GroupNorm(16, dim // 2),
            nn.ReLU(),
            nn.MaxPool3d(2, stride=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv3d(dim // 2, dim, kernel_size=(3, 3, 3), padding=1),
            nn.GroupNorm(16, dim),
            nn.ReLU(),
            nn.Conv3d(dim, dim, kernel_size=(3, 3, 3), padding=1),
            nn.GroupNorm(16, dim),
            nn.ReLU(),
            nn.MaxPool3d(2, stride=2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=(3, 3, 3), padding=1, stride=2),
            nn.GroupNorm(16, dim),
            nn.ReLU()
        )

    def forward(self, mri):
        conv1_out = self.conv1(mri)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        conv4_out = self.conv4(conv3_out)
        conv5_out = self.conv5(conv4_out)
        return conv5_out


#################################### sFCN ####################################
class SFCN(nn.Module):
    def __init__(self, channel_number=None):
        super(SFCN, self).__init__()
        if channel_number is None:
            channel_number = [32, 64, 128, 256, 256, 64]
        n_layer = len(channel_number)
        self.feature_extractor = nn.Sequential()
        for i in range(n_layer):
            if i == 0:
                in_channel = 1
            else:
                in_channel = channel_number[i - 1]
            out_channel = channel_number[i]
            if i < n_layer - 1:
                self.feature_extractor.add_module('conv_%d' % i,
                                                  self.conv_layer(in_channel,
                                                                  out_channel,
                                                                  maxpool=True,
                                                                  kernel_size=3,
                                                                  padding=1))
            else:
                self.feature_extractor.add_module('conv_%d' % i,
                                                  self.conv_layer(in_channel,
                                                                  out_channel,
                                                                  maxpool=False,
                                                                  kernel_size=1,
                                                                  padding=0))

    @staticmethod
    def conv_layer(in_channel, out_channel, maxpool=True, kernel_size=3, padding=0, maxpool_stride=2):
        if maxpool is True:
            layer = nn.Sequential(
                nn.Conv3d(in_channel, out_channel, padding=padding, kernel_size=kernel_size),
                nn.BatchNorm3d(out_channel),
                nn.MaxPool3d(2, stride=maxpool_stride),
                nn.ReLU(),
            )
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channel, out_channel, padding=padding, kernel_size=kernel_size),
                nn.BatchNorm3d(out_channel),
                nn.ReLU()
            )
        return layer

    def forward(self, x):
        x_f = self.feature_extractor(x)
        return x_f

#######################################ADViT#################################
class ADVIT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.to_2d_mri = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=(1, 1, 25), stride=(1, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 1, 2), stride=(1, 1, 2)),
            nn.Conv3d(32, 1, kernel_size=(1, 1, 25), stride=(1, 1, 1)),
            nn.BatchNorm3d(1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 1, 2), stride=(1, 1, 2))
        )
        self.to_2d_pet = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=(1, 1, 25), stride=(1, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 1, 2), stride=(1, 1, 2)),
            nn.Conv3d(32, 1, kernel_size=(1, 1, 25), stride=(1, 1, 1)),
            nn.BatchNorm3d(1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 1, 2), stride=(1, 1, 2))
        )
        self.vit_mri = ViT(
            image_size=128,
            patch_size=16,
            num_classes=2,
            channels=1,
            dim=192,
            depth=6,
            heads=3,
            mlp_dim=768,
            dropout=0.1,
            emb_dropout=0.1)
        self.vit_pet = ViT(
            image_size=128,
            patch_size=16,
            num_classes=2,
            channels=1,
            dim=192,
            depth=6,
            heads=3,
            mlp_dim=768,
            dropout=0.1,
            emb_dropout=0.1)
        self.fc = nn.Linear(192*2, 2)

    def forward(self, mri,pet):
        mri_out = self.to_2d_mri(mri)
        pet_out = self.to_2d_pet(pet)
        mri_out = rearrange(mri_out, 'b c h w d -> b c h (w d)')
        pet_out = rearrange(pet_out, 'b c h w d -> b c h (w d)')
        mri_out = self.vit_mri(mri_out)
        pet_out = self.vit_pet(pet_out)
        logits = self.fc(torch.cat([mri_out, pet_out], dim=-1))

        return logits