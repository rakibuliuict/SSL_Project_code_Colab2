import torch
from torch import nn
import torch.nn.functional as F


def center_crop_and_add(upsampled, encoder_feature):
    """Crop encoder feature map to match upsampled size and add."""
    up_shape = upsampled.shape[2:]
    enc_shape = encoder_feature.shape[2:]
    crop_slices = []
    for i in range(3):
        delta = enc_shape[i] - up_shape[i]
        if delta < 0:
            raise ValueError(f"Upsampled size {up_shape} is larger than encoder feature size {enc_shape} at dim {i}")
        start = delta // 2
        end = start + up_shape[i]
        crop_slices.append(slice(start, end))
    encoder_cropped = encoder_feature[:, :, crop_slices[0], crop_slices[1], crop_slices[2]]
    return upsampled + encoder_cropped


class ConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ConvBlock, self).__init__()
        ops = []
        for i in range(n_stages):
            in_channels = n_filters_in if i == 0 else n_filters_out
            ops.append(nn.Conv3d(in_channels, n_filters_out, kernel_size=3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                raise ValueError("Unknown normalization")
            ops.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        return self.conv(x)


class DownsamplingConvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=(2, 2, 1), normalization='none'):
        super(DownsamplingConvBlock, self).__init__()
        ops = [nn.Conv3d(n_filters_in, n_filters_out, kernel_size=3, stride=stride, padding=1)]
        if normalization == 'batchnorm':
            ops.append(nn.BatchNorm3d(n_filters_out))
        elif normalization == 'groupnorm':
            ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
        elif normalization == 'instancenorm':
            ops.append(nn.InstanceNorm3d(n_filters_out))
        elif normalization != 'none':
            raise ValueError("Unknown normalization")
        ops.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        return self.conv(x)


class InterpolateUpBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, normalization='none'):
        super(InterpolateUpBlock, self).__init__()
        ops = [nn.Conv3d(n_filters_in, n_filters_out, kernel_size=1)]
        if normalization == 'batchnorm':
            ops.append(nn.BatchNorm3d(n_filters_out))
        elif normalization == 'groupnorm':
            ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
        elif normalization == 'instancenorm':
            ops.append(nn.InstanceNorm3d(n_filters_out))
        elif normalization != 'none':
            raise ValueError("Unknown normalization")
        ops.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*ops)

    def forward(self, x, target_shape):
        x = F.interpolate(x, size=target_shape, mode='trilinear', align_corners=False)
        return self.conv(x)


class VNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=2, n_filters=16, normalization='instancenorm', has_dropout=False):
        super(VNet, self).__init__()
        self.has_dropout = has_dropout

        self.block_one = ConvBlock(1, n_channels, n_filters, normalization)
        self.block_one_dw = DownsamplingConvBlock(n_filters, n_filters * 2, normalization=normalization)

        self.block_two = ConvBlock(2, n_filters * 2, n_filters * 2, normalization)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization)

        self.block_three = ConvBlock(3, n_filters * 4, n_filters * 4, normalization)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization)

        self.block_four = ConvBlock(3, n_filters * 8, n_filters * 8, normalization)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization)

        self.block_five = ConvBlock(3, n_filters * 16, n_filters * 16, normalization)
        self.block_five_up = InterpolateUpBlock(n_filters * 16, n_filters * 8, normalization)

        self.block_six = ConvBlock(3, n_filters * 8, n_filters * 8, normalization)
        self.block_six_up = InterpolateUpBlock(n_filters * 8, n_filters * 4, normalization)

        self.block_seven = ConvBlock(3, n_filters * 4, n_filters * 4, normalization)
        self.block_seven_up = InterpolateUpBlock(n_filters * 4, n_filters * 2, normalization)

        self.block_eight = ConvBlock(2, n_filters * 2, n_filters * 2, normalization)
        self.block_eight_up = InterpolateUpBlock(n_filters * 2, n_filters, normalization)

        self.branchs = nn.ModuleList([
            nn.Sequential(
                ConvBlock(1, n_filters, n_filters, normalization),
                nn.Dropout3d(p=0.5) if has_dropout else nn.Identity(),
                nn.Conv3d(n_filters, n_classes, kernel_size=1)
            )
        ])

        if has_dropout:
            self.dropout = nn.Dropout3d(p=0.5)

    def encoder(self, x):
        x1 = self.block_one(x)
        x2 = self.block_one_dw(x1)
        x3 = self.block_two(x2)
        x4 = self.block_two_dw(x3)
        x5 = self.block_three(x4)
        x6 = self.block_three_dw(x5)
        x7 = self.block_four(x6)
        x8 = self.block_four_dw(x7)
        x9 = self.block_five(x8)
        if self.has_dropout:
            x9 = self.dropout(x9)
        return [x1, x3, x5, x7, x9]

    def decoder(self, features):
        x1, x3, x5, x7, x9 = features
        d1 = self.block_five_up(x9, x7.shape[2:])
        d1 = d1 + x7
        d2 = self.block_six(d1)

        d3 = self.block_six_up(d2, x5.shape[2:])
        d3 = d3 + x5
        d4 = self.block_seven(d3)

        d5 = self.block_seven_up(d4, x3.shape[2:])
        d5 = d5 + x3
        d6 = self.block_eight(d5)

        d7 = self.block_eight_up(d6, x1.shape[2:])
        d7 = d7 + x1

        out = [branch(d7) for branch in self.branchs]
        out.append(d2)
        return out

    def forward(self, x, turnoff_drop=False):
        if turnoff_drop:
            prev_state = self.has_dropout
            self.has_dropout = False
        features = self.encoder(x)
        out = self.decoder(features)
        if turnoff_drop:
            self.has_dropout = prev_state
        return out