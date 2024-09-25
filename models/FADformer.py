import torch
import torch.nn as nn

def get_residue(tensor , r_dim = 1):
    max_channel = torch.max(tensor, dim=r_dim, keepdim=True)
    min_channel = torch.min(tensor, dim=r_dim, keepdim=True)
    res_channel = max_channel[0] - min_channel[0]
    return res_channel


class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if kernel_size is None:
            kernel_size = patch_size

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=patch_size,
                              padding=(kernel_size - patch_size + 1) // 2, padding_mode='reflect')

    def forward(self, x):
        x = self.proj(x)
        return x


class PatchUnEmbed(nn.Module):
    def __init__(self, patch_size=4, out_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        self.out_chans = out_chans
        self.embed_dim = embed_dim

        if kernel_size is None:
            kernel_size = 1

        self.proj = nn.Sequential(
            nn.Conv2d(embed_dim, out_chans * patch_size ** 2, kernel_size=kernel_size,
                      padding=kernel_size // 2, padding_mode='reflect'),
            # nn.PixelShuffle(patch_size),
            nn.Conv2d(out_chans, out_chans, kernel_size=kernel_size,
                      padding=kernel_size // 2, padding_mode='reflect'),
        )

    def forward(self, x):
        x = self.proj(x)
        return x


class PatchUnEmbed_for_upsample(nn.Module):
    def __init__(self, patch_size=4, embed_dim=96, out_dim=64, kernel_size=None):
        super().__init__()
        self.embed_dim = embed_dim

        if kernel_size is None:
            kernel_size = 1

        self.proj = nn.Sequential(
            nn.Conv2d(embed_dim, out_dim * patch_size ** 2, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.PixelShuffle(patch_size),
        )

    def forward(self, x):
        x = self.proj(x)
        return x


class DownSample(nn.Module):
    """
    DownSample: Conv
    B*H*W*C -> B*(H/2)*(W/2)*(2*C)
    """

    def __init__(self, input_dim, output_dim, kernel_size=4, stride=2):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = output_dim

        self.proj = nn.Sequential(nn.Conv2d(input_dim, input_dim // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        x = self.proj(x)
        return x


class DownSample_RCP(nn.Module):
    """
    DownSample: Conv
    B*H*W*C -> B*(H/2)*(W/2)*(2*C)
    """

    def __init__(self, input_dim=4, output_dim=1, kernel_size=4, stride=2):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = output_dim

        self.proj = nn.Sequential(nn.PixelUnshuffle(2),
                                  nn.Conv2d(4, 1, kernel_size=3, stride=1, padding=1, bias=False),
                                  )


    def forward(self, x):
        x = self.proj(x)
        return x

class Upsample_RCP(nn.Module):
    def __init__(self, patch_size=4, embed_dim=96, out_dim=64, kernel_size=None):
        super().__init__()
        self.embed_dim = embed_dim

        if kernel_size is None:
            kernel_size = 1

        self.proj = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        x = self.proj(x)
        return x


class Prior_Gated_Feed_forward_Network(nn.Module):
    def __init__(
            self,
            dim,
            kernel_size=[1,3,5,7],
            se_ratio=4,
            local_size=8,
            scale_ratio=2,
            spilt_num=4
    ):
        super(Prior_Gated_Feed_forward_Network, self).__init__()
        self.dim = dim
        self.c_down_ratio = se_ratio
        self.size = local_size
        self.dim_sp = dim*scale_ratio//spilt_num
        self.conv_init = nn.Sequential(  # PW->DW->
            nn.Conv2d(dim, dim*2, 1),
            nn.GELU()
        )
        self.conv_fina = nn.Sequential(
            nn.Conv2d(dim*2, dim, 1),
            nn.GELU()
        )
        self.conv_dw = nn.Sequential(
            nn.Conv2d(dim*2, dim*2, kernel_size=3, padding=3 // 2, groups=dim*2,
                      padding_mode='reflect'),
            nn.GELU()
        )
        self.mask_in = nn.Sequential(
            nn.Conv2d(1, self.dim_sp, 1),
            nn.GELU()
        )
        self.mask_dw_conv_1 = nn.Sequential(
            nn.Conv2d(self.dim_sp//2, 1, kernel_size=3, padding=3 // 2, padding_mode='reflect'),
            nn.Sigmoid()
        )
        self.mask_dw_conv_2 = nn.Sequential(
            nn.Conv2d(self.dim_sp // 2, 1, kernel_size=5, padding=5 // 2, padding_mode='reflect'),
            nn.Sigmoid()
        )
        self.mask_out = nn.Sequential(
            nn.Conv2d(2, 1, 1),
            nn.GELU()
        )

    def forward(self, x, mask):
        x = self.conv_init(x)
        x = self.conv_dw(x)
        x = list(torch.split(x, self.dim, dim=1))
        mask = self.mask_in(mask)
        mask = list(torch.split(mask, self.dim_sp//2, dim=1))
        mask[0] = self.mask_dw_conv_1(mask[0])
        mask[1] = self.mask_dw_conv_2(mask[1])
        x[0] = mask[0] * x[0]
        x[1] = mask[1] * x[1]
        x = torch.cat(x, dim=1)
        x = self.conv_fina(x)
        mask = self.mask_out(torch.cat(mask, dim=1))

        return x, mask


class FourierUnit(nn.Module):

    def __init__(self, in_channels, out_channels, groups=1):
        super(FourierUnit, self).__init__()
        self.groups = groups
        self.conv_layer = torch.nn.Conv2d(in_channels=in_channels * 2, out_channels=out_channels * 2,
                                          kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)
        self.bn = torch.nn.BatchNorm2d(out_channels * 2)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        batch, c, h, w = x.size()

        # (batch, c, h, w/2+1, 2)
        ffted = torch.fft.rfft2(x, norm='ortho')
        x_fft_real = torch.unsqueeze(torch.real(ffted), dim=-1)
        x_fft_imag = torch.unsqueeze(torch.imag(ffted), dim=-1)
        ffted = torch.cat((x_fft_real, x_fft_imag), dim=-1)
        # (batch, c, 2, h, w/2+1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])

        ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
        ffted = self.relu(self.bn(ffted))

        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(
            0, 1, 3, 4, 2).contiguous()  # (batch,c, t, h, w/2+1, 2)
        ffted = torch.view_as_complex(ffted)

        output = torch.fft.irfft2(ffted, s=(h, w), norm='ortho')

        return output


class Freq_Fusion(nn.Module):
    def __init__(
            self,
            dim,
            kernel_size=[1,3,5,7],
            se_ratio=4,
            local_size=8,
            scale_ratio=2,
            spilt_num=4
    ):
        super(Freq_Fusion, self).__init__()
        self.dim = dim
        self.c_down_ratio = se_ratio
        self.size = local_size
        self.dim_sp = dim*scale_ratio//spilt_num
        self.conv_init_1 = nn.Sequential(  # PW
            nn.Conv2d(dim, dim, 1),
            nn.GELU()
        )
        self.conv_init_2 = nn.Sequential(  # DW
            nn.Conv2d(dim, dim, 1),
            nn.GELU()
        )
        self.conv_mid = nn.Sequential(
            nn.Conv2d(dim*2, dim, 1),
            nn.GELU()
        )
        self.FFC = FourierUnit(self.dim*2, self.dim*2)

        self.bn = torch.nn.BatchNorm2d(dim*2)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        x_1, x_2 = torch.split(x, self.dim, dim=1)
        x_1 = self.conv_init_1(x_1)
        x_2 = self.conv_init_2(x_2)
        x0 = torch.cat([x_1, x_2], dim=1)
        x = self.FFC(x0) + x0
        x = self.relu(self.bn(x))

        return x


class Fused_Fourier_Conv_Mixer(nn.Module):
    def __init__(
            self,
            dim,
            token_mixer_for_gloal=Freq_Fusion,
            mixer_kernel_size=[1,3,5,7],
            local_size=8
    ):
        super(Fused_Fourier_Conv_Mixer, self).__init__()
        self.dim = dim
        self.mixer_gloal = token_mixer_for_gloal(dim=self.dim, kernel_size=mixer_kernel_size,
                                 se_ratio=8, local_size=local_size)

        self.ca_conv = nn.Sequential(
            nn.Conv2d(2*dim, dim, 1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, padding_mode='reflect'),
            nn.GELU()
        )
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim // 4, dim, kernel_size=1),
            nn.Sigmoid()
        )
        self.conv_init = nn.Sequential(  # PW->DW->
            nn.Conv2d(dim, dim * 2, 1),
            nn.GELU()
        )
        self.dw_conv_1 = nn.Sequential(
            nn.Conv2d(self.dim, self.dim, kernel_size=3, padding=3 // 2,
                      groups=self.dim, padding_mode='reflect'),
            nn.GELU()
        )
        self.dw_conv_2 = nn.Sequential(
            nn.Conv2d(self.dim, self.dim, kernel_size=5, padding=5 // 2,
                      groups=self.dim, padding_mode='reflect'),
            nn.GELU()
        )


    def forward(self, x):
        x = self.conv_init(x)
        x = list(torch.split(x, self.dim, dim=1))
        x_local_1 = self.dw_conv_1(x[0])
        x_local_2 = self.dw_conv_2(x[0])
        x_gloal = self.mixer_gloal(torch.cat([x_local_1, x_local_2], dim=1))
        x = self.ca_conv(x_gloal)
        x = self.ca(x) * x

        return x


class FADBlock(nn.Module):
    def __init__(
            self,
            dim,
            norm_layer=nn.BatchNorm2d,
            token_mixer=Fused_Fourier_Conv_Mixer,
            kernel_size=[1,3,5,7],
            local_size=8
    ):
        super(FADBlock, self).__init__()
        self.dim = dim
        self.norm1 = torch.nn.BatchNorm2d(dim)
        self.norm2 = torch.nn.BatchNorm2d(dim)
        self.mixer = token_mixer(dim=self.dim, mixer_kernel_size=kernel_size, local_size=local_size)
        self.ffn = Prior_Gated_Feed_forward_Network(dim=self.dim)

    def forward(self, mix_input):
        x, mask = mix_input
        copy = x
        x = self.norm1(x)
        x = self.mixer(x)
        x = x + copy

        copy = x
        x = self.norm2(x)
        x, mask = self.ffn(x, mask)
        x = x + copy

        return x, mask


# need drop_path?
class FADStage(nn.Module):
    def __init__(
            self,
            depth=int,
            in_channels=int,
            mixer_kernel_size=[1,3,5,7],
            local_size=8
    ) -> None:
        """ Constructor method """
        # Call super constructor
        super(FADStage, self).__init__()
        # Init blocks
        self.blocks = nn.Sequential(*[
            FADBlock(
                dim=in_channels,
                norm_layer=nn.BatchNorm2d,
                token_mixer=Fused_Fourier_Conv_Mixer,
                kernel_size=mixer_kernel_size,
                local_size=local_size
            )
            for index in range(depth)
        ])

    def forward(self, mix_input):
        output = self.blocks(mix_input)
        return output


class FADBackbone(nn.Module):
    def __init__(self, in_chans=3, out_chans=3, patch_size=1,
                 embed_dim=[48, 96, 192, 96, 48], depth=[2, 2, 2, 2, 2],
                 local_size=[4, 4, 4, 4 ,4], embed_kernel_size=3,
                 downsample_kernel_size=None, upsample_kernel_size=None):
        super(FADBackbone, self).__init__()

        self.patch_size = patch_size
        if downsample_kernel_size is None:
            downsample_kernel_size = 4
        if upsample_kernel_size is None:
            upsample_kernel_size = 4

        self.patch_embed = PatchEmbed(patch_size=patch_size, in_chans=in_chans,
                                      embed_dim=embed_dim[0], kernel_size=embed_kernel_size)
        self.layer1 = FADStage(depth=depth[0], in_channels=embed_dim[0],
                               mixer_kernel_size=[1, 3, 5, 7], local_size=local_size[0])
        self.skip1 = nn.Conv2d(2*embed_dim[0], embed_dim[0], 1)
        self.downsample1 = DownSample(input_dim=embed_dim[0], output_dim=embed_dim[1],
                                      kernel_size=downsample_kernel_size, stride=2)
        self.down_rcp1 = DownSample_RCP()
        self.layer2 = FADStage(depth=depth[1], in_channels=embed_dim[1],
                               mixer_kernel_size=[1, 3, 5, 7], local_size=local_size[1])
        self.skip2 = nn.Conv2d(2*embed_dim[1], embed_dim[1], 1)
        self.downsample2 = DownSample(input_dim=embed_dim[1], output_dim=embed_dim[2],
                                      kernel_size=downsample_kernel_size, stride=2)
        self.down_rcp2 = DownSample_RCP()
        self.layer3 = FADStage(depth=depth[2], in_channels=embed_dim[2],
                               mixer_kernel_size=[1, 3, 5, 7], local_size=local_size[2])
        self.upsample1 = PatchUnEmbed_for_upsample(patch_size=2, embed_dim=embed_dim[2], out_dim=embed_dim[3])
        self.up_rcp1 = Upsample_RCP()
        self.layer4 = FADStage(depth=depth[3], in_channels=embed_dim[3],
                               mixer_kernel_size=[1, 3, 5, 7], local_size=local_size[3])
        self.upsample2 = PatchUnEmbed_for_upsample(patch_size=2, embed_dim=embed_dim[3],
                                                   out_dim=embed_dim[4])
        self.up_rcp2 = Upsample_RCP()
        self.layer5 = FADStage(depth=depth[4], in_channels=embed_dim[4],
                               mixer_kernel_size=[1, 3, 5, 7], local_size=local_size[4])
        self.patch_unembed = PatchUnEmbed(patch_size=patch_size, out_chans=out_chans,
                                          embed_dim=embed_dim[4], kernel_size=3)

    def forward(self, x):
        copy0 = x
        mask = get_residue(x) # B*1*H*W
        x = self.patch_embed(x)
        x, mask = self.layer1((x, mask))
        copy1 = x

        x = self.downsample1(x)
        mask = self.down_rcp1(mask)

        x, mask = self.layer2((x, mask))
        copy2 = x

        x = self.downsample2(x)
        mask = self.down_rcp2(mask)

        x, mask = self.layer3((x, mask))

        x = self.upsample1(x)
        mask = self.up_rcp1(mask)

        x = self.skip2(torch.cat([x, copy2], dim=1))
        x, mask = self.layer4((x, mask))

        x = self.upsample2(x)
        mask = self.up_rcp2(mask)

        x = self.skip1(torch.cat([x, copy1], dim=1))
        x, mask = self.layer5((x, mask))
        x = self.patch_unembed(x)

        x = copy0 + x
        return x


def FADformer_mini():
    return FADBackbone(
        embed_dim=[24, 48, 96, 48, 24],
        depth=[2, 3, 4, 3, 2],
        local_size=[4, 4, 4, 4, 4],
        embed_kernel_size=3
    )

def FADformer():
    return FADBackbone(
        embed_dim=[32, 64, 128, 64, 32],
        depth=[4, 8, 10, 8, 4],
        local_size=[4, 4, 4, 4, 4],
        embed_kernel_size=3
    )

