import torch as th
import torch.nn.functional as F
from torch import nn

from .size_adapter import SizeAdapter
import torch
# from .net_util import ChannelAttention, ChannelAttention_softmax, SpatialAttention, SpatialAttention_softmax,EN_Block,DE_Block,Self_Attention

from .net_util import *



class Coarse_Attention(nn.Module):
    """Modified version of Unet from SuperSloMo.

    Difference :
    1) diff=img-event.
    2)ca and sa
    """

    def __init__(self, inChannels):
        super(Coarse_Attention, self).__init__()
        self.conv = nn.Conv2d(inChannels*2, 2, kernel_size=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.pool=nn.AdaptiveAvgPool2d(1)

    def forward(self, img,event):

        x = torch.cat([img, event], dim=1)
        x=self.conv(x)
        x_i,x_e=x[:, 0:1, :, :], x[:, 1:2, :, :]
        x_i=self.sigmoid(x_i)
        x_e=self.sigmoid(x_e)
        x_i=self.pool(x_i)
        x_e=self.pool(x_e)
        g_img=x_i*img
        g_event=x_e*event

        return g_img, g_event






class Divide_Cross_Attention_Transformer(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias'):
        super(Divide_Cross_Attention_Transformer, self).__init__()
        self.num_heads=num_heads
        self.norm1_image = LayerNorm(dim, LayerNorm_type)
        self.norm1_event = LayerNorm(dim, LayerNorm_type)
        self.norm1_common = LayerNorm(dim, LayerNorm_type)
        self.norm1_differential = LayerNorm(dim, LayerNorm_type)
        self.attn_common = Divide_Cross_Transformer(dim, num_heads, bias)
        self.attn_differential = Divide_Cross_Transformer(dim, num_heads, bias)

        # mlp
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        mlp_hidden_dim = int(dim * ffn_expansion_factor)
        self.FFN_C = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0.)
        self.FFN_D = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0.)

        self.project_out1 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.project_out2 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, image,event):
        # image: b, c, h, w
        # event: b, c, h, w
        # return: b, c, h, w
        assert image.shape == event.shape, 'the shape of image doesnt equal to event'
        b, c , h, w = image.shape
        common=image+event
        differential=image-event
        ###########common################################

        attn_img_common,v_img_common=self.attn_common(self.norm1_image(image), self.norm1_common(common))
        attn_event_common,v_event_common=self.attn_common(self.norm1_event(event), self.norm1_common(common))

        attn_common_all=attn_img_common*attn_event_common
        out_common_img=attn_common_all @ v_img_common
        out_common_img = rearrange(out_common_img, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out_common_event=attn_common_all @ v_event_common
        out_common_event = rearrange(out_common_event, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out_common=out_common_img+out_common_event
        # out_common = rearrange(out_common, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out_common = self.project_out1(out_common)
        out_common = to_3d(out_common) # b, h*w, c
        out_common = out_common + self.FFN_C(self.norm1(out_common))
        out_common = to_4d(out_common, h, w)

        ###########differential################################

        attn_img_differential,v_img_differential=self.attn_differential(self.norm1_image(image), self.norm1_differential(differential))
        attn_event_differential,v_event_differential=self.attn_differential(self.norm1_event(event), self.norm1_differential(differential))
        out_differential_img=attn_img_differential @ v_img_differential
        out_differential_img = rearrange(out_differential_img, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out_differential_event=attn_event_differential @ v_event_differential
        out_differential_event = rearrange(out_differential_event, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out_differential=out_differential_img+out_differential_event
        # out_differential = rearrange(out_differential, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out_differential = self.project_out2(out_differential)
        out_differential = to_3d(out_differential) # b, h*w, c
        out_differential = out_differential + self.FFN_D(self.norm2(out_differential))
        out_differential = to_4d(out_differential, h, w)


        return out_common, out_differential


class Recombine_Cross_Attention_Transformer(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias'):
        super(Recombine_Cross_Attention_Transformer, self).__init__()

        self.norm1_common = LayerNorm(dim, LayerNorm_type)
        self.norm1_differential = LayerNorm(dim, LayerNorm_type)
        self.attn = Recombine_Cross_Transformer(dim, num_heads, bias)
        # mlp
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * ffn_expansion_factor)
        self.FFN_common = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0.)
        self.FFN_differential = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0.)

        self.project_out = nn.Conv2d(3*dim, dim, kernel_size=1, bias=bias)

    def forward(self, common, differential):
        # image: b, c, h, w
        # event: b, c, h, w
        # return: b, c, h, w
        assert common.shape == differential.shape, 'the shape of image doesnt equal to event'
        b, c , h, w = common.shape

        out_common, out_differential = self.attn(self.norm1_common(common), self.norm1_differential(differential))
        att_common = common + out_common
        att_differential = differential + out_differential

        # Linear_Projection
        att_common = to_3d(att_common)  # b, h*w, c
        att_common = att_common + self.FFN_common(self.norm1(att_common))
        att_common = to_4d(att_common, h, w)

        att_differential = to_3d(att_differential)  # b, h*w, c
        att_differential = att_differential + self.FFN_differential(self.norm2(att_differential))
        att_differential = to_4d(att_differential, h, w)

        fuse_add =att_common+att_differential
        fuse_product =att_common*att_differential
        fuse_max = torch.max(att_common,att_differential)
        fuse_cat = torch.cat((fuse_add, fuse_product,fuse_max), 1)
        fused=self.project_out(fuse_cat)



        return fused






class Encoder(nn.Module):
    """Modified version of Unet from SuperSloMo.

    Difference :
    1) there is an option to skip ReLU after the last convolution.
    2) there is a size adapter module that makes sure that input of all sizes
       can be processed correctly. It is necessary because original
       UNet can process only inputs with spatial dimensions divisible by 32.
    """

    def __init__(self, inChannels):
        super(Encoder, self).__init__()
        # self.inplanes = 32
        self.num_heads=4
        ######encoder部分
        ################################Resnet Image#######################################
        self.head = shallow_cell(inChannels)
        self.down1 = EN_Block(64, 128)  # 128
        self.down2 = EN_Block(128, 256)  # 64
        # self.down3 = EN_Block(128, 256)  # 32
        # self.down4 = EN_Block(256, 512)  # 16

    def forward(self, input):
        # Size adapter spatially augments input to the size divisible by 32.
        # x=torch.cat((input_img, input_event), 1)
        s0 = self.head(input)
        s1 = self.down1(s0)
        s2 = self.down2(s1)
        # s3 = self.down3(s2)
        # s4 = self.down4(s3)
        x = [s0, s1, s2]
        return x


class Decoder(nn.Module):
    """Modified version of Unet from SuperSloMo.
    """

    def __init__(self, outChannels):
        super(Decoder, self).__init__()
        ######Decoder
        # self.up1 = DE_Block(512, 256)
        # self.up2 = DE_Block(256, 128)
        self.up3 = DE_Block(256, 128)
        self.up4 = DE_Block(128, 64)

    def forward(self, input,skip):
        x=input
        # x = self.up1(x, skip[3])
        # x = self.up2(x, skip[2])
        x = self.up3(x, skip[1])
        x = self.up4(x, skip[0])

        # x = self.conv(x)
        # x=x+input_img
        return x


class Restoration(nn.Module):
    """Modified version of Unet from SuperSloMo.

    """

    def __init__(self, inChannels_img, inChannels_event,outChannels, args,ends_with_relu=False):
        super(Restoration, self).__init__()
        self._ends_with_relu = ends_with_relu
        self.num_heads=4
        ######encoder
        self.encoder_img=Encoder(inChannels_img)
        self.encoder_event=Encoder(inChannels_event)
        self.decoder = Decoder(outChannels)
        ######fusion
        self.Divide = Divide_Cross_Attention_Transformer(256, num_heads=self.num_heads,ffn_expansion_factor=4, bias=False,
                                                                                   LayerNorm_type='WithBias')

        self.Recombine = Recombine_Cross_Attention_Transformer(256, num_heads=self.num_heads,ffn_expansion_factor=4, bias=False,
                                                                                   LayerNorm_type='WithBias')

        self.conv = nn.Conv2d(64, outChannels, 3, stride=1, padding=1)


    def forward(self, input_img, input_event):


        ####  feature extraction
        en_img=self.encoder_img(input_img)
        en_event=self.encoder_event(input_event)

        #####Divide_and_ Recombine#########################

        ###Divide
        out_common, out_differential=self.Divide(en_img[-1],en_event[-1])
        ###Recombine
        out_fusion=self.Recombine(out_common,out_differential)

        #############decoder
        out=self.decoder(out_fusion,en_img)
        out = self.conv(out)
        out=out+input_img
        return out
