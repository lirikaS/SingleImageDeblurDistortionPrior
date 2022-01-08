import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PALayer(nn.Module):
    def __init__(self, channel, kernel_size=3):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
                nn.Conv2d(channel, channel // 8, kernel_size, padding=kernel_size//2, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 8, 1, kernel_size, padding=kernel_size//2, bias=True),
                nn.Sigmoid()
        )
    def forward(self, x):
        y = self.pa(x)
        return x * y

class CALayer(nn.Module):
    def __init__(self, channel, kernel_size=3):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
                nn.Conv2d(channel, channel // 8, kernel_size, padding=kernel_size//2, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 8, channel, kernel_size, padding=kernel_size//2, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y

class DeepAttentionBlock(nn.Module):
    def __init__(self, channel):
        super(DeepAttentionBlock, self).__init__()

        self.PA = PALayer(channel)        
        self.CA = CALayer(channel)

    def forward(self, x):
        output = self.PA(x)
        output = self.CA(output)
        return output
        
class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 -1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            #layers.append(nn.ReLU(inplace=True))
            layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
            
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

    
class RDB4s(nn.Module):
    
    def __init__(self, channel, gr=None,  num_conv=4, alpha = 0.1):    
        super(RDB4s, self).__init__()
        self.alpha = alpha
        self.num_conv = num_conv

        self.conv1 = BasicConv(channel, channel, kernel_size=3, stride=1, relu=True)
        self.conv2 = BasicConv(channel, channel, kernel_size=3, stride=1, relu=False)
        #self.conv3 = BasicConv(channel, channel, kernel_size=3, stride=1, relu=True)
        #self.conv4 = BasicConv(channel, channel, kernel_size=3, stride=1, relu=False)
        
        #self.convs = nn.ModuleList([BasicConv(channel, channel, kernel_size=3, stride=1, relu=True) for i in range(num_conv)])
        #self.conv1x1  = BasicConv(channel + gr*num_conv, channel, kernel_size=1, stride=1, relu=False)
        

    def forward(self, x):
         return self.conv2(self.conv1(x)) + x



class RRDB4s(nn.Module):    
    def __init__(self, channel, num_block=10):
        super(RRDB4s, self).__init__()
        gr = channel//4

        layers = [RDB4s(channel, gr) for _ in range(num_block)]
        self.layers = nn.ModuleList(layers)
        self.DeepAttentionBlock = DeepAttentionBlock(channel)

    def forward(self, x):
        out = x
        for l in self.layers:
            out = l(out)
        out = self.DeepAttentionBlock(out)
        return out
    
class FMD(nn.Module):   # Feature Merger Down-scale
    def __init__(self, channel):
        super(FMD, self).__init__()
        self.conv_ds = BasicConv(channel//2, channel, kernel_size=3, stride=2, relu=True)
        self.conv1 = BasicConv(2*channel, channel, kernel_size=3, stride=1, relu=True)
        self.conv2 = BasicConv(channel, channel, kernel_size=3, stride=1, relu=False)

    def forward(self, x1, x2):
        x2 = self.conv_ds(x2)
        x = self.conv1(torch.cat([x1, x2], dim=1))
        x = self.conv2(x)
        return x + x2

    
class FMU(nn.Module):   # Feature Merger Up-scale
    def __init__(self, channel):
        super(FMU, self).__init__()
        self.conv_us = BasicConv(channel*2, channel, kernel_size=4, stride=2, relu=True, transpose=True)
        self.conv1 = BasicConv(2*channel, channel, kernel_size=3, stride=1, relu=True)
        self.conv2 = BasicConv(channel, channel, kernel_size=3, stride=1, relu=False)
        
    def forward(self, x1, x2):
        x2 = self.conv_us(x2)
        x = self.conv1(torch.cat([x1, x2], dim=1))
        x = self.conv2(x)
        return x + x2
    
    
class DIPM(nn.Module):   # Feature Merger Down-scale
    def __init__(self, channel):
        super(DIPM, self).__init__()
        self.conv1 = BasicConv(2*channel, channel, kernel_size=3, stride=1, relu=True)
        self.conv2 = BasicConv(channel, channel, kernel_size=3, stride=1, relu=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        x = self.conv1(torch.cat([self.sigmoid(x1)*x2, x2], dim=1))
        x = self.conv2(x)
        return x
    
    
class DIPAM(nn.Module):   # Feature Merger Down-scale
    def __init__(self, channel):
        super(DIPAM, self).__init__()
        
        self.pa = nn.Sequential(
            BasicConv(channel, channel // 8, kernel_size=3, stride=1, relu=True),
            BasicConv(channel // 8, 1, kernel_size=3, stride=1, relu=False),
            nn.Sigmoid()
            )

        self.conv = BasicConv(3*channel, channel, kernel_size=3, stride=1, relu=True)

    def forward(self, x, dip):
        w = self.pa(dip)
        return self.conv(torch.cat([x * w, x, dip], dim=1))
    
class CHSUM(nn.Module):   # Feature Merger Down-scale
    def __init__(self, channel):
        super(CHSUM, self).__init__()
        self.conv_us = BasicConv(3, 3, kernel_size=4, stride=2, relu=True, transpose=True)
        self.pa = nn.Sequential(
            BasicConv(channel, channel // 8, kernel_size=3, stride=1, relu=True),
            BasicConv(channel // 8, 1, kernel_size=3, stride=1, relu=False),
            nn.Sigmoid()
            )

    def forward(self, x1, x2, fe):
        x2 = self.conv_us(x2)
        w = self.pa(fe)
        
        return w * x1 + (1-w) * x2


class MainNet(nn.Module):
    def __init__(self, num_RDB=4):
        super(MainNet, self).__init__()
        self.feat_extract = list()
        base_channel = 32
        num_blocks = 3

        DIP_module = DIPM #or DIPAM


        self.RDBs = nn.ModuleList([
            RDB4s(base_channel * 2),
            RDB4s(base_channel * 4),
            RDB4s(base_channel * 4),
            RDB4s(base_channel * 2)
        ])
        
        encode_convs = [BasicConv(3, base_channel*(2**i), kernel_size=3, relu=True, stride=1) for i in range(num_blocks)]
        decode_convs = [BasicConv(base_channel*(2**(num_blocks-1-i)), 3, kernel_size=3, relu=False, stride=1) for i in range(num_blocks)]
        self.convs = nn.ModuleList(
            encode_convs + decode_convs
        )
        
        encode_blocks = [RRDB4s(base_channel*(2**i), num_RDB) for i in range(num_blocks)]
        decode_blocks = [RRDB4s(base_channel*(2**(num_blocks-1-i)), num_RDB) for i in range(num_blocks)]
        self.blocks = nn.ModuleList(
            encode_blocks + decode_blocks
        )
        
        
        self.FMs = nn.ModuleList([
            FMD(base_channel * 2),
            FMD(base_channel * 4),
            FMU(base_channel * 2),
            FMU(base_channel),
        ])
        
        self.DIPMs = nn.ModuleList([
            DIP_module(base_channel),
            DIP_module(base_channel * 2),
            DIP_module(base_channel * 4)
        ])
        
        self.CHSUMs = nn.ModuleList([
            CHSUM(base_channel * 2),
            CHSUM(base_channel)
        ])

        self.RDBs.apply(self.weight_init)
        self.convs.apply(self.weight_init)
        self.blocks.apply(self.weight_init)
        self.FMs.apply(self.weight_init)
        self.DIPMs.apply(self.weight_init)
        self.CHSUMs.apply(self.weight_init)



    def forward(self, x):
        #B, C, H, W = x.size()
        x_2 = F.interpolate(x, scale_factor=0.5, mode='bilinear')
        x_4 = F.interpolate(x, scale_factor=0.25, mode='bilinear')
        
        #Main Encoder
        z_1 = self.convs[0](x)
        z_2 = self.RDBs[0](self.convs[1](x_2))
        z_4 = self.RDBs[1](self.convs[2](x_4))


        bo_1 = self.blocks[0](z_1)
        bo_2 = self.blocks[1](self.FMs[0](z_2, bo_1))
        bo_4 = self.blocks[2](self.FMs[1](z_4, bo_2))             
        
        outputs = list()

        z_4 = self.blocks[3](bo_4)
        res_4 = self.convs[3](self.RDBs[2](z_4)) + x_4
        outputs.append(torch.clamp(res_4, 0, 1))

        z_2 = self.blocks[4](self.FMs[2](bo_2, z_4))
        res_2 = self.convs[4](self.RDBs[3](z_2)) + self.CHSUMs[0](x_2, res_4, z_2)
        outputs.append(torch.clamp(res_2, 0, 1))

        z_1 = self.blocks[5](self.FMs[3](bo_1, z_2))
        res_1 = self.convs[5](z_1) + self.CHSUMs[1](x, res_2, z_1)
        outputs.append(torch.clamp(res_1, 0, 1))


        return outputs


    def weight_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0.0, 0.5 * math.sqrt(2. / n))
            if m.bias is not None:
                m.bias.data.zero_()
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif classname.find('Linear') != -1:
            n = m.weight.size(1)
            m.weight.data.normal_(0, 0.01)
            m.bias.data = torch.ones(m.bias.data.size())


            
class DIP_EST_NET(nn.Module):
    def __init__(self, in_channel, num_RDB=1):
        super(DIP_EST_NET, self).__init__()
        self.feat_extract = list()
        base_channel = 32
        num_blocks = 3


        self.RDBs = nn.ModuleList([
            RDB4s(base_channel * 2),
            RDB4s(base_channel * 4)
        ])
        
        encode_convs = [BasicConv(in_channel, base_channel*(2**i), kernel_size=3, relu=True, stride=1) for i in range(num_blocks)]
        decode_convs = [BasicConv(base_channel*(2**i), base_channel*(2**i), kernel_size=3, relu=False, stride=1) for i in range(num_blocks)]
        self.convs = nn.ModuleList(
            encode_convs + decode_convs
        )
        
        encode_blocks = [RRDB4s(base_channel*(2**i), num_RDB) for i in range(num_blocks)]
        self.blocks = nn.ModuleList(
            encode_blocks
        )
        
        
        self.FMs = nn.ModuleList([
            FMD(base_channel * 2),
            FMD(base_channel * 4)
        ])
        

        self.RDBs.apply(self.weight_init)
        self.convs.apply(self.weight_init)
        self.blocks.apply(self.weight_init)
        self.FMs.apply(self.weight_init)



    def forward(self, x):
        #B, C, H, W = x.size()
        x_2 = F.interpolate(x, scale_factor=0.5, mode='bilinear')
        x_4 = F.interpolate(x, scale_factor=0.25, mode='bilinear')
        
        #Main Encoder
        z_1 = self.convs[0](x)
        z_2 = self.RDBs[0](self.convs[1](x_2))
        z_4 = self.RDBs[1](self.convs[2](x_4))

        outputs = list()
        bo_1 = self.blocks[0](z_1)
        bo_2 = self.blocks[1](self.FMs[0](z_2, bo_1))
        bo_4 = self.blocks[2](self.FMs[1](z_4, bo_2))
        dip_1 = self.convs[3](bo_1)
        dip_2 = self.convs[4](bo_2)
        dip_4 = self.convs[5](bo_4)

        outputs.append(dip_1)
        outputs.append(dip_2)
        outputs.append(dip_4)
               
        return outputs


    def weight_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0.0, 0.5 * math.sqrt(2. / n))
            if m.bias is not None:
                m.bias.data.zero_()
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif classname.find('Linear') != -1:
            n = m.weight.size(1)
            m.weight.data.normal_(0, 0.01)
            m.bias.data = torch.ones(m.bias.data.size())



class build_net(nn.Module):
    def __init__(self):
        super(build_net, self).__init__()
        self.deblur_module = MainNet(10)
        #self.blur_module = MainNet(3)
        #self.bk_est_by_Blur = DIP_EST_NET(3)
        #self.bk_est_by_SharpBlur = DIP_EST_NET(6)
        
    def forward(self, blur, sharp, aux_sharp):
        sb = torch.cat([sharp, blur], dim=1)
        b = blur
        s = sharp
        s2 = aux_sharp

        deblur_res_by_B = self.deblur_module(blur)

        return deblur_res_by_B
