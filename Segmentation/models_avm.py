import torch
import torch.nn as nn

class conv_block(nn.Module):
    
    def __init__(self, in_ch, mid_ch, out_ch, norm_method='batch'):
        super(conv_block, self).__init__()

        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=True)
        self.norm_method = norm_method
        if norm_method=='group':
            self.gn1 = nn.GroupNorm(8,mid_ch)
        elif norm_method=='batch':
            self.bn1 = nn.BatchNorm2d(mid_ch)        
        self.conv2 = nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=True)
        if norm_method=='group':
            self.gn2 = nn.GroupNorm(8,out_ch)
        elif norm_method=='batch':
            self.bn2 = nn.BatchNorm2d(out_ch)        

    def forward(self, x):
        x = self.conv1(x)
        if self.norm_method=='group':
            x = self.gn1(x)
        elif self.norm_method=='batch':
            x = self.bn1(x)
        x = self.activation(x)
        
        x = self.conv2(x)
        if self.norm_method=='group':
            x = self.gn2(x)
        elif self.norm_method=='batch':
            x = self.bn2(x)
        output = self.activation(x)

        return output

    
#Nested Unet
class NestedUNet(nn.Module):

    def __init__(self, in_ch=1, out_ch=2, norm_method='batch'):
        super(NestedUNet, self).__init__()

        n1 = 32
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = conv_block(in_ch, filters[0], filters[0], norm_method)
        self.conv1_0 = conv_block(filters[0], filters[1], filters[1], norm_method)
        self.conv2_0 = conv_block(filters[1], filters[2], filters[2], norm_method)
        self.conv3_0 = conv_block(filters[2], filters[3], filters[3], norm_method)
        self.conv4_0 = conv_block(filters[3], filters[4], filters[4], norm_method)

        self.conv0_1 = conv_block(filters[0] + filters[1], filters[0], filters[0], norm_method)
        self.conv1_1 = conv_block(filters[1] + filters[2], filters[1], filters[1], norm_method)
        self.conv2_1 = conv_block(filters[2] + filters[3], filters[2], filters[2], norm_method)
        self.conv3_1 = conv_block(filters[3] + filters[4], filters[3], filters[3], norm_method)

        self.conv0_2 = conv_block(filters[0]*2 + filters[1], filters[0], filters[0], norm_method)
        self.conv1_2 = conv_block(filters[1]*2 + filters[2], filters[1], filters[1], norm_method)
        self.conv2_2 = conv_block(filters[2]*2 + filters[3], filters[2], filters[2], norm_method)

        self.conv0_3 = conv_block(filters[0]*3 + filters[1], filters[0], filters[0], norm_method)
        self.conv1_3 = conv_block(filters[1]*3 + filters[2], filters[1], filters[1], norm_method)

        self.conv0_4 = conv_block(filters[0]*4 + filters[1], filters[0], filters[0], norm_method)

        self.final = nn.Conv2d(filters[0], out_ch, kernel_size=1)


    def forward(self, x):
        
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.Up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.Up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.Up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.Up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.Up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.Up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.Up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.Up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.Up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.Up(x1_3)], 1))

        output = self.final(x0_4)
        return output


#Nested Unet
class UNet(nn.Module):
    
    def __init__(self, in_ch=1, out_ch=2, norm_method='batch'):
        super(UNet, self).__init__()

        n1 = 32
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = conv_block(in_ch, filters[0], filters[0], norm_method)
        self.conv1_0 = conv_block(filters[0], filters[1], filters[1], norm_method)
        self.conv2_0 = conv_block(filters[1], filters[2], filters[2], norm_method)
        self.conv3_0 = conv_block(filters[2], filters[3], filters[3], norm_method)
        self.conv4_0 = conv_block(filters[3], filters[4], filters[4], norm_method)
        self.conv3_1 = conv_block(filters[3] + filters[4], filters[3], filters[3], norm_method)
        self.conv2_2 = conv_block(filters[2] + filters[3], filters[2], filters[2], norm_method)
        self.conv1_3 = conv_block(filters[1] + filters[2], filters[1], filters[1], norm_method)
        self.conv0_4 = conv_block(filters[0] + filters[1], filters[0], filters[0], norm_method)
        self.final = nn.Conv2d(filters[0], out_ch, kernel_size=1)

    def forward(self, x):
        
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.Up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.Up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.Up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.Up(x1_3)], 1))

        output = self.final(x0_4)
        return output        
