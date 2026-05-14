"""
 @Time    : 2021/7/6 14:23
 @Author  : Haiyang Mei
 @E-mail  : mhy666@mail.dlut.edu.cn

 @Project : CVPR2021_PFNet
 @File    : PFNet.py
 @Function: Focus and Exploration Network

"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import backbone.resnet.resnet as resnet

###################################################################
# ################ Boundary-Guided Modules (IJCAI 2022) ############
###################################################################
class ECA(nn.Module):
    """
    Efficient Channel Attention (ECA): local cross-channel interaction.
    Used in BGNet (IJCAI'22) EFM local channel attention (LCA).
    """
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        t = int(abs((torch.log2(torch.tensor(channels, dtype=torch.float32)) + b) / gamma))
        k = t if t % 2 else t + 1
        k = max(k, 3)  # keep it stable for small C
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)

    def forward(self, x):
        # x: (B,C,H,W)
        y = F.adaptive_avg_pool2d(x, 1)  # (B,C,1,1)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))  # (B,1,C)
        y = torch.sigmoid(y.transpose(-1, -2).unsqueeze(-1))  # (B,C,1,1)
        return x * y


class EdgeAwareModule(nn.Module):
    """
    Edge-aware module (EAM) in BGNet (IJCAI'22):
    integrate low-level edge details + high-level location semantics.
    """
    def __init__(self, low_c=64, high_c=512, mid_low=64, mid_high=256, out_c=64):
        super().__init__()
        self.low_reduce = nn.Sequential(
            nn.Conv2d(low_c, mid_low, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(mid_low),
            nn.ReLU(inplace=True),
        )
        self.high_reduce = nn.Sequential(
            nn.Conv2d(high_c, mid_high, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(mid_high),
            nn.ReLU(inplace=True),
        )
        fuse_c = mid_low + mid_high
        self.fuse = nn.Sequential(
            nn.Conv2d(fuse_c, out_c, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )
        self.edge_pred = nn.Conv2d(out_c, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, f_low, f_high):
        # f_low: high-res (B,64,H/4,W/4) in PFNet's cr1
        # f_high: low-res (B,512,H/32,W/32) in PFNet's cr4
        low = self.low_reduce(f_low)
        high = self.high_reduce(f_high)
        high = F.interpolate(high, size=low.shape[2:], mode='bilinear', align_corners=True)
        fe = self.fuse(torch.cat([low, high], dim=1))  # (B,out_c,H/4,W/4)
        pe = self.edge_pred(fe)  # logits (B,1,H/4,W/4)
        return fe, pe


class EdgeGuidanceFeatureModule(nn.Module):
    """
    Edge-guidance feature module (EFM) in BGNet (IJCAI'22):
    fuse edge cues with multi-level features and apply local channel attention.
    """
    def __init__(self, feat_c, edge_c=64):
        super().__init__()
        self.edge_proj = nn.Sequential(
            nn.Conv2d(edge_c, feat_c, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(feat_c),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(feat_c, feat_c, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(feat_c),
            nn.ReLU(inplace=True),
        )
        self.eca = ECA(feat_c)
        self.out = nn.Sequential(
            nn.Conv2d(feat_c, feat_c, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(feat_c),
            nn.ReLU(inplace=True),
        )
        # residual scaling, initialized to 0 so that EFM starts as identity
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, fi, fe):
        # fi: (B,C,Hi,Wi), fe: (B,edge_c,He,We)
        fe_d = F.interpolate(fe, size=fi.shape[2:], mode='bilinear', align_corners=True)
        a = torch.sigmoid(self.edge_proj(fe_d))  # (B,C,Hi,Wi)
        fused = self.fuse(fi * a + fi)
        fused = self.eca(fused)
        residual = self.out(fused)
        return fi + self.gamma * residual

###################################################################
# ################## Channel Attention Block ######################
###################################################################
class CA_Block(nn.Module):
    def __init__(self, in_dim):
        super(CA_Block, self).__init__()
        self.chanel_in = in_dim
        self.gamma = nn.Parameter(torch.ones(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps (B X C X H X W)
            returns :
                out : channel attentive features
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out

###################################################################
# ################## Spatial Attention Block ######################
###################################################################
class SA_Block(nn.Module):
    def __init__(self, in_dim):
        super(SA_Block, self).__init__()
        self.chanel_in = in_dim
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.ones(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps (B X C X H X W)
            returns :
                out : spatial attentive features
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out

###################################################################
# ################## Context Exploration Block ####################
###################################################################
class Context_Exploration_Block(nn.Module):
    def __init__(self, input_channels):
        super(Context_Exploration_Block, self).__init__()
        self.input_channels = input_channels
        self.channels_single = int(input_channels / 4)

        self.p1_channel_reduction = nn.Sequential(
            nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p2_channel_reduction = nn.Sequential(
            nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p3_channel_reduction = nn.Sequential(
            nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p4_channel_reduction = nn.Sequential(
            nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())

        self.p1 = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, 1, 1, 0),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p1_dc = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())

        self.p2 = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, 3, 1, 1),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p2_dc = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())

        self.p3 = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, 5, 1, 2),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p3_dc = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=4, dilation=4),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())

        self.p4 = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, 7, 1, 3),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p4_dc = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=8, dilation=8),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())

        self.fusion = nn.Sequential(nn.Conv2d(self.input_channels, self.input_channels, 1, 1, 0),
                                    nn.BatchNorm2d(self.input_channels), nn.ReLU())

    def forward(self, x):
        p1_input = self.p1_channel_reduction(x)
        p1 = self.p1(p1_input)
        p1_dc = self.p1_dc(p1)

        p2_input = self.p2_channel_reduction(x) + p1_dc
        p2 = self.p2(p2_input)
        p2_dc = self.p2_dc(p2)

        p3_input = self.p3_channel_reduction(x) + p2_dc
        p3 = self.p3(p3_input)
        p3_dc = self.p3_dc(p3)

        p4_input = self.p4_channel_reduction(x) + p3_dc
        p4 = self.p4(p4_input)
        p4_dc = self.p4_dc(p4)

        ce = self.fusion(torch.cat((p1_dc, p2_dc, p3_dc, p4_dc), 1))

        return ce

###################################################################
# ##################### Positioning Module ########################
###################################################################
class Positioning(nn.Module):
    def __init__(self, channel):
        super(Positioning, self).__init__()
        self.channel = channel
        self.cab = CA_Block(self.channel)
        self.sab = SA_Block(self.channel)
        self.map = nn.Conv2d(self.channel, 1, 7, 1, 3)

    def forward(self, x):
        cab = self.cab(x)
        sab = self.sab(cab)
        map = self.map(sab)

        return sab, map

###################################################################
# ######################## Focus Module ###########################
###################################################################
class Focus(nn.Module):
    def __init__(self, channel1, channel2):
        super(Focus, self).__init__()
        self.channel1 = channel1
        self.channel2 = channel2

        self.up = nn.Sequential(nn.Conv2d(self.channel2, self.channel1, 7, 1, 3),
                                nn.BatchNorm2d(self.channel1), nn.ReLU(), nn.UpsamplingBilinear2d(scale_factor=2))

        self.input_map = nn.Sequential(nn.UpsamplingBilinear2d(scale_factor=2), nn.Sigmoid())
        self.output_map = nn.Conv2d(self.channel1, 1, 7, 1, 3)

        self.fp = Context_Exploration_Block(self.channel1)
        self.fn = Context_Exploration_Block(self.channel1)
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        self.bn1 = nn.BatchNorm2d(self.channel1)
        self.relu1 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(self.channel1)
        self.relu2 = nn.ReLU()

    def forward(self, x, y, in_map):
        # x; current-level features
        # y: higher-level features
        # in_map: higher-level prediction

        up = self.up(y)

        input_map = self.input_map(in_map)
        f_feature = x * input_map
        b_feature = x * (1 - input_map)

        fp = self.fp(f_feature)
        fn = self.fn(b_feature)

        refine1 = up - (self.alpha * fp)
        refine1 = self.bn1(refine1)
        refine1 = self.relu1(refine1)

        refine2 = refine1 + (self.beta * fn)
        refine2 = self.bn2(refine2)
        refine2 = self.relu2(refine2)

        output_map = self.output_map(refine2)

        return refine2, output_map

###################################################################
# ########################## NETWORK ##############################
###################################################################
class PFNet(nn.Module):
    def __init__(self, backbone_path=None):
        super(PFNet, self).__init__()
        # params

        # backbone
        resnet50 = resnet.resnet50(backbone_path)
        self.layer0 = nn.Sequential(resnet50.conv1, resnet50.bn1, resnet50.relu)
        self.layer1 = nn.Sequential(resnet50.maxpool, resnet50.layer1)
        self.layer2 = resnet50.layer2
        self.layer3 = resnet50.layer3
        self.layer4 = resnet50.layer4

        # channel reduction
        self.cr4 = nn.Sequential(nn.Conv2d(2048, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU())
        self.cr3 = nn.Sequential(nn.Conv2d(1024, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU())
        self.cr2 = nn.Sequential(nn.Conv2d(512, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU())
        self.cr1 = nn.Sequential(nn.Conv2d(256, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU())

        # boundary-guided modules (EAM + multi-level EFM)
        self.eam = EdgeAwareModule(low_c=64, high_c=512, out_c=64)
        self.efm4 = EdgeGuidanceFeatureModule(feat_c=512, edge_c=64)
        self.efm3 = EdgeGuidanceFeatureModule(feat_c=256, edge_c=64)
        self.efm2 = EdgeGuidanceFeatureModule(feat_c=128, edge_c=64)
        self.efm1 = EdgeGuidanceFeatureModule(feat_c=64, edge_c=64)

        # positioning
        self.positioning = Positioning(512)

        # focus
        self.focus3 = Focus(256, 512)
        self.focus2 = Focus(128, 256)
        self.focus1 = Focus(64, 128)

        for m in self.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = True

    def forward(self, x):
        # x: [batch_size, channel=3, h, w]
        layer0 = self.layer0(x)  # [-1, 64, h/2, w/2]
        layer1 = self.layer1(layer0)  # [-1, 256, h/4, w/4]
        layer2 = self.layer2(layer1)  # [-1, 512, h/8, w/8]
        layer3 = self.layer3(layer2)  # [-1, 1024, h/16, w/16]
        layer4 = self.layer4(layer3)  # [-1, 2048, h/32, w/32]

        # channel reduction
        cr4 = self.cr4(layer4)
        cr3 = self.cr3(layer3)
        cr2 = self.cr2(layer2)
        cr1 = self.cr1(layer1)

        # edge-aware semantics (EAM) and boundary-guided fusion (EFM)
        edge_feat, edge_logit = self.eam(cr1, cr4)  # (B,64,H/4,W/4), (B,1,H/4,W/4)
        # 为尽量保持 PFNet 原始行为，仅对中高层特征做引导，保留 cr1 直接进入 Focus1
        cr4 = self.efm4(cr4, edge_feat)
        cr3 = self.efm3(cr3, edge_feat)
        cr2 = self.efm2(cr2, edge_feat)

        # positioning
        positioning, predict4 = self.positioning(cr4)

        # focus
        focus3, predict3 = self.focus3(cr3, positioning, predict4)
        focus2, predict2 = self.focus2(cr2, focus3, predict3)
        focus1, predict1 = self.focus1(cr1, focus2, predict2)

        # rescale
        predict4 = F.interpolate(predict4, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict3 = F.interpolate(predict3, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict2 = F.interpolate(predict2, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict1 = F.interpolate(predict1, size=x.size()[2:], mode='bilinear', align_corners=True)

        # edge prediction (for auxiliary supervision)
        edge = F.interpolate(edge_logit, size=x.size()[2:], mode='bilinear', align_corners=True)

        if self.training:
            return predict4, predict3, predict2, predict1, edge

        return torch.sigmoid(predict4), torch.sigmoid(predict3), torch.sigmoid(predict2), torch.sigmoid(
            predict1)
