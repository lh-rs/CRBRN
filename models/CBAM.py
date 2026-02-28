import torch
import torch.nn as nn
import torchvision

def paramsInit(net):
    if isinstance(net, nn.Conv2d):
        nn.init.xavier_uniform_(net.weight.data)
        nn.init.constant_(net.bias.data, 0.00)
    if isinstance(net, nn.Linear):
        nn.init.xavier_uniform_(net.weight.data)
        nn.init.constant_(net.bias.data, 0.00)

class ChannelAttentionModule(nn.Module):
    def __init__(self, channel):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            # nn.Conv2d(channel, channel, 1,), # bias=False
            # nn.ReLU(),
            nn.Conv2d(channel, channel, 1, bias=False), #, bias=False
            # nn.Tanh()
        )
        self.sigmoid = nn.Sigmoid()
        paramsInit(self)
        
    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)

class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3),#, bias=False
            # nn.Tanh()
        )
        self.sigmoid = nn.Sigmoid()
        paramsInit(self)
        
        
    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out

class SA(nn.Module):
    def __init__(self, channel):
        super(SA, self).__init__()
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.spatial_attention(x) * x
        return out

class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out

        return out

class CBAM_Cross(nn.Module):
    def __init__(self, channel):
        super(CBAM_Cross, self).__init__()
        self.channel_attention_x = ChannelAttentionModule(channel)
        self.channel_attention_y = ChannelAttentionModule(channel)

        self.spatial_attention_x = SpatialAttentionModule()
        self.spatial_attention_y = SpatialAttentionModule()

    def forward(self, x,y):
        out_x = self.spatial_attention_y(y) * x   # 参数不共享
        out_y = self.spatial_attention_x(x) * y

        out_y = self.channel_attention_x(out_x) * out_y  # 参数不共享
        # out_x = self.channel_attention_x(out_y) * out_x  #原版
        out_x = self.channel_attention_y(out_y) * out_x
        return out_x,out_y
