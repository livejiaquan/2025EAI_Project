import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------
# Bottleneck Block
# ----------------------
class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, planes, out_channels, downsample=None, stride=1):
        super(Bottleneck, self).__init__()

        assert len(out_channels) == 3, "Bottleneck requires out_channels list of length 3"
        
        ################################################
        # Please replace ??? with the correct variable #            
        # example: in_channels, out_channels[0], ...   #
        ################################################
        # conv1: 1x1 卷積，降低通道數 (in_channels -> out_channels[0])
        self.conv1 = nn.Conv2d(in_channels, out_channels[0], kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels[0])

        # conv2: 3x3 卷積，處理特徵 (out_channels[0] -> out_channels[1])，stride可能為2做downsample
        self.conv2 = nn.Conv2d(out_channels[0], out_channels[1], kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels[1])

        # conv3: 1x1 卷積，增加通道數 (out_channels[1] -> out_channels[2])
        self.conv3 = nn.Conv2d(out_channels[1], out_channels[2], kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels[2])

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)
        return out

# ----------------------
# ResNet
# ----------------------
class ResNet(nn.Module):
    def __init__(self, block, layers, cfg, num_classes=1000, in_channels=3):
        super(ResNet, self).__init__()
        self.current_cfg_idx = 0

        # Conv1 - 使用 cfg[0] 來支援剪枝後的通道數
        self.conv1 = nn.Conv2d(in_channels, cfg[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(cfg[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.current_cfg_idx += 1  # 移動到下一個配置（第一個 Bottleneck）
        self.inplanes = cfg[0]  # 當前通道數從 cfg[0] 讀取

        # Layer1~Layer4
        self.layer1 = self._make_layer(block, 64, layers[0], cfg)
        self.layer2 = self._make_layer(block, 128, layers[1], cfg, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], cfg, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], cfg, stride=2)

        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(self.inplanes, num_classes)

    
    def _make_layer(self, block, planes, blocks, cfg, stride=1):
        #############################################################################
        # Figure out how to generate the correct layers and downsample based on cfg #
        #############################################################################
        downsample = None
        layers = []
        
        # 從 cfg 中讀取第一個 block 的輸出通道配置
        # 每個 Bottleneck 有 3 個卷積層，所以需要讀取 3 個通道數
        out_channels = [cfg[self.current_cfg_idx], 
                       cfg[self.current_cfg_idx + 1], 
                       cfg[self.current_cfg_idx + 2]]
        
        # 記住這個 layer 的標準輸出通道數（保證 identity shortcut）
        # 同一個 layer 內的所有 block 輸出必須相同
        layer_out_channels = out_channels[2]
        
        # 如果維度不匹配（stride != 1 或 通道數改變），需要 downsample
        # 這通常只發生在每個 layer 的第一個 block
        if stride != 1 or self.inplanes != layer_out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, layer_out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(layer_out_channels, affine=False)  # 使用 affine=False 以符合老師的參數數量
            )
        
        # 第一個 block（可能有 downsample 和 stride）
        layers.append(block(self.inplanes, planes, out_channels, downsample, stride))
        self.inplanes = layer_out_channels  # 更新當前通道數
        self.current_cfg_idx += 3  # 移動 cfg 索引（每個 block 有 3 個卷積層）
        
        # 剩餘的 blocks（stride=1，identity shortcut）
        # 關鍵：這些 block 的輸入輸出都必須是 layer_out_channels
        for i in range(1, blocks):
            out_channels = [cfg[self.current_cfg_idx], 
                           cfg[self.current_cfg_idx + 1], 
                           cfg[self.current_cfg_idx + 2]]
            
            # 確保 cfg 設計正確：同一 layer 內所有 block 的輸出必須相同
            assert out_channels[2] == layer_out_channels, \
                f"cfg 錯誤：block 輸出 {out_channels[2]} ≠ layer 輸出 {layer_out_channels}，會破壞 identity shortcut"
            
            # 因為輸入(self.inplanes) == 輸出(out_channels[2]) == layer_out_channels
            # 所以不需要 downsample，使用 identity shortcut
            layers.append(block(self.inplanes, planes, out_channels, None, 1))
            # self.inplanes 保持不變（仍然是 layer_out_channels）
            self.current_cfg_idx += 3
        
        return nn.Sequential(*layers) 

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def ResNet50(num_classes=10, in_channels=3, cfg=None):
    # cfg: number of channels for conv1 and the three conv layers in each Bottleneck
    if cfg is None:
        cfg = [64] + \
              [64, 64, 256]*3 + \
              [128, 128, 512]*4 + \
              [256, 256, 1024]*6 + \
              [512, 512, 2048]*3
    layers = [3, 4, 6, 3]
    return ResNet(Bottleneck, layers, cfg, num_classes=num_classes, in_channels=in_channels)

def ResNet101(num_classes=10, in_channels=3, cfg=None):
    # cfg: number of channels for conv1 and the three conv layers in each Bottleneck
    if cfg is None:
        cfg = [64] + \
              [64, 64, 256]*3 + \
              [128, 128, 512]*4 + \
              [256, 256, 1024]*23 + \
              [512, 512, 2048]*3
    layers = [3, 4, 23, 3]
    return ResNet(Bottleneck, layers, cfg, num_classes=num_classes, in_channels=in_channels)

def ResNet152(num_classes=10, in_channels=3, cfg=None):
    # cfg: number of channels for conv1 and the three conv layers in each Bottleneck
    if cfg is None:
        cfg = [64] + \
              [64, 64, 256]*3 + \
              [128, 128, 512]*8 + \
              [256, 256, 1024]*36 + \
              [512, 512, 2048]*3
    layers = [3, 8, 36, 3]
    return ResNet(Bottleneck, layers, cfg, num_classes=num_classes, in_channels=in_channels)