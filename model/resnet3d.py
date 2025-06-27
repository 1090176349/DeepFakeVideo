import os
import torch
import torch.nn as nn
from model.resnet3d50 import resnet3d50  # 自定义的 ResNet3D-50 模型实现
from modules.cnsdfm import CNSDFM
from modules.fotadm import FOTADM
from modules.quafm import QUAFM
from config import config

device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
# 指定预训练（知识蒸馏后）学生模型权重路径
backbone_checkpoint = os.path.join(config.MODEL_SAVE_PATH, "resnet3d50.pth")

class SDE_QNet(nn.Module):
    def __init__(self, num_classes=2):
        super(SDE_QNet, self).__init__()
        # 构造 ResNet3D-50 骨干网络（预设 num_classes 用于构造 fc 层，但后续不使用 fc）
        backbone = resnet3d50(num_classes=config.NUM_CLASSES)
        # 加载知识蒸馏后学生模型权重（仅加载匹配部分）
        state_dict = torch.load(backbone_checkpoint, map_location=device)
        backbone.load_state_dict(state_dict, strict=False)
        self.backbone = backbone

        # 定义动态分支拆分：先使用 stem 部分（conv1, bn1, relu, maxpool），然后再调用 layer1
        self.stem = nn.Sequential(
            backbone.conv1,   # 输出: [B, 64, T, H, W]
            backbone.bn1,
            backbone.relu,
            backbone.maxpool  # 输出: [B, 64, T, H/4, W/4]
        )
        self.layer1 = backbone.layer1   # 将 64 通道转换为 256 通道
        self.layer2 = backbone.layer2   # 输出: [B, 512, T, H, W]
        
        # 全局分支：从 layer3 开始
        self.layer3 = backbone.layer3   # 输出: [B, 1024, T, H, W]
        self.layer4 = backbone.layer4   # 输出: [B, 2048, T, H, W]
        self.avgpool = backbone.avgpool
        
        # 动态分支改进模块：输入通道为 512（来自 layer2）
        self.cnsdfm = CNSDFM(feature_dim=512)
        self.fotadm = FOTADM(in_channels=512)
        self.quafm = QUAFM(in_channels=512, out_channels=256)
        
        # 全局分支降维：将 2048 降为 256
        self.fc_global = nn.Linear(2048, 256)
        # 融合动态和全局分支特征（256+256=512）后分类
        self.fc_fuse = nn.Linear(512, num_classes)
    
    def forward(self, x):
        """
        输入 x: [B, 3, T, H, W]
        """
        B = x.size(0)
        # 动态分支：先经过 stem，再经过 layer1 得到 256 通道，再进入 layer2
        x_stem = self.stem(x)              # [16, 64, 16, 28, 28]
        x_l1 = self.layer1(x_stem)         # [16, 256, 16, 28, 28]
        x_dynamic = self.layer2(x_l1)      # [16, 512, 8, 14, 14]
        # 空间平均池化：对 H,W 维度做平均，保留时间 -> [B, 512, T]
        dynamic_seq = x_dynamic.mean(dim=[3,4]) #[16, 512, 8]
        refined_features, uncertainty = self.cnsdfm(dynamic_seq) #[16, 512, 8] 两个返回值都是这个shape
        anomaly_scores = self.fotadm(dynamic_seq, refined_features) #[16, 8]
        dynamic_feature = self.quafm(refined_features, uncertainty)  # [16, 256]
        
        # 全局分支：从 x_dynamic 继续传递 layer3, layer4, avgpool
        x_global = self.layer3(x_dynamic)    # [16, 1024, 4, 7, 7]
        x_global = self.layer4(x_global)     # [16, 2048, 2, 4, 4]
        x_global = self.avgpool(x_global)    # [16, 2048, 1, 1, 1]
        x_global = x_global.view(B, -1)      # [B, 2048]
        global_feature = self.fc_global(x_global)  # [B, 256]
        
        # 融合
        fused_features = torch.cat([dynamic_feature, global_feature], dim=1)  # [B, 512]
        logits = self.fc_fuse(fused_features)  # [B, num_classes]
        return logits, anomaly_scores

if __name__ == '__main__':
    model = SDE_QNet(num_classes=2)
    model = model.to(device)
    x = torch.randn(2, 3, 32, 112, 112).to(device)
    logits, anomaly = model(x)
    print("Logits shape:", logits.shape)
    print("Anomaly scores shape:", anomaly.shape)
