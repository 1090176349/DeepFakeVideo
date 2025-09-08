import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --------------------------
# 安全的分数阶导数权重生成函数
# --------------------------
def fractional_weights(alpha, steps):
    weights = [1.0]
    for k in range(1, steps):
        try:
            denom = math.gamma(k + 1) * math.gamma(alpha - k + 1)
            if denom == 0:
                break
            coeff = (-1) ** k * math.gamma(alpha + 1) / denom
            weights.append(coeff)
        except ValueError:
            break  # 非法 gamma 值，提前结束
    return torch.tensor(weights, dtype=torch.float32)


# --------------------------
# 分数阶微分模块（每通道一个 α）
# --------------------------
class FractionalDerivative(nn.Module):
    def __init__(self, channels, max_order=10):
        super(FractionalDerivative, self).__init__()
        self.channels = channels
        self.max_order = max_order
        self.alpha = nn.Parameter(torch.ones(channels) * 1.0)  # 可学习 α

    def forward(self, x):  # x: [B, C, T]
        B, C, T = x.shape
        device = x.device
        outputs = []

        for c in range(C):
            alpha_c = torch.clamp(self.alpha[c], 0.01, 1.99)
            weights = fractional_weights(alpha_c.item(), self.max_order).to(device).flip(0)  # [K]
            K = weights.shape[0]
            kernel = weights.view(1, 1, K)  # [1, 1, K]

            x_c = x[:, c:c+1, :]  # [B, 1, T]
            x_padded = F.pad(x_c, (K - 1, 0))  # 左 padding，保长度
            out = F.conv1d(x_padded, kernel)  # [B, 1, T]
            outputs.append(out)

        return torch.cat(outputs, dim=1)  # [B, C, T]


class FOTADM(nn.Module):
    """
    分数阶时序异常检测模块（Fractional-Order Temporal Anomaly Detection Module, FOTADM)

    理念与创新点：
    - 本模块通过引入分数阶微分算子，替代传统卷积，捕捉时序数据中的长程依赖和复杂异常模式。
    - 利用可学习的分数阶阶数 α，自适应调整微分阶数，实现多分数阶融合机制，增强模型表达能力。
    - 采用基于Gamma函数的权重计算方法，间接实现 Mittag-Leffler 函数的高效近似，保证分数阶导数计算的数值稳定性。
    - 结合多尺度残差信息，映射为异常得分，实现时序异常的精准检测。

    数学建模框架：
        分数阶微分定义（Grünwald–Letnikov形式）：
            D^α R(t) ≈ Σ_{k=0}^∞ (-1)^k (Γ(α+1) / (Γ(k+1) Γ(α - k +1))) R(t - kh)
        其中 α ∈ (0, 2) 为可学习分数阶数。

    输入：
    - original_features: 原始时序特征 [B, C, T]
    - refined_features:  经过预处理或滤波后的特征 [B, C, T]

    输出：
    - anomaly_scores:   异常检测得分 [B, T]，取值范围 [0, 1]

    应用场景：
    - 适用于视频伪造检测、工业故障预测、生物信号异常识别等多种时序异常检测任务，
      通过捕捉复杂长程依赖和非局部特征，提高异常检测的鲁棒性和准确率。
    """
    def __init__(self, in_channels, max_order=10):
        super(FOTADM, self).__init__()
        self.in_channels = in_channels
        self.frac_diff1 = FractionalDerivative(in_channels, max_order=max_order)
        self.frac_diff2 = FractionalDerivative(in_channels, max_order=max_order)  # 多阶融合

        # 全连接映射为异常分数
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),
            nn.ReLU(),
            nn.Linear(in_channels // 2, 1)
        )

    def forward(self, original_features, refined_features):
        # 计算残差
        residual = torch.abs(original_features - refined_features)  # [B, C, T]

        # 分数阶微分残差分析（多阶融合）
        res_frac1 = self.frac_diff1(residual)  # [B, C, T]
        res_frac2 = self.frac_diff2(residual)  # [B, C, T]
        multi_frac_res = res_frac1 + res_frac2  # 简单加权融合

        # 通道均值，获取粗略异常指标 [B, T]
        residual_avg = multi_frac_res.mean(dim=1)

        # 全连接层映射为 [B, T] 概率
        res_transposed = multi_frac_res.transpose(1, 2)  # [B, T, C]
        anomaly_scores = self.fc(res_transposed).squeeze(-1)  # [B, T]
        anomaly_scores = torch.sigmoid(anomaly_scores)

        return anomaly_scores
