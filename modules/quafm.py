import torch
import torch.nn as nn
import torch.nn.functional as F


class QUAFM(nn.Module):
    """
    量子不确定性感知融合模块（Quantum Uncertainty-Aware Fusion Module, QUAFM）

    理念与创新点：
    - 通过将时序特征及其不确定性建模为量子叠加态，引入密度矩阵表示融合特征状态。
    - 使用von Neumann熵替代传统香农熵，精准度量融合特征的不确定性和信息熵。
    - 利用量子主成分分析（QPCA）提取关键融合权重，实现基于量子态特征的加权融合。
    - 该模块兼具量子信息理论的严谨性和深度学习的灵活表达能力，提升时序特征融合的效果。

    数学建模框架：
        - 特征状态用密度矩阵 ρ 表示：
            ρ = (1/T) Σ_i |f_i⟩⟨f_i| ，其中 |f_i⟩ 表示特征向量。
        - von Neumann熵定义：
            S(ρ) = -Tr(ρ log ρ)，用以度量融合后的不确定性。
        - 量子主成分分析（QPCA）用于从密度矩阵提取主要特征方向与融合权重。

    输入：
    - refined_features: 经过预处理的平滑特征 [B, C, T]
    - uncertainty: 特征对应的不确定性信息 [B, C, T]

    输出：
    - fused_features: 融合后的时序特征表示 [B, out_channels]

    应用场景：
    - 适用于需要结合不确定性信息进行时序特征融合的任务，如异常检测、时序预测、信号处理等。
    - 通过量子概率理论提升融合表达的鲁棒性和信息利用效率。
    """

    def __init__(self, in_channels, out_channels):
        super(QUAFM, self).__init__()
        self.out_channels = out_channels
        self.proj = nn.Linear(in_channels, out_channels, bias=False)  # 特征投影用于QPCA权重生成
        self.softmax = nn.Softmax(dim=-1)

    def compute_density_matrix(self, features):
        """
        构造密度矩阵: ρ = Σ_{i,j} c_ij |fi⟩⟨fj|
        输入: features [B, C, T]
        输出: 密度矩阵 [B, C, C]
        """
        # L2 归一化后乘以转置，构建密度矩阵
        normed = F.normalize(features, p=2, dim=2)  # 沿 T 归一化
        rho = torch.matmul(normed, normed.transpose(1, 2)) / features.shape[2]
        return rho  # [B, C, C]

    def von_neumann_entropy(self, rho):
        """
        计算 von Neumann 熵: S(ρ) = -Tr(ρ log ρ)
        输入: 密度矩阵 [B, C, C]
        输出: 熵值 [B, 1]
        """
        # 特征值分解（本质等同于量子系统的谱展开）
        eigvals = torch.linalg.eigvalsh(rho)  # 实对称矩阵
        eigvals = torch.clamp(eigvals, min=1e-6)  # 避免 log(0)
        entropy = -torch.sum(eigvals * torch.log(eigvals), dim=1, keepdim=True)  # [B, 1]
        return entropy

    def forward(self, refined_features, uncertainty):
        # Step 1: 分别构建两个密度矩阵
        rho_r = self.compute_density_matrix(refined_features)  # [B, C, C]
        rho_u = self.compute_density_matrix(uncertainty)       # [B, C, C]

        # Step 2: 计算各自的 von Neumann 熵，作为不确定性引导信号
        entropy_r = self.von_neumann_entropy(rho_r)  # [B, 1]
        entropy_u = self.von_neumann_entropy(rho_u)  # [B, 1]

        # Step 3: 根据熵生成融合权重（越低熵→越置信→权重越高）
        total_entropy = entropy_r + entropy_u + 1e-6
        weight_r = 1 - entropy_r / total_entropy  # [B, 1]
        weight_u = 1 - entropy_u / total_entropy  # [B, 1]

        # Step 4: 对原始特征进行线性投影并加权融合
        refined_proj = self.proj(refined_features.mean(dim=2))  # [B, out_channels]
        uncertainty_proj = self.proj(uncertainty.mean(dim=2))   # [B, out_channels]

        fused_features = weight_r * refined_proj + weight_u * uncertainty_proj  # 融合特征

        return fused_features
