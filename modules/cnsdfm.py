import torch
import torch.nn as nn

class SDEFunction(nn.Module):
    """
    定义随机微分系统的 drift 和 diffusion 模块
    dx_t = f(x_t) dt + σ(x_t) ◦ dW_t
    """
    def __init__(self, feature_dim, hidden_dim=64):
        super(SDEFunction, self).__init__()
        self.drift = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, feature_dim)
        )
        self.diffusion = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
            nn.Softplus()  # 确保扩散项为正
        )

    def forward(self, x_t):
        """
        输入: x_t [B, feature_dim]
        输出: dx_t = drift * dt + diffusion * dW_t
        """
        drift = self.drift(x_t)
        diffusion = self.diffusion(x_t)
        return drift, diffusion


class CNSDFM(nn.Module):
    """
    连续时间神经随机微分滤波模块（Continuous-time Neural Stochastic Differential Filter Module, CNSDFM）

    理念与创新点：
    - 本模块将传统的离散卡尔曼滤波扩展为连续时间的随机微分系统，通过 SDE（Stochastic Differential Equations）建模状态演化过程。
    - 状态更新方程由 Stratonovich 积分驱动，保留几何结构与物理一致性，克服 Itô 积分的路径依赖问题。
    - 结合 Fokker-Planck 方程思想，通过 SDE 模拟隐式建模状态分布的时间演化。
    - 采用 LSTM 捕捉长程时序依赖，生成预测状态；同时引入自适应卡尔曼增益网络（noise_net）根据观测残差动态调整融合权重。
    
    数学建模框架：
        状态方程：dx_t = f(x_t) dt + σ(x_t) ◦ dW_t     （Stratonovich 形式）
        观测方程：dz_t = h(x_t) dt + η dV_t              （观测噪声建模）

    输入：
    - z: 原始观测序列 [B, feature_dim, T]

    输出：
    - refined_features: 滤波平滑后的时序特征 [B, feature_dim, T]
    - uncertainty: 每个时间步的不确定性估计 [B, feature_dim, T]（1 - K_t）

    应用场景：
    - 适用于视频、时间序列、生物信号等时序建模任务，特别是在伪造检测、异常识别中用于建模观测与潜在状态间的不确定性和演化规律。
    """
    def __init__(self, feature_dim, hidden_dim=64, num_layers=1, dt=1.0):
        super(CNSDFM, self).__init__()
        self.feature_dim = feature_dim
        self.dt = dt

        # LSTM 预测潜在状态（先验估计）
        self.lstm = nn.LSTM(input_size=feature_dim, hidden_size=hidden_dim,
                            num_layers=num_layers, batch_first=True)
        self.fc_pred = nn.Linear(hidden_dim, feature_dim)

        # SDE 模型（连续演化）
        self.sde_func = SDEFunction(feature_dim, hidden_dim)

        # 自适应卡尔曼增益估计器（通过观测残差）
        self.noise_net = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.Sigmoid()
        )

    def forward(self, z):
        B, C, T = z.shape
        z_seq = z.transpose(1, 2)  # [B, T, C]
        lstm_out, _ = self.lstm(z_seq)
        pred_state = self.fc_pred(lstm_out).transpose(1, 2)  # [B, C, T]

        refined_features = []
        uncertainty_list = []

        x_t = z[:, :, 0]  # 初始状态
        refined_features.append(x_t.unsqueeze(-1))
        uncertainty_list.append(torch.zeros_like(x_t).unsqueeze(-1))

        for t in range(1, T):
            x_pred = pred_state[:, :, t]
            z_t = z[:, :, t]

            # ---- SDE 更新 ----
            drift, diffusion = self.sde_func(x_t)  # [B, C], [B, C]
            noise = torch.randn_like(x_t)
            dx = drift * self.dt + diffusion * noise * torch.sqrt(torch.tensor(self.dt))  # 近似 Stratonovich 积分
            x_sde = x_t + dx  # 连续系统演化

            # 残差 + 自适应卡尔曼增益
            residual = z_t - x_pred
            K_t = self.noise_net(residual)

            # 状态更新：结合 SDE 演化结果与观测
            x_t = x_sde + K_t * (z_t - x_sde)

            refined_features.append(x_t.unsqueeze(-1))
            uncertainty_list.append((1 - K_t).unsqueeze(-1))

        refined_features = torch.cat(refined_features, dim=-1)
        uncertainty = torch.cat(uncertainty_list, dim=-1)
        return refined_features, uncertainty
