#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
预训练模型 V2
改进点：
1. 引入非对称卷积 (Asymmetric Convolution) 适应麻将横向牌理
2. 引入花色权重共享 (Weight Sharing) 利用万筒条同构性
3. 增加网络深度与宽度
"""

import torch
from torch import nn

class TileSelfAttention(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (channels // num_heads) ** -0.5
        
        # 1. 生成 Query, Key, Value
        self.qkv = nn.Linear(channels, channels * 3)
        self.proj = nn.Linear(channels, channels)
        
        # 2. 可学习的位置编码 (对应 4*9=36 个位置)
        self.pos_emb = nn.Parameter(torch.randn(1, 36, channels) * 0.02)

    def forward(self, x):
        """
        x: (B, C, 4, 9) -> CNN 提取出的特征图
        """
        B, C, H, W = x.shape
        N = H * W  # 36
        
        # Flatten: (B, C, H, W) -> (B, N, C)
        x = x.flatten(2).transpose(1, 2)
        
        # 加入位置编码：告诉网络这是"一万"还是"东风"
        x = x + self.pos_emb

        # 计算 Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention Score: (B, Heads, N, N)
        # 这张图就是"每张牌对其他牌的关注度"
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        # 聚合特征
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        
        # 还原形状 (B, C, 4, 9)
        x = x.transpose(1, 2).reshape(B, C, H, W)
        return x

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        # 1. Squeeze: 全局平均池化
        # 麻将的 4x9 很小，但这步能提取全图概况
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # 2. Excitation: 两层全连接，学习通道之间的依赖
        # reduction 用于控制参数量，防止过拟合
        mid_channels = max(channels // reduction, 8) # 保证中间层至少有8个神经元
        
        self.fc = nn.Sequential(
            nn.Linear(channels, mid_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, channels, bias=False),
            nn.Sigmoid() # 输出 0~1 的权重
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze -> (B, C, 1, 1) -> (B, C)
        y = self.avg_pool(x).view(b, c)
        # Excitation -> (B, C)
        y = self.fc(y).view(b, c, 1, 1)
        # Scale: 将权重乘回原特征图
        return x * y.expand_as(x)


class AsymResBlock(nn.Module):
    """
    非对称残差块：
    专门针对麻将设计的 Block，只进行横向卷积(1x3)和点卷积(1x1)。
    放弃 3x3 卷积以避免学习无意义的纵向（跨花色）空间关系。
    """
    def __init__(self, c, kernel_size=(1, 3), padding=(0, 1)):
        super().__init__()
        self.net = nn.Sequential(
            # 1. 横向卷积：提取顺子/刻子特征
            nn.Conv2d(c, c, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            
            # 2. 1x1 卷积：通道融合 (相当于全连接，整合Feature)
            nn.Conv2d(c, c, kernel_size=1, bias=False),
            nn.BatchNorm2d(c),
        )
        self.se = SEBlock(c, reduction=8)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
            out = self.net(x)
            
            # 在加回 x 之前，先进行通道加权
            out = self.se(out)
                
            return self.act(x + out)


class SuitSharedEncoder(nn.Module):
    """
    花色共享编码器：
    利用麻将“万筒条”逻辑一样的特性，共享权重处理前三行。
    第四行（字牌）单独处理。
    """
    def __init__(self, in_c, out_c):
        super().__init__()
        
        # 数牌处理塔 (万、筒、条 共享这套参数)
        # 输入形状将是 (B*3, in_c, 1, 9)
        self.suit_tower = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=(1, 3), padding=(0, 1), bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            AsymResBlock(out_c, kernel_size=(1, 3), padding=(0, 1)),
            AsymResBlock(out_c, kernel_size=(1, 3), padding=(0, 1))
        )

        # 字牌处理塔 (单独参数)
        # 字牌没有顺子逻辑，主要靠堆叠，所以多用 1x1，但也保留横向感受野看排列
        self.honor_tower = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=(1, 3), padding=(0, 1), bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            AsymResBlock(out_c, kernel_size=(1, 3), padding=(0, 1)) 
        )

    def forward(self, x):
        # x: (B, 60, 4, 9)
        B, C, H, W = x.shape
        
        # --- 1. 拆分数据 ---
        # suits: 取前3行 (B, 60, 3, 9)
        suits = x[:, :, 0:3, :]
        # honors: 取第4行 (B, 60, 1, 9)
        honors = x[:, :, 3:4, :]

        # --- 2. 处理数牌 (Weight Sharing) ---
        # 变换为 (B*3, 60, 1, 9) 以利用共享卷积
        suits_reshaped = suits.permute(0, 2, 1, 3).reshape(B * 3, C, 1, W)
        suits_features = self.suit_tower(suits_reshaped) # -> (B*3, out_c, 1, 9)
        # 还原回 (B, out_c, 3, 9)
        suits_features = suits_features.view(B, 3, -1, W).permute(0, 2, 1, 3)

        # --- 3. 处理字牌 ---
        honors_features = self.honor_tower(honors) # -> (B, out_c, 1, 9)

        # --- 4. 拼合 ---
        # 结果: (B, out_c, 4, 9)
        out = torch.cat([suits_features, honors_features], dim=2)
        return out


class PretrainModel(nn.Module):
    """
    改进版预训练模型
    """
    def __init__(self, hidden_dim=512, use_vec=True):
        super().__init__()
        self.use_vec = use_vec
        
        # -----------------------------------------------------------
        # 1. 初始特征压缩 (降维/特征筛选)
        # -----------------------------------------------------------
        self.pre_conv = nn.Sequential(
            nn.Conv2d(60, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # -----------------------------------------------------------
        # 2. 花色共享编码 (数牌权重共享，字牌独立)
        # -----------------------------------------------------------
        self.suit_encoder = SuitSharedEncoder(in_c=64, out_c=128)

        # -----------------------------------------------------------
        # 3. 主干网络 (Asymmetric ResBlocks)
        # -----------------------------------------------------------
        # 经过 SuitEncoder 后，通道数变为 128，空间仍为 4x9
        self.backbone = nn.Sequential(
            # 下采样通道融合：128 -> 256
            nn.Conv2d(128, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # 深层特征提取 (堆叠非对称残差块)
            *[AsymResBlock(256) for _ in range(8)],
            
            # 最后的 1x1 卷积压缩，准备 Flatten
            nn.Conv2d(256, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.attention = TileSelfAttention(channels=128) # 假设 backbone 输出 128 通道
        cnn_out_dim = 128 * 4 * 9  # = 4608

        # -----------------------------------------------------------
        # 4. 向量特征与融合
        # -----------------------------------------------------------
        if use_vec:
            self.vec_encoder = nn.Sequential(
                nn.Linear(117, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.15),
                nn.Linear(256, 256),
                nn.ReLU(inplace=True),
            )
            fusion_dim = cnn_out_dim + 256
        else:
            fusion_dim = cnn_out_dim

        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), # 改用 LayerNorm 这种更现代的归一化
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            # 再加一层增加非线性能力
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )

        # -----------------------------------------------------------
        # 5. 输出头
        # -----------------------------------------------------------
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, 235)
        )

        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, obs, vec=None, action_mask=None):
        # 1. 预处理
        x = self.pre_conv(obs.float())
        
        # 2. 花色分别处理
        x = self.suit_encoder(x)
        
        # 3. 全局特征提取 (此时 x 是 (B, 128, 4, 9))
        x = self.backbone(x)

        # --- 插入 Attention ---
        # 此时 x 仍然保持 (B, 128, 4, 9) 的空间结构，这样 Attention 才能知道谁挨着谁
        x = self.attention(x)
        
        # --- 关键：在这里 Flatten ---
        # 变成 (B, 4608)
        cnn_features = x.flatten(1) 
        
        # 4. 向量特征
        if self.use_vec and vec is not None:
            vec_features = self.vec_encoder(vec.float())
            # 此时 cnn_features 和 vec_features 都是 (B, N)，可以拼接
            features = torch.cat([cnn_features, vec_features], dim=1)
        else:
            features = cnn_features

        # 5. 融合
        hidden = self.fusion(features)

        # 6. 输出
        policy_logits = self.policy_head(hidden)
        value = self.value_head(hidden)

        if action_mask is not None:
            mask = action_mask.float()
            inf_mask = torch.clamp(torch.log(mask + 1e-45), min=-1e38, max=0)
            policy_logits = policy_logits + inf_mask

        return policy_logits, value