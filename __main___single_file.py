#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
__main__.py: Botzone单文件版本
所有代码合并在一个文件中，只需要额外上传模型文件
"""

import sys
import os
from collections import defaultdict
import numpy as np
import torch
from torch import nn

DEBUG_MODE = os.environ.get("MAHJONG_BOT_DEBUG", "0") == "1"

agent = None
seatWind = None
angang = None
zimo = False


def _safe_input():
    """返回一行输入，EOF时返回None"""
    try:
        return input()
    except EOFError:
        return None


def load_state_dict_compat(model, ckpt_path):
    sd = torch.load(ckpt_path, map_location="cpu")
    if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]
    if any(k.startswith("_orig_mod.") for k in sd.keys()):
        sd = {k[len("_orig_mod."):]: v for k, v in sd.items()}
    if any(k.startswith("module.") for k in sd.keys()):
        sd = {k[len("module."):]: v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    return model

# ============================================================
# Agent基类
# ============================================================

class MahjongGBAgent:
    observation_space = None
    action_space = None

    def __init__(self, seatWind):
        pass

    def request2obs(self, request):
        pass

    def action2response(self, action):
        pass


# ============================================================
# MahjongGB导入
# ============================================================

try:
    from MahjongGB import MahjongFanCalculator
except:
    pass


# ============================================================
# 基础FeatureAgent (6通道)
# ============================================================

class FeatureAgentBase(MahjongGBAgent):
    OBS_SIZE = 6
    ACT_SIZE = 235

    OFFSET_OBS = {
        'SEAT_WIND': 0,
        'PREVALENT_WIND': 1,
        'HAND': 2
    }
    OFFSET_ACT = {
        'Pass': 0,
        'Hu': 1,
        'Play': 2,
        'Chi': 36,
        'Peng': 99,
        'Gang': 133,
        'AnGang': 167,
        'BuGang': 201
    }
    TILE_LIST = [
        *('W%d'%(i+1) for i in range(9)),
        *('T%d'%(i+1) for i in range(9)),
        *('B%d'%(i+1) for i in range(9)),
        *('F%d'%(i+1) for i in range(4)),
        *('J%d'%(i+1) for i in range(3))
    ]
    OFFSET_TILE = {c: i for i, c in enumerate(TILE_LIST)}

    def __init__(self, seatWind):
        self.seatWind = seatWind
        self.hand = []
        self.packs = [[] for _ in range(4)]
        self.history = [[] for _ in range(4)]
        self.tileWall = [21] * 4
        self.shownTiles = defaultdict(int)
        self.wallLast = False
        self.isAboutKong = False
        self.prevalentWind = 0
        self.obs = np.zeros((self.OBS_SIZE, 36), dtype=np.int8)
        self.obs[self.OFFSET_OBS['SEAT_WIND']][self.OFFSET_TILE['F%d' % (self.seatWind + 1)]] = 1
        self.valid = []
        self.curTile = None
        self.tileFrom = None


# ============================================================
# 60通道FeatureAgent
# ============================================================

class FeatureAgent10M(FeatureAgentBase):
    """60通道+117维vec特征"""

    OBS_SIZE = 60
    VEC_SIZE = 117

    def __init__(self, seatWind):
        super().__init__(seatWind)
        self.obs = np.zeros((self.OBS_SIZE, 36), dtype=np.int8)
        self.obs[self.OFFSET_OBS['SEAT_WIND']][self.OFFSET_TILE['F%d' % (self.seatWind + 1)]] = 1

    def _obs(self):
        """返回观察字典"""
        return {
            'observation': self._get_extended_observation(),
            'vec': self._get_vector_features(),
            'action_mask': np.array([1 if i in self.valid else 0 for i in range(self.ACT_SIZE)], dtype=np.int8)
        }

    def _get_extended_observation(self):
        """构造60×4×9观察"""
        obs = np.zeros((60, 4, 9), dtype=np.int8)

        # 通道0-1: 风信息
        for i in range(36):
            r, c = divmod(i, 9)
            obs[0, r, c] = self.obs[self.OFFSET_OBS['SEAT_WIND'], i]
            obs[1, r, c] = self.obs[self.OFFSET_OBS['PREVALENT_WIND'], i]

        # 通道2-5: 手牌
        for i in range(36):
            r, c = divmod(i, 9)
            for j in range(4):
                if 2 + j < 6:
                    obs[2 + j, r, c] = self.obs[self.OFFSET_OBS['HAND'] + j, i]

        # 通道6-9: 各玩家打出的牌
        for player_id in range(4):
            for tile in self.history[player_id]:
                tile_idx = self.OFFSET_TILE.get(tile, -1)
                if tile_idx >= 0:
                    r, c = divmod(tile_idx, 9)
                    obs[6 + player_id, r, c] = min(obs[6 + player_id, r, c] + 1, 4)

        # 通道10-13: 副露明牌
        for player_id in range(4):
            for pack in self.packs[player_id]:
                if len(pack) >= 2:
                    tile = pack[1]
                    if tile != 'CONCEALED':
                        tile_idx = self.OFFSET_TILE.get(tile, -1)
                        if tile_idx >= 0:
                            r, c = divmod(tile_idx, 9)
                            obs[10 + player_id, r, c] = min(obs[10 + player_id, r, c] + 1, 4)

        # 通道14-17: 其他玩家手牌数
        for player_id in range(4):
            remaining = self.tileWall[player_id] if player_id < len(self.tileWall) else 13
            normalized = min(remaining / 13.0, 1.0)
            obs[14 + player_id, :, :] = int(normalized * 4)

        # 通道18-53: 剩余牌统计
        for tile_idx, tile in enumerate(self.TILE_LIST):
            shown = self.shownTiles.get(tile, 0)
            remaining = max(4 - shown, 0)
            r, c = divmod(tile_idx, 9)
            channel = 18 + tile_idx
            if channel < 54:
                obs[channel, r, c] = remaining

        return obs

    def _get_vector_features(self):
        """构造117维向量特征"""
        vec = np.zeros(117, dtype=np.float32)

        # 0-33: 手牌统计
        for tile in self.hand:
            tile_idx = self.OFFSET_TILE.get(tile, -1)
            if 0 <= tile_idx < 34:
                vec[tile_idx] = min(vec[tile_idx] + 0.25, 1.0)

        # 34-67: 已打出牌统计
        for tile, count in self.shownTiles.items():
            tile_idx = self.OFFSET_TILE.get(tile, -1)
            if 0 <= tile_idx < 34:
                vec[34 + tile_idx] = min(count / 4.0, 1.0)

        # 68-101: 副露牌统计
        for packs_list in self.packs:
            for pack in packs_list:
                if len(pack) >= 2 and pack[1] != 'CONCEALED':
                    tile_idx = self.OFFSET_TILE.get(pack[1], -1)
                    if 0 <= tile_idx < 34:
                        vec[68 + tile_idx] = min(vec[68 + tile_idx] + 0.25, 1.0)

        # 102-116: 其他特征
        vec[102] = self.prevalentWind / 3.0 if hasattr(self, 'prevalentWind') else 0.0
        vec[103] = self.seatWind / 3.0
        avg_wall = sum(self.tileWall) / 4.0 if self.tileWall else 21.0
        vec[104] = min(avg_wall / 21.0, 1.0)
        vec[105] = 1.0 if self.wallLast else 0.0
        vec[106] = len(self.hand) / 14.0
        vec[107] = len(self.packs[0]) / 4.0 if len(self.packs) > 0 else 0.0

        wan = sum(1 for t in self.hand if t.startswith('W'))
        tiao = sum(1 for t in self.hand if t.startswith('T'))
        bing = sum(1 for t in self.hand if t.startswith('B'))
        zi = sum(1 for t in self.hand if t[0] in ['F', 'J'])
        vec[108] = wan / 9.0
        vec[109] = tiao / 9.0
        vec[110] = bing / 9.0
        vec[111] = zi / 7.0
        vec[112:117] = 0.0

        return vec.astype(np.float16)

    def _hand_embedding_update(self):
        """更新手牌编码"""
        for i in range(4):
            self.obs[self.OFFSET_OBS['HAND'] + i] = 0
        for tile in self.hand:
            tile_idx = self.OFFSET_TILE[tile]
            for i in range(4):
                if self.obs[self.OFFSET_OBS['HAND'] + i, tile_idx] == 0:
                    self.obs[self.OFFSET_OBS['HAND'] + i, tile_idx] = 1
                    break

    def _check_mahjong(self, tile, isSelfDrawn=False, isAboutKong=False):
        """检查能否胡牌"""
        try:
            fans = MahjongFanCalculator(
                pack=tuple(self.packs[0]),
                hand=tuple(self.hand),
                winTile=tile,
                flowerCount=0,
                isSelfDrawn=isSelfDrawn,
                is4thTile=(self.shownTiles[tile] + isSelfDrawn) == 4,
                isAboutKong=isAboutKong,
                isWallLast=self.wallLast,
                seatWind=self.seatWind,
                prevalentWind=self.prevalentWind,
                verbose=True
            )
            fan_cnt = 0
            for fan_point, cnt, fan_name, fan_name_en in fans:
                fan_cnt += fan_point * cnt
            if fan_cnt < 8:
                raise Exception('Not Enough Fans')
        except:
            return False
        return True

    def request2obs(self, request):
        """协议解析"""
        t = request.split()
        if len(t) == 0:
            return

        if t[0] == 'Wind':
            self.prevalentWind = int(t[1])
            self.obs[self.OFFSET_OBS['PREVALENT_WIND']] = 0
            self.obs[self.OFFSET_OBS['PREVALENT_WIND']][self.OFFSET_TILE['F%d' % (self.prevalentWind + 1)]] = 1
            return

        if t[0] == 'Deal':
            self.hand = t[1:]
            self._hand_embedding_update()
            return

        if t[0] == 'Huang':
            self.valid = []
            return self._obs()

        if t[0] == 'Draw':
            self.tileWall[0] -= 1
            self.wallLast = self.tileWall[1] == 0
            tile = t[1]
            self.curTile = tile
            self.valid = []
            if self._check_mahjong(tile, isSelfDrawn=True, isAboutKong=self.isAboutKong):
                self.valid.append(self.OFFSET_ACT['Hu'])
            self.isAboutKong = False
            self.hand.append(tile)
            self._hand_embedding_update()
            for tile in set(self.hand):
                self.valid.append(self.OFFSET_ACT['Play'] + self.OFFSET_TILE[tile])
                if self.hand.count(tile) == 4 and not self.wallLast and self.tileWall[0] > 0:
                    self.valid.append(self.OFFSET_ACT['AnGang'] + self.OFFSET_TILE[tile])
            if not self.wallLast and self.tileWall[0] > 0:
                for pack_type, pack_tile, offer in self.packs[0]:
                    if pack_type == 'PENG' and pack_tile in self.hand:
                        self.valid.append(self.OFFSET_ACT['BuGang'] + self.OFFSET_TILE[pack_tile])
            return self._obs()

        if t[0] == 'Player':
            p = (int(t[1]) + 4 - self.seatWind) % 4
            if t[2] == 'Draw':
                self.tileWall[p] -= 1
                self.wallLast = self.tileWall[(p + 1) % 4] == 0
                return
            if t[2] == 'Invalid':
                self.valid = []
                return self._obs()
            if t[2] == 'Hu':
                self.valid = []
                return self._obs()
            if t[2] == 'Play':
                self.tileFrom = p
                self.curTile = t[3]
                self.shownTiles[self.curTile] += 1
                self.history[p].append(self.curTile)
                if p == 0:
                    if self.curTile in self.hand:
                        self.hand.remove(self.curTile)
                    self._hand_embedding_update()
                    return
                else:
                    self.valid = []
                    if self._check_mahjong(self.curTile):
                        self.valid.append(self.OFFSET_ACT['Hu'])
                    if not self.wallLast:
                        if self.hand.count(self.curTile) >= 2:
                            self.valid.append(self.OFFSET_ACT['Peng'] + self.OFFSET_TILE[self.curTile])
                            if self.hand.count(self.curTile) == 3 and self.tileWall[0] > 0:
                                self.valid.append(self.OFFSET_ACT['Gang'] + self.OFFSET_TILE[self.curTile])
                        color = self.curTile[0]
                        if p == 3 and color in 'WTB':
                            num = int(self.curTile[1])
                            tmp = []
                            for i in range(-2, 3):
                                tmp.append(color + str(num + i))
                            if tmp[0] in self.hand and tmp[1] in self.hand:
                                self.valid.append(self.OFFSET_ACT['Chi'] + 'WTB'.index(color) * 21 + (num - 3) * 3 + 2)
                            if tmp[1] in self.hand and tmp[3] in self.hand:
                                self.valid.append(self.OFFSET_ACT['Chi'] + 'WTB'.index(color) * 21 + (num - 2) * 3 + 1)
                            if tmp[3] in self.hand and tmp[4] in self.hand:
                                self.valid.append(self.OFFSET_ACT['Chi'] + 'WTB'.index(color) * 21 + (num - 1) * 3)
                    self.valid.append(self.OFFSET_ACT['Pass'])
                    return self._obs()
            if t[2] == 'Chi':
                tile = t[3]
                color = tile[0]
                num = int(tile[1])
                self.packs[p].append(('CHI', tile, int(self.curTile[1]) - num + 2))
                self.shownTiles[self.curTile] -= 1
                for i in range(-1, 2):
                    self.shownTiles[color + str(num + i)] += 1
                self.wallLast = self.tileWall[(p + 1) % 4] == 0
                if p == 0:
                    self.valid = []
                    self.hand.append(self.curTile)
                    for i in range(-1, 2):
                        self.hand.remove(color + str(num + i))
                    self._hand_embedding_update()
                    for tile in set(self.hand):
                        self.valid.append(self.OFFSET_ACT['Play'] + self.OFFSET_TILE[tile])
                    return self._obs()
                return
            if t[2] == 'UnChi':
                tile = t[3]
                color = tile[0]
                num = int(tile[1])
                self.packs[p].pop()
                self.shownTiles[self.curTile] += 1
                for i in range(-1, 2):
                    self.shownTiles[color + str(num + i)] -= 1
                if p == 0:
                    for i in range(-1, 2):
                        self.hand.append(color + str(num + i))
                    if self.curTile in self.hand:
                        self.hand.remove(self.curTile)
                    self._hand_embedding_update()
                return
            if t[2] == 'Peng':
                self.packs[p].append(('PENG', self.curTile, (4 + p - self.tileFrom) % 4))
                self.shownTiles[self.curTile] += 2
                self.wallLast = self.tileWall[(p + 1) % 4] == 0
                if p == 0:
                    self.valid = []
                    for _ in range(2):
                        if self.curTile in self.hand:
                            self.hand.remove(self.curTile)
                    self._hand_embedding_update()
                    for tile in set(self.hand):
                        self.valid.append(self.OFFSET_ACT['Play'] + self.OFFSET_TILE[tile])
                    return self._obs()
                return
            if t[2] == 'UnPeng':
                self.packs[p].pop()
                self.shownTiles[self.curTile] -= 2
                if p == 0:
                    for _ in range(2):
                        self.hand.append(self.curTile)
                    self._hand_embedding_update()
                return
            if t[2] == 'Gang':
                self.packs[p].append(('GANG', self.curTile, (4 + p - self.tileFrom) % 4))
                self.shownTiles[self.curTile] += 3
                if p == 0:
                    for _ in range(3):
                        if self.curTile in self.hand:
                            self.hand.remove(self.curTile)
                    self._hand_embedding_update()
                    self.isAboutKong = True
                return
            if t[2] == 'AnGang':
                tile = 'CONCEALED' if p else t[3]
                self.packs[p].append(('GANG', tile, 0))
                if p == 0:
                    self.isAboutKong = True
                    for _ in range(4):
                        if tile in self.hand:
                            self.hand.remove(tile)
                else:
                    self.isAboutKong = False
                return
            if t[2] == 'BuGang':
                tile = t[3]
                for i in range(len(self.packs[p])):
                    if tile == self.packs[p][i][1]:
                        self.packs[p][i] = ('GANG', tile, self.packs[p][i][2])
                        break
                self.shownTiles[tile] += 1
                if p == 0:
                    if tile in self.hand:
                        self.hand.remove(tile)
                    self._hand_embedding_update()
                    self.isAboutKong = True
                    return
                else:
                    self.valid = []
                    if self._check_mahjong(tile, isSelfDrawn=False, isAboutKong=True):
                        self.valid.append(self.OFFSET_ACT['Hu'])
                    self.valid.append(self.OFFSET_ACT['Pass'])
                    return self._obs()

        return

    def action2response(self, action):
        """动作转响应"""
        if action < self.OFFSET_ACT['Hu']:
            return 'Pass'
        if action < self.OFFSET_ACT['Play']:
            return 'Hu'
        if action < self.OFFSET_ACT['Chi']:
            return 'Play ' + self.TILE_LIST[action - self.OFFSET_ACT['Play']]
        if action < self.OFFSET_ACT['Peng']:
            t = (action - self.OFFSET_ACT['Chi']) // 3
            return 'Chi ' + 'WTB'[t // 7] + str(t % 7 + 2)
        if action < self.OFFSET_ACT['Gang']:
            return 'Peng'
        if action < self.OFFSET_ACT['AnGang']:
            return 'Gang'
        if action < self.OFFSET_ACT['BuGang']:
            return 'Gang ' + self.TILE_LIST[action - self.OFFSET_ACT['AnGang']]
        return 'BuGang ' + self.TILE_LIST[action - self.OFFSET_ACT['BuGang']]


# ============================================================
# 预训练模型
# ============================================================
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
# ============================================================
# Botzone交互
# ============================================================

def obs2response(model, obs):
    """模型推理"""
    if not isinstance(obs, dict):
        return "PASS"
    observation = obs.get("observation")
    vec = obs.get("vec")
    action_mask = obs.get("action_mask")
    if observation is None or vec is None or action_mask is None:
        if DEBUG_MODE:
            print("[DEBUG] obs missing keys, fallback PASS", file=sys.stderr)
        return "PASS"

    observation = torch.from_numpy(np.expand_dims(observation, 0))
    vec = torch.from_numpy(np.expand_dims(vec, 0))
    action_mask = torch.from_numpy(np.expand_dims(action_mask, 0))

    with torch.no_grad():
        logits, _ = model(observation, vec, action_mask)

    logits_np = logits.detach().numpy().flatten()
    mask_np = obs["action_mask"].flatten()
    valid_indices = np.flatnonzero(mask_np)
    if valid_indices.size == 0:
        action = agent.OFFSET_ACT['Pass']
    else:
        best_local = logits_np[valid_indices].argmax()
        action = int(valid_indices[best_local])
    response = agent.action2response(action)
    return response


def handle_request_line(model, request):
    """根据一行请求返回应该输出的响应"""
    global agent, seatWind, angang, zimo

    tokens = request.split()
    if not tokens:
        return "PASS"

    if tokens[0] == "0":
        seatWind = int(tokens[1])
        agent = FeatureAgent10M(seatWind)
        agent.request2obs("Wind %s" % tokens[2])
        return "PASS"

    if tokens[0] == "1":
        agent.request2obs(" ".join(["Deal", *tokens[5:]]))
        return "PASS"

    if tokens[0] == "2":
        obs = agent.request2obs("Draw %s" % tokens[1])
        if not obs:
            if DEBUG_MODE:
                print("[DEBUG] Draw obs missing, output PASS", file=sys.stderr)
            return "PASS"
        response = obs2response(model, obs)
        t = response.split()
        if t[0] == "Hu":
            return "HU"
        if t[0] == "Play":
            cand = t[1]
            # 兜底：若模型输出的牌不在手牌里，优先打刚摸的牌
            if agent is not None:
                if cand not in agent.hand:
                    fallback_tile = agent.curTile if agent and agent.curTile in agent.hand else (agent.hand[0] if agent and agent.hand else cand)
                    cand = fallback_tile
                if agent and cand in agent.hand:
                    try:
                        agent.hand.remove(cand)
                        agent._hand_embedding_update()
                        agent.curTile = None
                    except ValueError:
                        pass
            return "PLAY %s" % cand
        if t[0] == "Gang":
            angang = t[1]
            return "GANG %s" % t[1]
        if t[0] == "BuGang":
            return "BUGANG %s" % t[1]
        return "PASS"

    if tokens[0] == "3":
        p = int(tokens[1])
        action = tokens[2]

        if action == "DRAW":
            agent.request2obs("Player %d Draw" % p)
            zimo = True
            return "PASS"

        if action == "GANG":
            if p == seatWind and angang:
                agent.request2obs("Player %d AnGang %s" % (p, angang))
            elif zimo:
                agent.request2obs("Player %d AnGang" % p)
            else:
                agent.request2obs("Player %d Gang" % p)
            return "PASS"

        if action == "BUGANG":
            obs = agent.request2obs("Player %d BuGang %s" % (p, tokens[3]))
            if p == seatWind:
                return "PASS"
            if not obs:
                if DEBUG_MODE:
                    print("[DEBUG] BuGang obs missing, output PASS", file=sys.stderr)
                return "PASS"
            response = obs2response(model, obs)
            return "HU" if response == "Hu" else "PASS"

        zimo = False
        if action == "CHI":
            agent.request2obs("Player %d Chi %s" % (p, tokens[3]))
        elif action == "PENG":
            agent.request2obs("Player %d Peng" % p)

        obs = agent.request2obs("Player %d Play %s" % (p, tokens[-1]))
        if p == seatWind:
            return "PASS"
        if not obs:
            if DEBUG_MODE:
                print("[DEBUG] Opp play obs missing, output PASS", file=sys.stderr)
            return "PASS"

        response = obs2response(model, obs)
        t = response.split()
        if t[0] == "Hu":
            return "HU"
        if t[0] == "Pass":
            return "PASS"
        if t[0] == "Gang":
            angang = None
            return "GANG"
        if t[0] in ("Peng", "Chi"):
            follow_obs = agent.request2obs("Player %d " % seatWind + response)
            if not follow_obs:
                if DEBUG_MODE:
                    print("[DEBUG] Follow obs missing, output PASS", file=sys.stderr)
                return "PASS"
            follow_response = obs2response(model, follow_obs)
            extra = follow_response.split()[-1]
            agent.request2obs("Player %d Un" % seatWind + response)
            return " ".join([t[0].upper(), *t[1:], extra])
        return "PASS"

    return "PASS"


if __name__ == "__main__":
    # 加载模型
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model = PretrainModel(hidden_dim=256, use_vec=True)

    model_path = "/data/best_model_096.pt"
    if not os.path.exists(model_path):
        model_path = os.path.join(BASE_DIR, "data", "best_model_096.pt")

    load_state_dict_compat(model, model_path)
    model.eval()

    def emit_response(resp):
        if resp is None:
            resp = "PASS"
        print(resp)
        print(">>>BOTZONE_REQUEST_KEEP_RUNNING<<<")
        sys.stdout.flush()

    while True:
        line = _safe_input()
        if line is None:
            break
        line = line.strip()
        if not line:
            continue

        if DEBUG_MODE:
            print("[DEBUG] raw:", line, file=sys.stderr)

        if line.isdigit():
            turn = int(line)
            history = []
            for _ in range(2 * turn - 1):
                hist_line = _safe_input()
                if hist_line is None:
                    continue
                history.append(hist_line.strip())
            for idx, req in enumerate(history):
                if idx % 2 != 0:
                    continue
                if DEBUG_MODE:
                    print("[DEBUG] replay:", req, file=sys.stderr)
                response = handle_request_line(model, req)
                if idx == len(history) - 1:
                    emit_response(response)
            continue

        if DEBUG_MODE:
            print("[DEBUG] request:", line, file=sys.stderr)
            print("[DEBUG] tokens:", line.split(), file=sys.stderr)

        response = handle_request_line(model, line)
        emit_response(response)
