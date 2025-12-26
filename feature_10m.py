#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FeatureAgent10M - 扩展版特征提取器
提供60通道observation + 117维vec向量特征
兼容预训练数据处理脚本
"""

from feature import FeatureAgent
from collections import deque
import numpy as np


class FeatureAgent10M(FeatureAgent):
    """
    扩展特征Agent，在原有6通道基础上增加：
    - 更多手牌历史信息
    - 对手打牌历史
    - 剩余牌统计
    - 向量特征（vec）

    observation: 60×4×9 int8
    vec: 117 float16
    action_mask: 235 int8
    """

    OBS_SIZE = 60  # 从6扩展到60通道
    VEC_SIZE = 117  # 新增向量特征

    def __init__(self, seatWind):
        super().__init__(seatWind)
        # 扩展观察空间
        self.obs = np.zeros((self.OBS_SIZE, 36), dtype=np.int8)
        # 设置座次风
        self.obs[self.OFFSET_OBS['SEAT_WIND']][self.OFFSET_TILE['F%d' % (self.seatWind + 1)]] = 1
        self._ting_cache = {}
        self._ting_cache_order = deque()
        self._ting_cache_size = 4096

    def _obs(self):
        """返回观察字典，添加vec向量特征"""
        # 调用父类获取基础观察
        obs_dict = {
            'observation': self._get_extended_observation(),
            'vec': self._get_vector_features(),
            'action_mask': np.array([1 if i in self.valid else 0 for i in range(self.ACT_SIZE)], dtype=np.int8)
        }
        return obs_dict

    def _get_extended_observation(self):
        """
        构造60通道观察

        通道分配：
        0: 座次风 (SEAT_WIND)
        1: 场风 (PREVALENT_WIND)
        2-5: 手牌（最多4张同牌）
        6-9: 已打出的牌（4个玩家各1通道）
        10-13: 碰/杠的明牌
        14-17: 其他玩家手牌数
        18-53: 剩余牌堆统计（每种牌的剩余数量，0-4）
        54: 自己听牌（当前手牌可胡的牌）
        55-59: 保留通道
        """
        obs = np.zeros((60, 4, 9), dtype=np.int8)

        # 通道0-1: 风信息（从self.obs复制）
        for i in range(36):
            r, c = divmod(i, 9)
            obs[0, r, c] = self.obs[self.OFFSET_OBS['SEAT_WIND'], i]
            obs[1, r, c] = self.obs[self.OFFSET_OBS['PREVALENT_WIND'], i]

        # 通道2-5: 手牌
        for i in range(36):
            r, c = divmod(i, 9)
            for j in range(4):
                channel = 2 + j
                if channel < 6:
                    obs[channel, r, c] = self.obs[self.OFFSET_OBS['HAND'] + j, i]

        # 通道6-9: 各玩家打出的牌（从history统计）
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

        # 通道14-17: 其他玩家手牌数（归一化到0-1）
        for player_id in range(4):
            # 简化：用剩余牌墙数估计
            remaining = self.tileWall[player_id] if player_id < len(self.tileWall) else 13
            normalized = min(remaining / 13.0, 1.0)
            obs[14 + player_id, :, :] = int(normalized * 4)  # 0-4

        # 通道18-53: 剩余牌统计（每种牌一个通道，值为剩余数量0-4）
        for tile_idx, tile in enumerate(self.TILE_LIST):
            shown = self.shownTiles.get(tile, 0)
            remaining = max(4 - shown, 0)
            r, c = divmod(tile_idx, 9)
            channel = 18 + tile_idx
            if channel < 54:
                obs[channel, r, c] = remaining

        # 通道54: 自己听牌
        ting_mask = self._get_ting_mask()
        obs[54, :, :] = ting_mask

        return obs

    def _get_vector_features(self):
        """
        构造117维向量特征

        特征分组：
        0-33: 手牌统计（34种牌，每种0-4张）
        34-67: 已打出牌统计
        68-101: 明牌统计
        102-105: 场面信息（场风、座次风、剩余牌墙等）
        106-116: 其他统计特征
        """
        vec = np.zeros(117, dtype=np.float32)

        # 0-33: 手牌统计（归一化到0-1）
        for tile in self.hand:
            tile_idx = self.OFFSET_TILE.get(tile, -1)
            if 0 <= tile_idx < 34:
                vec[tile_idx] = min(vec[tile_idx] + 0.25, 1.0)  # 0/0.25/0.5/0.75/1.0

        # 34-67: 已打出牌统计
        for tile, count in self.shownTiles.items():
            tile_idx = self.OFFSET_TILE.get(tile, -1)
            if 0 <= tile_idx < 34:
                vec[34 + tile_idx] = min(count / 4.0, 1.0)

        # 68-101: 副露牌统计
        for packs_list in self.packs:
            for pack in packs_list:
                if len(pack) >= 2 and pack[1] != 'CONCEALED':
                    tile = pack[1]
                    tile_idx = self.OFFSET_TILE.get(tile, -1)
                    if 0 <= tile_idx < 34:
                        vec[68 + tile_idx] = min(vec[68 + tile_idx] + 0.25, 1.0)

        # 102: 场风（归一化）
        vec[102] = self.prevalentWind / 3.0 if hasattr(self, 'prevalentWind') else 0.0

        # 103: 座次风
        vec[103] = self.seatWind / 3.0

        # 104: 牌墙余量（估计）
        avg_wall = sum(self.tileWall) / 4.0 if self.tileWall else 21.0
        vec[104] = min(avg_wall / 21.0, 1.0)

        # 105: 是否海底
        vec[105] = 1.0 if self.wallLast else 0.0

        # 106-116: 手牌特征
        vec[106] = len(self.hand) / 14.0  # 手牌数
        vec[107] = len(self.packs[0]) / 4.0 if len(self.packs) > 0 else 0.0  # 副露数

        # 108-111: 各花色统计
        wan_count = sum(1 for t in self.hand if t.startswith('W'))
        tiao_count = sum(1 for t in self.hand if t.startswith('T'))
        bing_count = sum(1 for t in self.hand if t.startswith('B'))
        zi_count = sum(1 for t in self.hand if t[0] in ['F', 'J'])
        vec[108] = wan_count / 9.0
        vec[109] = tiao_count / 9.0
        vec[110] = bing_count / 9.0
        vec[111] = zi_count / 7.0

        # 112-116: 保留特征
        vec[112:117] = 0.0

        return vec.astype(np.float16)

    def _get_ting_mask(self):
        """
        返回当前手牌的听牌掩码 (4x9)。仅在手牌数 % 3 == 1 时计算。
        """
        if len(self.hand) % 3 != 1:
            return np.zeros((4, 9), dtype=np.int8)

        hand_counts = [0] * 34
        for tile in self.hand:
            idx = self.OFFSET_TILE.get(tile, -1)
            if 0 <= idx < 34:
                hand_counts[idx] += 1

        shown_counts = [0] * 34
        for tile, cnt in self.shownTiles.items():
            idx = self.OFFSET_TILE.get(tile, -1)
            if 0 <= idx < 34:
                shown_counts[idx] = cnt

        packs_sig = tuple(self.packs[0]) if self.packs else ()
        cache_key = (
            tuple(hand_counts),
            tuple(shown_counts),
            packs_sig,
            self.prevalentWind if hasattr(self, 'prevalentWind') else 0,
            self.seatWind,
            int(self.wallLast),
            int(self.isAboutKong),
        )
        cached = self._ting_cache.get(cache_key)
        if cached is not None:
            return cached.copy()

        mask = np.zeros((4, 9), dtype=np.int8)
        for tile in self.TILE_LIST:
            if self.shownTiles.get(tile, 0) >= 4:
                continue
            if self._check_mahjong(tile, isSelfDrawn=True, isAboutKong=self.isAboutKong):
                idx = self.OFFSET_TILE.get(tile, -1)
                if idx >= 0:
                    r, c = divmod(idx, 9)
                    mask[r, c] = 1

        if len(self._ting_cache_order) >= self._ting_cache_size:
            old_key = self._ting_cache_order.popleft()
            self._ting_cache.pop(old_key, None)
        self._ting_cache_order.append(cache_key)
        self._ting_cache[cache_key] = mask.copy()
        return mask
