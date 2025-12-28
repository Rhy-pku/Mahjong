import random
import unittest

import numpy as np
import torch

import mcts_self_play as mcts


class FakeModel:
    def __init__(self, logits, values):
        self.logits = torch.tensor(logits, dtype=torch.float32)
        self.values = torch.tensor(values, dtype=torch.float32)
        self.call_count = 0

    def __call__(self, obs_t):
        self.call_count += 1
        batch = obs_t.shape[0]
        logits = self.logits.unsqueeze(0).repeat(batch, 1)
        values = self.values.unsqueeze(0).repeat(batch, 1)
        return logits, values


class FakeEnv:
    def __init__(self, action_size, max_depth=1, action_mask=None):
        self.action_size = action_size
        self.max_depth = max_depth
        self.agent_names = [f"player_{i + 1}" for i in range(4)]
        self.hands = [[] for _ in range(4)]
        self.state = 0
        self.curPlayer = 0
        self.curTile = None
        self.done = False
        self.steps = 0
        self.last_action_dict = None
        if action_mask is None:
            self.action_mask = np.ones(self.action_size, dtype=np.int8)
        else:
            self.action_mask = np.array(action_mask, dtype=np.int8)

    def _obs(self):
        return {
            "player_1": {
                "observation": np.zeros((60, 4, 9), dtype=np.int8),
                "action_mask": self.action_mask.copy(),
            }
        }

    def step(self, action_dict):
        self.last_action_dict = dict(action_dict)
        self.steps += 1
        self.done = self.steps >= self.max_depth
        rewards = {name: 0 for name in self.agent_names}
        obs = {} if self.done else self._obs()
        return obs, rewards, self.done


class FakeMultiEnv(FakeEnv):
    def __init__(self, action_size, max_depth=1, action_mask=None):
        super().__init__(action_size=action_size, max_depth=max_depth, action_mask=action_mask)

    def _obs(self):
        return {
            "player_1": {
                "observation": np.zeros((60, 4, 9), dtype=np.int8),
                "action_mask": self.action_mask.copy(),
            },
            "player_2": {
                "observation": np.zeros((60, 4, 9), dtype=np.int8),
                "action_mask": self.action_mask.copy(),
            },
        }


class TestMCTSHelpers(unittest.TestCase):
    def test_candidate_actions_policy_mass(self):
        prior = np.array([0.6, 0.2, 0.1, 0.1], dtype=np.float32)
        valid = np.array([0, 1, 2, 3], dtype=np.int32)
        out = mcts._candidate_actions(
            prior=prior,
            valid=valid,
            n_visits=10,
            top_k=0,
            min_actions=1,
            policy_mass=0.7,
        )
        self.assertEqual(out.tolist(), [0, 1])

    def test_expand_masks_and_normalizes(self):
        node = mcts.MCTSNode(action_size=3, value_dim=1)
        mask = np.array([1, 0, 1], dtype=np.int8)
        priors = np.array([0.1, 0.2, 0.7], dtype=np.float32)
        node.expand(mask, priors)
        self.assertTrue(node.expanded)
        self.assertEqual(node.valid_actions.tolist(), [0, 2])
        self.assertAlmostEqual(node.prior[0], 0.125, places=6)
        self.assertAlmostEqual(node.prior[1], 0.0, places=6)
        self.assertAlmostEqual(node.prior[2], 0.875, places=6)

    def test_expand_fallbacks_to_uniform(self):
        node = mcts.MCTSNode(action_size=3, value_dim=1)
        mask = np.array([0, 0, 0], dtype=np.int8)
        priors = np.array([0.2, 0.3, 0.5], dtype=np.float32)
        node.expand(mask, priors)
        self.assertTrue(node.expanded)
        self.assertTrue(np.allclose(node.prior, np.array([1 / 3, 1 / 3, 1 / 3])))

    def test_apply_action_mask(self):
        logits = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
        mask = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
        masked = mcts._apply_action_mask(logits, mask)
        self.assertAlmostEqual(masked[0, 0].item(), 0.0, places=6)
        self.assertLess(masked[0, 1].item(), -50.0)

    def test_get_hu_action(self):
        self.assertEqual(mcts._get_hu_action(np.array([0, 1, 0], dtype=np.int8)), 1)
        self.assertIsNone(mcts._get_hu_action(np.array([1], dtype=np.int8)))
        self.assertIsNone(mcts._get_hu_action(np.array([1, 0], dtype=np.int8)))


class TestMCTSNodeSelect(unittest.TestCase):
    def test_select_prefers_high_q_when_c_puct_zero(self):
        node = mcts.MCTSNode(action_size=3, value_dim=1)
        node.prior = np.array([0.2, 0.3, 0.5], dtype=np.float32)
        node.valid_actions = np.array([0, 1, 2], dtype=np.int32)
        node.n_visits = 5
        node.nsa = np.array([1, 1, 1], dtype=np.int32)
        node.wsa[1, 0] = 1.0
        node.wsa[2, 0] = -1.0
        action = node.select(player=0, c_puct=0.0, top_k=0, min_actions=1, policy_mass=1.0)
        self.assertEqual(action, 1)


class TestMCTSAction(unittest.TestCase):
    def test_mcts_picks_argmax_prior(self):
        env = FakeEnv(action_size=3, max_depth=1, action_mask=[1, 0, 1])
        obs = env._obs()
        model = FakeModel(logits=[0.0, 1.0, 2.0], values=[0.0, 0.0, 0.0, 0.0])
        action, pi = mcts.mcts_action(
            env=env,
            obs_dict=obs,
            model=model,
            device=torch.device("cpu"),
            in_channels=72,
            value_dim=4,
            simulations=5,
            c_puct=1.0,
            reward_scale=1.0,
            determinize=False,
            rng=random.Random(0),
            temperature=0.0,
            top_k=1,
            min_actions=1,
            policy_mass=1.0,
            leaf_batch_size=1,
            mcts_player_ids=[0],
        )
        self.assertEqual(action, 2)
        self.assertAlmostEqual(float(np.sum(pi)), 1.0, places=6)

    def test_non_mcts_branch_skips_extra_inference(self):
        env = FakeEnv(action_size=3, max_depth=1, action_mask=[1, 0, 1])
        obs = env._obs()
        model = FakeModel(logits=[0.0, 1.0, 2.0], values=[0.0, 0.0, 0.0, 0.0])
        mcts.mcts_action(
            env=env,
            obs_dict=obs,
            model=model,
            device=torch.device("cpu"),
            in_channels=72,
            value_dim=4,
            simulations=2,
            c_puct=1.0,
            reward_scale=1.0,
            determinize=False,
            rng=random.Random(0),
            temperature=0.0,
            top_k=0,
            min_actions=1,
            policy_mass=1.0,
            leaf_batch_size=1,
            mcts_player_ids=[1],
        )
        self.assertEqual(model.call_count, 1)


class TestEvalLeafBatch(unittest.TestCase):
    def test_eval_leaf_batch_expands_and_backprops(self):
        env = FakeEnv(action_size=3, max_depth=1, action_mask=[1, 0, 1])
        obs = env._obs()["player_1"]
        root = mcts.MCTSNode(action_size=3, value_dim=4)
        node = mcts.MCTSNode(action_size=3, value_dim=4)
        path = [(root, 2)]
        model = FakeModel(logits=[0.0, 0.0, 1.0], values=[0.5, 0.0, 0.0, 0.0])
        leaf_batch = [(node, path, 0, obs, env)]
        mcts._eval_leaf_batch(
            leaf_batch=leaf_batch,
            root=root,
            model=model,
            device=torch.device("cpu"),
            in_channels=72,
            value_dim=4,
        )
        self.assertTrue(node.expanded)
        self.assertEqual(root.n_visits, 1)
        self.assertEqual(root.nsa[2], 1)
        self.assertAlmostEqual(root.wsa[2, 0], 0.5, places=6)


class TestSimulateToLeaf(unittest.TestCase):
    def test_simulate_to_leaf_returns_leaf_when_unexpanded(self):
        env = FakeEnv(action_size=3, max_depth=1)
        obs = env._obs()
        root = mcts.MCTSNode(action_size=3, value_dim=4)
        model = FakeModel(logits=[0.0, 0.0, 0.0], values=[0.0, 0.0, 0.0, 0.0])
        kind, payload = mcts._simulate_to_leaf(
            env=env,
            obs_dict=obs,
            root=root,
            model=model,
            device=torch.device("cpu"),
            in_channels=72,
            value_dim=4,
            c_puct=1.0,
            reward_scale=1.0,
            top_k=0,
            min_actions=1,
            policy_mass=1.0,
            mcts_player_ids=[0],
        )
        self.assertEqual(kind, "leaf")
        node, path, player, leaf_obs, leaf_env = payload
        self.assertIs(node, root)
        self.assertEqual(path, [])
        self.assertEqual(player, 0)
        self.assertEqual(leaf_obs, obs["player_1"])
        self.assertIs(leaf_env, env)

    def test_simulate_to_leaf_non_mcts_uses_prior_argmax(self):
        env = FakeEnv(action_size=3, max_depth=1)
        obs = env._obs()
        root = mcts.MCTSNode(action_size=3, value_dim=4)
        root.expand(obs["player_1"]["action_mask"], np.array([0.1, 0.7, 0.2], dtype=np.float32))
        root.n_visits = 3
        root.nsa = np.array([1, 1, 1], dtype=np.int32)
        root.wsa[2, 0] = 2.0
        model = FakeModel(logits=[0.0, 0.0, 0.0], values=[0.0, 0.0, 0.0, 0.0])
        kind, payload = mcts._simulate_to_leaf(
            env=env,
            obs_dict=obs,
            root=root,
            model=model,
            device=torch.device("cpu"),
            in_channels=72,
            value_dim=4,
            c_puct=0.0,
            reward_scale=1.0,
            top_k=0,
            min_actions=1,
            policy_mass=1.0,
            mcts_player_ids=[1],
        )
        self.assertEqual(kind, "terminal")
        _, path = payload
        self.assertEqual(path[-1][1], 1)

    def test_simulate_to_leaf_mcts_uses_select(self):
        env = FakeEnv(action_size=3, max_depth=1)
        obs = env._obs()
        root = mcts.MCTSNode(action_size=3, value_dim=4)
        root.expand(obs["player_1"]["action_mask"], np.array([0.1, 0.7, 0.2], dtype=np.float32))
        root.n_visits = 3
        root.nsa = np.array([1, 1, 1], dtype=np.int32)
        root.wsa[2, 0] = 2.0
        model = FakeModel(logits=[0.0, 0.0, 0.0], values=[0.0, 0.0, 0.0, 0.0])
        kind, payload = mcts._simulate_to_leaf(
            env=env,
            obs_dict=obs,
            root=root,
            model=model,
            device=torch.device("cpu"),
            in_channels=72,
            value_dim=4,
            c_puct=0.0,
            reward_scale=1.0,
            top_k=0,
            min_actions=1,
            policy_mass=1.0,
            mcts_player_ids=[0],
        )
        self.assertEqual(kind, "terminal")
        _, path = payload
        self.assertEqual(path[-1][1], 2)

    def test_simulate_to_leaf_multi_agent_uses_policy_actions(self):
        env = FakeMultiEnv(action_size=3, max_depth=1, action_mask=[1, 1, 1])
        obs = env._obs()
        root = mcts.MCTSNode(action_size=3, value_dim=4)
        model = FakeModel(logits=[0.0, 2.0, 1.0], values=[0.0, 0.0, 0.0, 0.0])
        kind, payload = mcts._simulate_to_leaf(
            env=env,
            obs_dict=obs,
            root=root,
            model=model,
            device=torch.device("cpu"),
            in_channels=72,
            value_dim=4,
            c_puct=1.0,
            reward_scale=1.0,
            top_k=0,
            min_actions=1,
            policy_mass=1.0,
            mcts_player_ids=[0],
        )
        self.assertEqual(kind, "terminal")
        self.assertIsNotNone(env.last_action_dict)
        self.assertEqual(env.last_action_dict["player_1"], 1)
        self.assertEqual(env.last_action_dict["player_2"], 1)


if __name__ == "__main__":
    unittest.main()
