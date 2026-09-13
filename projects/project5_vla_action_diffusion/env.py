"""
2D Reaching with Distractors env.

State:
- agent position (2D)
- target position (2D)
- distractor positions (n_dist × 2D)

Observation:
- 64×64 RGB image (rendered scene)
- proprioception: agent position (2D)

Action: 2D delta position (continuous, [-0.1, 0.1]^2 after clipping)

Success: reach within 0.05 of target
Collision: agent center within 0.05 of any distractor

Note: This is a toy env — no physics, no friction, just point-mass motion.
"""
import numpy as np


class Reach2DEnv:
    """2D point-mass reaching with distractors."""

    BOUND = 1.0
    AGENT_RADIUS = 0.04
    TARGET_RADIUS = 0.06
    DISTRACTOR_RADIUS = 0.07
    SUCCESS_THRESHOLD = 0.06
    COLLISION_THRESHOLD = 0.10  # 中心距离阈值
    MAX_STEPS = 100
    ACTION_SCALE = 0.05  # action 在 [-1, 1] 之间，env 内部乘 ACTION_SCALE

    def __init__(self, n_distractors=2, image_size=64, seed=None):
        self.n_distractors = n_distractors
        self.image_size = image_size
        self.rng = np.random.RandomState(seed)
        self.reset()

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        # Random positions, ensure agent not too close to target/distractor
        while True:
            self.agent_pos = self.rng.uniform(-0.8, 0.8, 2)
            self.target_pos = self.rng.uniform(-0.8, 0.8, 2)
            dist_to_target = np.linalg.norm(self.agent_pos - self.target_pos)
            if dist_to_target > 0.4:
                break
        # Distractors: place them roughly between agent and target to make task non-trivial
        self.distractors = []
        for _ in range(self.n_distractors):
            for _ in range(50):  # max attempts
                pos = self.rng.uniform(-0.8, 0.8, 2)
                if (np.linalg.norm(pos - self.agent_pos) > 0.2
                    and np.linalg.norm(pos - self.target_pos) > 0.2):
                    self.distractors.append(pos)
                    break
        self.distractors = np.array(self.distractors) if len(self.distractors) else np.zeros((0, 2))
        self.step_count = 0
        return self.observation()

    def step(self, action):
        """action: 2D in [-1, 1], will be scaled to [-ACTION_SCALE, ACTION_SCALE]."""
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        new_pos = self.agent_pos + action * self.ACTION_SCALE
        new_pos = np.clip(new_pos, -self.BOUND, self.BOUND)
        self.agent_pos = new_pos
        self.step_count += 1

        # Check success / collision
        dist_to_target = np.linalg.norm(self.agent_pos - self.target_pos)
        success = dist_to_target < self.SUCCESS_THRESHOLD
        collision = False
        if len(self.distractors) > 0:
            dists = np.linalg.norm(self.distractors - self.agent_pos, axis=1)
            collision = (dists < self.COLLISION_THRESHOLD).any()

        done = bool(success or collision or self.step_count >= self.MAX_STEPS)
        reward = float(success) - float(collision)
        info = {
            "success": bool(success),
            "collision": bool(collision),
            "step_count": self.step_count,
            "dist_to_target": float(dist_to_target),
        }
        return self.observation(), reward, done, info

    # ------------------------------------------------------------------
    # Observation rendering
    # ------------------------------------------------------------------
    def observation(self):
        """Return dict with image, proprio and goal.

        - image: (64, 64, 3) uint8，场景渲染图。**干扰物只出现在这里**
        - state: (2,) agent 位置，本体感知（proprioception）
        - goal:  (2,) target 位置

        为什么把 goal 单独给出来、却不给 distractor 位置：

        阶段 2 的 state-only baseline 用 (state, goal) 作条件——它知道目标在哪，
        所以能学会朝目标走，但看不见干扰物，必然撞上去。阶段 3 接上视觉后，
        干扰物的信息只能从 image 里来。两个阶段的成功率差值，就是"视觉带来了什么"
        这个问题的答案。如果把 distractor 位置也塞进 state，这个对照实验就没有意义了。
        """
        return {
            "image": self.render(),
            "state": self.agent_pos.copy().astype(np.float32),
            "goal": self.target_pos.copy().astype(np.float32),
        }

    def _world_to_pixel(self, pos):
        # pos in [-1, 1], pixel in [0, image_size]
        x = (pos[0] + 1) / 2 * self.image_size
        y = (1 - (pos[1] + 1) / 2) * self.image_size  # flip y
        return int(x), int(y)

    def _draw_circle(self, img, cx, cy, radius_px, color):
        for dy in range(-radius_px, radius_px + 1):
            for dx in range(-radius_px, radius_px + 1):
                if dx * dx + dy * dy <= radius_px * radius_px:
                    x, y = cx + dx, cy + dy
                    if 0 <= x < self.image_size and 0 <= y < self.image_size:
                        img[y, x] = color

    def _draw_square(self, img, cx, cy, half_size, color):
        x0 = max(0, cx - half_size)
        x1 = min(self.image_size, cx + half_size + 1)
        y0 = max(0, cy - half_size)
        y1 = min(self.image_size, cy + half_size + 1)
        img[y0:y1, x0:x1] = color

    def render(self):
        """Render 64×64 RGB image showing scene."""
        img = np.ones((self.image_size, self.image_size, 3), dtype=np.uint8) * 240
        # Distractors: grey squares
        for d in self.distractors:
            cx, cy = self._world_to_pixel(d)
            half = max(3, int(self.DISTRACTOR_RADIUS * self.image_size / 2))
            self._draw_square(img, cx, cy, half, np.array([120, 120, 120], dtype=np.uint8))
        # Target: red star (drawn as red square for simplicity)
        cx, cy = self._world_to_pixel(self.target_pos)
        half = max(3, int(self.TARGET_RADIUS * self.image_size / 2))
        self._draw_square(img, cx, cy, half, np.array([230, 30, 30], dtype=np.uint8))
        # Agent: blue circle
        cx, cy = self._world_to_pixel(self.agent_pos)
        radius_px = max(2, int(self.AGENT_RADIUS * self.image_size / 2))
        self._draw_circle(img, cx, cy, radius_px, np.array([30, 80, 230], dtype=np.uint8))
        return img


# ---------------------------------------------------------------------------
# Expert policy (用 A* / spline / heuristic 生成 demos)
# ---------------------------------------------------------------------------
def expert_policy(env, smoothness=0.5):
    """
    Heuristic expert: walk toward target, avoid distractors via simple potential field.
    Returns: list of (obs, action) pairs.
    """
    obs_list, action_list = [], []
    for _ in range(env.MAX_STEPS):
        attractor = env.target_pos - env.agent_pos  # 朝目标
        repulsor = np.zeros(2)
        for d in env.distractors:
            diff = env.agent_pos - d
            dist = np.linalg.norm(diff) + 1e-6
            if dist < 0.3:
                repulsor += diff / dist**3 * 0.02  # 排斥力
        v = attractor + repulsor
        v = v / (np.linalg.norm(v) + 1e-6)
        # Add smoothness (use previous direction info implicitly via small noise)
        action = v * 0.7 + env.rng.randn(2) * 0.05
        action = np.clip(action, -1, 1)

        obs_list.append(env.observation())
        action_list.append(action.astype(np.float32))
        _, _, done, info = env.step(action)
        if done:
            break
    return obs_list, action_list, info


if __name__ == "__main__":
    env = Reach2DEnv(n_distractors=2, seed=42)
    obs = env.reset()
    print(f"Image: {obs['image'].shape}, dtype={obs['image'].dtype}")
    print(f"State: {obs['state']}")
    print(f"Target: {env.target_pos}, Distractors: {env.distractors}")
    # Roll out expert
    obs_list, act_list, info = expert_policy(env)
    print(f"Rollout: {len(obs_list)} steps, success={info['success']}")
