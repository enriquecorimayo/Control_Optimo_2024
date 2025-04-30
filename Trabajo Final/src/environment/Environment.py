import gymnasium as gym
from gymnasium import spaces
import numpy as np
from utils.AvellanedaUtils import optimal_spread, orders_intensity_lambda


class AvellanedaEnv(gym.Env):
    def __init__(self, s0, T, dt, sigma, k, A, kappa, seed=0, is_discrete=True):
        super().__init__()

        self.dw_stats = None
        self.ws = None
        self.rewards = None
        self.c_ = None
        self.n = None
        self.t = None
        self.q = None
        self.s = None
        self.w = None
        self.s0 = s0
        self.T = T
        self.dt = dt
        self.sigma = sigma
        self.k = k
        self.A = A
        self.kappa = kappa
        self.is_discrete = is_discrete
        self.tick_size = 0.1
        self.sqrtdt = np.sqrt(dt)

        self.observation_space = spaces.Box(
            low=np.array([0.0, -np.inf, 0.0], dtype=np.float32),
            high=np.array([np.inf, np.inf, T], dtype=np.float32),
            dtype=np.float32
        )
        self.num_ticks = int(1.0 / self.tick_size)
        self.action_space = spaces.Discrete(self.num_ticks ** 2)
        # self.action_space = spaces.Discrete(21)
        self.seed(seed)

    def seed(self, seed=None):
        self.np_random, _ = gym.utils.seeding.np_random(seed)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)

        self.s = self.s0
        self.q = 0.0
        self.t = 0.0
        self.w = 0.0
        self.n = int(self.T / self.dt)
        self.c_ = 0.0
        self.rewards = []
        self.ws = []
        self.dw_stats = []

        obs = np.array((0, self.q, self.T - self.t), dtype=np.float32)
        return obs, {"s": self.s}

    def step(self, action):
        bid = self.s
        ask = self.s
        if self.is_discrete:
            bid_tick = action // self.num_ticks
            ask_tick = action % self.num_ticks
            bid_spread = bid_tick * self.tick_size
            ask_spread = ask_tick * self.tick_size
            bid -= bid_spread
            ask += ask_spread
        else:
            despl = float(action)
            ba_spread = optimal_spread(self.kappa, self.sigma, self.T - self.t, self.k)
            bid += - despl - ba_spread / 2
            ask += - despl + ba_spread / 2

        lambda_b = orders_intensity_lambda(self.A, self.k, self.s - bid)
        lambda_a = orders_intensity_lambda(self.A, self.k, ask - self.s)

        dnb = 1 if self.np_random.uniform() <= lambda_b * self.dt else 0
        dna = 1 if self.np_random.uniform() <= lambda_a * self.dt else 0
        self.q += dnb - dna

        self.c_ += -dnb * bid + dna * ask
        old_s = self.s
        self.s += self.sigma * self.sqrtdt * (1 if self.np_random.uniform() < 0.5 else -1)
        ret = (self.s - old_s) / old_s
        previous_w = self.w
        self.w = self.c_ + self.q * self.s
        dw = self.w - previous_w

        self.dw_stats.append(dw)
        mu = np.mean(self.dw_stats)
        reward = dw - self.kappa / 2 * (dw - mu) ** 2

        self.t += self.dt
        terminated = self.t >= self.T
        truncated = False

        obs = np.array((ret, self.q, self.T - self.t), dtype=np.float32)
        info = {"w": self.w, "s": self.s}

        return obs, reward, terminated, truncated, info
