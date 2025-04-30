import numpy as np

from utils.AvellanedaUtils import optimal_spread


class SimulationResults:
    def __init__(self, bids, asks, inv, s):
        self.bids = bids
        self.asks = asks
        self.inv = inv
        self.s = s


def run_agents(envs, rl_agent, opt_avellaneda_agent, sym_agent):
    env = envs[0]

    obs, w_rl = env.reset()
    bids_rl = np.zeros(env.n)
    asks_rl = np.zeros(env.n)
    ss_rl = np.zeros(env.n)
    qs_rl = np.zeros(env.n)
    final = False
    i = 0

    total_reward_rl = 0.0
    while not final:
        raw_action = rl_agent.predict(obs, deterministic=True)
        ss_rl[i] = w_rl["s"]
        qs_rl[i] = obs[1]

        action = raw_action[0]
        bid_tick = action // env.num_ticks
        ask_tick = action % env.num_ticks
        bid_spread = bid_tick * env.tick_size
        ask_spread = ask_tick * env.tick_size

        bids_rl[i] = ss_rl[i] - bid_spread
        asks_rl[i] = ss_rl[i] + ask_spread

        obs, reward, final, _, w_rl = env.step(action)
        inv_rl = obs[1]
        i += 1
        total_reward_rl += reward
    sim_res = SimulationResults(bids_rl, asks_rl, qs_rl, ss_rl)
    env = envs[1]

    obs, w_opt = env.reset()
    bids_opt = np.zeros(env.n)
    asks_opt = np.zeros(env.n)
    ds_opt = np.zeros(env.n)
    spread_opt = np.zeros(env.n)
    ss_opt = np.zeros(env.n)
    qs_opt = np.zeros(env.n)
    final = False
    i = 0

    total_reward_opt = 0.0
    while not final:
        action_opt = opt_avellaneda_agent.step(obs)

        ds_opt[i] = action_opt
        spread_opt[i] = optimal_spread(env.kappa, env.sigma, env.T - env.t, env.k)

        ss_opt[i] = obs[0]
        qs_opt[i] = obs[1]

        bids_opt[i] = ss_opt[i] - ds_opt[i] - spread_opt[i] / 2
        asks_opt[i] = ss_opt[i] - ds_opt[i] + spread_opt[i] / 2

        obs, reward, final, _, w_opt = env.step(action_opt)
        inv_opt = obs[1]
        total_reward_opt += reward
        i += 1

    env = envs[2]

    obs, _ = env.reset()
    bids_sym = np.zeros(env.n)
    asks_sym = np.zeros(env.n)
    ds_sym = np.zeros(env.n)
    spread_sym = np.zeros(env.n)
    ss_sym = np.zeros(env.n)
    w_sym = np.zeros(env.n)
    qs_sym = np.zeros(env.n)
    final = False
    i = 0

    total_reward_sym = 0.0
    while not final:
        action_sym = sym_agent.step(obs)

        ds_sym[i] = action_sym
        spread_sym[i] = optimal_spread(env.kappa, env.sigma, env.T - env.t, env.k)

        ss_sym[i] = obs[0]
        qs_sym[i] = obs[1]

        bids_sym[i] = ss_sym[i] - ds_sym[i] - spread_sym[i] / 2
        asks_sym[i] = ss_sym[i] - ds_sym[i] + spread_sym[i] / 2

        obs, reward, final, _, w_sym = env.step(action_sym)
        inv_sym = obs[1]
        i += 1
        total_reward_sym += reward

    return w_rl['w'], w_opt['w'], w_sym[
        'w'], total_reward_rl, total_reward_opt, total_reward_sym, inv_rl, inv_opt, inv_sym, sim_res
