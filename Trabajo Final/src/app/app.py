import numpy as np
from utils.AvellanedaUtils import optimal_spread

actions_num = 21  # MS: So the range of possibilities goes from 0.3% to 3% from TOB
max_abs_dif = 4
max_abs_spread = 20


def run_env_agent_comp(envs, agent_rl, agent_opt, agent_sym):
    env = envs[0]

    obs, _ = env.reset()
    bids_rl = np.zeros(env.n)
    asks_rl = np.zeros(env.n)
    ss_rl = np.zeros(env.n)
    ws_rl = np.zeros(env.n)
    qs_rl = np.zeros(env.n)
    final = False
    i = 0

    total_reward_rl = 0.0
    while not final:
        action_rl = agent_rl.predict(obs, deterministic=True)
        ss_rl[i] = obs[0]
        qs_rl[i] = obs[1]

        despl = (action_rl[0] - (actions_num - 1) / 2) * max_abs_dif / (actions_num - 1)
        ba_spread = optimal_spread(env.beta, env.sigma, env.T - env.t, env.k)

        bids_rl[i] = ss_rl[i] - despl - ba_spread / 2
        asks_rl[i] = ss_rl[i] - despl + ba_spread / 2

        obs, reward, final, _, w_rl = env.step(action_rl[0])
        i += 1
        total_reward_rl += reward

    env = envs[1]

    obs, _ = env.reset()
    bids_opt = np.zeros(env.n)
    asks_opt = np.zeros(env.n)
    ds_opt = np.zeros(env.n)
    spread_opt = np.zeros(env.n)
    ss_opt = np.zeros(env.n)
    ws_opt = np.zeros(env.n)
    qs_opt = np.zeros(env.n)
    final = False
    i = 0

    total_reward_opt = 0.0
    while not final:
        action_opt = agent_opt.step(obs)

        ds_opt[i] = action_opt
        spread_opt[i] = optimal_spread(env.beta, env.sigma, env.T - env.t, env.k)

        ss_opt[i] = obs[0]
        qs_opt[i] = obs[1]

        bids_opt[i] = ss_opt[i] - ds_opt[i] - spread_opt[i] / 2
        asks_opt[i] = ss_opt[i] - ds_opt[i] + spread_opt[i] / 2

        obs, reward, final, _, w_opt = env.step(action_opt)
        total_reward_opt += reward
        i += 1

    env = envs[2]

    obs, _ = env.reset()
    bids_sym = np.zeros(env.n)
    asks_sym = np.zeros(env.n)
    ds_sym = np.zeros(env.n)
    spread_sym = np.zeros(env.n)
    ss_sym = np.zeros(env.n)
    ws_sym = np.zeros(env.n)
    qs_sym = np.zeros(env.n)
    final = False
    i = 0

    total_reward_sym = 0.0
    while not final:
        action_sym = agent_sym.step(obs)

        ds_sym[i] = action_sym
        spread_sym[i] = optimal_spread(env.beta, env.sigma, env.T - env.t, env.k)

        ss_sym[i] = obs[0]
        qs_sym[i] = obs[1]

        bids_sym[i] = ss_sym[i] - ds_sym[i] - spread_sym[i] / 2
        asks_sym[i] = ss_sym[i] - ds_sym[i] + spread_sym[i] / 2

        obs, reward, final, _, w_sym = env.step(action_sym)
        i += 1
        total_reward_sym += reward

    return w_rl['w'], w_opt['w'], w_sym['w'], total_reward_rl, total_reward_opt, total_reward_sym
