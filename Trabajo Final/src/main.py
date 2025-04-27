import numpy as np
from matplotlib import pyplot as plt
from stable_baselines3 import DQN
from environment.Environment import AvellanedaEnv
from environment.State import s0, sigma, T, dt, beta, k, A, kappa
from app.app import run_env_agent_comp
from agents.AvellanedaAgent import AvellanedaAgent
from agents.SymmetricAgent import SymmetricAgent

from app import app

if __name__ == "__main__":
    model = DQN.load("logs/best_model.zip")
    number_of_sims = 1000

    n = int(T / dt)
    ws_rl = np.zeros(number_of_sims)
    ws_opt = np.zeros(number_of_sims)
    ws_sym = np.zeros(number_of_sims)
    tr_rl = np.zeros(number_of_sims)
    tr_opt = np.zeros(number_of_sims)
    tr_sym = np.zeros(number_of_sims)

    envs = [AvellanedaEnv(s0, T, dt, sigma, beta, k, A, kappa),
            AvellanedaEnv(s0, T, dt, sigma, beta, k, A, kappa, seed=0, is_discrete=False),
            AvellanedaEnv(s0, T, dt, sigma, beta, k, A, kappa, seed=0, is_discrete=False)]
    for i in range(number_of_sims):
        if i % 10 == 0:
            print(str(i / 10) + "%")
        ws_rl[i], ws_opt[i], ws_sym[i], tr_rl[i], tr_opt[i], tr_sym[i] = run_env_agent_comp(envs, model,
                                                                                            AvellanedaAgent(beta, sigma,
                                                                                                            k),
                                                                                            SymmetricAgent(beta, sigma,
                                                                                                           k))
    # Figure 2 (p. 222)
    fig = plt.figure(figsize=(20, 8), dpi=100, facecolor='w', edgecolor='k')
    plt.hist([ws_opt, ws_rl], bins=30, edgecolor='black', label=['Optimum', 'RL Discrete'])
    plt.grid()
    plt.legend()
    plt.xlabel("Wealth")
    plt.ylabel("Frequency")
    plt.title("Accumulated wealth histogram")
    plt.savefig("fig1_wealth_histogram.png")  # Save figure 1

    fig2 = plt.figure(figsize=(20, 8), dpi=100, facecolor='w', edgecolor='k')
    plt.hist([tr_opt, tr_sym, tr_rl], bins=30, edgecolor='black', label=['Optimum', 'Symmetric', 'RL Discrete'])
    plt.grid()
    plt.legend()
    plt.xlabel("Total Reward")
    plt.ylabel("Frequency")
    plt.title("Accumulated wealth histogram")
    plt.savefig("fig2_reward_histogram_3bars.png")  # Save figure 2

    fig3 = plt.figure(figsize=(20, 8), dpi=100, facecolor='w', edgecolor='k')
    plt.hist([tr_opt, tr_rl], bins=30, edgecolor='black', label=['Optimum', 'RL Discrete'])
    plt.grid()
    plt.legend()
    plt.xlabel("Total Reward")
    plt.ylabel("Frequency")
    plt.title("Accumulated wealth histogram")
    plt.savefig("fig3_reward_histogram_2bars.png")  # Save figure 3

    plt.show()  # Show all figures

    print("Optimo:")
    print(np.mean(ws_opt))
    print(np.std(ws_opt))
    print(np.mean(ws_opt) / np.std(ws_opt))
    print("Simetrico:")
    print(np.mean(ws_sym))
    print(np.std(ws_sym))
    print(np.mean(ws_sym) / np.std(ws_sym))
    print("RL:")
    print(np.mean(ws_rl))
    print(np.std(ws_rl))
    print(np.mean(ws_rl) / np.std(ws_rl))

    print()

    utility_avellaneda = np.mean(-np.exp(-beta * ws_opt))
    utility_rl = np.mean(-np.exp(-beta * ws_rl))

    print("Optimum utility function value: \t{}".format(utility_avellaneda))
    print("Symmetric utility function value: \t{}".format(np.mean(-np.exp(-beta * ws_sym))))
    print("RL utility function value: \t\t{}".format(utility_rl))

    print("Optimo:")
    print(np.mean(tr_opt))
    print(np.std(tr_opt))
    print(np.mean(tr_opt) / np.std(tr_opt))

    print("Simetrico:")
    print(np.mean(tr_sym))
    print(np.std(tr_sym))
    print(np.mean(tr_sym) / np.std(tr_sym))

    print("RL:")
    print(np.mean(tr_rl))
    print(np.std(tr_rl))
    print(np.mean(tr_rl) / np.std(tr_rl))