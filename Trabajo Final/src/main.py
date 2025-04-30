import numpy as np
from stable_baselines3 import DQN
from environment.Environment import AvellanedaEnv
from environment.State import s0, sigma, T, dt, k, A, kappa
from app.app import run_agents
from agents.OptimalMMAvellanedaAgent import OptimalMMAvellanedaAgent
from agents.SymmetricAgent import SymmetricAgent
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.io as pio


def save_plotly(fig, filename):
    pio.write_html(fig, file=filename, auto_open=False)


if __name__ == "__main__":
    model = DQN.load("logs/risk_aversion_0_1/best_model.zip")
    number_of_sims = 1000

    n = int(T / dt)
    ws_rl = np.zeros(number_of_sims)
    ws_opt = np.zeros(number_of_sims)
    ws_sym = np.zeros(number_of_sims)
    tr_rl = np.zeros(number_of_sims)
    tr_opt = np.zeros(number_of_sims)
    tr_sym = np.zeros(number_of_sims)
    inv_rl = np.zeros(number_of_sims)
    inv_opt = np.zeros(number_of_sims)
    inv_sym = np.zeros(number_of_sims)
    simulations = [None] * number_of_sims

    envs = [AvellanedaEnv(s0, T, dt, sigma, k, A, kappa),
            AvellanedaEnv(s0, T, dt, sigma, k, A, kappa, seed=0, is_discrete=False),
            AvellanedaEnv(s0, T, dt, sigma, k, A, kappa, seed=0, is_discrete=False)]
    for i in range(number_of_sims):
        if i % 10 == 0:
            print(str(i / 10) + "%")
        ws_rl[i], ws_opt[i], ws_sym[i], tr_rl[i], tr_opt[i], tr_sym[i], inv_rl[i], inv_opt[i], inv_sym[
            i], simulations[i] = run_agents(
            envs, model,
            OptimalMMAvellanedaAgent(
                kappa, sigma, k),
            SymmetricAgent(kappa, sigma,
                           k))
    # Figure 2 (p. 222)
    x = np.arange(200)

    # Gráficos individuales por simulación
    for i in range(3):
        sim = simulations[i]

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                            subplot_titles=["One Path Simulation", "Inventory"])

        # Mid price, bid, ask
        fig.add_trace(go.Scatter(x=x, y=sim.s, name='mid price'), row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=sim.bids, name='bid'), row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=sim.asks, name='ask'), row=1, col=1)

        # Inventory
        fig.add_trace(go.Scatter(x=x, y=sim.inv, name='inventory', line=dict(color='purple')), row=2, col=1)

        fig.update_layout(height=600, width=1000, title_text=f"Simulation {i + 1}")
        save_plotly(fig, f"results/risk_aversion_0_1/one_path_sim_{i + 1}.html")

    # Histograma: Wealth
    # Determine the combined range
    xmin = min(np.min(ws_opt), np.min(ws_rl))
    xmax = max(np.max(ws_opt), np.max(ws_rl))
    nbins = 50
    bin_size = (xmax - xmin) / nbins

    fig1 = go.Figure()
    fig1.add_trace(go.Histogram(x=ws_opt, name='Optimum',
                                xbins=dict(start=xmin, end=xmax, size=bin_size)
                                ))
    fig1.add_trace(go.Histogram(x=ws_rl, name='RL Discrete',
                                xbins=dict(start=xmin, end=xmax, size=bin_size)
                                ))

    fig1.update_layout(barmode='overlay',
                       title="Accumulated wealth histogram",
                       xaxis_title="Wealth",
                       yaxis_title="Frequency")
    fig1.update_traces(opacity=0.75)
    save_plotly(fig1, "results/risk_aversion_0_1/fig1_wealth_histogram.html")

    # Histograma: Total Reward (3 bars)
    # Calcular los valores mínimo y máximo combinados
    xmin = min(np.min(tr_opt), np.min(tr_sym), np.min(tr_rl))
    xmax = max(np.max(tr_opt), np.max(tr_sym), np.max(tr_rl))
    nbins = 50
    bin_size = (xmax - xmin) / nbins

    fig2 = go.Figure()
    fig2.add_trace(go.Histogram(
        x=tr_opt,
        name='Optimum',
        xbins=dict(start=xmin, end=xmax, size=bin_size)
    ))
    fig2.add_trace(go.Histogram(
        x=tr_sym,
        name='Symmetric',
        xbins=dict(start=xmin, end=xmax, size=bin_size)
    ))
    fig2.add_trace(go.Histogram(
        x=tr_rl,
        name='RL Discrete',
        xbins=dict(start=xmin, end=xmax, size=bin_size)
    ))

    fig2.update_layout(
        barmode='overlay',
        title="Total Reward Histogram (3 bars)",
        xaxis_title="Total Reward",
        yaxis_title="Frequency"
    )
    fig2.update_traces(opacity=0.75)

    save_plotly(fig2, "results/risk_aversion_0_1/fig2_reward_histogram_3bars.html")

    # Calcular el rango común
    xmin = min(np.min(tr_opt), np.min(tr_rl))
    xmax = max(np.max(tr_opt), np.max(tr_rl))
    nbins = 50
    bin_size = (xmax - xmin) / nbins

    # Histograma: Total Reward (2 bars)
    fig3 = go.Figure()
    fig3.add_trace(go.Histogram(
        x=tr_opt,
        name='Optimum',
        xbins=dict(start=xmin, end=xmax, size=bin_size)
    ))
    fig3.add_trace(go.Histogram(
        x=tr_rl,
        name='RL Discrete',
        xbins=dict(start=xmin, end=xmax, size=bin_size)
    ))

    fig3.update_layout(
        barmode='overlay',
        title="Total Reward Histogram (2 bars)",
        xaxis_title="Total Reward",
        yaxis_title="Frequency"
    )
    fig3.update_traces(opacity=0.75)

    save_plotly(fig3, "results/risk_aversion_0_1/fig3_reward_histogram_2bars.html")

    print("#########################")
    print("Riqueza acumulada óptimo: \t")
    print("Riqueza acumulada Simétrico: \t")
    print("Riqueza acumulada Q-Learning: \t")
    print("#########################")

    print("Óptimo:")
    print(np.mean(ws_opt))
    print(np.std(ws_opt))
    print(np.mean(ws_opt) / np.std(ws_opt))
    print("Simétrico:")
    print(np.mean(ws_sym))
    print(np.std(ws_sym))
    print(np.mean(ws_sym) / np.std(ws_sym))
    print("RL:")
    print(np.mean(ws_rl))
    print(np.std(ws_rl))
    print(np.mean(ws_rl) / np.std(ws_rl))

    print()

    utility_avellaneda = np.mean(-np.exp(-kappa * ws_opt))
    utility_rl = np.mean(-np.exp(-kappa * ws_rl))

    print("#########################")
    print("valor de la función de utilidad óptimo: \t{}".format(utility_avellaneda))
    print("valor de la función de utilidad simétrico: \t{}".format(np.mean(-np.exp(-kappa * ws_sym))))
    print("valor de la función de utilidad Q-learning: \t\t{}".format(utility_rl))
    print("#########################")

    print("Óptimo:")
    print(np.mean(tr_opt))
    print(np.std(tr_opt))
    print(np.mean(tr_opt) / np.std(tr_opt))

    print("Simétrico:")
    print(np.mean(tr_sym))
    print(np.std(tr_sym))
    print(np.mean(tr_sym) / np.std(tr_sym))

    print("Q-Learning:")
    print(np.mean(tr_rl))
    print(np.std(tr_rl))
    print(np.mean(tr_rl) / np.std(tr_rl))

    print("#########################")
    print("Inventario óptimo: \t")
    print("Inventario Simétrico: \t")
    print("Inventario Q-Learning: \t")
    print("#########################")
    print("Óptimo:")
    print(np.mean(inv_opt))
    print(np.std(inv_opt))
    print(np.mean(inv_opt) / np.std(inv_opt))

    print("Simétrico:")
    print(np.mean(inv_sym))
    print(np.std(inv_sym))
    print(np.mean(inv_sym) / np.std(inv_sym))

    print("RL:")
    print(np.mean(inv_rl))
    print(np.std(inv_rl))
    print(np.mean(inv_rl) / np.std(inv_rl))
# Guardar como HTML en vez de PNG
