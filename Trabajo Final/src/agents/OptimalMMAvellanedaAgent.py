from utils.AvellanedaUtils import optimal_spread_callback, optimal_reservation_price_callback


class OptimalMMAvellanedaAgent:
    def __init__(self, kappa, sigma, k):
        self.optimal_spread_callback = optimal_spread_callback(kappa, sigma, k)
        self.optimal_reservation_price_callback = optimal_reservation_price_callback(sigma, kappa)

    def step(self, obs):
        reservation = self.optimal_reservation_price_callback(obs[2], obs[0], obs[1])
        displacement = obs[0] - reservation

        return displacement
