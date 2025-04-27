from utils.AvellanedaUtils import optimal_spread_callback, optimal_reservation_price_callback


class AvellanedaAgent:
    def __init__(self, beta, sigma, k):
        self.spread_func = optimal_spread_callback(beta, sigma, k)
        self.r_func = optimal_reservation_price_callback(sigma, beta)

    def act(self, observation):
        spread = self.spread_func(observation[2])
        r_ = self.r_func(observation[2], observation[0], observation[1])

        bid = r_ - spread / 2
        ask = r_ + spread / 2

        ds = observation[0] - r_

        # return spread, ds
        return ds

    def step(self, observation):
        return self.act(observation)

# agent = AvellanedaAgent(gamma, sigma, k)
