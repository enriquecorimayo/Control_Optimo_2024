from utils.AvellanedaUtils import optimal_spread_callback


# symmetrical policy agent as per Avellaneda

class SymmetricAgent:
    def __init__(self, beta, sigma, k):
        self.spread_func = optimal_spread_callback(beta, sigma, k)

    def act(self, observation):
        # spread = self.spread_func(observation[2])
        return 0

    def step(self, observation):
        return 0

# symmetric_agent = SymmetricAgent(gamma, sigma, k)
