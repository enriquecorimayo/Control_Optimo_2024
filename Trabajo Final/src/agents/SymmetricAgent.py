from utils.AvellanedaUtils import optimal_spread_callback


# symmetrical policy agent as per Avellaneda

class SymmetricAgent:
    def __init__(self, kappa, sigma, k):
        self.spread_callback = optimal_spread_callback(kappa, sigma, k)

    def step(self, observation):
        return 0

# symmetric_agent = SymmetricAgent(gamma, sigma, k)
