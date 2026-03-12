import numpy as np

class BinomialTree():
    """
    Additive binomial tree pricing model with identical up/down step sizes
    """
    def __init__(self, r_f: float, sigma: float, s0: float, T: float, n: int):
        self.r_f = r_f
        self.sigma = sigma
        self.s0 = s0
        self.T = T
        self.n = n
        self.dt = T / n

        # step size delta r
        self.d_r = np.sqrt(((self.r_f - (self.sigma ** 2) / 2) * self.dt) ** 2 + (self.sigma ** 2) * self.dt)

        # risk neutral probability of up move
        self.p_u = 0.5 * (self.r_f - (self.sigma ** 2) / 2) * self.dt / self.d_r + 0.5

        self.stock_tree = np.zeros((self.n + 1, self.n + 1))

        for i in range(self.n + 1):
            for j in range(i + 1):
                self.stock_tree[j, i] = self.s0 + self.d_r * (i - 2 * j)


    # pricing
    def price_option(self, K: float, call: bool, american: bool) -> np.float64:
        """
        Price a european or american call/put using the instance's stock price tree
        @param K: strike price
        @param call: whether pricing a call (True) or put (False)
        @param american: whether pricing an american (True) or european (False)
        """
        option_tree = np.zeros((self.n + 1, self.n + 1))

        # terminal payoff
        if call:
            for j in range(self.n + 1):
                option_tree[j, self.n] = max(np.exp(self.stock_tree[j, self.n]) - K, 0)
        else:
            for j in range(self.n + 1):
                option_tree[j, self.n] = max(K - np.exp(self.stock_tree[j, self.n]), 0)

        # iterate backwards to calculate discounted expected value at each node
        for i in reversed(range(self.n)):
            for j in range(i + 1):
                hold_val = np.exp(-self.r_f * self.dt) * (self.p_u * option_tree[j, i + 1] + (1 - self.p_u) * option_tree[j + 1, i + 1])

                if american:
                    exercise_val = max(np.exp(self.stock_tree[j, i]) - K, 0) if call else max(K - np.exp(self.stock_tree[j, i]), 0)
                    option_tree[j, i] = max(hold_val, exercise_val)
                else:
                    option_tree[j, i] = hold_val
        
        return option_tree[0, 0]
    
    # pricing
    def price_chooser_option(self, K: float, american: bool) -> np.float64:
        """
        Price a european or american chooser option using the instance's stock price tree
        @param K: strike price
        @param american: whether pricing an american (True) or european (False)
        """
        option_tree = np.zeros((self.n + 1, self.n + 1))

        # terminal payoff
        for j in range(self.n + 1):
            option_tree[j, self.n] = abs(np.exp(self.stock_tree[j, self.n]) - K)

        # iterate backwards to calculate discounted expected value at each node
        for i in reversed(range(self.n)):
            for j in range(i + 1):
                hold_val = np.exp(-self.r_f * self.dt) * (self.p_u * option_tree[j, i + 1] + (1 - self.p_u) * option_tree[j + 1, i + 1])

                if american:
                    exercise_val = abs(np.exp(self.stock_tree[j, i]) - K)
                    option_tree[j, i] = max(hold_val, exercise_val)
                else:
                    option_tree[j, i] = hold_val
        
        return option_tree[0, 0]


class TrinomialTree():
    pass


if __name__ == "__main__":
    ## BINOMIAL TREE CHECKS
    # # double check within range of log(multiplicative CRR u factor)
    # tree = BinomialTree(0.05, 0.2, 100, 1, 5)
    # print(tree.d_r)
    # print(tree.p_u)
    # h = tree.sigma * np.sqrt(tree.dt)
    # print(h)
    # print((np.exp(tree.r_f * tree.dt) - np.exp(-h)) / (np.exp(h) - np.exp(-h)))
    # print(tree.stock_tree)

    S0 = 100
    K = 105
    sigma = 0.25
    r = 0.05
    T = 1
    steps = 2

    tree = BinomialTree(r, sigma, np.log(S0), T, steps)

    print(tree.price_option(K, call=False, american=False))
    print(tree.price_option(K, call=False, american=True))
    print(tree.price_option(K, call=True, american=False))
    print(tree.price_option(K, call=True, american=True))


