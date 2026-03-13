import numpy as np

from FE621.pricing.black_scholes import BlackScholes
from FE621.pricing.tree import TrinomialTree


class ExplicitFD:
    """
    Explicit Finite Difference Method

    Using the first parameterization from the book (last term approximated via node (i + 1, j))
    and solving for d_x based on notes from class that the optimal choice matches trinomial tree
    O(∆x^2 + ∆t) convergence
    """
    def __init__(self, r_f: float, sigma: float, s0: float, T: float, n: int):
        self.r_f = r_f
        self.sigma = sigma
        self.s0 = np.log(s0)
        self.T = T
        self.n = n
        self.dt = T / n

        # step size delta x
        self.d_x = self.sigma * np.sqrt(3 * self.dt)

        # calculate "probabilities" for approximation
        a = (self.sigma / self.d_x) ** 2
        b = (self.r_f - (self.sigma ** 2) / 2) / self.d_x

        self.p_u = (a + b) * self.dt / 2
        self.p_m = 1 - a * self.dt - self.r_f * self.dt
        self.p_d = (a - b) * self.dt / 2
    

    def price_option(self, K: float, call: bool) -> np.float64:
        """
        Price a european call/put option
        """
        price_grid = np.zeros((2 * self.n + 1, self.n + 1))

        # terminal payoff
        if call:
            for j in range(2 * self.n + 1):
                price_grid[j, self.n] = max(np.exp(self.s0 + (self.n - j) * self.d_x) - K, 0)
        else:
            for j in range(2 * self.n + 1):
                price_grid[j, self.n] = max(K - np.exp(self.s0 + (self.n - j) * self.d_x), 0)

        # iterate backwards to calculate discounted expected value at each node
        for i in reversed(range(self.n)):
            for j in range(self.n - i, self.n + i + 1):
                price_grid[j, i] = (
                    self.p_u * price_grid[j - 1, i + 1] + 
                    self.p_m * price_grid[j, i + 1] + 
                    self.p_d * price_grid[j + 1, i + 1]
                )

        # print(price_grid)

        return price_grid[self.n, 0]


class ImplicitFD:
    """
    Implicit Finite Difference Method

    O(∆x^2 + ∆t) convergence
    """
    pass


class CrankNicolsonFD:
    """
    Crank-Nicolson Finite Difference Method
    """
    pass


if __name__ == "__main__":
    S0 = 100
    K = 105
    sigma = 0.25
    r = 0.05
    T = 1
    steps = 1000

    tree = TrinomialTree(r, sigma, S0, T, steps)
    print(tree.price_option(K, call=True, american=False))
    print(BlackScholes.call(S0, K, T, r, sigma))

    efd = ExplicitFD(r, sigma, S0, T, steps)
    print(efd.price_option(K, call=True))