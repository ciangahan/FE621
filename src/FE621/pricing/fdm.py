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
    def __init__(self, r_f: float, div: float, sigma: float, s0: float, T: float, n: int):
        self.r_f = r_f
        self.div = div
        self.sigma = sigma
        self.s0 = np.log(s0)
        self.T = T
        self.n = n
        self.dt = T / n

        # step size delta x
        self.d_x = self.sigma * np.sqrt(3 * self.dt)

        # calculate "probabilities" for approximation
        a = (self.sigma / self.d_x) ** 2
        b = (self.r_f - self.div - (self.sigma ** 2) / 2) / self.d_x

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

        return price_grid[self.n, 0]


class ImplicitFD:
    """
    Implicit Finite Difference Method

    O(∆x^2 + ∆t) convergence
    """
    @staticmethod
    def price_option(S0, K, r, div, sigma, T, N, Nj, dx, call:bool, american:bool):
        """
        Price option via Implicit Finite Difference Method
        @param S0: initial stock price
        @param K: strike
        @param r: risk free rate (log)
        @param div: dividend rate (log)
        @param sigma: stock volatility
        @param T: time to maturity
        @param N: Number of time steps
        @param Nj: Number of log stock price steps
        @param dx: Log stock price step size
        """
        dt = T/N
        mu = r - div - 0.5 * (sigma ** 2)
        edx = np.exp(dx)

        pu = -0.5 * dt * ((sigma / dx) ** 2 + mu / dx)
        pm = 1 + dt * (sigma / dx) ** 2 + r * dt
        pd = -0.5 * dt * ((sigma / dx) ** 2 - mu / dx)

        stock_prices = np.zeros(2 * Nj + 1)
        stock_prices[-1] = S0 * np.exp(-Nj * dx)
        for j in reversed(range(2 * Nj)):
            stock_prices[j] = stock_prices[j+1] * edx

        price_grid = np.zeros((2 * Nj + 1, N + 1))

        # terminal payoff
        if call:
            for j in range(2 * Nj + 1):
                price_grid[j, N] = max(stock_prices[j] - K, 0)
            lambda_U = stock_prices[0] - stock_prices[1]
            lambda_L = 0
        else:
            for j in range(2 * Nj + 1):
                price_grid[j, N] = max(K - stock_prices[j], 0)
            lambda_U = 0
            lambda_L = stock_prices[-1] - stock_prices[-2]
        
        for i in reversed(range(N)):
            ImplicitFD.solve_implicit_tridiagonal_system(price_grid, i, pu, pm, pd, lambda_L, lambda_U, Nj)

            if american:
                if call:
                    for j in range(2 * Nj + 1):
                        price_grid[j, i] = max(price_grid[j, i], max(stock_prices[j] - K, 0))
                else:
                    for j in range(2 * Nj + 1):
                        price_grid[j, i] = max(price_grid[j, i], max(K - stock_prices[j], 0))

        return price_grid[Nj, 0]


    @staticmethod
    def solve_implicit_tridiagonal_system(price_grid, i, pu, pm, pd, lambda_L, lambda_U, Nj):
        # pm primes, p primes when subsitituting up 
        pmp = np.zeros(2 * Nj + 1)
        pp = np.zeros(2 * Nj + 1)

        pmp[2 * Nj - 1] = pm + pd
        pp[2 * Nj - 1] = price_grid[2 * Nj - 1, i + 1] + pd * lambda_L

        for j in np.arange(2 * Nj - 2, 0, -1):
            pmp[j] = pm - pu * pd / pmp[j + 1]
            pp[j] = price_grid[j, i + 1] - pp[j + 1] * pd/pmp[j + 1]

        price_grid[0, i] = (pp[1] + pmp[1] * lambda_U) / (pu + pmp[1])
        price_grid[1, i] = price_grid[0, i] - lambda_U
        
        for j in range(2, 2 * Nj):
            price_grid[j, i] = (pp[j] - pu * price_grid[j-1, i]) / pmp[j]
        
        price_grid[2 * Nj, i] = price_grid[2 * Nj - 1, i] - lambda_L

        return


class CrankNicolsonFD:
    """
    Crank-Nicolson Finite Difference Method
    """
    @staticmethod
    def price_option(S0, K, r, div, sigma, T, N, Nj, dx, call:bool, american:bool):
        """
        Price option via Crank-Nicolson Finite Difference Method
        @param S0: initial stock price
        @param K: strike
        @param r: risk free rate (log)
        @param div: dividend rate (log)
        @param sigma: stock volatility
        @param T: time to maturity
        @param N: Number of time steps
        @param Nj: Number of log stock price steps
        @param dx: Log stock price step size
        """
        dt = T/N
        mu = r - div - 0.5 * (sigma ** 2)
        edx = np.exp(dx)

        pu = -0.25 * dt * ((sigma / dx) ** 2 + mu / dx)
        pm = 1 + 0.5 * dt * (sigma / dx) ** 2 + 0.5 * r * dt
        pd = -0.25 * dt * ((sigma / dx) ** 2 - mu / dx)

        stock_prices = np.zeros(2 * Nj + 1)
        stock_prices[-1] = S0 * np.exp(-Nj * dx)
        for j in reversed(range(2 * Nj)):
            stock_prices[j] = stock_prices[j+1] * edx

        price_grid = np.zeros((2 * Nj + 1, N + 1))

        # terminal payoff
        if call:
            for j in range(2 * Nj + 1):
                price_grid[j, N] = max(stock_prices[j] - K, 0)
            lambda_U = stock_prices[0] - stock_prices[1]
            lambda_L = 0
        else:
            for j in range(2 * Nj + 1):
                price_grid[j, N] = max(K - stock_prices[j], 0)
            lambda_U = 0
            lambda_L = stock_prices[-1] - stock_prices[-2]
        
        for i in reversed(range(N)):
            CrankNicolsonFD.solve_cn_tridiagonal_system(price_grid, i, pu, pm, pd, lambda_L, lambda_U, Nj)

            if american:
                if call:
                    for j in range(2 * Nj + 1):
                        price_grid[j, i] = max(price_grid[j, i], max(stock_prices[j] - K, 0))
                else:
                    for j in range(2 * Nj + 1):
                        price_grid[j, i] = max(price_grid[j, i], max(K - stock_prices[j], 0))

        return price_grid[Nj, 0]


    @staticmethod
    def solve_cn_tridiagonal_system(price_grid, i, pu, pm, pd, lambda_L, lambda_U, Nj):
        # pm primes, p primes when subsitituting up 
        pmp = np.zeros(2 * Nj + 1)
        pp = np.zeros(2 * Nj + 1)

        pmp[2 * Nj - 1] = pm + pd
        pp[2 * Nj - 1] = -pu * price_grid[2 * Nj - 2, i + 1] - (pm - 2) * price_grid[2 * Nj - 1, i + 1] - pd * price_grid[2 * Nj, i + 1] + pd * lambda_L

        for j in np.arange(2 * Nj - 2, 0, -1):
            pmp[j] = pm - pu * pd / pmp[j + 1]
            pp[j] = -pu * price_grid[j - 1, i + 1] - (pm - 2) * price_grid[j, i+1] - pd * price_grid[j + 1, i + 1] - pp[j + 1] * pd/pmp[j + 1]

        price_grid[0, i] = (pp[1] + pmp[1] * lambda_U) / (pu + pmp[1])
        price_grid[1, i] = price_grid[0, i] - lambda_U
        
        for j in range(2, 2 * Nj):
            price_grid[j, i] = (pp[j] - pu * price_grid[j-1, i]) / pmp[j]
        
        price_grid[2 * Nj, i] = price_grid[2 * Nj - 1, i] - lambda_L

        return


if __name__ == "__main__":
    S0 = 100
    K = 100
    sigma = 0.2
    r = 0.06
    T = 1
    div = 0.0
    steps = 1000

    tree = TrinomialTree(r, sigma, S0, T, steps)
    print(tree.price_option(K, call=False, american=False))
    print(BlackScholes.put(S0, K, T, r, sigma))

    efd = ExplicitFD(r, div, sigma, S0, T, steps)
    print(efd.price_option(K, call=False))


    print(ImplicitFD.price_option(S0, K, r, div, sigma, T, N=1000, Nj=1000, dx=0.002, call=False, american=False))
    print(CrankNicolsonFD.price_option(S0, K, r, div, sigma, T, N=1000, Nj=1000, dx=0.002, call=False, american=False))