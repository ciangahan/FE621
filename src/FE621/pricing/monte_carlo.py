import numpy as np
import time

from abc import ABC, abstractmethod

from FE621.pricing.black_scholes import BlackScholes


class MonteCarloSimulator(ABC):
    """
    Path generation abstract class
    """
    @abstractmethod
    def simulate(self, **kwargs) -> tuple[np.float64, np.float64, float]:
        """
        @param paths: price (or log price) paths of the stock price
        @returns array of payoffs
        """
        pass


class GBMEuropean(MonteCarloSimulator):
    """
    Geometric Brownian Motion European Option simulator
    """

    def __init__(self, r, s0, k, T, sigma):
        """
        @param r: risk free rate
        @param s0: initial stock price
        @param T: time to maturity
        @param sigma: stock price vol
        """
        self.r = r
        self.s0 = np.log(s0)
        self.k = k
        self.T = T
        self.sigma = sigma


    def simulate(self, m, av=False, call=True) -> tuple[np.float64, np.float64, float]:
        """
        @param m: number of simulations
        @returns: 2d array of log price paths
        """
        start = time.perf_counter()
        sqrt_T = np.sqrt(self.T)
        normals = np.random.normal(0, 1, size=m)
        if av:
            normals = np.concatenate((normals, -normals), axis=0)
        stock_vals = np.exp((normals * self.sigma * sqrt_T + (self.r - self.sigma ** 2 / 2) * self.T) + self.s0)

        if call:
            payoffs = np.exp(-self.r * self.T) * np.maximum(stock_vals - self.k, 0)
        else:
            payoffs = np.exp(-self.r * self.T) * np.maximum(self.k - stock_vals, 0)
        end = time.perf_counter()
        option_val = np.average(payoffs)
        option_stderr = np.std(payoffs) / np.sqrt(len(payoffs))
        
        return (option_val, option_stderr, end - start)


class GBMEuropeanCV(MonteCarloSimulator):
    """
    Geometric Brownian Motion European Option simulator with Control Variate
    """

    def __init__(self, r, s0, k, T, sigma):
        """
        @param r: risk free rate
        @param s0: initial stock price
        @param T: time to maturity
        @param sigma: stock price vol
        """
        self.r = r
        self.s0 = np.log(s0)
        self.k = k
        self.T = T
        self.sigma = sigma


    def simulate(self, m, n, av=False, call=True) -> tuple[np.float64, np.float64, float]:
        """
        @param m: number of simulations
        @returns: 2d array of log price paths
        """
        start = time.perf_counter()
        dt = self.T / n
        sqrt_dt = np.sqrt(dt)
        normals = np.random.normal(0, 1, size=(m, n))
        if av:
            normals = np.concatenate((normals, -normals), axis=0)
        paths = np.exp((normals * self.sigma * sqrt_dt + (self.r - self.sigma ** 2 / 2) * dt).cumsum(axis=1) + self.s0)
        paths = np.pad(paths, ((0, 0), (1, 0)), constant_values=np.exp(self.s0))

        # time to maturity for each path value
        ts = np.zeros_like(paths) + np.linspace(self.T, 0, n + 1)

        if call:
            deltas = BlackScholes.delta_call(paths[:,:-1], self.k, ts[:,:-1], self.r, self.sigma)
        else:
            deltas = BlackScholes.delta_put(paths[:,:-1], self.k, ts[:,:-1], self.r, self.sigma)

        # increments from delta hedges (S_ti+1 - S_ti exp(rdt))
        incs = paths[:,1:] - paths[:,:-1] * np.exp(self.r * dt)
        # adjustment (sum of delta_i * inc_i * exp(r(T - t_i+1)))
        adj = ((deltas * incs) * np.exp(self.r * ts[:,1:])).sum(axis=1)

        if call:
            raw_payoffs = np.maximum(paths[:,-1] - self.k, 0)
        else:
            raw_payoffs = np.maximum(self.k - paths[:,-1], 0)

        # C0 e^(rT) = CT - sum(0, n-1)[delta c_ti * (s_ti+1 - s_ti e^rdt) * e^r(T-t_i+1)]
        payoffs = (raw_payoffs - adj) * np.exp(-self.r * self.T)
        end = time.perf_counter()
        option_val = np.average(payoffs)
        option_stderr = np.std(payoffs) / np.sqrt(len(payoffs))
        
        return (option_val, option_stderr, end - start)


class CEVEuropean(MonteCarloSimulator):
    """
    CEV model simulation for european option
    """
    def __init__(self, r, s0, k, T, sigma, beta, l, mu_j, sigma_j):
        self.r = r
        self.s0 = s0
        self.k = k
        self.T = T
        self.sigma = sigma
        self.beta = beta
        self.l = l
        self.mu_j = mu_j
        self.sigma_j = sigma_j
        self.batch_size = 20000

    def simulate_path_batch(self, batch_size, n, call=True):
        # generate jump locations and sizes
        dt = self.T / n
        sqrt_dt = np.sqrt(dt)

        normals = np.random.standard_normal((batch_size, n))

        jumps_per_path = np.random.poisson(self.l * self.T, size=batch_size)
        dys = np.zeros((batch_size, n))

        total_jumps = jumps_per_path.sum()
        
        if total_jumps > 0:
            # generate all jumps and jump positions at once, then add them to the paths
            jump_sizes = np.random.normal(self.mu_j, self.sigma_j, size=total_jumps)

            jump_times = np.random.uniform(0, self.T, size=total_jumps)
            # convert jump times to index of list
            jump_steps = np.floor(jump_times / dt).astype(int)

            # find path number for each jump
            path_indices = np.repeat(np.arange(batch_size),
                                     jumps_per_path)

            # distribute jump sizes
            np.add.at(dys, (path_indices, jump_steps), jump_sizes)

        rs = np.zeros((batch_size, n+1))
        rs[:, 0] = np.log(self.s0)

        for i in range(1, n + 1):
            s_prev = np.exp(rs[:, i-1])
            
            # discretized steps
            cev_vol = self.sigma * (s_prev ** (self.beta / 2 - 1))
            
            drift = (self.r - self.l * (np.exp(self.mu_j + 0.5 * self.sigma_j**2) - 1) 
                    - 0.5 * cev_vol**2)
            
            diffusion = cev_vol * sqrt_dt * normals[:, i-1]
            
            rs[:, i] = rs[:, i-1] + (drift * dt + diffusion + dys[:, i-1])

        sT = np.exp(rs[:, -1])

        if call:
            payoffs = np.maximum(sT - self.k, 0)
        else:
            payoffs = np.maximum(self.k - sT, 0)

        return payoffs


    # batching simulations for increased efficiency (else memory is overloaded at high sim count)
    def simulate(self, m, n, call=True):
        payoffs = np.array([])

        batches = np.ceil(m / self.batch_size).astype(int)

        for batch in np.arange(batches):
            size = min(m - batch * self.batch_size, self.batch_size)
            payoffs = np.concat((payoffs, self.simulate_path_batch(size, n, call)))
        
        payoffs = payoffs * np.exp(-self.r * self.T)

        option_val = np.average(payoffs)
        option_stderr = np.std(payoffs) / np.sqrt(len(payoffs))

        return option_val, option_stderr


class GBMAsian(MonteCarloSimulator):
    """
    Geometric Brownian Motion Asian Option simulator
    """

    def __init__(self, r, d, s0, k, T, sigma):
        """
        @param r: risk free rate
        @param d: dividend rate
        @param s0: initial stock price
        @param T: time to maturity
        @param sigma: stock price vol
        """
        self.r = r
        self.d = d
        self.s0 = np.log(s0)
        self.k = k
        self.T = T
        self.sigma = sigma
        self.batch_size = 20000
    

    def simulate_path_batch(self, batch_size, n, call=True):
        # generate jump locations and sizes
        dt = self.T / n
        sqrt_dt = np.sqrt(dt)

        normals = np.random.standard_normal((batch_size, n))

        # log stock price: diffusion + drift
        rs = (normals * sqrt_dt * self.sigma + (self.r - self.d - 0.5 * self.sigma ** 2) * dt).cumsum(axis=1) + self.s0

        s = np.exp(rs)
        if call:
            payoffs = np.maximum(s.mean(axis=1) - self.k, 0)
        else:
            payoffs = np.maximum(self.k - s.mean(axis=1), 0)
        
        return payoffs


    def simulate(self, m, n, call=True):
        """
        @param m: number of trials
        @param n: number of monitoring points (from t_1 = T/n to t_n = T evenly spread)
        """
        payoffs = np.array([])

        batches = np.ceil(m / self.batch_size).astype(int)

        for batch in np.arange(batches):
            size = min(m - batch * self.batch_size, self.batch_size)
            payoffs = np.concat((payoffs, self.simulate_path_batch(size, n, call)))
        
        payoffs = payoffs * np.exp(-self.r * self.T)

        option_val = np.average(payoffs)
        option_stderr = np.std(payoffs) / np.sqrt(len(payoffs))

        return option_val, option_stderr


class GBMUOBarrier(MonteCarloSimulator):
    """
    Geometric Brownian Motion Up & Out Barrier Option simulator
    """

    def __init__(self, r, d, s0, k, h, T, sigma):
        """
        @param r: risk free rate
        @param d: dividend rate
        @param s0: initial stock price
        @param T: time to maturity
        @param sigma: stock price vol
        """
        self.r = r
        self.d = d
        self.s0 = np.log(s0)
        self.k = k
        self.h = h
        self.T = T
        self.sigma = sigma
        self.batch_size = 20000
    

    def simulate_path_batch(self, batch_size, n, call=True):
        # generate jump locations and sizes
        dt = self.T / n
        sqrt_dt = np.sqrt(dt)

        normals = np.random.standard_normal((batch_size, n))

        # log stock price: diffusion + drift
        rs = (normals * sqrt_dt * self.sigma + (self.r - self.d - 0.5 * self.sigma ** 2) * dt).cumsum(axis=1) + self.s0

        s = np.exp(rs)

        if call:
            payoffs = np.maximum(s[:,-1] - self.k, 0)
        else:
            payoffs = np.maximum(self.k - s[:,-1], 0)
        
        barrier_indicator = s.max(axis=1) <= self.h
        
        return payoffs * barrier_indicator


    def simulate(self, m, n, call=True):
        """
        @param m: number of trials
        @param n: number of monitoring points (from t_1 = T/n to t_n = T evenly spread)
        """
        payoffs = np.array([])

        batches = np.ceil(m / self.batch_size).astype(int)

        for batch in np.arange(batches):
            size = min(m - batch * self.batch_size, self.batch_size)
            payoffs = np.concat((payoffs, self.simulate_path_batch(size, n, call)))
        
        payoffs = payoffs * np.exp(-self.r * self.T)

        option_val = np.average(payoffs)
        option_stderr = np.std(payoffs) / np.sqrt(len(payoffs))

        return option_val, option_stderr
