import numpy as np

from scipy.stats import norm

from FE621.utils import root_bisection, root_newton

class BlackScholes:
    @staticmethod
    def _d1_d2(S, K, t, r, sigma):
        """
        d1 and d2 helper function
        """
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
        d2 = d1 - sigma * np.sqrt(t)
        return d1, d2

    @staticmethod
    def call(S, K, t, r, sigma):
        d1, d2 = BlackScholes._d1_d2(S, K, t, r, sigma)
        return S * norm.cdf(d1) - K * np.exp(-r * t) * norm.cdf(d2)

    @staticmethod
    def put(S, K, t, r, sigma):
        d1, d2 = BlackScholes._d1_d2(S, K, t, r, sigma)
        return K * np.exp(-r * t) * norm.cdf(-d2) - S * norm.cdf(-d1)


    # todo: finite difference approximation for each of these
    @staticmethod
    def delta_call(S, K, t, r, sigma):
        d1, _ = BlackScholes._d1_d2(S, K, t, r, sigma)
        return norm.cdf(d1)

    @staticmethod
    def delta_put(S, K, t, r, sigma):
        d1, _ = BlackScholes._d1_d2(S, K, t, r, sigma)
        return norm.cdf(d1) - 1

    @staticmethod
    def vega(S, K, t, r, sigma):
        d1, _ = BlackScholes._d1_d2(S, K, t, r, sigma)
        return S * norm.pdf(d1) * np.sqrt(t)
    
    @staticmethod
    def gamma(S, K, t, r, sigma):
        d1, _ = BlackScholes._d1_d2(S, K, t, r, sigma)
        return norm.pdf(d1) / (S * sigma * np.sqrt(t))


    @staticmethod
    def iv_call_bisection(S, K, t, r, mkt_price, log_iter=False):
        # create objective function to find root
        def call_objective(sigma):
            return BlackScholes.call(S, K, t, r, sigma) - mkt_price

        # providing conservative bounds to ensure vol is bracketed
        return root_bisection(call_objective, 0.000001, 20, log_iter=log_iter)

    @staticmethod
    def iv_put_bisection(S, K, t, r, mkt_price, log_iter=False):
        # create objective function to find root
        def put_objective(sigma):
            return BlackScholes.put(S, K, t, r, sigma) - mkt_price

        # providing conservative bounds to ensure vol is bracketed
        return root_bisection(put_objective, 0.000001, 20, log_iter=log_iter)


    @staticmethod
    def iv_call_newton(S, K, t, r, mkt_price, log_iter=False):
        def call_objective(sigma):
            return BlackScholes.call(S, K, t, r, sigma) - mkt_price
        
        def call_derivative(sigma):
            return BlackScholes.vega(S, K, t, r, sigma)
        
        return root_newton(call_objective, call_derivative, 1, log_iter=log_iter)


    @staticmethod
    def iv_put_newton(S, K, t, r, mkt_price, log_iter=False):
        def put_objective(sigma):
            return BlackScholes.put(S, K, t, r, sigma) - mkt_price

        def put_derivative(sigma):
            return BlackScholes.vega(S, K, t, r, sigma)

        return root_newton(put_objective, put_derivative, 1, log_iter=log_iter)
        


if __name__ == "__main__":
    print(BlackScholes.iv_call_bisection(100, 100, 1, 0.05, 10.45, log_iter=True))
    print(BlackScholes.iv_put_bisection(100, 100, 1, 0.05, 5.57))
    print(BlackScholes.iv_call_newton(100, 100, 1, 0.05, 10.45))
    print(BlackScholes.iv_put_newton(100, 100, 1, 0.05, 5.57))