import numpy as np

from FE621.pricing.black_scholes import BlackScholes

class BinomialTree():
    """
    Additive binomial tree pricing model with identical up/down step sizes
    """
    def __init__(self, r_f: float, sigma: float, s0: float, T: float, n: int):
        self.r_f = r_f
        self.sigma = sigma
        self.s0 = np.log(s0)
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
    """
    Additive trinomial tree pricing model with identical up/down step sizes
    """
    def __init__(self, r_f: float, sigma: float, s0: float, T: float, n: int):
        self.r_f = r_f
        self.sigma = sigma
        self.s0 = np.log(s0)
        self.T = T
        self.n = n
        self.dt = T / n
        self.disc = np.exp(-self.r_f * self.dt) # precomputed time step discount

        # step size delta r
        self.d_r = self.sigma * np.sqrt(3 * self.dt)

        # calculate risk neutral probabilities
        mu = self.r_f - (self.sigma ** 2) / 2
        a = mu * self.dt / self.d_r
        b = 1/3 + (mu * self.dt / self.d_r) ** 2

        self.p_u = (a + b) / 2
        self.p_m = 1 - b
        self.p_d = (b - a) / 2

        # set up stock tree as centralized triangle in 2d array
        self.stock_tree = np.zeros((2 * self.n + 1, self.n + 1))

        for i in range(self.n + 1):
            for j in range(self.n - i, self.n + i + 1):
                self.stock_tree[j, i] = self.s0 + self.d_r * (self.n - j)
    

    # pricing
    def price_option(self, K: float, call: bool, american: bool) -> np.float64:
        """
        Price a european call/put option with the given stock tree
        """
        option_tree = np.zeros((2 * self.n + 1, self.n + 1))

        # terminal payoff
        if call:
            for j in range(2 * self.n + 1):
                option_tree[j, self.n] = max(np.exp(self.stock_tree[j, self.n]) - K, 0)
        else:
            for j in range(2 * self.n + 1):
                option_tree[j, self.n] = max(K - np.exp(self.stock_tree[j, self.n]), 0)

        # iterate backwards to calculate discounted expected value at each node
        for i in reversed(range(self.n)):
            for j in range(self.n - i, self.n + i + 1):
                hold_val = self.disc * (
                    self.p_u * option_tree[j - 1, i + 1] + 
                    self.p_m * option_tree[j, i + 1] + 
                    self.p_d * option_tree[j + 1, i + 1]
                )

                if american:
                    exercise_val = max(np.exp(self.stock_tree[j, i]) - K, 0) if call else max(K - np.exp(self.stock_tree[j, i]), 0)
                    option_tree[j, i] = max(hold_val, exercise_val)
                else:
                    option_tree[j, i] = hold_val

        return option_tree[self.n, 0]


    def price_barrier_option_knock_out(self, K: float, H: float, call: bool) -> np.float64:
        """
        Price a european knock out barrier call/put option with the given stock tree

        Given the knock-out style of the option, pricing is simpler than the knock-in case below.
        If the option is above the barrier, its value is 0 (hit is an absorbing state) and if not
        it is the discounted expectation of all future states.
        """
        assert self.s0 != H, "Barrier cannot be current option price; use standard option pricing instead"

        # first determine whether the barrier is up or down
        up = self.s0 < np.log(H)

        option_tree = np.zeros((2 * self.n + 1, self.n + 1))

        # terminal payoff
        if call:
            for j in range(2 * self.n + 1):
                stock_val = np.exp(self.stock_tree[j, self.n])

                # Barrier cross condition: stock >= H and H > S0 or stock <= H and H < S0
                if (stock_val >= H and up) or (stock_val <= H and not up):
                    option_tree[j, self.n] = 0
                else:
                    option_tree[j, self.n] = max(stock_val - K, 0)
        else:
            for j in range(2 * self.n + 1):
                stock_val = np.exp(self.stock_tree[j, self.n])

                if (stock_val >= H and up) or (stock_val <= H and not up):
                    option_tree[j, self.n] = 0
                else:
                    option_tree[j, self.n] = max(K - stock_val, 0)

        # iterate backwards to calculate discounted expected value at each node
        for i in reversed(range(self.n)):
            for j in range(self.n - i, self.n + i + 1):
                stock_val = np.exp(self.stock_tree[j, i])

                # like above, if stock value is above, it and all future paths are "hit"
                if (stock_val >= H and up) or (stock_val <= H and not up):
                    option_tree[j, self.n] = 0
                else:
                    hold_val = self.disc * (
                        self.p_u * option_tree[j - 1, i + 1] + 
                        self.p_m * option_tree[j, i + 1] + 
                        self.p_d * option_tree[j + 1, i + 1]
                    )

                    option_tree[j, i] = hold_val
        
        # print(option_tree)

        return option_tree[self.n, 0]

    
    def price_barrier_option_knock_in(self, K: float, H: float, call: bool) -> np.float64:
        """
        Price a european knock in barrier call/put option with the given stock tree

        If a node's stock price is across the barrier, it just carries a hit state
        If a node's stock price is not across, it carries a hit state (discounted of all hit states)
        and no-hit state (discounted of all no-hit states). If a no-hit state has a next node that
        crosses the barrier, that node's hit state is used instead of no-hit for the current node's value

        ex:
               - hit (crosses barrier)
        no-hit - no hit
               - no hit
        
        the current node's no-hit option value is the discounted EV of the hit/no/no states respectively
        """
        assert self.s0 != H, "Barrier cannot be current option price; use standard option pricing instead"

        # first determine whether the option is up or down
        up = self.s0 < np.log(H)

        # using two trees: hit state and no hit state as described above
        option_tree_hit = np.zeros((2 * self.n + 1, self.n + 1))
        option_tree_no_hit = np.zeros((2 * self.n + 1, self.n + 1))

        # terminal payoff
        if call:
            for j in range(2 * self.n + 1):
                stock_val = np.exp(self.stock_tree[j, self.n])

                # Barrier cross condition: stock > H and H > S0 or stock < H and H < S0
                # no-hit terminal payoff is already initialized to zero for all cols
                option_tree_hit[j, self.n] = max(stock_val - K, 0)
        else:
            for j in range(2 * self.n + 1):
                stock_val = np.exp(self.stock_tree[j, self.n])

                option_tree_hit[j, self.n] = max(K - stock_val, 0)

        # iterate backwards to calculate discounted expected value at each node
        for i in reversed(range(self.n)):
            for j in range(self.n - i, self.n + i + 1):
                stock_val = np.exp(self.stock_tree[j, i])

                option_tree_hit[j, i] = self.disc * (
                    self.p_u * option_tree_hit[j - 1, i + 1] + 
                    self.p_m * option_tree_hit[j, i + 1] + 
                    self.p_d * option_tree_hit[j + 1, i + 1]
                )

                # if not currently across barrier, fill in no-hit case
                if (stock_val <= H and up) or (stock_val >= H and not up):
                    # if up and stock's up move crosses
                    if up:
                        up_val = np.exp(self.stock_tree[j - 1, i + 1])
                        if up_val >= H:
                            option_tree_no_hit[j, i] = self.disc * (
                                self.p_u * option_tree_hit[j - 1, i + 1] + 
                                self.p_m * option_tree_no_hit[j, i + 1] + 
                                self.p_d * option_tree_no_hit[j + 1, i + 1]
                            )
                        else:
                            option_tree_no_hit[j, i] = self.disc * (
                                self.p_u * option_tree_no_hit[j - 1, i + 1] + 
                                self.p_m * option_tree_no_hit[j, i + 1] + 
                                self.p_d * option_tree_no_hit[j + 1, i + 1]
                            )
                    else:
                        down_val = np.exp(self.stock_tree[j + 1, i + 1])
                        if down_val <= H:
                            option_tree_no_hit[j, i] = self.disc * (
                                self.p_u * option_tree_no_hit[j - 1, i + 1] + 
                                self.p_m * option_tree_no_hit[j, i + 1] + 
                                self.p_d * option_tree_hit[j + 1, i + 1]
                            )
                        else:
                            option_tree_no_hit[j, i] = self.disc * (
                                self.p_u * option_tree_no_hit[j - 1, i + 1] + 
                                self.p_m * option_tree_no_hit[j, i + 1] + 
                                self.p_d * option_tree_no_hit[j + 1, i + 1]
                            )
        
        # print("H", np.log(H))
        # print(option_tree_hit)
        # print(option_tree_no_hit)

        return option_tree_no_hit[self.n, 0]




if __name__ == "__main__":
    ## BINOMIAL TREE CHECKS
    # # double check within range of log(multiplicative CRR u factor)
    # tree = BinomialTree(0.05, 0.25, 100, 1, 2)
    # print(tree.d_r)
    # print(tree.p_u)
    # print(tree.stock_tree)

    S0 = 100
    K = 105
    sigma = 0.25
    r = 0.05
    T = 1
    steps = 200

    bin_tree = BinomialTree(r, sigma, S0, T, steps)

    # print(bin_tree.stock_tree)

    # print(bin_tree.price_option(K, call=True, american=False))
    # print(tree.price_option(K, call=False, american=True))
    # print(tree.price_option(K, call=True, american=False))
    # print(tree.price_option(K, call=True, american=True))

    tree = TrinomialTree(r, sigma, S0, T, steps)

    # print(tree.p_u)
    # print(tree.p_m)
    # print(tree.p_d)

    # print(tree.stock_tree)

    print(tree.price_option(K, call=False, american=False)) # at 2 steps, 135 is first tier up
    up_out = tree.price_barrier_option_knock_out(K, K - 50, call=False)
    up_in = tree.price_barrier_option_knock_in(K, K - 50, call=False)
    print(up_out)
    print(up_in)
    print(up_out + up_in)
    # print(tree.price_barrier_option_knock_out(K, K + 10, call=True))
    # print(tree.price_barrier_option_knock_out(K, K + 20, call=True))
    # print(tree.price_barrier_option_knock_out(K, K + 50, call=True))
    # print(BlackScholes.up_out_call(S0, K, K + 5, T, r, sigma))
    # print(BlackScholes.up_out_call(S0, K, K + 10, T, r, sigma))
    # print(BlackScholes.up_out_call(S0, K, K + 20, T, r, sigma))
    # print(BlackScholes.up_out_call(S0, K, K + 50, T, r, sigma))




