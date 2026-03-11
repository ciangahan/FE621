import numpy as np

from collections.abc import Callable

def root_bisection(func:Callable, a:float, b:float, epsilon:float=1e-6, log_iter:bool=False):
    """
    Use the bisection method to find the root of a continuous function
    @param func: Objective function
    @param a: Initial bound (a and b must bracket a root, that is f(a)f(b) < 0)
    @param b: Initial bound (a and b must bracket a root, that is f(a)f(b) < 0)
    @param epsilon: convergence tolerance (>0)
    """
    fa = func(a)
    fb = func(b)

    if fa * fb > 0:
        # print(f"Warning: Supplied bounds {a} and {b} do not bracket a root")
        if log_iter:
            return np.nan, np.nan
        else:
            return np.nan
    
    i = 0

    while abs(a - b) > epsilon:
        i += 1
        mid = (a + b) / 2
        fmid = func(mid)

        if fmid == 0:
            a = mid
            break
        elif fa * fmid < 0:
            fb = fmid
            b = mid
        else:
            fa = fmid
            a = mid

    if log_iter:
        return a, i
    else:
        return a


def root_newton(func:Callable, derivative:Callable, x:float, epsilon:float=1e-6, max_iter:int=100, log_iter:bool=False):
    """
    Newton method for root-finding of continuous, differentiable function
    @param func: objective function
    @param derivative: derivative of objective function
    @param x: initial guess
    @param epsilon: convergence tolerance
    @param max_iter: max iterations before giving up
    @param log_iter: Log/output number of iterations to converge
    """

    for i in range(max_iter):
        fx = func(x)
        
        if abs(fx) < epsilon:
            if log_iter:
                return x, i + 1
            else:
                return x
        
        fpx = derivative(x)

        x = x - fx / fpx

    if log_iter:
        return np.nan, np.nan
    else:
        return np.nan
    

if __name__ == "__main__":
    """
    Brief tests for each function
    """
    print(root_bisection(lambda x: (x-2)**3, 0, 5))
    print(root_newton(lambda x: (x-2)**3, lambda x: 3 * (x-2) ** 2, 5))