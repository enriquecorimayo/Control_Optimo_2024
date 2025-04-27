import numpy as np


def optimal_spread(kappa, sigma, T_t, k):
    return kappa * sigma ** 2 * (T_t) + 2 / kappa * np.log(1 + kappa / k)


def optimal_reservation_price(kappa, sigma, T_t, s, inv):
    return s - inv * kappa * sigma ** 2 * (T_t)


def orders_intensity_lambda(A, k, d):
    return A * np.exp(-k * d)


def optimal_spread_callback(kappa, sigma, k):
    return lambda T_t: optimal_spread(kappa, sigma, T_t, k)


def optimal_reservation_price_callback(sigma, kappa):
    return lambda T_t, s, inv: optimal_reservation_price(kappa, sigma, T_t, s, inv)
