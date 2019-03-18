
import pytest

from pynominate import nominate
import numpy as np
from scipy.optimize import minimize


def smooth_loss_grad(par, x, lambdaval):
    return -2.0 * (x - par) + \
        nominate.dwnominate_spline_penalty_grad(np.reshape(par, (-1, 2)), lambdaval).flatten()


def smooth_loss(par, x, lambdaval):
    return np.sum(np.square(x - par)) + \
        nominate.dwnominate_spline_penalty(np.reshape(par, (-1, 2)), lambdaval)


def test_spline_penalty():
    t = np.array(range(20)) + 0.0
    x = np.square(t)

    startval = np.array(range(10,30)) + 0.0
    lambdaval = 0
    res = minimize(
        smooth_loss,
        startval,
        jac=smooth_loss_grad,
        method="SLSQP",
        args=(x, lambdaval)
    )

    assert pytest.approx(res["x"]) == x

    # Large lambda
    t = np.array(range(2)) + 0.0
    x = np.square(t)

    startval = [10, 13]  # np.array(range(10,30)) + 0.0
    lambdaval = 1e10
    res = minimize(
        smooth_loss,
        startval,
        jac=smooth_loss_grad,
        method="SLSQP",
        args=(x, lambdaval)
    )

    assert np.mean(np.reshape(res["x"], (-1, 2)), axis=1) == np.mean(np.reshape(x, (-1, 2)), axis=1)


