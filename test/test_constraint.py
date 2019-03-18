
import pytest

from pynominate import nominate
import numpy as np
from scipy.optimize import minimize

idpts = [[-1.1, 0], [0.0, 0.0]]

print(nominate.circle_constraint_idpt_spline(idpts))
print(nominate.circle_constraint_idpt_spline_grad(idpts))
