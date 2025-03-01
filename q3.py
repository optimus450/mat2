import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp, solve_bvp
def competition_model(t, z):
    x, y = z
    dxdt = x * (2 - 0.4*x - 0.3*y)
    dydt = y * (1 - 0.1*y - 0.3*x)
    return [dxdt, dydt]

cases = [(1.5, 3.5, 'a'), (1, 1, 'b'), (2, 7, 'c'), (4.5, 0.5, 'd')]
fig, axs = plt.subplots(2, 2, figsize=(15, 10))

for i, (x0, y0, case) in enumerate(cases):
    sol = solve_ivp(competition_model, [0, 50], [x0, y0], method='RK45', dense_output=True)
    t_vals = np.linspace(0, 50, 500)
    x_vals, y_vals = sol.sol(t_vals)
    ax = axs[i//2, i%2]
    ax.plot(t_vals, x_vals, 'b-', label='x(t)')
    ax.plot(t_vals, y_vals, 'r-', label='y(t)')
    ax.set_title(f'Case {case}')
    ax.legend()
    ax.grid()

plt.tight_layout()
plt.show()