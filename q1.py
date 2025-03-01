import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp, solve_bvp

# 1. Initial Value Problems (IVPs)
def ivp1(t, y):
    return t * np.exp(3 * t) - 2 * y

def exact_solution1(t):
    return (1/5) * t * np.exp(3*t) - (1/25) * np.exp(3*t) + (1/25) * np.exp(-2*t)

t_span = (0, 1)
t_eval = np.linspace(0, 1, 50)
y0 = [0]

sol1 = solve_ivp(ivp1, t_span, y0, method='RK45', t_eval=t_eval)
exact_y1 = exact_solution1(t_eval)
error1 = np.abs(sol1.y[0] - exact_y1)
print(error1)

plt.plot(t_eval, sol1.y[0], label='Numerical Solution')
plt.plot(t_eval, exact_y1, '--', label='Exact Solution')
plt.xlabel('t')
plt.ylabel('y')
plt.legend()
plt.title('IVP Solution and Exact Comparison')
plt.show()


import numpy as np
from scipy.integrate import odeint, solve_ivp
import matplotlib.pyplot as plt

# Problem 1i
def dy_dt_1i(y, t):
    return t * np.exp(3*t) - 2*y

t_1i = np.linspace(0, 1, 100)
y0_1i = 0

# Solve with odeint
y_odeint_1i = odeint(dy_dt_1i, y0_1i, t_1i)

# Solve with solve_ivp
sol_ivp_1i = solve_ivp(lambda t, y: dy_dt_1i(y, t), [0, 1], [y0_1i], t_eval=t_1i)
y_ivp_1i = sol_ivp_1i.y[0]

# Exact solution
def y_exact_1i(t):
    return (1/3)*t*np.exp(3*t) - (1/25)*np.exp(3*t) + (1/25)*np.exp(-2*t)

y_exact_vals_1i = y_exact_1i(t_1i)

# Plot
plt.figure(figsize=(10,6))
plt.plot(t_1i, y_odeint_1i, 'b--', label='odeint')
plt.plot(t_1i, y_ivp_1i, 'g--', label='solve_ivp')
plt.plot(t_1i, y_exact_vals_1i, 'r-', label='Exact')
plt.title('Problem 1i: Solution Comparison')
plt.xlabel('t')
plt.ylabel('y(t)')
plt.legend()
plt.grid()
plt.show()

# Problem 1ii
def dy_dt_1ii(t, y):
    return 1 + (t - y)**2

t_1ii = np.linspace(2, 3, 100)
y0_1ii = [1]

# Solve with odeint
def dy_dt_odeint_1ii(y, t):
    return 1 + (t - y)**2

y_odeint_1ii = odeint(dy_dt_odeint_1ii, y0_1ii, t_1ii)

# Solve with solve_ivp
sol_ivp_1ii = solve_ivp(dy_dt_1ii, [2, 3], y0_1ii, t_eval=t_1ii)
y_ivp_1ii = sol_ivp_1ii.y[0]

# Exact solution
def y_exact_1ii(t):
    return t + 1/(1 - t)

y_exact_vals_1ii = y_exact_1ii(t_1ii)

# Plot
plt.figure(figsize=(10,6))
plt.plot(t_1ii, y_odeint_1ii, 'b--', label='odeint')
plt.plot(t_1ii, y_ivp_1ii, 'g--', label='solve_ivp')
plt.plot(t_1ii, y_exact_vals_1ii, 'r-', label='Exact')
plt.title('Problem 1ii: Solution Comparison')
plt.xlabel('t')
plt.ylabel('y(t)')
plt.legend()
plt.grid()
plt.show()