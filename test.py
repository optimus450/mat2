import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp, solve_bvp
from scipy.integrate import solve_bvp

# solve_bvp
def ode_bvp(x, y):
    return np.vstack((y[1], 100*y[0]))

def bc(ya, yb):
    return np.array([ya[0]-1, yb[0]-np.exp(-10)])

x_guess = np.linspace(0, 1, 10)
y_guess = np.zeros((2, 10))
y_guess[0] = np.exp(-10 * x_guess)

sol_bvp = solve_bvp(ode_bvp, bc, x_guess, y_guess)
x_bvp = np.linspace(0, 1, 100)
y_bvp = sol_bvp.sol(x_bvp)[0]

# Shooting Method
def shooting_ode(t, y):
    return [y[1], 100*y[0]]

sol1 = solve_ivp(shooting_ode, [0,1], [1,0], t_eval=x_bvp)
sol2 = solve_ivp(shooting_ode, [0,1], [0,1], t_eval=x_bvp)
alpha = (np.exp(-10) - sol1.y[0,-1]) / sol2.y[0,-1]
y_shooting = sol1.y[0] + alpha * sol2.y[0]

# Plot
plt.figure(figsize=(10,6))
plt.plot(x_bvp, y_bvp, 'r-', label='solve_bvp')
plt.plot(x_bvp, y_shooting, 'b--', label='Shooting')
plt.plot(x_bvp, np.exp(-10*x_bvp), 'k-', label='Exact')
plt.legend()
plt.title('Problem 6: BVP Solutions')
plt.grid()
plt.show()