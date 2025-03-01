import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp, solve_bvp
def lotka_volterra(t, z):
    x, y = z
    dxdt = -0.1*x + 0.02*x*y
    dydt = 0.2*y - 0.025*x*y
    return [dxdt, dydt]

initial_conditions = [6, 6]
t = np.linspace(0, 50, 1000)
sol2 = solve_ivp(lotka_volterra, (0, 50), initial_conditions, t_eval=t)

plt.plot(t, sol2.y[0], label='Predators')
plt.plot(t, sol2.y[1], label='Prey')
plt.xlabel('Time')
plt.ylabel('Population (thousands)')
plt.legend()
plt.title('Lotka-Volterra Predator-Prey Model')
plt.show()

# Identifying equal populations
diff = np.abs(sol2.y[0] - sol2.y[1])
equal_time = t[np.argmin(diff)]
print(f'First time when predator and prey populations are equal: {equal_time:.2f}')



import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp, solve_bvp
def lotka_volterra(t, z):
    x, y = z
    dxdt = -0.1 * x + 0.02 * x * y
    dydt = 0.2 * y - 0.025 * x * y
    return [dxdt, dydt]

z0 = [6, 6]
t_span = [0, 20]

def event_x_equal_y(t, z):
    x, y = z
    return x - y
event_x_equal_y.terminal = False

sol = solve_ivp(lotka_volterra, t_span, z0, events=event_x_equal_y, dense_output=True)

event_times = sol.t_events[0]
valid_times = [t for t in event_times if t > 0]
if valid_times:
    first_equal_time = valid_times[0]
    print(f"First time t > 0 when x(t) = y(t): {first_equal_time:.4f}")

t_vals = np.linspace(0, sol.t[-1], 300)
z_vals = sol.sol(t_vals)
x_vals, y_vals = z_vals[0], z_vals[1]

plt.figure(figsize=(10,6))
plt.plot(t_vals, x_vals, 'b-', label='Predators (x)')
plt.plot(t_vals, y_vals, 'r-', label='Prey (y)')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('Problem 2: Lotka-Volterra Model')
plt.legend()
plt.grid()
plt.show()