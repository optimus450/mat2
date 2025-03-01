import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp, solve_bvp
def system_ode(t, w):
    u1, u2, u3, v1, v2, v3 = w
    return [u2, u3, -2*v2**2 + v1, v2, v3, -u3**3 + v2 + u1 + np.sin(t)]

w0 = [0, 0, 0, 0, 0, 0]  # Placeholder ICs
sol = solve_ivp(system_ode, [0, 10], w0, method='RK45')

plt.figure(figsize=(10,6))
plt.plot(sol.t, sol.y[0], 'b-', label='x1(t)')
plt.plot(sol.t, sol.y[3], 'r-', label='x2(t)')
plt.xlabel('Time')
plt.title('Problem 5: System Solution')
plt.legend()
plt.grid()
plt.show()