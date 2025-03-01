def pendulum_ode(t, y, g, L):
    theta, omega = y
    return [omega, -(g/L) * np.sin(theta)]

g, L = 32.17, 2
y0 = [np.pi/6, 0]
t_eval = np.arange(0, 2.1, 0.1)

sol = solve_ivp(pendulum_ode, [0, 2], y0, args=(g, L), t_eval=t_eval)

print("Theta values:")
for t, theta in zip(t_eval, sol.y[0]):
    print(f"t={t:.1f}s: theta={theta:.6f} rad")

plt.figure(figsize=(10,6))
plt.plot(sol.t, sol.y[0], 'b-o')
plt.xlabel('Time (s)')
plt.ylabel('Theta (rad)')
plt.title('Problem 4: Pendulum Motion')
plt.grid()
plt.show()