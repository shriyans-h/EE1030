"""
Verify: y'' + xy = 0,  y(0)=1,  y'(0)=0
Numerical (RK45) vs power series solution.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ── ODE as first-order system ──────────────────────────────
def ode(t, Y):
    y, yp = Y
    return [yp, -t * y]          # y'' = -x*y

x_span = (0, 4)
x_eval = np.linspace(*x_span, 500)
sol = solve_ivp(ode, x_span, [1, 0], t_eval=x_eval, method='RK45', rtol=1e-10)

# ── Power series (first 4 non-zero terms) ─────────────────
def series(x):
    return (1
            - x**3 / 6
            + x**6 / 180
            - x**9 / 12960)

xs = x_eval
ys_series = series(xs)

# ── Recurrence coefficients (cross-check) ─────────────────
a = np.zeros(30)
a[0] = 1.0          # a0=1, a1=0, a2=0
for n in range(27):
    a[n + 3] = -a[n] / ((n + 3) * (n + 2))

print("First 10 non-zero coefficients (multiples of 3):")
for k in range(10):
    idx = 3 * k
    print(f"  a[{idx:2d}] = {a[idx]:+.8e}")

# ── Plot ───────────────────────────────────────────────────
plt.figure(figsize=(8, 5))
plt.plot(sol.t, sol.y[0], 'b-',  lw=2.5, label='Numerical (RK45)')
plt.plot(xs,    ys_series, 'r--', lw=2,   label='Power series (4 terms)')
plt.axhline(0, color='k', lw=0.6)
plt.xlabel('x'); plt.ylabel('y(x)')
plt.title(r"$y'' + xy = 0$,  $y(0)=1$,  $y'(0)=0$")
plt.legend(); plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig('verify_1_2_52.png', dpi=150)
plt.show()
print("Plot saved → verify_1_2_52.png")