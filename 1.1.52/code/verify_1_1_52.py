"""
Verify: Sallen-Key 2nd-order LPF
R1=R2=22k, C1=C2=330pF, Rf=10k, Rg=17k
Expected: f0≈21.9kHz, K≈4dB, 40dB/decade rolloff, Butterworth
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# ── Parameters ────────────────────────────────────────────
R  = 22e3        # Ω
C  = 330e-12     # F
Rf = 10e3        # Ω  (output → inverting input)
Rg = 17e3        # Ω  (inverting input → GND)

K     = 1 + Rf / Rg
w0    = 1 / (R * C)
f0    = w0 / (2 * np.pi)
Q     = 1 / (3 - K)

print("=" * 40)
print(f"K          = {K:.6f}  ({20*np.log10(K):.2f} dB)")
print(f"f0         = {f0/1e3:.3f} kHz")
print(f"omega0     = {w0:.2f} rad/s")
print(f"Q          = {Q:.4f}  (1/sqrt(2) = {1/np.sqrt(2):.4f})")
print(f"Filter type: {'Butterworth' if abs(Q - 1/np.sqrt(2)) < 0.01 else 'Other'}")
print("=" * 40)

# ── Transfer function H(s) = K*w0^2 / (s^2 + w0*(3-K)*s + w0^2) ──
num = [K * w0**2]
den = [1, w0 * (3 - K), w0**2]

sys = signal.TransferFunction(num, den)

# ── Bode plot ─────────────────────────────────────────────
f = np.logspace(2, 7, 2000)
w = 2 * np.pi * f
_, H = signal.freqs(num, den, worN=w)
mag_dB = 20 * np.log10(np.abs(H))

fig, ax = plt.subplots(figsize=(9, 5))
ax.semilogx(f / 1e3, mag_dB, 'b-', lw=2.5, label='Sallen-Key LPF (computed)')

# Mark key points
ax.axhline(20 * np.log10(K),      color='gray', ls='--', lw=1, label=f'DC gain = {20*np.log10(K):.1f} dB')
ax.axhline(20 * np.log10(K) - 3,  color='green', ls='--', lw=1, label=f'DC gain − 3 dB = {20*np.log10(K)-3:.1f} dB')
ax.axvline(f0 / 1e3, color='red', ls='--', lw=1, label=f'f₀ = {f0/1e3:.2f} kHz')

# Annotate -40dB/decade slope
f_slope = np.array([1e5, 1e6])
ref_dB  = 20 * np.log10(K) - 40  # one decade past f0
ax.semilogx(f_slope / 1e3, [ref_dB, ref_dB - 40], 'k:', lw=1.5, label='−40 dB/decade')

ax.set_xlabel('Frequency (kHz)', fontsize=12)
ax.set_ylabel('|Vo/Vi| (dB)', fontsize=12)
ax.set_title('Bode Plot — Sallen-Key LPF\n'
             r'$R_1=R_2=22\,k\Omega,\;C_1=C_2=330\,pF,\;R_f=10\,k\Omega,\;R_g=17\,k\Omega$',
             fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, which='both', alpha=0.35)
ax.set_xlim([0.1, 1e4])
ax.set_ylim([-60, 10])

plt.tight_layout()
plt.savefig('bode_1_1_52.png', dpi=150)
plt.show()
print("Saved → bode_1_1_52.png")

# ── Verify gain exactly at f0 ─────────────────────────────
_, H_at_f0 = signal.freqs(num, den, worN=[w0])
print(f"\nGain at f0: {20*np.log10(abs(H_at_f0[0])):.3f} dB  "
      f"(expected {20*np.log10(K)-3:.3f} dB)")
