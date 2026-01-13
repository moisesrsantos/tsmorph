"""Example to compare linear morphing vs DBA morphing and save a comparison image.

Run:
    python examples/compare_morphing.py

This will generate examples/morph_comparison.png
"""
import os, sys
# Ensure project root is on sys.path so we import the local tsmorph package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import matplotlib.pyplot as plt
from tsmorph import TSmorph

os.makedirs(os.path.dirname(__file__), exist_ok=True)

# Create two synthetic series that are similar but temporally misaligned
x = np.linspace(0, 6 * np.pi, 200)
S = np.sin(x) + 0.1 * np.random.RandomState(0).normal(size=len(x))
# target has a local pattern shifted and scaled
T = np.sin(x * 1.05 + 0.8) * 0.8 + 0.2 * np.random.RandomState(1).normal(size=len(x)) + 0.2

granularity = 6
morph_linear = TSmorph(S, T, granularity=granularity)
morph_dba = TSmorph(S, T, granularity=granularity)

# Generate morphed series with simple timing
import time
start = time.perf_counter()
df_linear = morph_linear.fit(use_dba=False)
mid = time.perf_counter()
df_dba = morph_dba.fit(use_dba=True, n_iter=10)
end = time.perf_counter()
print(f"Linear fit time: {mid - start:.4f}s | DBA fit time: {end - mid:.4f}s")

# Choose some intermediate series to plot
cols = df_linear.columns
indices = [0, len(cols) // 2, len(cols) - 1]

fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Linear morphing plot
axs[0].plot(S, color='black', alpha=0.8, label='Source (S)')
axs[0].plot(T, color='grey', alpha=0.8, label='Target (T)')
for idx in indices:
    axs[0].plot(df_linear[cols[idx]].values, label=f'Linear {cols[idx]}', linewidth=1.5)
axs[0].set_title('Linear Morphing (no DBA)')
axs[0].legend(loc='upper right')

# DBA morphing plot
axs[1].plot(S, color='black', alpha=0.8, label='Source (S)')
axs[1].plot(T, color='grey', alpha=0.8, label='Target (T)')
for idx in indices:
    axs[1].plot(df_dba[cols[idx]].values, label=f'DBA {cols[idx]}', linewidth=1.5)
axs[1].set_title('DBA-aligned Morphing')
axs[1].legend(loc='upper right')

plt.tight_layout()
output = os.path.join(os.path.dirname(__file__), 'morph_comparison.png')
plt.savefig(output, dpi=200)
print(f"Saved comparison image to: {output}")
