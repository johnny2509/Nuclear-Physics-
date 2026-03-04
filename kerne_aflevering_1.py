
#%%

import numpy as np

epsilon = 2.225          # MeV
R = 1.9                  # fm

kap = (np.sqrt((2*1.66*10**(-27)/2) / ((6.626*10**(-34))/(2*np.pi))**2 * 2.225*10**6*1.602*10**(-19))) * 10**(-15)
print(kap)

# %%

from scipy.optimize import root_scalar

def func(U0):
    k = (np.sqrt(((2*1.66*10**(-27)/2)*(U0-epsilon)*10**6*1.602*10**(-19) / ((6.626*10**(-34))/(2*np.pi))**2)))* 10**(-15)
    return k / np.tan(k * R) + kap

sol = root_scalar(func, bracket=[10,100])
print(sol.root)

# %%

import numpy as np
import matplotlib.pyplot as plt

k = (np.sqrt(((2*1.66*10**(-27)/2)*(sol.root-epsilon)*10**6*1.602*10**(-19) / ((6.626*10**(-34))/(2*np.pi))**2)))* 10**(-15)
print(k)

A = (R/2 - np.sin(2*k*R)/(4*k) + np.sin(k*R)**2/(2*kap))**(-0.5)
print(A)

r_in = np.linspace(0, R, 100)
r_out = np.linspace(R, 17, 100)

u_in = A * np.sin(k * r_in)
u_out = A * np.sin(k * R) * np.exp(-kap * (r_out - R))

plt.plot(r_in, u_in, label='Inside the nucleus')
plt.plot(r_out, u_out, label='Outside the nucleus')
plt.axvline(x=R, color='k', linestyle='--', label='$R= 1.9$ fm')
plt.axhline(y=0, color='k', linestyle='-')
plt.xlabel('r (fm)')
plt.ylabel('u(r)')
plt.title('Radial wavefunction of the deuteron')
plt.legend()
plt.ylim(-0.1, 0.63)
plt.show()

#%%

# task 2

#hbar2_2m = (2*1.66*10**(-27) / ((6.626*10**(-34))/(2*np.pi))**2) * 10**(-15)
hbar = 197.327 # MeV*fm
m = 939.56 # MeV/c^2
hbar2_2m = ((hbar**2) / (2*m))#*(3*10**8)**2 #*10**6*1.602*10**(-19)#) * 10**(-15)
print(hbar2_2m)

# %%

# epsilon finder

from scipy.optimize import root_scalar

U0 = 40 # MeV
A = 25 # number of nucleons

hbar = 197.327 # MeV*fm
m = 939.56 # MeV/c^2
hbar2_2m = (hbar**2) / (2*m) # MeV*fm^2

def func2(epsilon):

    R = 1.4 * A**(1/3) # fm
    
    kappa = np.sqrt(epsilon / hbar2_2m)

    k = np.sqrt((U0 - epsilon) / hbar2_2m)

    return k / np.tan(k * R) + kappa

sol2 = root_scalar(func2, bracket=[0.1, U0-0.1])
print(sol2.root)

# %%

from scipy.optimize import fsolve

roots = []
for As in [25, 50, 75, 100, 125, 150, 175, 200]:
    root = fsolve(func2, As)[0]
    if 0.1 < root < U0-0.1 and not any(abs(root - r) < 1e-6 for r in roots):
        roots.append(root)
print(roots)

# %%

import numpy as np
from scipy.optimize import root_scalar

U0 = 40.0       # MeV
hbar = 197.327  # MeV fm
m = 939.56      # MeV
hbar2_2m = hbar**2 / (2*m)

# helper function for a given mass number A

def find_roots_for_A(A, U0=U0, hbar2_2m=hbar2_2m):
    R = 1.4 * A**(1/3)  # fm

    def f(epsilon):
        k = np.sqrt((U0 - epsilon) / hbar2_2m)
        kappa = np.sqrt(epsilon / hbar2_2m)
        return k * np.cos(k * R) + kappa * np.sin(k * R)

    eps_grid = np.linspace(1e-4, U0 - 1e-4, 5000)
    vals = f(eps_grid)

    roots = []
    for i in range(len(eps_grid) - 1):
        if vals[i] * vals[i + 1] < 0:
            sol = root_scalar(f, bracket=[eps_grid[i], eps_grid[i + 1]])
            roots.append(sol.root)

    return np.unique(np.round(roots, 6))

# radial wavefunction plotter for a given epsilon and A

def plot_radial_wavefunction(epsilon, A, U0=U0, hbar2_2m=hbar2_2m):
    R = 1.4 * A**(1/3)
    kappa = np.sqrt(epsilon / hbar2_2m)
    k = np.sqrt((U0 - epsilon) / hbar2_2m)

    # normalization constant derived from continuity at R
    A_norm = (R/2 - np.sin(2*k*R)/(4*k) + np.sin(k*R)**2/(2*kappa))**(-0.5)

    r_in = np.linspace(0, R, 200)
    r_out = np.linspace(R, 5*R, 200)

    u_in = A_norm * np.sin(k * r_in)
    u_out = A_norm * np.sin(k * R) * np.exp(-kappa * (r_out - R))

    plt.figure(figsize=(6,4))
    plt.plot(r_in, u_in, label=fr'$\epsilon={-epsilon:.3f}\,$MeV')
    plt.plot(r_out, u_out, ls='--', label=None)
    plt.axvline(x=R, color='k', ls='--')
    plt.axhline(y=0, color='k', ls='-')
    plt.xlabel('$r$ (fm)', size=12)
    plt.ylabel('$u(r)', size=12)
    plt.title(f'$A = {A}$')
    plt.legend()
    plt.tight_layout()
    plt.show()

# return radial arrays for an epsilon and A (no plotting)
def radial_arrays(epsilon, A, U0=U0, hbar2_2m=hbar2_2m):
    R = 1.4 * A**(1/3)
    kappa = np.sqrt(epsilon / hbar2_2m)
    k = np.sqrt((U0 - epsilon) / hbar2_2m)
    A_norm = (R/2 - np.sin(2*k*R)/(4*k) + np.sin(k*R)**2/(2*kappa))**(-0.5)
    r_in = np.linspace(0, R, 200)
    r_out = np.linspace(R, 5*R, 200)
    u_in = A_norm * np.sin(k * r_in)
    u_out = A_norm * np.sin(k * R) * np.exp(-kappa * (r_out - R))
    return R, r_in, r_out, u_in, u_out

import matplotlib.pyplot as plt

# select a few sample mass numbers and gather data for plotting
plot_As = []
plot_eps = []
# group the A values into two sets of four for subplots
group1 = [25, 50, 75, 100]
group2 = [125, 150, 175, 200]

for idx, group in enumerate((group1, group2)):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    for ax, A in zip(axes, group):
        rts = find_roots_for_A(A)
        if len(rts) > 0:
            neg_roots = -rts
            print(f"A={A}, s-wave binding energies (MeV): {neg_roots}")
            # compute radius for the current nucleus
            R = 1.4 * A**(1/3)
            ax.axhline(y=0, color='k', ls='-')
            ax.axvline(x=R, color='k', ls='-')  # vertical line at boundary
            for eps in neg_roots:
                plot_As.append(A)
                plot_eps.append(eps)
                R, r_in, r_out, u_in, u_out = radial_arrays(-eps, A)
                ax.plot(r_in, u_in, label=fr'$\epsilon={eps:.3f}\ \mathrm{{MeV}}$')
                ax.plot(r_out, u_out, ls='--', label=None)
            ax.set_title(f'A={A}')
            ax.set_xlabel('r (fm)', size=12)
            ax.set_ylabel('u(r)', size=12)
            ax.legend()
    fig.tight_layout()
    # set x limits for this figure group
    xlim_range = (-1, 20) if idx == 0 else (-1, 25)
    for ax in axes:
        ax.set_xlim(xlim_range)
    plt.show()

# make a scatter plot of A vs. epsilon roots
if plot_As:
    plt.figure(figsize=(8, 5))
    plt.scatter(plot_As, plot_eps, s=10)
    plt.xlabel('${A}$', size=14)
    plt.ylabel('${\epsilon}$ (MeV)', size=14)
    plt.title('${\epsilon}$ for various $A$', size=16)
    plt.grid(True)
    plt.show()
# %%

import numpy as np
import matplotlib.pyplot as plt

k = (np.sqrt(((2*1.66*10**(-27)/2)*(sol.root-epsilon)*10**6*1.602*10**(-19) / ((6.626*10**(-34))/(2*np.pi))**2)))* 10**(-15)
print(k)

A = (R/2 - np.sin(2*k*R)/(4*k) + np.sin(k*R)**2/(2*kap))**(-0.5)
print(A)

r_in = np.linspace(0, R, 100)
r_out = np.linspace(R, 17, 100)

u_in = A * np.sin(k * r_in)
u_out = A * np.sin(k * R) * np.exp(-kap * (r_out - R))

plt.plot(r_in, u_in, label='Inside the nucleus')
plt.plot(r_out, u_out, label='Outside the nucleus')
plt.axvline(x=R, color='k', linestyle='--', label='$R= 1.9$ fm')
plt.axhline(y=0, color='k', linestyle='-')
plt.xlabel('r (fm)')
plt.ylabel('u(r)')
plt.title('Radial wavefunction of the deuteron')
plt.legend()
plt.ylim(-0.1, 0.63)
plt.show()

#%%

# task 3: epsilon = 0

epsilon = 0.0
hbar = 197.327 # MeV*fm
m = 939.56 # MeV/c^2
hbar2_2m = (hbar**2) / (2*m) # MeV*fm^2

R = 1.4 * A**(1/3) # fm
    
kappa = np.sqrt(epsilon / hbar2_2m)
k = np.sqrt((U0 - epsilon) / hbar2_2m)
print(f"kappa: {kappa}, k: {k}")
# %%
