
#%%

from matplotlib import pyplot as plt
import numpy as np

E1 = 0.0 # closed shell energy [MeV]
E2 = np.linspace(-2.0, 2.0, 200) #  MeV
V = 0.3 # coupling potential [MeV]

def Eplus(E1, E2, V):
    return 1/2 * ((E1 + E2) + np.sqrt((E1 - E2)**2 - 4*E1*E2 + 4*V**2))

def Eminus(E1, E2, V):
    return 1/2 * ((E1 + E2) - np.sqrt((E1 - E2)**2 - 4*E1*E2 + 4*V**2))

plt.plot(E2, Eplus(E1, E2, V), label='E+ (upper 0+ state)')
plt.plot(E2, Eminus(E1, E2, V), label='E- (lower 0+ state)')
plt.xlabel('E2 (MeV)')
plt.ylabel('Energy (MeV)')
plt.title('Eigenenergies in the interval E2 = [-2, 2] MeV')
plt.axhline(0, color='gray', linestyle='--', label='E1 = 0 MeV')
plt.legend()

#%%

def one_p(E1, E2, V):
    return 1 / (1 + (Eplus(E1, E2, V) - E1)**2 / V**2)

def one_m(E1, E2, V):
    return 1 / (1 + (Eminus(E1, E2, V) - E1)**2 / V**2)

def two_p(E1, E2, V):
    return 1 / (1 + (Eplus(E1, E2, V) - E2)**2 / V**2)

def two_m(E1, E2, V):
    return 1 / (1 + (Eminus(E1, E2, V) - E2)**2 / V**2)

plt.figure()
#plt.plot(E2, one_p(E1, E2, V), label=r'$c^2_{1,+}$')
plt.plot(E2, one_m(E1, E2, V), label=r'$c^2_{1,-}$')
#plt.plot(E2, two_p(E1, E2, V), label=r'$c^2_{2,+}$')
plt.plot(E2, two_m(E1, E2, V), label=r'$c^2_{2,-}$')
plt.xlabel('E2 (MeV)')
plt.ylabel('Probability')
plt.title('Probability amplitudes for the lowest 0+ state')
plt.axhline(0.5, color='gray', linestyle='--')
plt.legend(loc=5, prop={'size': 14})
# %%
