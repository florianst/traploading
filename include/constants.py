import matplotlib
import matplotlib.pyplot as plt
import time
from scipy.integrate import trapz
from scipy.optimize import fsolve
import numpy as np

matplotlib.rcParams['font.family'] = 'Latin Modern Roman'
matplotlib.rcParams['font.size'] = 15

params = {'text.usetex': False, 'mathtext.fontset': 'cm'} # for the fonts in the legend an inline latex
plt.rcParams.update(params)

# UNITS
W  = 1
A  = 1
m  = 1
um = 1e-6
nm = 1e-9
mm = 1e-3
uK = 1e-6

# CONSTANTS
pi   = np.pi
mRb  = 87*1.67e-27
mK   = 40*1.67e-27
hbar = 1.0546e-34
c    = 3e8
e    = 1.6e-19
me   = 9.1e-31
epsilon0 = 8.854e-12
mu0  = 4*pi*1e-7
kB   = 1.38e-23
lambda_Dipole = 1064*nm
omega_Dipole = (2*pi*c)/lambda_Dipole


# SET UP THE POTENTIALS - Rubidium
lambda_RbD1 = 794.98*nm # Rubidium D1 line
lambda_RbD2 = 780.24*nm # Rubidium D2 line
gamma_RbD1  = 36.1e6
gamma_RbD2  = 38.1e6
omega_RbD1  = (2*pi*c)/lambda_RbD1
omega_RbD2  = (2*pi*c)/lambda_RbD2
URbDipole0  = -((pi*c**2)/(2*omega_RbD1**3))*gamma_RbD1*(1/(omega_RbD1 - omega_Dipole) + 1/(omega_RbD1 + omega_Dipole)) - (pi*c**2)/(2*omega_RbD2**3)*gamma_RbD2*(2/(omega_RbD2 - omega_Dipole) + 2/(omega_RbD2 + omega_Dipole))


# SET UP THE POTENTIALS - Potassium
lambda_KD1 = 770.11*nm # Potassium D1 line
lambda_KD2 = 766.7*nm # Potassium D2 line
gamma_KD1  = 37.4e6
gamma_KD2  = 37.9e6
omega_KD1  = (2*pi*c)/lambda_KD1
omega_KD2  = (2*pi*c)/lambda_KD2
UKDipole0  = -((pi*c**2)/(2*omega_KD1**3))*gamma_KD1*(1/(omega_KD1 - omega_Dipole) + 1/(omega_KD1 + omega_Dipole)) - (pi*c**2)/(2*omega_KD2**3)*gamma_KD2*(2/(omega_KD2 - omega_Dipole) + 2/(omega_KD2 + omega_Dipole))