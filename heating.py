from include.constants import * # contains everything which is needed across files
from include.experiment import * # contains all fixed parameters of the simulated experiment

Pdipole = 4*W
Icoil   = 27*A
crossed = False

def V1(x, y, z):  # Initial Potential: dipole + quadrupole. x, y, z all in m
    return mRb * g * z + mu * B(x, y, z, Icoil)

def V2(x, y, z):  # Initial Potential: dipole + quadrupole. x, y, z all in m
    return mRb * g * z + mu * B(x, y, z, Icoil) + RbDipolePotential(x, y, z - z0, Pdipole, 50 * um, 50 * um, crossed=crossed)

def V3(x, y, z):  # Final Potential: quadrupole trap switched off
    return mRb * g * z                          + RbDipolePotential(x, y, z - z0, Pdipole, 50 * um, 50 * um, crossed=crossed)

Vinit = V1
Vfin  = V3


# ADIABATIC TEMPERATURE CHANGE
def I_integrand(x, y, z, Tact): # x, y, z all in m, T in K
    if (Tact==T1): return np.exp(-1.0/(kB*Tact)*Vinit(x, y, z))
    else:          return np.exp(-1.0/(kB*Tact)*Vfin(x, y, z))

def I(T):
    acc = 100
    x = np.linspace(-1000*um, 1000*um, acc)
    y = np.linspace(-300*um, 300*um, acc)
    z = np.linspace(-300*um, 100*um, acc)

    integral = trapz(trapz(trapz(I_integrand(x[:,None][:,None], y[:,None], z, T), z), y), x)
    return integral

def J_integrand(x, y, z, Tact): # x, y, z all in m, T in K
    if (Tact==T1): return Vinit(x, y, z)*np.exp(-1.0/(kB*Tact)*Vinit(x, y, z))
    else:          return Vfin(x, y, z)*np.exp(-1.0/(kB*Tact)*Vfin(x, y, z))

def J(T):
    acc = 100
    x = np.linspace(-1000*um, 1000*um, acc)
    y = np.linspace(-300*um, 300*um, acc)
    z = np.linspace(-300*um, 100*um, acc)

    integral = trapz(trapz(trapz(J_integrand(x[:, None][:, None], y[:, None], z, T), z), y), x)
    return integral

def eq_temperature(T2, T1):
    return 1.0/T1*J(T1)/I(T1) - 1.0/T2*J(T2)/I(T2) - kB*np.log(1.0*T2**(3.0/2)*I(T2)/(T1**(3.0/2)*I(T1)))


# PLOT
starttime = time.perf_counter()
diagnose = False
acc = 90
z0= -90*um
g = 0 # simulate without gravity

do_batchplots = True

if (do_batchplots):
    # plot heating ratios for a range of dipole beam powers (or coil currents, etc.)
    T1=10*uK
    x = np.linspace(-300 * um, 300 * um, acc)
    y = np.linspace(-300 * um, 300 * um, acc)
    z = np.linspace(-300 * um + z0, 300 * um + z0, acc)
    eq_temperature = np.vectorize(eq_temperature)
    T2s = np.array([])
    for Pdipole in np.arange(0.7, 12, 1):  # Icoil in np.arange(20, 110, 10)
        T_guess = T1 * 1.2  # if the solution does not properly converge, tweak this
        T_sol, = fsolve(eq_temperature, T_guess, args=(T1,))
        # print(eq_temperature(T_sol, T1)) # check that this is reasonably close to zero
        if (eq_temperature(T_sol, T1) > 1e-30): T_sol = 'err'
        print('Pdipole={0:1.0f}A, T1={1:1.5f} uK, T2={2:1.5f} uK, ratio={3:1.5f}'.format(Pdipole, T1 / uK, T_sol / uK, T_sol / T1))
        T2s = np.append(T2s, T_sol / T1)
    print(T2s)
    plt.scatter(np.arange(0.7, 12, 1), T2s)
    plt.ylabel('heating $T_2/T_1$')
    plt.xlabel('dipole beam power (W)')
    plt.title('$I_{coil}$=27 A')
    plt.xlim((0,12))

    plt.title('Complete cycle, $x_0={0:1.0f}\mu$m, $P_d={1:1.0f}$W, runtime {2:1.1f}s'.format(z0/um, Pdipole, stoptime - starttime))
    plt.legend()
    plt.show()

else:
    # plot an overview of the potentials involved
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13, 5))

    x = np.arange(-2000*um, 2000*um, 5*um)
    ax1.plot(x/um, 1/kB*Vinit(x, 0, z0)/uK, label="initial")
    ax1.plot(x/um, 1/kB*Vfin(x, 0, z0)/uK, label="final")
    ax1.set_xlabel('x ($\mu m$)')
    ax1.set_ylabel('V ($k_B \; \mu K$)')
    ax1.set_ylim((-200, 100))
    ax1.legend()

    z = np.arange(-300*um, 300*um, 8*um)
    ax2.plot(z/um, 1/kB*Vinit(0, 0, z)/uK)
    ax2.plot(z/um, 1/kB*Vfin(0, 0, z)/uK)
    ax2.set_ylim((-200, 100))
    ax2.set_xlabel('z ($\mu m$)')
    ax2.tick_params(labelleft=False)

    y = np.arange(-300*um, 300*um, 8*um)
    ax3.plot(y/um, 1/kB*Vinit(0, y, z0)/uK, label="initial")
    ax3.plot(y/um, 1/kB*Vfin(0, y, z0)/uK, label="final")
    ax3.set_ylim((-200, 100))
    ax3.set_xlabel('y ($\mu m$)')
    ax3.tick_params(labelleft=False)

    plt.show()