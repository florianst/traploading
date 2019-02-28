from include.constants  import * # contains everything which is needed across files
from include.experiment import * # contains all fixed parameters of the simulated experiment

Pdipole = 4*W
Icoil   = 27*A
crossed = False

def V1(x, y, z):  # Initial Potential: only quadrupole. x, y, z all in m
    return mRb * g * z + mu * B(x, y, z, Icoil)

def V2(x, y, z):  # Intermediate Potential: dipole + quadrupole. x, y, z all in m
    return mRb * g * z + mu * B(x, y, z, Icoil) + RbDipolePotential(x, y, z - z0, Pdipole, 50 * um, 50 * um, crossed=crossed)

def V3(x, y, z):  # Final Potential: quadrupole trap switched off. x, y, z all in m
    return mRb * g * z                          + RbDipolePotential(x, y, z - z0, Pdipole, 50 * um, 50 * um, crossed=crossed)

Vinit = V1
Vfin  = V3

# ADIABATIC TEMPERATURE CHANGE
def I_integrand(x, y, z, Tact): # x, y, z all in m, T in K
    if (Tact==T1): return np.exp(-1.0/(kB*Tact)*Vinit(x, y, z))
    else:          return np.exp(-1.0/(kB*Tact)*Vfin(x, y, z))

def I(T):
    acc = 100 # grain of the sampling space (number of points)
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


# TRANSFER EFFICIENCY
acc = 90        # accuracy (number of data points for numerical integration between llim and ulim). For typical potentials, acc/llim ~ 0.15 seems to be a reasonable ratio

def f(E):
    return 2.0*np.sqrt(1.0*E/pi)*(kB*T)**(-3.0/2)*np.exp(-E/(kB*T))

def normalisation_integrand(x, y, z, T): # x, y, z all in m, T in K
    return np.exp(-Vinit(x, y, z)/(kB*T))

def n(x, y, z, normalisation):
    return np.exp(-Vinit(x, y, z)/(kB*T))/normalisation

def frac_below(x, y, z):
    # have to create separate linspaces - np.linspace doesn't work for array arguments (is not vectorized)!
    E = np.zeros((acc, acc, acc, 200))
    for zindex, zval in enumerate(z):
        for yindex, yval in enumerate(y):
            for xindex, xval in enumerate(x):
                if (-(Vfin(xval[0][0], yval[0], zval)) > 0.0): # only integrate where Vfin < 0 so as to avoid negative energies
                    E[xindex, yindex, zindex] = np.linspace(0.0, min([15*kB*T,-(Vfin(xval[0][0], yval[0], zval))]), 200)  # in order to increase performance, reduce the integration range for narrow distributions (low T)

    integral = trapz(f(E), E)
    return integral

def eta_integrand(x, y, z, normalisation):
    return n(x, y, z, normalisation)*frac_below(x, y, z)

def eta(T1):
    # normalisation is T-dependent, if we don't want to recalculate it for every single (x, y, z) we have to hand it down to the n function manually
    global normalisation, T, x, y, z

    # take into account the adiabatic T change happening between the potentials (see Temperature.py)
    temptime = time.perf_counter()
    T_guess = T1 * 0.5  # if the solution does not properly converge, tweak this
    T, = fsolve(eq_temperature, T_guess, args=(T1,))
    print('Adiabatic V change from T1={0:1.4f}uK to T={1:1.4f}uK, calculation took {2:1.1f}s'.format(T1 / uK, T / uK, time.perf_counter() - temptime))

    # now calculate the transfer efficiency at this temperature
    temptime = time.perf_counter()
    normalisation = trapz(trapz(trapz(normalisation_integrand(x[:, None][:, None], y[:, None], z, T), z), y), x)

    if (diagnose):
        # plot potentials, density destributions and energy distribution for diagnosis purposes
        fig, ((ax1, ax2, ax3, ax0), (ax4, ax5, ax6, axE)) = plt.subplots(2, 4, figsize=(13, 5))

        xplot = np.arange(-2000*um, 2000*um, 5*um)
        ax1.plot(xplot/um, 1/kB*Vinit(xplot, 0, z0)/uK, label="initial")
        ax1.plot(xplot/um, 1/kB*Vfin(xplot, 0, z0)/uK, label="final")
        ax1.set_xlabel('x ($\mu m$)')
        ax1.set_ylabel('V ($k_B \; \mu K$)')
        ax1.set_ylim((-200, 100))
        ax1.legend()
        ax4.plot(xplot/um, n(xplot, 0, z0, normalisation))
        ax4.set_xlabel('x ($\mu m$)')
        ax4.set_ylabel('n')

        zplot = np.arange(-300*um, 300*um, 8*um)
        ax2.plot(zplot/um, 1/kB*Vinit(0, 0, zplot)/uK)
        ax2.plot(zplot/um, 1/kB*Vfin(0, 0, zplot)/uK)
        ax2.set_ylim((-200, 100))
        ax2.set_xlabel('z ($\mu m$)')
        ax2.tick_params(labelleft=False)
        ax5.plot(zplot/um, n(0, 0, zplot, normalisation))
        ax5.set_xlabel('z ($\mu m$)')
        ax5.set_ylabel('n')

        yplot = np.arange(-300*um, 300*um, 8*um)
        ax3.plot(yplot/um, 1/kB*Vinit(0, yplot, z0)/uK, label="initial")
        ax3.plot(yplot/um, 1/kB*Vfin(0, yplot, z0)/uK, label="final")
        ax3.set_ylim((-200, 100))
        ax3.set_xlabel('y ($\mu m$)')
        ax3.tick_params(labelleft=False)
        ax6.plot(yplot/um, n(0, yplot, z0, normalisation))
        ax6.set_xlabel('y ($\mu m$)')
        ax6.set_ylabel('n')

        En = np.arange(0,150*uK*kB, 1*uK*kB)
        axE.plot(En/uK/kB, f(En))
        axE.set_xlabel('f(E)')
        axE.set_ylabel('E (uK kB)')

        plt.show()

    returnval = trapz(trapz(trapz(eta_integrand(x[:,None][:,None], y[:,None], z, normalisation), z), y), x)
    print('Transfer efficiency eta={0:1.5f}, calculation took {1:1.1f}s'.format(returnval, time.perf_counter() - temptime))
    return returnval



# PLOT
starttime = time.perf_counter()
diagnose = False
do_batchplots = True

if (do_batchplots):
    # plot transfer efficiency depending on temperature for several dipole beam offsets z0
    for z0 in [-140*um, -90*um, -50*um, -0*um]:
        print('z0='+str(z0 / um)+'um, Pdipole = '+str(Pdipole)+'W')
        eq_temperature = np.vectorize(eq_temperature)
        etas = np.array([])
        x = np.linspace(-300 * um, 300 * um, acc)
        y = np.linspace(-300 * um, 300 * um, acc)
        z = np.linspace(-300 * um + z0, 300 * um + z0, acc)
        for T1 in np.arange(1*uK, 21*uK, 1.5*uK):
            etas = np.append(etas, eta(T1))
        stoptime = time.perf_counter()
        print(etas)
        plt.plot(np.arange(1, 21, 1.5), etas, '-o', label='$z_0={0:1.0f}\mu$m'.format(z0/um))
    plt.ylim(0.0, 1.0)
    plt.ylabel('Loading efficiency $\eta$')
    plt.xlabel('Initial Temperature $T_1$ ($\mu$K)')
    print('Complete run took %.2f s' % (stoptime - starttime))
    plt.title('Complete cycle, $x_0={0:1.0f}\mu$m, $P_d={1:1.0f}$W, runtime {2:1.1f}s'.format(z0/um, Pdipole, stoptime - starttime))
    plt.legend()
    plt.show()

else:
    # plot an overview of the potentials involved
    z0 = -50*um

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



