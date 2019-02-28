from include.constants  import * # contains everything which is needed across files

# SET UP THE POTENTIALS
a = 0.03*m  # coil radius
b = 31.5*mm # half of coil separation
m = 75      # number of coil windings
g = 9.81
mu = (e*hbar)/(2*me)
B0 = 0
for w in range(1, m+1): # sum over single coil windings
    B0 = B0 + (3*mu0*(30*mm + w*0.6*mm)**2*b)/(2*((30*mm + w*0.6*mm)**2 + b**2)**(5.0/2))

def B(x, y, z, I): # magnetic field value at a given position x,y,z m in space for a coil current of I Ampere
    return 1.0*I*B0*np.sqrt(z**2 + y**2 + 4*x**2)


# SET UP THE POTENTIALS - beams
def wh(z, w0h): # Gaussian waist evolution for horizontal waist
    return w0h*np.sqrt(1 + (z/(pi*w0h**2/lambda_Dipole))**2)

def wv(z, w0v): # Gaussian waist evolution for vertical waist
    return w0v*np.sqrt(1 + (z/(pi*w0v**2/lambda_Dipole))**2)

def SingleRbDipole(h, v, z, P, w0h, w0v): # Potential created by single dipole beam
    return URbDipole0*(2*P)/(pi*wh(z, w0h)*wv(z, w0v))*np.exp(-((2*h**2)/wh(z, w0h)**2) - (2*v**2)/wv(z, w0v)**2)

def RbDipolePotential(x, y, z, P, w0h, w0v, crossed=False):
    if (crossed): P=P/2
    return SingleRbDipole(z, y, x, P, w0h, w0v)+(SingleRbDipole(x, y, z, P, w0h, w0v) if crossed else 0)
