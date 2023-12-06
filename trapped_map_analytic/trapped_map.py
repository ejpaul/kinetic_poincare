import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar
from simsopt.field.boozermagneticfield import BoozerAnalytic
from simsopt.field.tracing import trace_particles_boozer, MaxToroidalFluxStoppingCriterion,MinToroidalFluxStoppingCriterion
from simsopt.util.constants import ALPHA_PARTICLE_MASS, FUSION_ALPHA_PARTICLE_ENERGY,ALPHA_PARTICLE_CHARGE
from os.path import exists

charge = ALPHA_PARTICLE_CHARGE
mass = ALPHA_PARTICLE_MASS
Ekin = FUSION_ALPHA_PARTICLE_ENERGY
v0 = np.sqrt(2*Ekin/mass)

B0 = 5.78895707
G0 = 80.3790491989699
iota0 = -1.1634045741388
etabar = -0.12987855
N = -4
psi0 = 6.79643
field = BoozerAnalytic(B0=B0,G0=G0,iota0=iota0,etabar=etabar,N=N,psi0=psi0,B0z=1e-3)
nfp = 4

sign_vpar = 1.0 # should be +/- 1. sign(vpar)
pitch_angle = 0.0 # lambda = v_perp^2/(v^2 B) = const. along trajectory
nzeta_poinc = 1 # Number of zeta initial conditions for poincare
ns_poinc = 100 # Number of s initial conditions for poincare
tol = 1e-8 # gc integration tolerance
Npts = 1000 # number of points for poincare plotting
tmax = 1e-2 # time limit for gc integration

chi_mirror = 1.2*np.pi/2 # chi = theta - helicity*nfp * zeta -> this determines mirror point
s_mirror = 0.6 # radial point for mirroring
zeta_mirror = 0 # toroidal angle for mirroring
helicity = -1 # helicity for symmetry. Should be +1, -1, or 0
pointsize = 0.2 # marker size for Poincare plot
leftbound = 0.1*np.pi # left boundary for chi_mirror root solve
rightbound = 0.9*np.pi # right boundary for chi_mirror root

def trapped_map(point):
    """
    Integates the gc equations from one mirror point to the next mirror point.
    point contains the [s, chi, zeta] coordinates and returns the same coordinates
    after mapping.
    """
    points = np.zeros((1,3))
    points[:,0] = point[0]
    points[:,1] = point[1] + nfp * helicity * point[2]
    points[:,2] = point[2]

    gc_tys, gc_zeta_hits = trace_particles_boozer(field, points, [0], tmax=tmax, mass=mass, charge=charge,
            Ekin=Ekin, zetas=[], vpars=[0], tol=tol, stopping_criteria=[MinToroidalFluxStoppingCriterion(0.01),MaxToroidalFluxStoppingCriterion(1.0)],
            forget_exact_path=False,mode='gc_vac',vpars_stop=True,zetas_stop=False)
    for i in range(len(gc_zeta_hits[0])):
        if gc_zeta_hits[0][i,1]>=0:
            # If idx=0, then vpars was hit
            # If `idx<0`, then `stopping_criteria[int(-idx)-1]` was hit.
            point[0] = gc_zeta_hits[0][i,2]
            point[1] = gc_zeta_hits[0][i,3] - nfp * helicity * gc_zeta_hits[0][i,4]
            point[2] = gc_zeta_hits[0][i,4]
            return point
        else:
            point[0] = -1.
            point[1] = -1.
            point[2] = -1.
            return point
    point[0] = -1.
    point[1] = -1.
    point[2] = -1.
    return point

points = np.zeros((1,3))
# Choose one mirror point
points[:,0] = s_mirror
points[:,1] = chi_mirror + helicity * nfp * zeta_mirror
points[:,2] = zeta_mirror

field.set_points(points)
modBcrit = field.modB()[0,0]

# We now compute all of the chi mirror points for each s and zeta
def chi_mirror_func(s,zeta):
    points[:,0] = s
    points[:,2] = zeta
    points[:,1] = chi_mirror + zeta*nfp*helicity # Initial guess for bounce point
    # Peform optimization to find bounce point
    def diffmodB(chi):
        return (modB_func(chi)-modBcrit)
    def graddiffmodB(chi):
        points[:,1] = chi + zeta*nfp*helicity
        field.set_points(points)
        return field.dmodBdtheta()[0,0]
    def modB_func(chi):
        points[:,1] = chi + zeta*nfp*helicity
        field.set_points(points)
        return field.modB()[0,0]
        # Make sure that bracket has different signs
    leftvalue = diffmodB(leftbound)
    rightvalue = diffmodB(rightbound)
    if (np.sign(leftvalue) == np.sign(rightvalue)):
        print('s = ',s)
        print('zeta/(2*pi/nfp) = ',zeta/(2*np.pi/nfp))
        print('leftvalue: ',leftvalue)
        print('rightvalue: ',rightvalue)
        print('Unable to bracket the root')
        chis_plot = np.linspace(leftbound,rightbound,100)
        return -1
    sol = root_scalar(diffmodB,fprime=graddiffmodB,bracket=(leftbound,rightbound))
    if (np.abs(diffmodB(sol.root))<1e-10):
        return sol.root
    else:
        print('root value: ',np.abs(diffmodB(sol.root)))
        print('Error occured in finding mirror point!')
        print('s = ',s)
        print('zeta = ',zeta)
        return -1

zetas = np.linspace(0,2*np.pi/nfp,nzeta_poinc)
s = np.linspace(0.1,0.9,ns_poinc)
zetas2d,s2d = np.meshgrid(zetas,s)
zetas2d = zetas2d.flatten()
s2d = s2d.flatten()

chis2d = []
s2d_all = []
zetas2d_all = []
# For each point, find the mirror point in chi
for i in range(len(zetas2d.flatten())):
    chi = chi_mirror_func(s2d.flatten()[i],zetas2d.flatten()[i])
    if (chi != -1):
        chis2d.append(chi)
        s2d_all.append(s2d.flatten()[i])
        zetas2d_all.append(zetas2d.flatten()[i])

s2d = s2d_all 
zetas2d = zetas2d_all

Ntrj = len(chis2d)

"""
We now iterate over the Poincare initial conditions and perform the mapping.
"""
chis_all = []
zetas_all = []
s_all = []
freq_all = []
rad_all = []
for itrj in range(Ntrj):
    chis_filename = 'chis_poinc_'+str(itrj)+'.txt'
    s_filename = 's_poinc_'+str(itrj)+'.txt'
    zetas_filename = 'zetas_poinc_'+str(itrj)+'.txt'
    if (exists(chis_filename) and exists(s_filename)):
        chis_plot = np.loadtxt(chis_filename)
        zetas_plot = np.loadtxt(zetas_filename)
        s_plot = np.loadtxt(s_filename)
        if chis_plot.size < 2:
            chis_plot = np.array([chis_plot])
            zetas_plot = np.array([zetas_plot])
            s_plot = np.array([s_plot])
    else:
        tr = [s2d[itrj],chis2d[itrj],zetas2d[itrj]]
        chis_plot = []
        zetas_plot = []
        s_plot = []
        s_plot.append(tr[0])
        chis_plot.append(tr[1])
        zetas_plot.append(tr[2])
        for jj in range(Npts):
            tr = trapped_map(tr)
            if tr[0] != -1:
                tr = trapped_map(tr) # apply map twice to get upper bounce point
                if tr[0] != -1:
                    s_plot.append(tr[0])
                    chis_plot.append(tr[1])
                    zetas_plot.append(tr[2])
                else:
                    break
            else:
                break
        np.savetxt(chis_filename,chis_plot)
        np.savetxt(s_filename,s_plot)
        np.savetxt(zetas_filename,zetas_plot)

    if (len(zetas_plot)>1):
        delta_zeta = (np.asarray(zetas_plot[1::])-np.asarray(zetas_plot[0:-1]))/(2*np.pi/nfp)
        freq_all.append(np.mean(np.asarray(delta_zeta))) # Compute effective frequency of each orbit
        rad_all.append(s2d[itrj])
    chis_all.append(chis_plot)
    zetas_all.append(zetas_plot)
    s_all.append(s_plot)

    plt.figure(1)
    plt.xlabel(r'$\zeta$')
    plt.ylabel(r'$s$')
    plt.xlim([0,2*np.pi/nfp])
    plt.ylim([0,1])
    for i in range(len(chis_all)):
        plt.scatter(np.mod(zetas_all[i],2*np.pi/nfp), s_all[i], marker='o',s=pointsize,edgecolors='none')
    plt.savefig('map_poincare.pdf')
