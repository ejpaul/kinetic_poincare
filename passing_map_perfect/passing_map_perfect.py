import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root, root_scalar
import sys
from simsopt._core.util import parallel_loop_bounds
from simsopt.util.mpi import MpiPartition
from simsopt.field.boozermagneticfield import BoozerRadialInterpolant, InterpolatedBoozerField
from simsopt.field.tracing import trace_particles_boozer, MaxToroidalFluxStoppingCriterion,MinToroidalFluxStoppingCriterion
from simsopt.mhd import Vmec
from simsopt.util.constants import ALPHA_PARTICLE_MASS, FUSION_ALPHA_PARTICLE_ENERGY,ALPHA_PARTICLE_CHARGE
import matplotlib
from scipy.interpolate import interp1d
from os.path import exists
from booz_xform import Booz_xform
import os

cm = plt.get_cmap('gist_rainbow')
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    mpi = MpiPartition(comm_world=comm)
except ImportError:
    comm = None
    mpi = None
import time

if comm is not None:
    if comm.rank == 0:
        verbose = True
    else:
        verbose = False
else:
    verbose = True

filename = 'wout_new_QH_aScaling.nc' # vmec filename - is read unless boozmn_filename exists
boozmn_filename = "boozmn.nc" # booz_xform filename

charge = ALPHA_PARTICLE_CHARGE
mass = ALPHA_PARTICLE_MASS
Ekin = FUSION_ALPHA_PARTICLE_ENERGY
v0 = np.sqrt(2*Ekin/mass)

pitch_angle = 0.0 # lambda = v_perp^2/(v^2 B) = const. along trajectory
nchi_poinc = 1 # Number of zeta initial conditions for poincare
ns_poinc = 120 # Number of s initial conditions for poincare
tol = 1e-10 # gc integration tolerance
Npts = 500 # number of points for poincare plotting
tmax = 1e-2 # time limit for gc integration
rescale = False # if True, vmec equilibrium is rescaled to ARIES-CS size and field strength
mpol = 48 # poloidal resolution for booz_xform
ntor = 48 # toroidal resolution for booz_xform
ns_interp = 30 # number of radial grid points for interpolation
ntheta_interp = 30 # number of poloidal grid points for interpolation
nzeta_interp = 30 # number of toroidal grid points for interpolation
order = 3 # order for interpolation
helicity = -1 # helicity for symmetry. Should be +1, -1, or 0
pointsize = 0.2 # marker size for Poincare plot
const_bool = True # Make plot to check that modB is constant along constant chi curve

# If boozmn.nc exists, read it
if exists(boozmn_filename):
    equil = Booz_xform()
    equil.verbose = 0
    equil.read_boozmn(boozmn_filename)
    if verbose:
        print('Read boozmn')
    nfp = equil.nfp
# Otherwise, we will generate from vmec
else:
    vmec = Vmec(filename)
    if rescale:
        vmec.run()
        # Rescale equilibrium
        Vt = 444.385920765916
        Bt = 5.86461234641553
        Vc = vmec.wout.volume_p
        Bc = vmec.wout.volavgB
        phic = vmec.wout.phi[-1]
        boundary = vmec._boundary

        dofs = boundary.get_dofs()
        dofs *= (Vt/Vc)**(1/3)

        boundary.set_dofs(dofs)
        phic *= (Vt/Vc)**(2/3) * (Bt/Bc)
        vmec.indata.phiedge = phic
        vmec.run()
    equil = vmec
    nfp = equil.wout.nfp

time1 = time.time()

# Compute min and max B as a function of radius
bri = BoozerRadialInterpolant(equil,order,mpol=mpol,ntor=ntor,no_K=True,mpi=mpi,
                              verbose=verbose,N=helicity*nfp)

time2 = time.time()
if verbose:
    print('BRI time: ',time2-time1)

time1 = time.time()

# Compute min and max B as a function of radius
nfp = bri.nfp
degree = 3
srange = (0, 1, ns_interp)
thetarange = (0, np.pi, ntheta_interp)
zetarange = (0, 2*np.pi/nfp, nzeta_interp)
field = InterpolatedBoozerField(bri, degree, srange, thetarange, zetarange, True, nfp=nfp, stellsym=True)

time2 = time.time()
if verbose:
    print('IBF time: ',time2-time1)

# Plot modB as a funtion of the symmetry angle. This is a check that helicity is correct.
if const_bool:
    zetas_grid = np.linspace(0,2*np.pi/nfp,100)
    points = np.zeros((len(zetas_grid),3))
    points[:,0] = 0.5
    points[:,1] = nfp*helicity*zetas_grid
    points[:,2] = zetas_grid
    field.set_points(points)
    plt.figure()
    plt.plot(zetas_grid,field.modB()[:,0])
    plt.xlabel(r'$\zeta$')
    plt.ylabel(r'$B$')
    plt.savefig('constB_perfect.png')

def passing_map(point):
    """
    Integates the gc equations from one mirror point to the next mirror point.
    point contains the [s, chi, vpar] coordinates and returns the same coordinates
    after mapping.
    """
    points = np.zeros((1,3))
    points[:,0] = point[0]
    points[:,1] = point[1]
    points[:,2] = 0
    gc_tys, gc_zeta_hits = trace_particles_boozer(field, points, [point[2]], tmax=tmax, mass=mass, charge=charge,
            Ekin=Ekin, zetas=[0], vpars=[0], omegas=[0], tol=tol, stopping_criteria=[MinToroidalFluxStoppingCriterion(0.01),MaxToroidalFluxStoppingCriterion(1.0)],
            forget_exact_path=False,mode='gc_vac',vpars_stop=True,zetas_stop=True)
    for i in range(len(gc_zeta_hits[0])):
        # The second condition excludes trapped particles
        if (gc_zeta_hits[0][i,1]>=0 and ((np.mod(gc_zeta_hits[0][i,4],2*np.pi)<1e-10) or (np.abs(np.mod(gc_zeta_hits[0][i,4],2*np.pi)-2*np.pi)<1e-10))):
            # If idx=0, then zetas or vpars was hit
            # If `idx<0`, then `stopping_criteria[int(-idx)-1]` was hit.
            point[0] = gc_zeta_hits[0][i,2]
            point[1] = gc_zeta_hits[0][i,3]
            point[2] = gc_zeta_hits[0][i,5]
            return point
        else:
            print('vpar: ',gc_zeta_hits[0][i,5])
            print('zeta: ',np.mod(gc_zeta_hits[0][i,4],2*np.pi))
            point[0] = -1.
            point[1] = -1.
            point[2] = -1.
            return point
    point[0] = -1.
    point[1] = -1.
    point[2] = -1.
    return point

"""
Function to compute vparallel given (s,chi)
"""
def vpar_func(s,chi):
    point = np.zeros((1,3))
    point[0,0] = s
    point[0,1] = chi
    field.set_points(point)
    modB = field.modB()[0,0]
    # Skip any trapped particles
    if (1 - pitch_angle*modB < 0):
        return None
    else:
        return v0*np.sqrt(1 - pitch_angle*modB)

s = np.linspace(0.1,0.9,ns_poinc)
chis = np.linspace(0,2*np.pi,nchi_poinc)
s, chis = np.meshgrid(s,chis)
s = s.flatten()
chis = chis.flatten()

first, last = parallel_loop_bounds(comm, len(s))
# For each point, find value of vpar such that lambda = vperp^2/(v^2 B)
chis_all = []
s_all = []
vpar_all = []
for i in range(first,last):
    vpar = vpar_func(s[i],chis[i])
    if vpar is not None:
        s_all.append(s[i])
        chis_all.append(chis[i])
        vpar_all.append(vpar)

if comm is not None:
    chis = [i for o in comm.allgather(chis_all) for i in o]
    s = [i for o in comm.allgather(s_all) for i in o]
    vpar = [i for o in comm.allgather(vpar_all) for i in o]

time2 = time.time()
if verbose:
    print('chi_mirror time: ',time2-time1)

Ntrj = len(chis)

time1 = time.time()

"""
We now iterate over the Poincare initial conditions and perform the mapping.
"""
chis_all = []
vpars_all = []
s_all = []
freq_all = []
rad_all = []
first, last = parallel_loop_bounds(comm, Ntrj)
for itrj in range(first,last):
    chis_filename = 'chis_poinc_perfect_'+str(itrj)+'.txt'
    s_filename = 's_poinc_perfect_'+str(itrj)+'.txt'
    vpars_filename = 'vpars_poinc_perfect_'+str(itrj)+'.txt'
    if (exists(chis_filename) and exists(s_filename)):
        chis_plot = np.loadtxt(chis_filename)
        vpars_plot = np.loadtxt(vpars_filename)
        s_plot = np.loadtxt(s_filename)
        if chis_plot.size < 2:
            chis_plot = np.array([chis_plot])
            vpars_plot = np.array([vpars_plot])
            s_plot = np.array([s_plot])
    else:
        tr = [s[itrj],chis[itrj],vpar[itrj]]
        chis_plot = []
        vpars_plot = []
        s_plot = []
        s_plot.append(tr[0])
        chis_plot.append(tr[1])
        vpars_plot.append(tr[2])
        for jj in range(Npts):
            tr = passing_map(tr)
            if tr[0] != -1:
                s_plot.append(tr[0])
                chis_plot.append(tr[1])
                vpars_plot.append(tr[2])
            else:
                break
        np.savetxt(chis_filename,chis_plot)
        np.savetxt(s_filename,s_plot)
        np.savetxt(vpars_filename,vpars_plot)

    if (len(vpars_plot)>1):
        delta_chi = (np.asarray(chis_plot[1::])-np.asarray(chis_plot[0:-1]))/(2*np.pi)
        freq_all.append(np.mean(np.asarray(delta_chi))) # Compute effective frequency of each orbit
        rad_all.append(s[itrj])
    chis_all.append(chis_plot)
    vpars_all.append(vpars_plot)
    s_all.append(s_plot)

if comm is not None:
    freq_all = [i for o in comm.allgather(freq_all) for i in o]
    rad_all = [i for o in comm.allgather(rad_all) for i in o]

if comm is not None:
    chis_all = [i for o in comm.allgather(chis_all) for i in o]
    vpars_all = [i for o in comm.allgather(vpars_all) for i in o]
    s_all = [i for o in comm.allgather(s_all) for i in o]
if verbose:
    plt.figure(1)
    plt.xlabel(r'$\chi$')
    plt.ylabel(r'$s$')
    plt.xlim([0,2*np.pi])
    plt.ylim([0,1])
    for i in range(len(chis_all)):
        plt.scatter(np.mod(chis_all[i],2*np.pi), s_all[i], marker='o',s=pointsize,edgecolors='none')
    plt.savefig('map_poincare_perfect.pdf')

time2 = time.time()
if verbose:
    print('poincare time: ',time2-time1)

if verbose:
    plt.figure()
    plt.plot(rad_all, freq_all)
    plt.xlabel(r'$s$')
    plt.ylabel(r'$\omega$')
    plt.savefig('frequency_perfect.png')

    np.savetxt('omega_perfect.txt',freq_all)
    np.savetxt('rad_perfect.txt',rad_all)
