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

sign_vpar = 1.0 # should be +/- 1. sign(vpar)
pitch_angle = 0.0 # lambda = v_perp^2/(v^2 B) = const. along trajectory
nchi_poinc = 1 # Number of zeta initial conditions for poincare
ns_poinc = 120 # Number of s initial conditions for poincare
tol = 1e-10 # gc integration tolerance
Npts = 500 # number of points for poincare plotting
tmax = 1e-2 # time limit for gc integration
perisland = 4 # Number of points to plot per island
qmax = 80 # maximum mode for rational orbits p/q
pmax = 80 # maximum mode for rational orbit p/q
delta_freq = 0.0 # buffer in freq for rational values to search for
NUM_COLORS = 30 # number of colors to plot
radius_max = 0.99 # maximum radius to look for a given n/m frequency
radius_min = 0.01 # minimum radius to look for a given n/m frequency
xtol = 1e-4 # tolerance for root solve
ftol = 1e-4 # tolerance for root solve
rescale = False # if True, vmec equilibrium is rescaled to ARIES-CS size and field strength
factor = 0.1 # factor hyperparameter for root solve
markevery = 1 # markevery n points for Poincare plot
mpol = 48 # poloidal resolution for booz_xform
ntor = 48 # toroidal resolution for booz_xform
ns_interp = 30 # number of radial grid points for interpolation
ntheta_interp = 30 # number of poloidal grid points for interpolation
nzeta_interp = 30 # number of toroidal grid points for interpolation
order = 3 # order for interpolation
chi_mirror = 1.2*np.pi/2 # chi = theta - helicity*nfp * zeta -> this determines mirror point
s_mirror = 0.6 # radial point for mirroring
zeta_mirror = 0 # toroidal angle for mirroring
helicity = -1 # helicity for symmetry. Should be +1, -1, or 0
pointsize = 0.2 # marker size for Poincare plot
markersize = 3 # marker size for pseudomap curves
leftbound = 0.1*np.pi # left boundary for chi_mirror root solve
rightbound = 0.9*np.pi # right boundary for chi_mirror root
min_max_bool = True # Make plot to show min and max B as a function of radius
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
                              verbose=verbose)

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
    plt.savefig('constB.png')

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
        return sign_vpar*v0*np.sqrt(1 - pitch_angle*modB)

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
    chis_filename = 'chis_poinc_'+str(itrj)+'.txt'
    s_filename = 's_poinc_'+str(itrj)+'.txt'
    vpars_filename = 'vpars_poinc_'+str(itrj)+'.txt'
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

# Take omega.txt and rad.txt from "perfect" calculation
if exists('omega_perfect.txt') and exists('rad_perfect.txt'):
    if verbose:
        freq_all = np.loadtxt('omega_perfect.txt')
        rad_all = np.loadtxt('rad_perfect.txt')
    else:
        freq_all = None
        rad_all = None
    freq_all = comm.bcast(freq_all, root=0)
    rad_all = comm.bcast(rad_all, root=0)
else:
    if comm is not None:
        freq_all = [i for o in comm.allgather(freq_all) for i in o]
        rad_all = [i for o in comm.allgather(rad_all) for i in o]

if comm is not None:
    chis_all = [i for o in comm.allgather(chis_all) for i in o]
    s_all = [i for o in comm.allgather(s_all) for i in o]
if verbose:
    plt.figure(1)
    plt.xlabel(r'$\chi$')
    plt.ylabel(r'$s$')
    plt.xlim([0,2*np.pi])
    plt.ylim([0,1])
    for i in range(len(chis_all)):
        plt.scatter(np.mod(chis_all[i],2*np.pi), s_all[i], marker='o',s=pointsize,edgecolors='none')
    plt.savefig('map_poincare.pdf')

time2 = time.time()
if verbose:
    print('poincare time: ',time2-time1)

"""
Here we find all rationals p/q within frequency range.
"""
freq_min = np.min(freq_all)-delta_freq
freq_max = np.max(freq_all)+delta_freq
fpp = []
fqq = []
ratio = []
for iq in range(1,qmax):
    for jp in range(-pmax,pmax):
        # the last condition is to prevent multiple copies
        if (nfp*jp/iq > freq_min and nfp*jp/iq < freq_max and np.any(np.isclose(ratio,nfp*jp/iq))==0):
            fpp.append(nfp*jp)
            fqq.append(iq)
            ratio.append(nfp*jp/iq)

if verbose:
    np.savetxt('fpp.txt',np.asarray(fpp))
    np.savetxt('fqq.txt',np.asarray(fqq))
    plt.figure()
    plt.plot(rad_all, freq_all)
    for im in range(len(ratio)):
        plt.axhline(ratio[im],color='black',linestyle='--')
    plt.xlabel(r'$s$')
    plt.ylabel(r'$\omega$')
    plt.savefig('frequency.png')

point = np.zeros((3,))

"""
This returns the pseudomap, obtained by adding nu to the radial coordinate
returned from the true trapped map.
angle = chi
radius = s
normal = nu
radiusnormal = [s,nu]
"""
def pseudomap(angleradiusnormal):
    angleradius = angleradiusnormal[0:2]
    s_point = angleradius[1]
    chi_point = angleradius[0]
    if (angleradius[0]!=-1):
        vpar_point = vpar_func(s_point,chi_point)
        if (vpar_point is None):
            return angleradiusnormal
        point = np.zeros((3,))
        point[0] = max(s_point,0.01) # Ensure that we don't cross through 0
        point[1] = chi_point
        point[2] = vpar_point
        point = passing_map(point)
        if (point[0]<radius_min): # Ensure that we don't cross through 0 or -1 error raised
            return angleradiusnormal
        angleradius[0] = point[1]
        angleradius[1] = point[0]
        # Add nu to r
        angleradius[1] = angleradius[1] +  angleradiusnormal[2]
        # nu does not change
        angleradiusnormal = [ angleradius[0], angleradius[1], angleradiusnormal[2] ]
    return angleradiusnormal

"""
Here we seek a p/q periodic orbit from the pseudomap.
radiusnormal = [s,nu]
"""
def qqpseudomap(radiusnormal, chi, pp, qq):
    radiusnormalorig = [ chi, radiusnormal[0], radiusnormal[1] ]
    radiusnormalnew = radiusnormalorig
    for ii in range(qq): # Perform qq pseodomaps
        radiusnormalnew = pseudomap(radiusnormalnew)
    # Return error between mapped chi and chi0 + 2*np.pi*p
    if radiusnormalnew[0] != -1:
        error = radiusnormalorig[0:2] + np.asarray([pp*2*np.pi,0.0]) - radiusnormalnew[0:2]
    else:
        # An error ocurred -> return a large penalty
        error = np.asarray([1e3,1e3])
    return error

# We construct a spline in order to find a guess for the radius from p/q
freq_spline = interp1d(freq_all,rad_all,fill_value="extrapolate",bounds_error=False)

"""
Here we construct a guess for the radial location of a p/q orbit based on the
frequency profile. If count>0, a previous guess failed, so we instead just try
10 equally-spaced guesses.
"""
def guess_radius(freq_guess,count=0):
    if count > 0:
        guesses = np.linspace(0.1,0.9,10)
        guess = guesses[count]
        print('trying again with ',guess)
        return guess
    guess = freq_spline(freq_guess)
    if (guess<=radius_min):
        return radius_min
    if (guess>=radius_max):
        return radius_max
    return guess

marker_list = []
linestyle_list = []
for i in range(NUM_COLORS):
    if (i%4 == 0):
        marker_list.append('1')
        linestyle_list.append('--')
    elif (i%4 == 1):
        marker_list.append('|')
        linestyle_list.append('-')
    elif (i%4 == 2):
        marker_list.append('+')
        linestyle_list.append(':')
    elif (i%4 == 3):
        marker_list.append('X')
        linestyle_list.append('-.')

time1 = time.time()

"""
Loop over p/q orbits, and find the pseudomap (i.e., determine nu) for each
value of chi.
"""
first, last = parallel_loop_bounds(comm, len(fpp))
chis_all = []
radius_all = []
nus_all = []
pp_all = []
qq_all = []
for i in range(first,last):
    pp = fpp[i]
    qq = fqq[i]
    chi_filename = 'chi_p_'+str(pp)+'_q_'+str(qq)+'.txt'
    nu_filename = 'nu_p_'+str(pp)+'_q_'+str(qq)+'.txt'
    radius_filename = 'radius_p_'+str(pp)+'_q_'+str(qq)+'.txt'
    read = False
    if (exists(chi_filename) and exists(nu_filename) and exists(radius_filename) and os.path.getsize(chi_filename)>0 and os.path.getsize(nu_filename)>0 and os.path.getsize(radius_filename)>0):
        chis = np.loadtxt(chi_filename)
        nus = np.loadtxt(nu_filename)
        radius = np.loadtxt(radius_filename)
        if (len(radius)>0):
            read = True
    if not read:
        radialguess = guess_radius(pp/qq)
        print('radial guess: '+str(radialguess))
        nuguess = 0
        chis = []
        radius = []
        nus = []
        count = 0
        ii = 0
        while (ii < perisland):
            chi = ii * 2*np.pi / (qq * perisland ) # chi for pseudo orbit
            rn = np.array([ radialguess, nuguess ])
            sol = root(qqpseudomap, rn, args=(chi,pp,qq), method='hybr', options={'maxfev':1000, 'eps':1e-4, 'factor':factor, 'xtol':xtol})
            rn = sol.x
            if (sol.success and np.linalg.norm(sol.fun) < ftol):
                chis.append(np.mod(chi,2*np.pi))
                radius.append(rn[0])
                nus.append(rn[1])
                # Now apply the map qq times to complete the Poincare section
                angleradiusnormal = np.array([chis[-1],radius[-1],nus[-1]])
                for iter in range(int(qq)):
                    angleradiusnormal = pseudomap(angleradiusnormal)
                    angleradiusnormal[0] = np.mod(angleradiusnormal[0],2*np.pi)
                    chis.append(angleradiusnormal[0])
                    radius.append(angleradiusnormal[1])
                    nus.append(angleradiusnormal[2])
                # Now add in chi = 2*pi
                chis.append(2*np.pi)
                radius.append(radius[0])
                nus.append(nus[0])
                # Update radius and nu guesses
                radialguess = rn[0]
                nuguess = rn[1]
                count = 0
            else:
                print('count: ',count)
                count = count + 1
                if count < 10:
                    radialguess = guess_radius(pp/qq,count=count)
                    nuguess = 0
                    continue
                else:
                    # If this one did not converge, don't try any more theta points
                    break
            count = 0
            ii += 1
        # Now sort the result by chi
        chis = np.array(chis)
        nus = np.array(nus)
        radius = np.array(radius)
        indices = np.argsort(chis)
        chis = chis[indices]
        nus = nus[indices]
        radius = radius[indices]
        np.savetxt(chi_filename,chis)
        np.savetxt(nu_filename,nus)
        np.savetxt(radius_filename,radius)

    radius = np.asarray(radius)
    nus = np.asarray(nus)
    chis = np.asarray(chis)
    if (len(radius)>0):
        chis_all.append(chis)
        nus_all.append(nus)
        radius_all.append(radius)
        pp_all.append(pp)
        qq_all.append(qq)

if comm is not None:
    chis_all = [i for o in comm.allgather(chis_all) for i in o]
    nus_all = [i for o in comm.allgather(nus_all) for i in o]
    radius_all = [i for o in comm.allgather(radius_all) for i in o]
    pp_all = [i for o in comm.allgather(pp_all) for i in o]
    qq_all = [i for o in comm.allgather(qq_all) for i in o]
    if comm.rank == 0:
        f = open('summary.txt','w')
        plt.figure(1)
        ax = plt.gca()
        ax.set_prop_cycle(color=[cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)],marker=marker_list)
        for i in range(len(radius_all)):
            f.write('{}{}{}\n'.format(str(int(pp_all[i])).ljust(10),str(int(qq_all[i])).ljust(10),np.max(np.abs(nus_all[i]))))
            plt.plot(chis_all[i],radius_all[i],markevery=markevery,label=str(int(pp_all[i]))+', '+str(int(qq_all[i])),linewidth=1.0,markersize=markersize)
        lgd = plt.legend(bbox_to_anchor=(1.0,1.0),ncol=2)
        plt.savefig('poincare_with_pseudomap.pdf',bbox_extra_artists=(lgd,), bbox_inches='tight')

time2 = time.time()
if verbose:
    print('pseudomap time: ',time2-time1)
