import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton, root, minimize, Bounds, root_scalar
import sys
from simsopt._core.util import parallel_loop_bounds
from simsopt.util.mpi import MpiPartition
from simsopt.field.boozermagneticfield import BoozerRadialInterpolant, InterpolatedBoozerField
from simsopt.field.tracing import trace_particles_boozer, ZetaStoppingCriterion, MaxToroidalFluxStoppingCriterion,MinToroidalFluxStoppingCriterion,VparStoppingCriterion,ToroidalTransitStoppingCriterion
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

filename = 'wout_new_QH_aScaling.nc'
boozmn_filename = "boozmn.nc"

charge = ALPHA_PARTICLE_CHARGE
mass = ALPHA_PARTICLE_MASS
Ekin = FUSION_ALPHA_PARTICLE_ENERGY
v0 = np.sqrt(2*Ekin/mass)

nzeta_poinc = 1 # initial cond for poincare
ns_poinc = 120 # initial cond for poincare
tol = 1e-10 # integration tolerance
Npts = 500 # points for poincare plotting
tmax = 1e-2 # time limit for integration
perisland = 4 # Number of points per island
mmax = 20 # maximum mode for island
nmax = 20 # maximum mode for island
delta_freq = 0.0 # buffer in freq for rational values to search for
NUM_COLORS = 30 # number of colors to plot
vpar_sign = 1
radius_max = 0.01
radius_min = 0.99
xtol = 1e-4
ftol = 1e-4
rescale = False
factor = 0.1
markevery = 1 # markevery n points for
mpol = 48 # poloidal resolution for booz_xform
ntor = 48 # toroidal resolution for booz_xform
ns_interp = 30 # number of radial grid points for interpolation
ntheta_interp = 30 # number of poloidal grid points for interpolation
nzeta_interp = 30 # number of toroidal grid points for interpolation
order = 3 # order for interpolation
chi_mirror = np.pi/2 # chi = theta - helicity*nfp * zeta
s_mirror = 0.6 # radial point for mirroring
zeta_mirror = 0 # toroidal angle for mirroring
helicity = 1 # helicity for symmetry. Should be +1 or -1
markersize = 0.2 # marker size for Poincare plot

# If boozmn.nc exists, read it
if exists(boozmn_filename):
    equil = Booz_xform()
    equil.verbose = 0
    equil.read_boozmn(boozmn_filename)
    if verbose:
        print('Read boozmn')
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

time1 = time.time()

bri = BoozerRadialInterpolant(equil,order,mpol=mpol,ntor=ntor,no_K=True,mpi=mpi,verbose=verbose)

time2 = time.time()
if verbose:
    print('BRI time: ',time2-time1)

time1 = time.time()

nfp = bri.nfp
degree = 3
srange = (0, 1, ns_interp)
thetarange = (0, np.pi, ntheta_interp)
zetarange = (0, 2*np.pi/nfp, nzeta_interp)
field = InterpolatedBoozerField(bri, degree, srange, thetarange, zetarange, True, nfp=nfp, stellsym=True)

time2 = time.time()
if verbose:
    print('IBF time: ',time2-time1)

def trapped_map(point):
    """ Point contains the [s, chi, zeta] coordinates and returns the same coordinates
        after mapping """
    if point[0] != -1.:
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
        print('error!')
    return point

points = np.zeros((1,3))
# Choose one mirror point
points[:,0] = s_mirror
points[:,1] = chi_mirror + helicity * nfp * zeta_mirror
points[:,2] = zeta_mirror

field.set_points(points)
modB = field.modB()[0,0]
mu = (Ekin/mass)/modB

zetas = np.linspace(0,2*np.pi/nfp,nzeta_poinc)
s = np.linspace(0.1,0.9,ns_poinc)
zetas2d,s2d = np.meshgrid(zetas,s)
zetas2d = zetas2d.flatten()
s2d = s2d.flatten()
def chi_mirror_func(s,zeta):
    points[:,0] = s
    points[:,2] = zeta
    points[:,1] = chi_mirror + zeta*nfp*helicity # Initial guess for bounce point
    # Peform optimization to find bounce point
    def diffmodB(chi):
        return (modB_func(chi)-modB)
    def graddiffmodB(chi):
        points[:,1] = chi + zeta*nfp*helicity
        field.set_points(points)
        return field.dmodBdtheta()[0,0]
    def modB_func(chi):
        points[:,1] = chi + zeta*nfp*helicity
        field.set_points(points)
        return field.modB()[0,0]
    try:
        sol = root_scalar(diffmodB,fprime=graddiffmodB,bracket=(0.01*np.pi,0.99*np.pi))
    except:
        print('Error occured in finding mirror point!')
        print('s = ',s)
        print('zeta = ',zeta)
        return -1
    if (np.abs(diffmodB(sol.root))<1e-10):
        return sol.root
    else:
        print('root value: ',np.abs(diffmodB(sol.root)))
        print('Error occured in finding mirror point!')
        print('s = ',s)
        print('zeta = ',zeta)
        return -1

time1 = time.time()

chis2d = []
first, last = parallel_loop_bounds(comm, len(zetas2d.flatten()))
# For each point, find the mirror point in theta
for i in range(first,last):
    chi = chi_mirror_func(s2d.flatten()[i],zetas2d.flatten()[i])
    if (chi != -1):
        chis2d.append(chi)

if comm is not None:
    chis2d = [i for o in comm.allgather(chis2d) for i in o]

time2 = time.time()
if verbose:
    print('theta_mirror time: ',time2-time1)

Ntrj = len(chis2d)

time1 = time.time()

chis_all = []
zetas_all = []
s_all = []
freq_all = []
rad_all = []
first, last = parallel_loop_bounds(comm, Ntrj)
for itrj in range(first,last):
    chis_filename = 'chis_poinc_'+str(itrj)+'.txt'
    s_filename = 's_poinc_'+str(itrj)+'.txt'
    zetas_filename = 'zetas_poinc_'+str(itrj)+'.txt'
    if (exists(chis_filename) and exists(s_filename)):
        chis_plot = np.loadtxt(chis_filename)
        zetas_plot = np.loadtxt(zetas_filename)
        s_plot = np.loadtxt(s_filename)
    else:
        tr = [s2d[itrj],chis2d[itrj],zetas2d[itrj]]
        chis_plot = []
        zetas_plot = []
        s_plot = []
        for jj in range(Npts):
            tr = trapped_map(tr)
            tr = trapped_map(tr) # apply map twice to get upper bounce point
            chis_plot.append(tr[1])
            s_plot.append(tr[0])
            zetas_plot.append(tr[2])
        np.savetxt(chis_filename,chis_plot)
        np.savetxt(s_filename,s_plot)
        np.savetxt(zetas_filename,zetas_plot)

    zetas_plot = np.array(zetas_plot)
    delta_zeta = (zetas_plot[1::]-zetas_plot[0:-1])/(2*np.pi/nfp)
    if (np.all(delta_zeta!=0)):
        freq_all.append(np.mean(delta_zeta)) # Compute effective frequency of each orbit
        rad_all.append(s2d[itrj])
        chis_all.append(chis_plot)
        zetas_all.append(zetas_plot)
        s_all.append(s_plot)

if comm is not None:
    chis_all = [i for o in comm.allgather(chis_all) for i in o]
    zetas_all = [i for o in comm.allgather(zetas_all) for i in o]
    s_all = [i for o in comm.allgather(s_all) for i in o]
    freq_all = [i for o in comm.allgather(freq_all) for i in o]
    rad_all = [i for o in comm.allgather(rad_all) for i in o]
if verbose:
    plt.figure(1)
    plt.xlabel(r'$\zeta$')
    plt.ylabel(r'$s$')
    plt.xlim([0,2*np.pi/nfp])
    plt.ylim([0,1])
    for i in range(len(chis_all)):
        plt.scatter(np.mod(zetas_all[i],2*np.pi/nfp), s_all[i], marker='o',s=markersize,edgecolors='none')
    plt.savefig('map_poincare.pdf')

time2 = time.time()
if verbose:
    print('poincare time: ',time2-time1)

freq_min = np.min(freq_all)-delta_freq
freq_max = np.max(freq_all)+delta_freq

fpp = []
fqq = []
ratio = []
for im in range(1,mmax):
    for jn in range(nmax):
        if (jn/im > freq_min and jn/im < freq_max and np.any(np.isclose(ratio,jn/im))==0):
            fpp.append(np.sign(freq_all[0])*jn)
            fqq.append(im)
            ratio.append(jn/im)

if verbose:
    plt.figure(2)
    plt.plot(rad_all, freq_all)
    for im in range(len(ratio)):
        plt.axhline(ratio[im],color='black',linestyle='--')
    plt.xlabel(r'$s$')
    plt.ylabel(r'$\omega$')
    plt.savefig('frequency.png')

point = np.zeros((3,))

def pseudomap(angleradiusnormal):
    point = np.zeros((3,))
    angleradius = angleradiusnormal[0:2]
    s_point = angleradius[1]
    zeta_point = angleradius[0]
    if (np.all(angleradius!=-1)):
        chi_point = chi_mirror_func(s_point,zeta_point)
        if (chi_point == -1):
            return angleradiusnormal
        point[0] = max(s_point,0.01) # Ensure that we don't cross through 0
        point[1] = chi_point
        point[2] = zeta_point
        point = trapped_map(point)
        if (np.any(point == -1)):
            return angleradiusnormal
        point[0] = max(point[0],0.01) # Ensure that we don't cross through 0
        point = trapped_map(point)
        if (np.any(point == -1)):
            return angleradiusnormal
        angleradius[0] = point[2]
        angleradius[1] = point[0]
        # Add nu to r
        angleradius[1] = angleradius[1] +  angleradiusnormal[2]
        # nu does not change
        angleradiusnormal = [ angleradius[0], angleradius[1], angleradiusnormal[2] ]
    return angleradiusnormal

def qqpseudomap(radiusnormal, theta, pp, qq):
    radiusnormalorig = [ theta, radiusnormal[0], radiusnormal[1] ]
    radiusnormalnew = radiusnormalorig
    for ii in range(qq): # Perform qq pseodomaps
        radiusnormalnew = pseudomap(radiusnormalnew)
    # Return error between mapped theta and theta0 + 2*np.pi*p
    if radiusnormalnew[0] != -1:
        error = radiusnormalorig[0:2] + np.asarray([pp*2*np.pi/nfp,0.0]) - radiusnormalnew[0:2]
    else:
        error = np.array([radiusnormalnew[0],radiusnormalnew[1]])
    return error

freq_spline = interp1d(freq_all,rad_all,fill_value="extrapolate",bounds_error=False)
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

# Loop over rationals
first, last = parallel_loop_bounds(comm, len(fpp))
zetas_all = []
radius_all = []
nus_all = []
pp_all = []
qq_all = []
for i in range(first,last):
    pp = fpp[i]
    qq = fqq[i]
    zeta_filename = 'zeta_p_'+str(pp)+'_q_'+str(qq)+'.txt'
    nu_filename = 'nu_p_'+str(pp)+'_q_'+str(qq)+'.txt'
    radius_filename = 'radius_p_'+str(pp)+'_q_'+str(qq)+'.txt'
    read = False
    if (exists(zeta_filename) and exists(nu_filename) and exists(radius_filename) and os.path.getsize(zeta_filename)>0 and os.path.getsize(nu_filename)>0 and os.path.getsize(radius_filename)>0):
        zetas = np.loadtxt(zeta_filename)
        nus = np.loadtxt(nu_filename)
        radius = np.loadtxt(radius_filename)
        if (len(radius)>0):
            read = True
    if not read:
        radialguess = guess_radius(pp/qq)
        print('radial guess: '+str(radialguess))
        nuguess = 0
        zetas = []
        radius = []
        nus = []
        count = 0
        ii = 0
        while (ii < perisland):
            zeta = ii * 2*np.pi / ( nfp * qq * perisland ) # zeta for pseudo orbit
            rn = np.array([ radialguess, nuguess ])
            sol = root(qqpseudomap, rn, args=(zeta,pp,qq), method='hybr', options={'maxfev':1000, 'eps':1e-4, 'factor':factor, 'xtol':xtol})
            rn = sol.x
            if (sol.success and np.linalg.norm(sol.fun) < ftol):
                zetas.append(np.mod(zeta,2*np.pi/nfp))
                radius.append(rn[0])
                nus.append(rn[1])
                # Now apply the map qq times to complete the Poincare section
                angleradiusnormal = np.array([zetas[-1],radius[-1],nus[-1]])
                for iter in range(qq):
                    angleradiusnormal = pseudomap(angleradiusnormal)
                    angleradiusnormal[0] = np.mod(angleradiusnormal[0],2*np.pi/nfp)
                    zetas.append(angleradiusnormal[0])
                    radius.append(angleradiusnormal[1])
                    nus.append(angleradiusnormal[2])
                # Now add in zeta = 2*pi/nfp
                zetas.append(2*np.pi/nfp)
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
                    radialguess = guess_radius(np.abs(pp/qq),count=count)
                    nuguess = 0
                    continue
                else:
                    # If this one did not converge, don't try any more theta points
                    break
            count = 0
            ii += 1
        # Now sort the result by zeta
        zetas = np.array(zetas)
        nus = np.array(nus)
        radius = np.array(radius)
        indices = np.argsort(zetas)
        zetas = zetas[indices]
        nus = nus[indices]
        radius = radius[indices]
        np.savetxt(zeta_filename,zetas)
        np.savetxt(nu_filename,nus)
        np.savetxt(radius_filename,radius)

    radius = np.asarray(radius)
    nus = np.asarray(nus)
    zetas = np.asarray(zetas)
    if (len(radius)>0):
        zetas_all.append(zetas)
        nus_all.append(nus)
        radius_all.append(radius)
        pp_all.append(pp)
        qq_all.append(qq)

if comm is not None:
    zetas_all = [i for o in comm.allgather(zetas_all) for i in o]
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
            plt.plot(zetas_all[i],radius_all[i],markevery=markevery,label=str(int(pp_all[i]))+', '+str(int(qq_all[i])),linewidth=1.0,markersize=1)
        lgd = plt.legend(bbox_to_anchor=(1.0,1.0),ncol=2)
        plt.savefig('poincare_with_map.pdf',bbox_extra_artists=(lgd,), bbox_inches='tight')

time2 = time.time()
if verbose:
    print('pseudomap time: ',time2-time1)
