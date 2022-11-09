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

filename = 'wout_mattqa1_000_000000.nc'
boozmn_filename = "boozmn.nc"

charge = ALPHA_PARTICLE_CHARGE
mass = ALPHA_PARTICLE_MASS
Ekin = FUSION_ALPHA_PARTICLE_ENERGY
v0 = np.sqrt(2*Ekin/mass)

nzeta_poinc = 10 # initial cond for poincare
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
markevery = 1

order = 3
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

bri = BoozerRadialInterpolant(equil,order,mpol=48,ntor=48,no_K=True,mpi=mpi)

time2 = time.time()
if verbose:
    print('BRI time: ',time2-time1)

time1 = time.time()

nfp = bri.bx.nfp
degree = 3
srange = (0, 1, 30)
thetarange = (0, np.pi, 30)
zetarange = (0, 2*np.pi/nfp, 30)
field = InterpolatedBoozerField(bri, degree, srange, thetarange, zetarange, True, nfp=nfp, stellsym=True)

time2 = time.time()
if verbose:
    print('IBF time: ',time2-time1)

# Compute Bmin and Bmax as a function of radius
thetas = np.linspace(0,2*np.pi,100)
s = np.linspace(0,1,101)
thetas2d,s2d = np.meshgrid(thetas,s)
points = np.zeros((len(thetas2d.flatten()),3))
points[:,0] = s2d.flatten()
points[:,1] = thetas2d.flatten()

time1 = time.time()

field.set_points(points)
modB = field.modB()[:,0].reshape(np.shape(thetas2d))
minB = np.min(modB,axis=1)
maxB = np.max(modB,axis=1)

time2 = time.time()
if verbose:
    print('min/max time: ',time2-time1)

points = np.zeros((1,3))
def trapped_map(point):
    if point[0] != -1.:
        points[:,0] = point[0]
        points[:,1] = point[1]
        points[:,2] = point[2]

        gc_tys, gc_zeta_hits = trace_particles_boozer(field, points, [0], tmax=tmax, mass=mass, charge=charge,
                Ekin=Ekin, zetas=[], vpars=[0], tol=tol, stopping_criteria=[MinToroidalFluxStoppingCriterion(0.01),MaxToroidalFluxStoppingCriterion(1.0)],
                forget_exact_path=False,mode='gc_vac',vpars_stop=True,zetas_stop=False)
        for i in range(len(gc_zeta_hits[0])):
            if gc_zeta_hits[0][i,1]>=0:
                # If idx=0, then vpars was hit
                # If `idx<0`, then `stopping_criteria[int(-idx)-1]` was hit.
                point[0] = gc_zeta_hits[0][i,2]
                point[1] = gc_zeta_hits[0][i,3]
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

# Choose one mirror point
points[:,0] = 0.6
points[:,1] = np.pi/2
points[:,2] = 0

field.set_points(points)
modB = field.modB()[0,0]
mu = (Ekin/mass)/modB

if verbose:
    plt.figure(0)
    plt.plot(s,minB)
    plt.plot(s,maxB)
    plt.axhline(modB)
    plt.xlabel(r'$s$')
    plt.ylabel(r'$B$')
    plt.xlim([0,1])

    plt.savefig('modB.png')

zetas = np.linspace(0,2*np.pi/nfp,nzeta_poinc)
s = np.linspace(0.1,0.9,ns_poinc)
zetas2d,s2d = np.meshgrid(zetas,s)
zetas2d = zetas2d.flatten()
s2d = s2d.flatten()
def theta_mirror(s,zeta):
    points[:,0] = s
    points[:,2] = zeta
    points[:,1] = np.pi/2 # Initial guess for bounce point
    # Peform optimization to find bounce point
    def diffmodB(theta):
        return (modB_func(theta)-modB)
    def graddiffmodB(theta):
        points[:,1] = theta
        field.set_points(points)
        return field.dmodBdtheta()[0,0]
    def modB_func(theta):
        points[:,1] = theta
        field.set_points(points)
        return field.modB()[0,0]
    try:
        sol = root_scalar(diffmodB,fprime=graddiffmodB,bracket=(0.99*np.pi,0.01*np.pi))
    except:
        print('Error occured!')
        print('s = ',s)
        print('zeta = ',zeta)
        return -1
    if (np.abs(diffmodB(sol.root))<1e-10):
        return sol.root
    else:
        print('root value: ',np.abs(diffmodB(sol.root)))
        print('Error occured!')
        print('s = ',s)
        print('zeta = ',zeta)
        return -1

time1 = time.time()

thetas2d = []
first, last = parallel_loop_bounds(comm, len(zetas2d.flatten()))
# For each point, find the mirror point in theta
for i in range(first,last):
    theta = theta_mirror(s2d.flatten()[i],zetas2d.flatten()[i])
    if (theta != -1):
        thetas2d.append(theta)

if comm is not None:
    thetas2d = [i for o in comm.allgather(thetas2d) for i in o]

time2 = time.time()
if verbose:
    print('theta_mirror time: ',time2-time1)

Ntrj = len(thetas2d)
s2d = s2d.flatten()
zetas2d = zetas2d.flatten()

time1 = time.time()

thetas_all = []
zetas_all = []
s_all = []
freq_all = []
rad_all = []
first, last = parallel_loop_bounds(comm, Ntrj)
for itrj in range(first,last):
    thetas_filename = 'thetas_poinc_'+str(itrj)+'.txt'
    s_filename = 's_poinc_'+str(itrj)+'.txt'
    zetas_filename = 'zetas_poinc_'+str(itrj)+'.txt'
    if (exists(thetas_filename) and exists(s_filename)):
        thetas_plot = np.loadtxt(thetas_filename)
        zetas_plot = np.loadtxt(zetas_filename)
        s_plot = np.loadtxt(s_filename)
    else:
        tr = [s2d[itrj],thetas2d[itrj],zetas2d[itrj]]
        thetas_plot = []
        zetas_plot = []
        s_plot = []
        for jj in range(Npts):
            tr = trapped_map(tr)
            tr = trapped_map(tr) # apply map twice to get upper bounce point
            thetas_plot.append(tr[1])
            s_plot.append(tr[0])
            zetas_plot.append(tr[2])
        np.savetxt(thetas_filename,thetas_plot)
        np.savetxt(s_filename,s_plot)
        np.savetxt(zetas_filename,zetas_plot)

    zetas_plot = np.array(zetas_plot)
    delta_zeta = (zetas_plot[1::]-zetas_plot[0:-1])/(2*np.pi/nfp)
    if (np.all(delta_zeta!=0)):
        freq_all.append(np.mean(delta_zeta)) # Compute effective frequency of each orbit
        rad_all.append(s2d[itrj])
        thetas_all.append(thetas_plot)
        zetas_all.append(zetas_plot)
        s_all.append(s_plot)

if comm is not None:
    thetas_all = [i for o in comm.allgather(thetas_all) for i in o]
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
    for i in range(len(thetas_all)):
        plt.scatter(np.mod(zetas_all[i],2*np.pi/nfp), s_all[i], marker='o',s=0.2,edgecolors='none')
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
        theta_point = theta_mirror(s_point,zeta_point)
        if (theta_point == -1):
            return angleradiusnormal
        point[0] = max(s_point,0.01) # Ensure that we don't cross through 0
        point[1] = theta_point
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
        error = 1e1*np.array([radiusnormalnew[0],radiusnormalnew[1]])
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
thetas_all = []
radius_all = []
nus_all = []
pp_all = []
qq_all = []
for i in range(first,last):
    pp = fpp[i]
    qq = fqq[i]
    theta_filename = 'theta_p_'+str(pp)+'_q_'+str(qq)+'.txt'
    nu_filename = 'nu_p_'+str(pp)+'_q_'+str(qq)+'.txt'
    radius_filename = 'radius_p_'+str(pp)+'_q_'+str(qq)+'.txt'
    read = False
    if (exists(theta_filename) and exists(nu_filename) and exists(radius_filename) and os.path.getsize(theta_filename)>0 and os.path.getsize(nu_filename)>0 and os.path.getsize(radius_filename)>0):
        thetas = np.loadtxt(theta_filename)
        nus = np.loadtxt(nu_filename)
        radius = np.loadtxt(radius_filename)
        if (len(radius)>0):
            read = True
    if not read:
        radialguess = guess_radius(pp/qq)
        print('radial guess: '+str(radialguess))
        nuguess = 0
        thetas = []
        radius = []
        nus = []
        count = 0
        ii = 0
        while (ii < perisland):
            theta = ii * 2*np.pi / ( nfp * qq * perisland ) # Theta for pseudo orbit
            rn = np.array([ radialguess, nuguess ])
            sol = root(qqpseudomap, rn, args=(theta,pp,qq), method='hybr', options={'maxfev':1000, 'eps':1e-4, 'factor':factor, 'xtol':xtol})
            rn = sol.x
            if (sol.success and np.linalg.norm(sol.fun) < ftol):
                thetas.append(np.mod(theta,2*np.pi/nfp))
                radius.append(rn[0])
                nus.append(rn[1])
                # Now apply the map qq times to complete the Poincare section
                angleradiusnormal = np.array([thetas[-1],radius[-1],nus[-1]])
                for iter in range(qq):
                    angleradiusnormal = pseudomap(angleradiusnormal)
                    angleradiusnormal[0] = np.mod(angleradiusnormal[0],2*np.pi/nfp)
                    thetas.append(angleradiusnormal[0])
                    radius.append(angleradiusnormal[1])
                    nus.append(angleradiusnormal[2])
                # Now add in zeta = 2*pi/nfp
                thetas.append(2*np.pi/nfp)
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
        # Now sort the result by theta
        thetas = np.array(thetas)
        nus = np.array(nus)
        radius = np.array(radius)
        indices = np.argsort(thetas)
        thetas = thetas[indices]
        nus = nus[indices]
        radius = radius[indices]
        np.savetxt(theta_filename,thetas)
        np.savetxt(nu_filename,nus)
        np.savetxt(radius_filename,radius)

    radius = np.asarray(radius)
    nus = np.asarray(nus)
    thetas = np.asarray(thetas)
    if (len(radius)>0):
        thetas_all.append(thetas)
        nus_all.append(nus)
        radius_all.append(radius)
        pp_all.append(pp)
        qq_all.append(qq)

if comm is not None:
    thetas_all = [i for o in comm.allgather(thetas_all) for i in o]
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
            plt.plot(thetas_all[i],radius_all[i],markevery=markevery,label=str(int(pp_all[i]))+', '+str(int(qq_all[i])),linewidth=1.0,markersize=1)
        lgd = plt.legend(bbox_to_anchor=(1.0,1.0),ncol=2)
        plt.savefig('poincare_with_map.pdf',bbox_extra_artists=(lgd,), bbox_inches='tight')

time2 = time.time()
if verbose:
    print('pseudomap time: ',time2-time1)
