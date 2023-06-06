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

boozmn_filename = "boozmn_ARIES.nc" # booz_xform filename
ns_interp = 30 # number of radial grid points for interpolation
ntheta_interp = 30 # number of poloidal grid points for interpolation
nzeta_interp = 30 # number of toroidal grid points for interpolation
order = 3 # order for interpolation
chi_mirror = 1.2*np.pi/2 # chi = theta - helicity*nfp * zeta -> this determines mirror point
s_mirror = 0.6 # radial point for mirroring
zeta_mirror = 0 # toroidal angle for mirroring
helicity = 0
mpol = 48 # poloidal resolution for booz_xform
ntor = 48

equil = Booz_xform()
equil.verbose = 0
equil.read_boozmn(boozmn_filename)
nfp = equil.nfp

bri = BoozerRadialInterpolant(equil,order,mpol=mpol,ntor=ntor,no_K=True,mpi=mpi,verbose=verbose)
nfp = bri.nfp
degree = 3
srange = (0, 1, ns_interp)
thetarange = (0, np.pi, ntheta_interp)
zetarange = (0, 2*np.pi/nfp, nzeta_interp)
field = InterpolatedBoozerField(bri, degree, srange, thetarange, zetarange, True, nfp=nfp, stellsym=True)

#points = np.zeros((len(zetas_grid),3))
s=0.5
theta0=np.linspace(0,2*np.pi,200)
for i in np.arange(0,200):
     point0=np.zeros((1,3))
     point0[:,0]=s
     point0[:,1]=theta0[i]
     point0[:,2]=0.0
     field.set_points(point0)
     iota=field.iota()[0,0]
     zetas_grid = np.linspace(0,2*np.pi/iota,100)
     thetas=theta0[i]+iota*zetas_grid
     points=np.zeros((len(zetas_grid),3))
     points[:,0]=s
     points[:,1]=thetas
     points[:,2]=zetas_grid
     field.set_points(points)
     modB0=field.modB()[:,0]

     plt.clf()
     plt.plot(zetas_grid,modB0)
     plt.xlabel('$\zeta$')
     plt.ylabel('|B|')
     plt.ylim(4.95,6.7)
     plt.title(r'$\theta$='+str(theta0[i]))
     plt.savefig('modB_fieldline_'+str(i)+'.png')
