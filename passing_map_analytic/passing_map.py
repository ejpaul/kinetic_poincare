import numpy as np
import matplotlib.pyplot as plt
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

sign_vpar = 1.0 # should be +/- 1. sign(vpar)
pitch_angle = 0.0 # lambda = v_perp^2/(v^2 B) = const. along trajectory
nchi_poinc = 1 # Number of zeta initial conditions for poincare
ns_poinc = 100 # Number of s initial conditions for poincare
tol = 1e-8 # gc integration tolerance
Npts = 1000 # number of points for poincare plotting
tmax = 1e-2 # time limit for gc integration

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

# For each point, find value of vpar such that lambda = vperp^2/(v^2 B)
chis_all = []
s_all = []
vpar_all = []
for i in range(len(s)):
    vpar = vpar_func(s[i],chis[i])
    if vpar is not None:
        s_all.append(s[i])
        chis_all.append(chis[i])
        vpar_all.append(vpar)

chis = chis_all 
s = s_all 
vpar = vpar_all 

Ntrj = len(chis)

"""
We now iterate over the Poincare initial conditions and perform the mapping.
"""
chis_all = []
vpars_all = []
s_all = []
freq_all = []
rad_all = []
for itrj in range(Ntrj):
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

plt.figure(1)
plt.xlabel(r'$\chi$')
plt.ylabel(r'$s$')
plt.xlim([0,2*np.pi])
plt.ylim([0,1])
for i in range(len(chis_all)):
    plt.scatter(np.mod(chis_all[i],2*np.pi), s_all[i], marker='o',s=0.1,edgecolors='none')
plt.savefig('map_poincare.pdf')

