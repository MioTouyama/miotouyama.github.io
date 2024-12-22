---
title: A Quick Introduction to the Code of PySCF Multiconfiguration Calculations
date: 2024-12-22 20:21:00 +0900
categories: [PySCF, Intro]
tags: [doc]     # TAG names should always be lowercase
---

This is an example.

```python
from pyscf import gto, lib 

def weight(n):
    return [1 / n] * n

lib.num_threads(24) # https://github.com/pyscf/pyscf/issues/1041 # %nproc=24

mol = gto.M()
mol.atom = """
 C                  0.85174954   -3.49907914    0.00000000
 C                  2.24690954   -3.49907914    0.00000000
 C                  2.94444754   -2.29132814    0.00000000
 C                  2.24679354   -1.08281914   -0.00119900
 C                  0.85196854   -1.08289714   -0.00167800
 C                  0.15436754   -2.29110314   -0.00068200
 F                  4.29444731   -2.29122993    0.00077832
 F                 -1.19563243   -2.29087847   -0.00090299
 F                  0.17680620   -4.66824601    0.00055247
 F                  2.92151711   -4.66843880    0.00161437
 F                  2.92223542    0.08605992   -0.00127143

 F                  0.17667208    0.08606536   -0.00284785

"""
# mol.atom = """
#   Cr                  0.00000000    0.00000000    0.00000000
#   Cr                  2.00000000    0.00000000    0.00000000
# """
mol.max_memory = 80000 # MB, %mem=80GB
mol.output = 'test.out'
mol.basis = 'def2-svp'
mol.charge = 0
mol.spin = 0 # Nalpha - Nbeta, doublet: 1, triplet: 2, ...
mol.verbose = 4
# https://github.com/pyscf/pyscf/blob/master/examples/gto/00-input_mole.py
# mol.cart = True
# turn on symm for multi state with mixed spin
# mol.symmetry = True
mol.build()

from pyscf import scf
mf = scf.HF(mol) # .apply(dft.addons.remove_linear_dep_)
# mf = scf.ROHF(mol) # start from high spin
mf.chkfile = 'pyscf.chk'

# https://github.com/pyscf/pyscf/issues/865
# 1e-13 default, but for organic, decrease to -10 or -11
# int=acc2e=12 in G16
mf.direct_scf_tol = 1e-10
# https://gaussian.com/scf/
# scf=conver=8 in G16, rms of ddm < 1e-8, max ddm < 1e-6
# not check energy, but typically dE < 1e-16
# https://www.faccts.de/docs/orca/6.0/manual/contents/detailed/scfconv.html
# orca, tightSCF
# dE < 1e-8, dE1 < 1e-5
# https://pyscf.org/_modules/pyscf/scf/hf.html
# "norm_gorb = numpy.linalg.norm(mf.get_grad(mo_coeff, mo_occ, fock))"
# abs(e_tot-last_hf_e) < conv_tol or norm_gorb < conv_tol_grad
# conv_tol_grad = sqrt(conv_tol)
# 1e-9, 1e-4.5 as default
mf.conv_tol = 1e-9
# mf.max_cycle=100

# https://github.com/pyscf/pyscf/blob/master/examples/df/00-with_df.py
from pyscf import df
mf = df.density_fit(mf)

mf.kernel()

# https://github.com/pyscf/pyscf/blob/master/examples/mcscf/43-avas.py
# from pyscf.mcscf import avas
# nact, nele, mo_coeff = avas.avas(mf, ["C 2p", "C 3p", "C 2s", "C 3s"])
nact, nele = 6, 6 # can calc another spin diff. from mol (naele, nbele)

from pyscf import mcscf
mc = mcscf.CASSCF(mf, nact, nele)
mc.kernel()
mo_coeff = mc.mo_coeff

mol.basis = 'def2-tzvp'
mol.build()

from pyscf import mcpdft
mc = mcpdft.CASSCF(mf, 'tPBE', nact, nele) # change to DMRG when > (15, 15)
# mc.otxc = 'ftPBE' # 'tBLYP', 'ftBLYP'
# 'tPBE0', 'ftPBE0'
# from pyscf import dft
# mc.grids.radi_method = dft.gauss_chebyshev
# mc.grids.level = 5 

# https://block2.readthedocs.io/en/latest/user/dmrg-scf.html#preparation
from pyscf import dmrgscf
dmrgscf.settings.BLOCKEXE = '/shared_apps/anaconda3/envs/env_pyscfbin/bin/block2main'
dmrgscf.settings.MPIPREFIX = 'mpirun -n 6 --bind-to none'
mc = dmrgscf.DMRGSCF(mf, nact, nele)
# same as init a normal mcscf obj and then give dmrg fci solver
# mc = mcscf.CASSCF(mf, nact, nele)
# mc.fcisolver = dmrgscf.DMRGCI(mol)
mc.fcisolver.threads = 4
# general setting of fci solver
# mc.fcisolver.lindep = 1e-12
# mc.fcisolver.max_cycle = 50
# mc.fcisolver.conv_tol = 1e-8

# https://github.com/pyscf/pyscf/blob/master/doc_legacy/source/mcscf.rst
mc.canonicalization = True

# from pyscf import fci
# mc.fcisolver = fci.SCI(mol)

# same spin
nstate = 6
# state-average
mc = mc.state_average_(weight(nstate))
# For MC-PDFT, multi-state is availabe, option: XMS, CMS
# https://github.com/pyscf/pyscf-forge/blob/master/examples/mcpdft/16-multi_state.py
mc = mc.multi_state(weight(nstate), 'xms')
mc = mc.multi_state(weight(nstate), 'cms')
# https://github.com/pyscf/pyscf-forge/blob/master/examples/mcpdft/42-linearized-pdft.py
mc = mc.multi_state(weight(nstate), 'lin')

# diff. spin
nspin, nstate = 2, [6, 6]
from pyscf import fci
solvers = [fci.direct_spin1_symm.FCI(mol), fci.direct_spin0_symm.FCI(mol)]
# solvers = [dmrgscf.DMRGCI(mol) for _ in range(nspin)]
solvers[0] = fci.addons.fix_spin(solvers[0], ss=0)
solvers[1].spin = 2
solvers[1] = fci.addons.fix_spin(solvers[1], ss=2)
from os import sep
for i, (s, n) in enumerate(zip(solvers, nstate)):
    s.nroots = n
    # for DMRG
    # s.runtimeDir = lib.param.TMPDIR + sep + str(i)
    # s.scratchDirectory = lib.param.TMPDIR + sep + str(i)
    # s.memory = int(mol.max_memory / 1000) # GB
    # s.threads = 2
# For CAS
from pyscf import mcscf
mcscf.state_average_mix_(mc, solvers, weight(sum(nstate)))
# For MC-PDFT, multi-state uses another keyword
mc = mc.multi_state_mix(solvers, weight(sum(nstate)), 'lin')

# General setting
mc.chkfile = 'pyscf.chk'
# dE < 1e-8 in G16 and orb???
# MCSCF converge criterion is similar to HF/KS, but needs additional conditions
# https://github.com/pyscf/pyscf/blob/master/pyscf/mcscf/mc1step.py#L465
mc.conv_tol = 1e-12
mc.conv_tol_grad = 1e-6
# https://github.com/pyscf/pyscf/blob/master/pyscf/mcscf/__init__.py
# increasing micro might reduce the total macro, default 4/50
# mc.max_cycle_micro = 4
# mc.max_cycle_macro = 50 # same as mc.max_cycle = 300
# https://github.com/pyscf/pyscf/blob/master/examples/mcscf/03-natural_orbital.py
# When .natorb is set, the natural orbitals may NOT be sorted by the active space occupancy.
mc.natorb = True
# mc.sort_mo([5,6,8,9]) # select orb, not rotate like Gaussian, 1based id
# mc.frozen = 2 # or [0,1,15] # 0based
# mc.fix_spin_(ss=2)

# https://pyscf.org/user/mcscf.html
# RI not turned on for MCSCF by default
mc = df.density_fit(mc)

mc.kernel()
# if use lower basis and project to higher
# https://github.com/pyscf/pyscf/blob/master/examples/mcscf/14-project_init_guess.py
# mc.kernel(mcscf.project_init_guess(mc, mo_coeff))

# for those hard to converge
# https://github.com/pyscf/pyscf/issues/912
from mcscf import mc2step
conv, e_tot, e_cas, ci, mo_coeff, mo_energy = mc2step.kernel(mc, mc.mo_coeff) # ci is fcivec

# do not know why fcivec is not saved,
# even dumped to chk and read by the following restart work, fcivec is not succeeded
# lib.chkfile.dump('pyscf.chk', 'mcscf/ci', mc.ci)

mc.analyze()

# mol = lib.chkfile.load_mol('pyscf.chk')
# mol.output = 'test.out'
# mol.build()
# mf = scf.HF(mol)
# set other property, remember density_fit
# mf.__dict__.update(lib.chkfile.load('pyscf.chk', 'scf'))
# dm = mf.make_rdm1()
# mf.kernel(dm)

# create mc, set other property, remember density_fit
# https://github.com/pyscf/pyscf/blob/master/examples/mcscf/13-load_chkfile.py
# mc.__dict__.update(lib.chkfile.load('pyscfback.chk', 'mcscf'))
# mc.kernel()

# change on-top func run after mc.kernel()
# mc.compute_pdft_energy_(otxc='ftPBE')
# For MC-PDFT, CAS energy stored in mc.e_mcscf

g = mc.Gradients() # grad obj created after mc converged
g.kernel()

mc_scanner = mc.as_scanner() # state=
mol.set_geom_()
mc_scanner(mol)

g_scanner = g.as_scanner() # state=
mol.set_geom_()
g_scanner(mol)

# analytical hess of cas has not been implemented, Dec. 16th, 2024
# numerical hess
import numpy as np
N = mol.natm * 3
hess = np.zeros((N, N))
d = 0.001
for i in range(N):
    xyz = mol.atom_coords().copy()
    xyz[i // 3, i % 3] += d
    mol.set_geom_(xyz, unit='Bohr')
    e, g1 = g_scanner(mol)
    xyz[i // 3, i % 3] -= 2 * d
    mol.set_geom_(xyz, unit='Bohr')
    e, g2 = g_scanner(mol)
    hess[i] = (g1.ravel() - g2.ravel()) / (2 * d)
hess = 0.5 * (hess + hess.T)

# https://github.com/pyscf/pyscf/blob/master/examples/hessian/10-thermochemistry.py
from pyscf.hessian import thermo
freq_info = thermo.harmonic_analysis(mol, hess.reshape(mol.natm, mol.natm, 3, 3))
thermo_info = thermo.thermo(mf, freq_info['freq_au'], 298.15, 101325)

# https://pyscf.org/user/geomopt.html
# do analysis for each step
def cb(envs):
    mc = envs['g_scanner'].base
    mc.analyze(verbose=4)

# for geomeTRIC
conv_params = {'convergence_energy': 1e-6, # Eh
               'convergence_grms': 3e-4, # Eh/Bohr
               'convergence_gmax': 4.5e-4, # Eh/Bohr
               'convergence_drms': 1.2e-3 / 1.8897259886, # Angstrom
               'convergence_dmax': 1.8e-3 / 1.8897259886, # Angstrom
               'transition': True,
               'hessian': True,
               'trust': 0.02,
               'tmax': 0.06,
               'constraints': 'constraints.txt'}

opt = g_scanner.optimizer(solver='geomeTRIC')
opt.callback = cb
opt.kernel(conv_params)

mf.mo_coeff = mc.mo_coeff
mc = mcscf.CASCI(mf, nact, nele)
mc.fcisolver.nroots = nstate
mc.natorb = True
# mc.fix_spin_(ss=2)
mc.kernel()

from pyscf import mrpt
energies = []
# for default FCI
mr = mrpt.nevpt2.NEVPT(mc)
for i in range(nstate):
    mr.kernel(root=i)
    # for MS-PDFT, e_mcscf is not available, recalculate it
    energies.append(mr.e_tot[i] + mc.e_mcscf[i])
# for DMRG
mo_coeffs, ci, mo_energies = [None] * nstate, [None] * nstate, [None] * nstate,
for i in range(nstate):
    mo_coeffs[i], ci[i], mo_energies[i] = mc.canonicalize(mc.mo_coeff, ci=mc.ci[i], cas_natorb=False)
mc.ci = ci
for i in range(nstate):
    mc.mo_coeff, mc.mo_energy = mo_coeffs[i], mo_energies[i]
    mr = mrpt.nevpt2.NEVPT(mc)
    mr.canonicalized = True
    mr.compress_approx()
    mr.kernel(root=i)
    energies.append(mr.e_tot[i] + mc.e_mcscf[i])
```
