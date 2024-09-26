import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--indir', type=str, default='4AA')
parser.add_argument('--outdir', type=str, default='/data/cb/scratch/share/mdgen/4AA_sims')
parser.add_argument('--worker_id', type=int, default=0)
parser.add_argument('--num_workers', type=int, default=1)
parser.add_argument('--pdb_id', nargs='*', default=[])
parser.add_argument('--joblist', type=str, default='splits/1k_4AA.csv')
parser.add_argument('--implicit', action='store_true')
parser.add_argument('--job_id', nargs='*', default=[])
parser.add_argument('--sim_ns', type=float, default=100)
parser.add_argument('--print_freq', type=int, default=1000)
parser.add_argument('--save_ps', type=float, default=0.1)
parser.add_argument('--friction_coeff', type=float, default=0.3)
parser.add_argument('--md_device', type=str, default="CUDA")
args = parser.parse_args()

import openmm, mdtraj, sys, os
from openmm.app import PDBFile, ForceField, Modeller, PME, HBonds, Simulation, StateDataReporter
from openmm import unit, LangevinMiddleIntegrator, Platform, MonteCarloBarostat
import numpy as np
import pandas as pd


dt = 2 * unit.femtosecond
total_steps = int((args.sim_ns * unit.nanosecond) / dt)
save_interval = int((args.save_ps * unit.picosecond) / dt)
print(f"Running for {total_steps} steps")
print(f"Saving every {save_interval} steps")
print(f"Will save {int(total_steps / save_interval)} frames")


def make(aa):
    if '_' in aa:
        aa = aa.split('_')[0]
    print(f'Making {aa}')
    from pymol import cmd
    cmd.reinitialize()
    cmd.fab(aa, hydro=0)
    cmd.save(f'{args.indir}/{aa}.pdb')
    print(f'Fixing {aa}')
    from pdbfixer import PDBFixer
    fixer = PDBFixer(filename=f'{args.indir}/{aa}.pdb')
    fixer.missingResidues = {}
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    with open(f'{args.indir}/{aa}.pdb', 'w') as f:
        PDBFile.writeFile(fixer.topology, fixer.positions, f, True)

    # subprocess.run(['pdbfixer', f'{args.indir}/{aa}.pdb', '--add-atoms=heavy'])
    # subprocess.run(['mv', 'output.pdb', f'{args.indir}/{aa}.pdb'])

def do(name):
    os.makedirs(f"{args.outdir}/{name}", exist_ok=True)

    aa = name.split('_')[0]
    if not os.path.exists(f"{args.indir}/{aa}.pdb"):
        make(aa)
    pdb = PDBFile(f"{args.indir}/{aa}.pdb")
    if args.implicit:
        forcefield = ForceField('amber14-all.xml', 'implicit/gbn2.xml')
    else:
        forcefield = ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
    modeller = Modeller(pdb.topology, pdb.positions)
    modeller.addHydrogens(forcefield, pH=7)

    if args.implicit:
        system = forcefield.createSystem(modeller.topology, constraints=HBonds)
    else:
        modeller.addSolvent(forcefield, padding=1.0 * unit.nanometer)
        system = forcefield.createSystem(modeller.topology, nonbondedMethod=PME,
                                     nonbondedCutoff=1.0 * unit.nanometer,
                                     constraints=HBonds)
        
    integrator = LangevinMiddleIntegrator(350 * unit.kelvin, args.friction_coeff / unit.picosecond, dt)
    simulation = Simulation(modeller.topology, system, integrator,
                            platform=Platform.getPlatformByName(args.md_device))
    simulation.context.setPositions(modeller.positions)

    top = mdtraj.Topology.from_openmm(modeller.topology)
    print(f'System with {top.n_atoms} atoms')
    mask = top.select("protein and not type H")
    print(f'Reporting {len(mask)} atoms')
    reporter = mdtraj.reporters.HDF5Reporter(f'{args.outdir}/{name}/{name}.h5', reportInterval=save_interval,
                                             atomSubset=mask)

    print('Minimizing energy')
    simulation.minimizeEnergy()

    print("Running NVT")
    simulation.reporters.append(StateDataReporter(
        sys.stdout,
        reportInterval=args.print_freq,
        step=True,
        potentialEnergy=True,
        temperature=True,
        volume=True,
        speed=True,
        remainingTime=True,
        totalSteps=total_steps + 10000
    ))
    simulation.reporters.append(StateDataReporter(
        open(f'{args.outdir}/{name}/{name}.out', 'w'),
        reportInterval=1000,
        step=True,
        potentialEnergy=True,
        temperature=True,
        volume=True,
        speed=True,
        remainingTime=True,
        totalSteps=total_steps + 10000
    ))
    simulation.step(10000)

    if not args.implicit:
        print("Running NPT")
        system.addForce(MonteCarloBarostat(1 * unit.bar, 350 * unit.kelvin))
        simulation.context.reinitialize(preserveState=True)
        
    simulation.reporters.append(reporter)
    simulation.step(total_steps)
    reporter.close()

    print("Converting to XTC")
    traj = mdtraj.load(f'{args.outdir}/{name}/{name}.h5')
    traj.superpose(traj)
    traj.save(f'{args.outdir}/{name}/{name}.xtc')
    traj[0].save(f'{args.outdir}/{name}/{name}.pdb')

if args.pdb_id:
    jobs = args.pdb_id
else:
    jobs = np.array(pd.read_csv(args.joblist, index_col='name').index)
    if args.job_id:
        jobs = jobs.index[list(map(int, args.job_id))]
    else:
        jobs = jobs[args.worker_id::args.num_workers]
        
for name in jobs:
    do(name)


