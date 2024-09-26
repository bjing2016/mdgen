import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--split', type=str, default='splits/atlas.csv')
parser.add_argument('--sim_dir', type=str, default='/data/cb/scratch/datasets/atlas')
parser.add_argument('--outdir', type=str, default='./data_atlas')
parser.add_argument('--num_workers', type=int, default=1)
parser.add_argument('--suffix', type=str, default='')
parser.add_argument('--atlas', action='store_true')
parser.add_argument('--stride', type=int, default=1)
args = parser.parse_args()

import mdtraj, os, tqdm
import pandas as pd 
from multiprocessing import Pool
import numpy as np
from mdgen import residue_constants as rc

os.makedirs(args.outdir, exist_ok=True)

df = pd.read_csv(args.split, index_col='name')
names = df.index

def main():
    jobs = []
    for name in names:
        if os.path.exists(f'{args.outdir}/{name}{args.suffix}.npy'): continue
        jobs.append(name)

    if args.num_workers > 1:
        p = Pool(args.num_workers)
        p.__enter__()
        __map__ = p.imap
    else:
        __map__ = map
    for _ in tqdm.tqdm(__map__(do_job, jobs), total=len(jobs)):
        pass
    if args.num_workers > 1:
        p.__exit__(None, None, None)

# def prot_to_frames(ca_coords, c_coords, n_coords):
#     prot_frames = Rigid.from_3_points(
#         torch.from_numpy(c_coords),
#         torch.from_numpy(ca_coords),
#         torch.from_numpy(n_coords),
#     )
#     rots = torch.eye(3)
#     rots[0, 0] = -1
#     rots[2, 2] = -1
#     rots = Rotation(rot_mats=rots)
#     return prot_frames.compose(Rigid(rots, None))


def traj_to_atom14(traj):
    arr = np.zeros((traj.n_frames, traj.n_residues, 14, 3), dtype=np.float16)
    for i, resi in enumerate(traj.top.residues):
        for at in resi.atoms:
            if at.name not in rc.restype_name_to_atom14_names[resi.name]:
                print(resi.name, at.name, 'not found'); continue
            j = rc.restype_name_to_atom14_names[resi.name].index(at.name)
            arr[:,i,j] = traj.xyz[:,at.index] * 10.0
    return arr

if args.atlas:
    def do_job(name):
        for i in [1,2,3]:
            traj = mdtraj.load(f'{args.atlas_dir}/{name}/{name}_prod_R{i}_fit.xtc', top=f'{args.atlas_dir}/{name}/{name}.pdb') 
            traj.atom_slice([a.index for a in traj.top.atoms if a.element.symbol != 'H'], True)
            traj.superpose(traj)
            arr = traj_to_atom14(traj)
            np.save(f'{args.outdir}/{name}_R{i}{args.suffix}.npy', arr[::args.stride])
else:
    def do_job(name):
        traj = mdtraj.load(f'{args.atlas_dir}/{name}/{name}.xtc', top=f'{args.atlas_dir}/{name}/{name}.pdb')
        traj.superpose(traj)
        arr = traj_to_atom14(traj)
        np.save(f'{args.outdir}/{name}{args.suffix}.npy', arr[::args.stride])

if __name__ == "__main__":
    main()