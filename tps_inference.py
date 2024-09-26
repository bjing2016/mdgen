import argparse
import copy
import json
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--sim_ckpt', type=str, default=None, required=True)
parser.add_argument('--data_dir', type=str, default='share/4AA_data')
parser.add_argument('--mddir', type=str, default='/data/cb/scratch/share/mdgen/4AA_sims')
parser.add_argument('--suffix', type=str, default='')
parser.add_argument('--pdb_id', nargs='*', default=[])
parser.add_argument('--num_frames', type=int, default=1000)
parser.add_argument('--num_batches', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--out_dir', type=str, default=".")
parser.add_argument('--split', type=str, default='splits/4AA_test.csv')
parser.add_argument('--chunk_idx', type=int, default=0)
parser.add_argument('--n_chunks', type=int, default=1)
args = parser.parse_args()
import mdgen.analysis
import os, torch, mdtraj, tqdm
from mdgen.geometry import atom14_to_atom37, atom37_to_torsions
from mdgen.tensor_utils import tensor_tree_map

from mdgen.residue_constants import restype_order
from mdgen.wrapper import NewMDGenWrapper
from mdgen.dataset import atom14_to_frames
import pandas as pd
import contextlib
import numpy as np

@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)

os.makedirs(args.out_dir, exist_ok=True)

def get_sample(arr, seqres, start_idxs, end_idxs, start_state, end_state, num_frames=1000):
    start_idx = np.random.choice(start_idxs, 1).item()
    end_idx = np.random.choice(end_idxs, 1).item()

    start_arr = np.copy(arr[start_idx:start_idx + 1]).astype(np.float32)
    end_arr = np.copy(arr[end_idx:end_idx + 1]).astype(np.float32)
    seqres = torch.tensor([restype_order[c] for c in seqres])

    start_frames = atom14_to_frames(torch.from_numpy(start_arr))
    start_atom37 = torch.from_numpy(atom14_to_atom37(start_arr, seqres)).float()
    start_torsions, start_torsion_mask = atom37_to_torsions(start_atom37, seqres[None])
    
    end_frames = atom14_to_frames(torch.from_numpy(end_arr))
    end_atom37 = torch.from_numpy(atom14_to_atom37(end_arr, seqres)).float()
    end_torsions, end_torsion_mask = atom37_to_torsions(end_atom37, seqres[None])
    L = start_frames.shape[1]
    traj_torsions = start_torsions.expand(num_frames, -1, -1, -1).clone()
    traj_torsions[-1] = end_torsions

    traj_trans = start_frames._trans.expand(num_frames, -1, -1).clone()
    traj_trans[-1] = end_frames._trans

    traj_rots = start_frames._rots._rot_mats.expand(num_frames, -1, -1, -1).clone()
    traj_rots[-1] = end_frames._rots._rot_mats

    mask = torch.ones(L)
    return {
        'torsions': traj_torsions,
        'torsion_mask': start_torsion_mask[0],
        'trans': traj_trans,
        'rots': traj_rots,
        'seqres': seqres,
        'start_idx': start_idx,
        'end_idx': end_idx,
        'start_state': start_state,
        'end_state': end_state,
        'mask': mask,  # (L,)
    }

def do(model, name, seqres):
    print('doing', name)
    if os.path.exists(f'{args.out_dir}/{name}_metadata.json'):
        return
    if os.path.exists(f'{args.out_dir}/{name}_metadata.pkl'):
        pkl_metadata = pickle.load(open(f'{args.out_dir}/{name}_metadata.pkl', 'rb'))
        msm = pkl_metadata['msm']
        cmsm = pkl_metadata['cmsm']
        ref_kmeans = pkl_metadata['ref_kmeans']
    else:
        with temp_seed(137):
            feats, ref = mdgen.analysis.get_featurized_traj(f'{args.mddir}/{name}/{name}', sidechains=True)
            tica, _ = mdgen.analysis.get_tica(ref)
            kmeans, ref_kmeans = mdgen.analysis.get_kmeans(tica.transform(ref))
            try:
                msm, pcca, cmsm = mdgen.analysis.get_msm(ref_kmeans, nstates=10)
            except Exception as e:
                print('ERROR', e, name, flush=True)
                return
        pickle.dump({
            'msm': msm,
            'cmsm': cmsm,
            'tica': tica,
            'pcca': pcca,
            'kmeans': kmeans,
            'ref_kmeans': ref_kmeans,
        }, open(f'{args.out_dir}/{name}_metadata.pkl', 'wb'))

    flux_mat = cmsm.transition_matrix * cmsm.pi[None, :]
    flux_mat[flux_mat < 0.0000001] = np.inf  # set 0 flux to inf so we do not choose that as the argmin
    start_state, end_state = np.unravel_index(np.argmin(flux_mat, axis=None), flux_mat.shape)
    ref_discrete = msm.metastable_assignments[ref_kmeans]
    start_idxs = np.where(ref_discrete == start_state)[0]
    end_idxs = np.where(ref_discrete == end_state)[0]
    if (ref_discrete == start_state).sum() == 0 or (ref_discrete == end_state).sum() == 0:
        print('No start or end state found for ', name, 'skipping...')
        return

    arr = np.lib.format.open_memmap(f'{args.data_dir}/{name}.npy', 'r')

    metadata = []
    for i in tqdm.tqdm(range(args.num_batches), desc='num batch'):
        batch_list = []
        for _ in range(args.batch_size):
            batch_list.append(
                get_sample(arr, seqres, copy.deepcopy(start_idxs), end_idxs, start_state, end_state, num_frames=args.num_frames))
        batch = next(iter(torch.utils.data.DataLoader(batch_list, batch_size=args.batch_size)))
        batch = tensor_tree_map(lambda x: x.cuda(), batch)
        print('Start tps for ', name, 'with start coords', batch['trans'][0, 0, 0])
        print('Start tps for ', name, 'with end coords', batch['trans'][0, -1, 0])
        atom14s, _ = model.inference(batch)
        for j in range(args.batch_size):
            idx = i * args.batch_size + j
            path = os.path.join(args.out_dir, f'{name}_{idx}.pdb')
            atom14_to_pdb(atom14s[j].cpu().numpy(), batch['seqres'][0].cpu().numpy(), path)

            traj = mdtraj.load(path)
            traj.superpose(traj)
            traj.save(os.path.join(args.out_dir, f'{name}_{idx}.xtc'))
            traj[0].save(os.path.join(args.out_dir, f'{name}_{idx}.pdb'))
            metadata.append({
                'name': name,
                'start_idx': batch['start_idx'][j].cpu().item(),
                'end_idx': batch['end_idx'][j].cpu().item(),
                'start_state': batch['start_state'][j].cpu().item(),
                'end_state': batch['end_state'][j].cpu().item(),
                'path': path,
            })
    json.dump(metadata, open(f'{args.out_dir}/{name}_metadata.json', 'w'))


@torch.no_grad()
def main():
    model = NewMDGenWrapper.load_from_checkpoint(args.sim_ckpt)
    model.eval().to('cuda')
    df = pd.read_csv(args.split, index_col='name')
    names = np.array(df.index)

    chunks = np.array_split(names, args.n_chunks)
    chunk = chunks[args.chunk_idx]
    print('#' * 20)
    print(f'RUN NUMBER: {args.chunk_idx}, PROCESSING IDXS {args.chunk_idx * len(chunk)}-{(args.chunk_idx + 1) * len(chunk)}')
    print('#' * 20)
    for name in tqdm.tqdm(chunk, desc='num peptides'):
        if args.pdb_id and name not in args.pdb_id:
            continue
        do(model, name, df.seqres[name])


main()