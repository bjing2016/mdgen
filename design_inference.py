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
parser.add_argument('--num_frames', type=int, default=100)
parser.add_argument('--num_batches', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--out_dir', type=str, default=".")
parser.add_argument('--random_start_idx', action='store_true')
parser.add_argument('--split', type=str, default='splits/4AA_test.csv')
parser.add_argument('--chunk_idx', type=int, default=0)
parser.add_argument('--n_chunks', type=int, default=1)
args = parser.parse_args()
import mdgen.analysis
import os, torch, mdtraj, tqdm
import numpy as np
from mdgen.geometry import atom14_to_frames, atom14_to_atom37, atom37_to_torsions
from mdgen.tensor_utils import tensor_tree_map

from mdgen.residue_constants import restype_order, restype_atom37_mask
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

def get_sample(arr, seqres, start_idxs, start_state, end_state, num_frames=100):
    start_idx = np.random.choice(start_idxs, 1).item()
    if args.random_start_idx:
        start_idx = np.random.randint(low=0,high=len(arr)-num_frames)
    end_idx = start_idx + num_frames

    arr = np.copy(arr[start_idx: end_idx]).astype(np.float32)
    seqres = torch.tensor([restype_order[c] for c in seqres])

    frames = atom14_to_frames(torch.from_numpy(arr))
    atom37 = torch.from_numpy(atom14_to_atom37(arr, seqres)).float()
    torsions, torsion_mask = atom37_to_torsions(atom37, seqres[None])

    L = frames.shape[1]


    mask = torch.ones(L)
    return {
        'torsions': torsions,
        'torsion_mask': torsion_mask[0],
        'trans': frames._trans,
        'rots': frames._rots._rot_mats,
        'seqres': seqres,
        'start_idx': start_idx,
        'end_idx': end_idx,
        'start_state': start_state,
        'end_state': end_state,
        'mask': mask,  # (L,)
    }

def do(model, name, seqres):
    print('doing', name)
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
    np.fill_diagonal(flux_mat, 0)
    start_state, end_state = np.unravel_index(np.argmax(flux_mat, axis=None), flux_mat.shape)
    ref_discrete = msm.metastable_assignments[ref_kmeans]
    
    arr = np.lib.format.open_memmap(f'{args.data_dir}/{name}.npy', 'r')
    if model.args.frame_interval:
        arr = arr[::model.args.frame_interval]
        ref_discrete = ref_discrete[::model.args.frame_interval]
    
    is_start = ref_discrete == start_state
    is_end = ref_discrete == end_state

    trans_indices = is_start[:-args.num_frames] * is_end[args.num_frames:]
    start_idxs = np.where(trans_indices)[0]
    if (trans_indices).sum() == 0:
        print('No transition path found for ', name, 'skipping...')
        return

    

    metadata = []
    for i in tqdm.tqdm(range(args.num_batches), desc='num batch'):
        batch_list = []
        for _ in range(args.batch_size):
            batch_list.append(
                get_sample(arr, seqres, copy.deepcopy(start_idxs), start_state, end_state, num_frames=args.num_frames))
        batch = next(iter(torch.utils.data.DataLoader(batch_list, batch_size=args.batch_size)))
        batch = tensor_tree_map(lambda x: x.cuda(), batch)


        print('Start tps for ', name, 'with start coords', batch['trans'][0, 0, 0], 'and with end coords', batch['trans'][0, -1, 0])
        atom14s, aa_out = model.inference(batch)
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
                'aa_out': aa_out[j].cpu().numpy().tolist(),  # 'aa_out': 'aa_out',
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