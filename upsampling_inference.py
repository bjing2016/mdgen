import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=str, default=None, required=True)
parser.add_argument('--data_dir', type=str, default=None, required=True)
parser.add_argument('--suffix', type=str, default='_i100')
parser.add_argument('--pdb_id', nargs='*', default=[])
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--out_dir', type=str, default=".")
parser.add_argument('--split', type=str, default='splits/4AA_implicit_test.csv')
args = parser.parse_args()

import os, torch, mdtraj, tqdm
import numpy as np
from mdgen.geometry import atom14_to_frames, atom14_to_atom37, atom37_to_torsions
from mdgen.residue_constants import restype_order
from mdgen.tensor_utils import tensor_tree_map
from mdgen.wrapper import NewMDGenWrapper
from mdgen.utils import atom14_to_pdb
import pandas as pd




os.makedirs(args.out_dir, exist_ok=True)



def get_batch(name, seqres, num_frames):
    arr = np.lib.format.open_memmap(f'{args.data_dir}/{name}{args.suffix}.npy', 'r')
    arr = np.copy(arr).astype(np.float32)

    frames = atom14_to_frames(torch.from_numpy(arr))
    seqres = torch.tensor([restype_order[c] for c in seqres])
    atom37 = torch.from_numpy(atom14_to_atom37(arr, seqres[None])).float()
    torsions, torsion_mask = atom37_to_torsions(atom37, seqres[None])
    L = frames.shape[1]
    mask = torch.ones(L)
    return {
        'torsions': torsions,
        'torsion_mask': torsion_mask[0],
        'trans': frames._trans,
        'rots': frames._rots._rot_mats,
        'seqres': seqres,
        'mask': mask, # (L,)
    }

def split_batch(item, num_frames=1000, cond_interval=100):
    total_frames = item['torsions'].shape[0] * cond_interval
    batches = []
    total_items = int(total_frames / num_frames)
    cond_frames = int(num_frames / cond_interval)
    for i in tqdm.trange(total_items):
        new_batch = {
            'torsions': torch.zeros(num_frames, 4, 7, 2),
            'torsion_mask': item['torsion_mask'],
            'trans': torch.zeros(num_frames, 4, 3),
            'rots': torch.zeros(num_frames, 4, 3, 3),
            'seqres': item['seqres'],
            'mask': item['mask'],
        }
        new_batch['rots'][:] = torch.eye(3)
        new_batch['torsions'][::cond_interval] = item['torsions'][i*cond_frames:(i+1)*cond_frames]
        new_batch['trans'][::cond_interval] = item['trans'][i*cond_frames:(i+1)*cond_frames]
        new_batch['rots'][::cond_interval] = item['rots'][i*cond_frames:(i+1)*cond_frames]
        batches.append(new_batch)
    return batches
    
def do(model, name, seqres):

    item = get_batch(name, seqres, num_frames = model.args.num_frames)
    
    items = split_batch(item, num_frames=model.args.num_frames, cond_interval=model.args.cond_interval)
    
    loader = torch.utils.data.DataLoader(items, shuffle=False, batch_size=args.batch_size)

    all_atom14 = []
    for batch in tqdm.tqdm(loader):
        batch = tensor_tree_map(lambda x: x.cuda(), batch)  
        atom14, _ = model.inference(batch)
        all_atom14.extend(atom14)
        
    all_atom14 = torch.cat(all_atom14)
    
    path = os.path.join(args.out_dir, f'{name}.pdb')
    atom14_to_pdb(all_atom14.cpu().numpy(), batch['seqres'][0].cpu().numpy(), path)
    
    traj = mdtraj.load(path)
    traj.superpose(traj)
    traj.save(os.path.join(args.out_dir, f'{name}.xtc'))
    traj[0].save(os.path.join(args.out_dir, f'{name}.pdb'))

@torch.no_grad()
def main():
    model = NewMDGenWrapper.load_from_checkpoint(args.ckpt)
    model.eval().to('cuda')
    
    
    df = pd.read_csv(args.split, index_col='name')
    for name in df.index:
        if args.pdb_id and name not in args.pdb_id:
            continue
        do(model, name, df.seqres[name])
        

main()