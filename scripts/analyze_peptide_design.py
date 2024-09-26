import numpy as np
import os
import argparse
from mdgen.residue_constants import aatype_to_str_sequence, restype_order
import json
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--mddir', type=str, default='share/4AA_sims')
parser.add_argument('--data_dir', type=str, default='share/4AA_sims_replica')
parser.add_argument('--pdbdir', type=str, required=True)
parser.add_argument('--split', type=str, default='splits/4AA_test.csv') 
args = parser.parse_args()



names = os.listdir(args.pdbdir)
names = list(set([nam[:4] for nam in names if 'metadata.json' in nam]))

metadatas = {name: json.load(open(f'{args.pdbdir}/{name}_metadata.json', 'rb')) for name in names}

res = {}
for name in names:
    metadata = metadatas[name]
    res[name] = [traj['aa_out'][0] for traj in metadata]

designed_names = {}
max_cond_recovery = 0
max_design_recovery = 0

all_recovery = 0
design_recovery = 0
final_design_recovery = 0
most_frequent_middle_recovery = 0
cond_recovery = 0
for name in tqdm(names):
    max_aa = []
    name_numeric = np.array([restype_order[l] for l in name])
    pred = np.array(res[name])
    pred_str = [aatype_to_str_sequence(nam[1:-1]) for nam in pred]
    
    design_middles, index, counts = np.unique(np.array(pred_str), return_counts=True, return_index=True)
    most_freq_idx = index[np.argmax(counts)]
    #design_middle = design_middles[np.argmax(counts)]
    most_freq_pred = pred[most_freq_idx]
    design_middle = aatype_to_str_sequence(most_freq_pred[1:-1])
    most_frequent_middle_recovery += (most_freq_pred == name_numeric)[1:-1].mean()
        
    recovery = (pred == name_numeric[None, :])
    design_recovery += recovery[:, 1:-1].mean()
    cond_recovery += np.concatenate([recovery[:, -1], recovery[:, 0]]).mean()

    final_design_idx = np.argsort(recovery[:,0].astype(float) + recovery[:,-1].astype(float))[0]
    final_design_name = pred[final_design_idx]
    final_design_recovery += (name_numeric[1:-1] == final_design_name[1:-1]).mean()
    
    for i in range(4):
        letters, counts = np.unique(np.array(res[name])[:,i], return_counts=True)
        max_aa.append(letters[np.argmax(counts)])
    max_aa = np.array(max_aa)
    max_cond_recovery += ((name_numeric[0] == max_aa[0]).astype(float) + (name_numeric[-1] == max_aa[-1]).astype(float)) / 2
    max_design_recovery += (name_numeric[1:-1] == max_aa[1:-1]).mean()

    designed_name = name[0] + design_middle + name[-1]
    

    metadata = metadatas[name]
    start_idx = metadata[most_freq_idx]['start_idx']
    end_idx = metadata[most_freq_idx]['end_idx']

    designed_names[name] = {
        'designed_name': designed_name,
        'start_idx': start_idx,
        'end_idx': end_idx,
        'start_state': metadata[most_freq_idx]['start_state'],
        'end_state': metadata[most_freq_idx]['end_state']
    }


cond_recovery = cond_recovery / len(names)
design_recovery = design_recovery / len(names)

max_cond_recovery = max_cond_recovery / len(names)
max_design_recovery = max_design_recovery / len(names)

final_design_recovery = final_design_recovery/ len(names)
most_frequent_middle_recovery = most_frequent_middle_recovery/ len(names)

print('cond_recovery', cond_recovery)
print('max_cond_recovery', max_cond_recovery)

print('design_recovery', design_recovery)
print('max_design_recovery', max_design_recovery)

print('final_design_recovery', final_design_recovery)
print('most_frequent_middle_recovery', most_frequent_middle_recovery)
