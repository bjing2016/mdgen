import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--pdbdir", required=True)
parser.add_argument("--mddir", default='share/4AA_sims_implicit')
parser.add_argument('--pdb_id', nargs='*', default=[])
args = parser.parse_args()

import mdgen.analysis
import tqdm, os
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acovf
import numpy as np
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

def do(name):
    feats, ref = mdgen.analysis.get_featurized_traj(f'{args.mddir}/{name}/{name}', sidechains=True, cossin=False)
    feats, traj = mdgen.analysis.get_featurized_traj(f'{args.pdbdir}/{name}', sidechains=True, cossin=False)


    md_autocorr = {}
    our_autocorr = {}
    subsample_autocorr = {}
    for i, feat in enumerate(feats.describe()):
        md_autocorr[feat] = acovf(np.sin(ref[:,i]), demean=False, adjusted=True) + acovf(np.cos(ref[:,i]), demean=False, adjusted=True)
        our_autocorr[feat] = acovf(np.sin(traj[:,i]), demean=False, adjusted=True) + acovf(np.cos(traj[:,i]), demean=False, adjusted=True)
        subsample_autocorr[feat] = acovf(np.sin(ref[::100,i]), demean=False, adjusted=True) + acovf(np.cos(ref[::100,i]), demean=False, adjusted=True)
        
    lags = 0.1 * (1 + np.arange(1000000))
    subsample_lags = 10 * (1 + np.arange(1000000))

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    
    for i, key in enumerate([key for key in md_autocorr if 'CHI' in key]):
        toplot = md_autocorr[key][1:]
        axs[0].plot(lags[:len(toplot)], toplot, color=colors[i%len(colors)])
    
        toplot = subsample_autocorr[key][1:]
        axs[0].scatter(subsample_lags[:len(toplot)], toplot, color=colors[i%len(colors)], label=key)
    
        toplot = our_autocorr[key][1:]
        axs[0].plot(lags[:len(toplot)], toplot, color=colors[i%len(colors)], linestyle='--')
    axs[0].set_title(f"{name} sidechains")
    axs[0].set_xscale('log')
    axs[0].set_xlim(0.1, 100)
    axs[0].set_ylim(0.5, 1)
    axs[0].set_xlabel('ps')
    axs[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3, fontsize=6)

    for i, key in enumerate([key for key in md_autocorr if 'CHI' not in key]):
        toplot = md_autocorr[key][1:]
        axs[1].plot(lags[:len(toplot)], toplot, color=colors[i%len(colors)])
    
        toplot = subsample_autocorr[key][1:]
        axs[1].scatter(subsample_lags[:len(toplot)], toplot, color=colors[i%len(colors)], label=key)
    
        toplot = our_autocorr[key][1:]
        axs[1].plot(lags[:len(toplot)], toplot, color=colors[i%len(colors)], linestyle='--')
    axs[1].set_title(f"{name} backbones")

    axs[1].set_xscale('log')
    axs[1].set_xlim(0.1, 100)
    axs[1].set_ylim(0.5, 1)
    axs[1].set_xlabel('ps')
    axs[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3, fontsize=6)

    fig.savefig(f"{args.pdbdir}/{name}.pdf", bbox_inches='tight', pad_inches=0)
    
if args.pdb_id:
    pdb_id = args.pdb_id
else:
    pdb_id = [nam.split('.')[0] for nam in os.listdir(args.pdbdir) if '.pdb' in nam]


for name in tqdm.tqdm(pdb_id): 
    if os.path.exists(f"{args.pdbdir}/{name}.pdf"): continue
    try:
        do(name)
    except Exception as e:
        print(name, e)