import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--mddir', type=str, default='share/4AA_sims')
parser.add_argument('--pdbdir', type=str, required=True) 
parser.add_argument('--save', action='store_true')
parser.add_argument('--plot', action='store_true')
parser.add_argument('--save_name', type=str, default='out.pkl')
parser.add_argument('--pdb_id', nargs='*', default=[])
parser.add_argument('--no_msm', action='store_true')
parser.add_argument('--no_decorr', action='store_true')
parser.add_argument('--no_traj_msm', action='store_true')
parser.add_argument('--truncate', type=int, default=None)
parser.add_argument('--msm_lag', type=int, default=10)
parser.add_argument('--ito', action='store_true')
parser.add_argument('--num_workers', type=int, default=1)

args = parser.parse_args()

import mdgen.analysis
import pyemma, tqdm, os, pickle
from scipy.spatial.distance import jensenshannon
from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acovf, acf
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

def main(name):
    out = {}
    np.random.seed(137)
    fig, axs = plt.subplots(4, 4, figsize=(20, 20))

    ### BACKBONE torsion marginals PLOT ONLY
    if args.plot:
        feats, traj = mdgen.analysis.get_featurized_traj(f'{args.pdbdir}/{name}', sidechains=False, cossin=False)
        if args.truncate: traj = traj[:args.truncate]
        feats, ref = mdgen.analysis.get_featurized_traj(f'{args.mddir}/{name}/{name}', sidechains=False, cossin=False)
        pyemma.plots.plot_feature_histograms(ref, feature_labels=feats, ax=axs[0,0], color=colors[0])
        pyemma.plots.plot_feature_histograms(traj, ax=axs[0,0], color=colors[1])
        axs[0,0].set_title('BB torsions')

    
    ### JENSEN SHANNON DISTANCES ON ALL TORSIONS
    feats, traj = mdgen.analysis.get_featurized_traj(f'{args.pdbdir}/{name}', sidechains=True, cossin=False)
    if args.truncate: traj = traj[:args.truncate]
    feats, ref = mdgen.analysis.get_featurized_traj(f'{args.mddir}/{name}/{name}', sidechains=True, cossin=False)

    out['features'] = feats.describe()

    out['JSD'] = {}
    for i, feat in enumerate(feats.describe()):
        ref_p = np.histogram(ref[:,i], range=(-np.pi, np.pi), bins=100)[0]
        traj_p = np.histogram(traj[:,i], range=(-np.pi, np.pi), bins=100)[0]
        out['JSD'][feat] = jensenshannon(ref_p, traj_p)

    for i in [1,3]:
        ref_p = np.histogram2d(*ref[:,i:i+2].T, range=((-np.pi, np.pi),(-np.pi,np.pi)), bins=50)[0]
        traj_p = np.histogram2d(*traj[:,i:i+2].T, range=((-np.pi, np.pi),(-np.pi,np.pi)), bins=50)[0]
        out['JSD']['|'.join(feats.describe()[i:i+2])] = jensenshannon(ref_p.flatten(), traj_p.flatten())

    ############ Torsion decorrelations
    if args.no_decorr:
        pass
    else:
        out['md_decorrelation'] = {}
        for i, feat in enumerate(feats.describe()):
            
            autocorr = acovf(np.sin(ref[:,i]), demean=False, adjusted=True, nlag=100000) + acovf(np.cos(ref[:,i]), demean=False, adjusted=True, nlag=100000)
            baseline = np.sin(ref[:,i]).mean()**2 + np.cos(ref[:,i]).mean()**2
            # E[(X(t) - E[X(t)]) * (X(t+dt) - E[X(t+dt)])] = E[X(t)X(t+dt) - E[X(t)]X(t+dt) - X(t)E[X(t+dt)] + E[X(t)]E[X(t+dt)]] = E[X(t)X(t+dt)] - E[X]**2
            lags = 1 + np.arange(len(autocorr))
            if 'PHI' in feat or 'PSI' in feat:
                axs[0,1].plot(lags, (autocorr - baseline) / (1-baseline), color=colors[i%len(colors)])
            else:
                axs[0,2].plot(lags, (autocorr - baseline) / (1-baseline), color=colors[i%len(colors)])
    
            out['md_decorrelation'][feat] = (autocorr.astype(np.float16) - baseline) / (1-baseline)
           
        axs[0,1].set_title('Backbone decorrelation')
        axs[0,2].set_title('Sidechain decorrelation')
        axs[0,1].set_xscale('log')
        axs[0,2].set_xscale('log')
    
        out['our_decorrelation'] = {}
        for i, feat in enumerate(feats.describe()):
            
            autocorr = acovf(np.sin(traj[:,i]), demean=False, adjusted=True, nlag=1 if args.ito else 1000) + acovf(np.cos(traj[:,i]), demean=False, adjusted=True, nlag=1 if args.ito else 1000)
            baseline = np.sin(traj[:,i]).mean()**2 + np.cos(traj[:,i]).mean()**2
            # E[(X(t) - E[X(t)]) * (X(t+dt) - E[X(t+dt)])] = E[X(t)X(t+dt) - E[X(t)]X(t+dt) - X(t)E[X(t+dt)] + E[X(t)]E[X(t+dt)]] = E[X(t)X(t+dt)] - E[X]**2
            lags = 1 + np.arange(len(autocorr))
            if 'PHI' in feat or 'PSI' in feat:
                axs[1,1].plot(lags, (autocorr - baseline) / (1-baseline), color=colors[i%len(colors)])
            else:
                axs[1,2].plot(lags, (autocorr - baseline) / (1-baseline), color=colors[i%len(colors)])
    
            out['our_decorrelation'][feat] = (autocorr.astype(np.float16) - baseline) / (1-baseline)
    
        axs[1,1].set_title('Backbone decorrelation')
        axs[1,2].set_title('Sidechain decorrelation')
        axs[1,1].set_xscale('log')
        axs[1,2].set_xscale('log')

    ####### TICA #############
    feats, traj = mdgen.analysis.get_featurized_traj(f'{args.pdbdir}/{name}', sidechains=True, cossin=True)
    if args.truncate: traj = traj[:args.truncate]
    feats, ref = mdgen.analysis.get_featurized_traj(f'{args.mddir}/{name}/{name}', sidechains=True, cossin=True)

    tica, _ = mdgen.analysis.get_tica(ref)
    ref_tica = tica.transform(ref)
    traj_tica = tica.transform(traj)
    
    tica_0_min = min(ref_tica[:,0].min(), traj_tica[:,0].min())
    tica_0_max = max(ref_tica[:,0].max(), traj_tica[:,0].max())

    tica_1_min = min(ref_tica[:,1].min(), traj_tica[:,1].min())
    tica_1_max = max(ref_tica[:,1].max(), traj_tica[:,1].max())
    
    ref_p = np.histogram(ref_tica[:,0], range=(tica_0_min, tica_0_max), bins=100)[0]
    traj_p = np.histogram(traj_tica[:,0], range=(tica_0_min, tica_0_max), bins=100)[0]
    out['JSD']['TICA-0'] = jensenshannon(ref_p, traj_p)

    ref_p = np.histogram2d(*ref_tica[:,:2].T, range=((tica_0_min, tica_0_max),(tica_1_min, tica_1_max)), bins=50)[0]
    traj_p = np.histogram2d(*traj_tica[:,:2].T, range=((tica_0_min, tica_0_max),(tica_1_min, tica_1_max)), bins=50)[0]
    out['JSD']['TICA-0,1'] = jensenshannon(ref_p.flatten(), traj_p.flatten())
    
    #### 1,0, 1,1 TICA FES
    if args.plot:
        pyemma.plots.plot_free_energy(*ref_tica[::100, :2].T, ax=axs[2,0], cbar=False)
        pyemma.plots.plot_free_energy(*traj_tica[:, :2].T, ax=axs[2,1], cbar=False)
        axs[2,0].set_title('TICA FES (MD)')
        axs[2,1].set_title('TICA FES (ours)')


    ####### TICA decorrelation ########
    if args.no_decorr:
        pass
    else:
        # x, adjusted=False, demean=True, fft=True, missing='none', nlag=None
        autocorr = acovf(ref_tica[:,0], nlag=100000, adjusted=True, demean=False)
        out['md_decorrelation']['tica'] = autocorr.astype(np.float16)
        if args.plot:
            axs[0,3].plot(autocorr)
            axs[0,3].set_title('MD TICA')
        
    
        autocorr = acovf(traj_tica[:,0], nlag=1 if args.ito else 1000, adjusted=True, demean=False)
        out['our_decorrelation']['tica'] = autocorr.astype(np.float16)
        if args.plot:
            axs[1,3].plot(autocorr)
            axs[1,3].set_title('Traj TICA')

    ###### Markov state model stuff #################
    if not args.no_msm:
        kmeans, ref_kmeans = mdgen.analysis.get_kmeans(tica.transform(ref))
        try:
            msm, pcca, cmsm = mdgen.analysis.get_msm(ref_kmeans, nstates=10)
    
            out['kmeans'] = kmeans
            out['msm'] = msm
            out['pcca'] = pcca
            out['cmsm'] = cmsm
        
            traj_discrete = mdgen.analysis.discretize(tica.transform(traj), kmeans, msm)
            ref_discrete = mdgen.analysis.discretize(tica.transform(ref), kmeans, msm)
            out['traj_metastable_probs'] = (traj_discrete == np.arange(10)[:,None]).mean(1)
            out['ref_metastable_probs'] = (ref_discrete == np.arange(10)[:,None]).mean(1)
            ######### 
        
            msm_transition_matrix = np.eye(10)
            for a, i in enumerate(cmsm.active_set):
                for b, j in enumerate(cmsm.active_set):
                    msm_transition_matrix[i,j] = cmsm.transition_matrix[a,b]
    
            out['msm_transition_matrix'] = msm_transition_matrix
            out['pcca_pi'] = pcca._pi_coarse
        
            msm_pi = np.zeros(10)
            msm_pi[cmsm.active_set] = cmsm.pi
            out['msm_pi'] = msm_pi
            
            if args.no_traj_msm:
                pass
            else:
                
                traj_msm = pyemma.msm.estimate_markov_model(traj_discrete, lag=args.msm_lag)
                out['traj_msm'] = traj_msm
        
                traj_transition_matrix = np.eye(10)
                for a, i in enumerate(traj_msm.active_set):
                    for b, j in enumerate(traj_msm.active_set):
                        traj_transition_matrix[i,j] = traj_msm.transition_matrix[a,b]
                out['traj_transition_matrix'] = traj_transition_matrix
                
            
                traj_pi = np.zeros(10)
                traj_pi[traj_msm.active_set] = traj_msm.pi
                out['traj_pi'] = traj_pi
                
        except Exception as e:
            print('ERROR', e, name, flush=True)
    
    if args.plot:
        fig.savefig(f'{args.pdbdir}/{name}.pdf')
    
    return name, out

if args.pdb_id:
    pdb_id = args.pdb_id
else:
    pdb_id = [nam.split('.')[0] for nam in os.listdir(args.pdbdir) if '.pdb' in nam and not '_traj' in nam]
pdb_id = [nam for nam in pdb_id if os.path.exists(f'{args.pdbdir}/{nam}.xtc')]
print('number of trajectories', len(pdb_id))

    
if args.num_workers > 1:
    p = Pool(args.num_workers)
    p.__enter__()
    __map__ = p.imap
else:
    __map__ = map
out = dict(tqdm.tqdm(__map__(main, pdb_id), total=len(pdb_id)))
if args.num_workers > 1:
    p.__exit__(None, None, None)

if args.save:
    with open(f"{args.pdbdir}/{args.save_name}", 'wb') as f:
        f.write(pickle.dumps(out))


