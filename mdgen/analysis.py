import json
import os

import numpy as np
import pyemma
from tqdm import tqdm

def get_featurizer(name, sidechains=False, cossin=True):
    feat = pyemma.coordinates.featurizer(name+'.pdb')
    feat.add_backbone_torsions(cossin=cossin)
    if sidechains:
        feat.add_sidechain_torsions(cossin=cossin)
    return feat
    
def get_featurized_traj(name, sidechains=False, cossin=True):
    feat = pyemma.coordinates.featurizer(name+'.pdb')
    feat.add_backbone_torsions(cossin=cossin)
    if sidechains:
        feat.add_sidechain_torsions(cossin=cossin)
    traj = pyemma.coordinates.load(name+'.xtc', features=feat)
    return feat, traj

def get_featurized_atlas_traj(name, sidechains=False, cossin=True):
    feat = pyemma.coordinates.featurizer(name+'.pdb')
    feat.add_backbone_torsions(cossin=cossin)
    if sidechains:
        feat.add_sidechain_torsions(cossin=cossin)
    traj = pyemma.coordinates.load(name+'_prod_R1_fit.xtc', features=feat)
    return feat, traj

def get_tica(traj, lag=1000):
    tica = pyemma.coordinates.tica(traj, lag=lag, kinetic_map=True)
    # lag time 100 ps = 0.1 ns
    return tica, tica.transform(traj)

def get_kmeans(traj):
    kmeans = pyemma.coordinates.cluster_kmeans(traj, k=100, max_iter=100, fixed_seed=137)
    return kmeans, kmeans.transform(traj)[:,0]

def get_msm(traj, lag=1000, nstates=10):
    msm = pyemma.msm.estimate_markov_model(traj, lag=lag)
    pcca = msm.pcca(nstates)
    assert len(msm.metastable_assignments) == 100
    cmsm = pyemma.msm.estimate_markov_model(msm.metastable_assignments[traj], lag=lag)
    return msm, pcca, cmsm

def discretize(traj, kmeans, msm):
    return msm.metastable_assignments[kmeans.transform(traj)[:,0]]

def load_tps_ensemble(name, directory):
    metadata = json.load(open(os.path.join(directory, f'{name}_metadata.json'),'rb'))
    all_feats = []
    all_traj = []
    for i, meta_dict in tqdm(enumerate(metadata)):
        feats, traj = get_featurized_traj(f'{directory}/{name}_{i}', sidechains=True)
        all_feats.append(feats)
        all_traj.append(traj)
    return all_feats, all_traj


def sample_tp(trans, start_state, end_state, traj_len, n_samples):
    s_1 = start_state
    s_N = end_state
    N = traj_len

    s_t = np.ones(n_samples, dtype=int) * s_1
    states = [s_t]
    for t in range(1, N - 1):
        numerator = np.linalg.matrix_power(trans, N - t - 1)[:, s_N] * trans[s_t, :]
        probs = numerator / np.linalg.matrix_power(trans, N - t)[s_t, s_N][:, None]
        s_t = np.zeros(n_samples, dtype=int)
        for n in range(n_samples):
            s_t[n] = np.random.choice(np.arange(len(trans)), 1, p=probs[n])
        states.append(s_t)
    states.append(np.ones(n_samples, dtype=int) * s_N)
    return np.stack(states, axis=1)


def get_tp_likelihood(tp, trans):
    N = tp.shape[1]
    n_samples = tp.shape[0]
    s_N = tp[0, -1]
    trans_probs = []
    for i in range(N - 1):
        t = i + 1
        s_t = tp[:, i]
        numerator = np.linalg.matrix_power(trans, N - t - 1)[:, s_N] * trans[s_t, :]
        probs = numerator / np.linalg.matrix_power(trans, N - t)[s_t, s_N][:, None]

        s_tp1 = tp[:, i + 1]
        trans_prob = probs[np.arange(n_samples), s_tp1]
        trans_probs.append(trans_prob)
    probs = np.stack(trans_probs, axis=1)
    probs[np.isnan(probs)] = 0
    return probs


def get_state_probs(tp, num_states=10):
    stationary = np.bincount(tp.reshape(-1), minlength=num_states)
    return stationary / stationary.sum()