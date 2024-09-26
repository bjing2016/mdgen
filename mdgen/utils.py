import numpy as np
import scipy
import torch
from . import protein
from .geometry import atom14_to_atom37

def get_offsets(ref_frame, rigids):
    B, T, L = rigids.shape
    if T > 500000:
        offsets1 = ref_frame.invert().compose(rigids[:, : 500000]).to_tensor_7()
        offsets2 = ref_frame.invert().compose(rigids[:, 500000:]).to_tensor_7()
        return torch.cat([offsets1, offsets2], 1)
    else:
        return ref_frame.invert().compose(rigids).to_tensor_7()

def simplex_proj(seq):
    """Algorithm from https://arxiv.org/abs/1309.1541 Weiran Wang, Miguel Á. Carreira-Perpiñán"""
    Y = seq.reshape(-1, seq.shape[-1])
    N, K = Y.shape
    X, _ = torch.sort(Y, dim=-1, descending=True)
    X_cumsum = torch.cumsum(X, dim=-1) - 1
    div_seq = torch.arange(1, K + 1, dtype=Y.dtype, device=Y.device)
    Xtmp = X_cumsum / div_seq.unsqueeze(0)

    greater_than_Xtmp = (X > Xtmp).sum(dim=1, keepdim=True)
    row_indices = torch.arange(N, dtype=torch.long, device=Y.device).unsqueeze(1)
    selected_Xtmp = Xtmp[row_indices, greater_than_Xtmp - 1]

    X = torch.max(Y - selected_Xtmp, torch.zeros_like(Y))
    return X.view(seq.shape)

class DirichletConditionalFlow:
    def __init__(self, K=20, alpha_min=1, alpha_max=100, alpha_spacing=0.01):
        self.alphas = np.arange(alpha_min, alpha_max + alpha_spacing, alpha_spacing)
        self.beta_cdfs = []
        self.bs = np.linspace(0, 1, 1000)
        for alph in self.alphas:
            self.beta_cdfs.append(scipy.special.betainc(alph, K-1, self.bs))
        self.beta_cdfs = np.array(self.beta_cdfs)
        self.beta_cdfs_derivative = np.diff(self.beta_cdfs, axis=0) / alpha_spacing
        self.alpha_spacing = alpha_spacing
        self.K = K

    def c_factor(self, bs, alpha):
        # if the bs is close to the edge of the simplex in one of its entries, then we want the c factor to be 0 for high alphas.
        # That is the rationale for why we return 0s in the case of an overflow.

        beta = scipy.special.beta(alpha, self.K - 1) # betafunction(alpha, K-1)
        beta_div = np.where(bs < 1, beta / ((1 - bs) ** (self.K - 1)), 0)
        beta_div_full = np.where((bs ** (alpha - 1)) > 0, beta_div / (bs ** (alpha - 1)), 0)

        I_func = self.beta_cdfs_derivative[np.argmin(np.abs(alpha - self.alphas))]
        interp = -np.interp(bs, self.bs, I_func)

        final = interp * beta_div_full
        return final

def atom14_to_pdb(atom14, aatype, path):
    prots = []
    for i, pos in enumerate(atom14):
        pos = atom14_to_atom37(pos, aatype)
        prots.append(create_full_prot(pos, aatype=aatype))
    with open(path, 'w') as f:
        f.write(prots_to_pdb(prots))


def create_full_prot(
        atom37: np.ndarray,
        aatype=None,
        b_factors=None,
    ):
    assert atom37.ndim == 3
    assert atom37.shape[-1] == 3
    assert atom37.shape[-2] == 37
    n = atom37.shape[0]
    residue_index = np.arange(n)
    atom37_mask = np.sum(np.abs(atom37), axis=-1) > 1e-7
    if b_factors is None:
        b_factors = np.zeros([n, 37])
    if aatype is None:
        aatype = np.zeros(n, dtype=int)
    chain_index = np.zeros(n, dtype=int)
    return protein.Protein(
        atom_positions=atom37,
        atom_mask=atom37_mask,
        aatype=aatype,
        residue_index=residue_index,
        b_factors=b_factors,
        chain_index=chain_index
    )


def prots_to_pdb(prots):
    ss = ''
    for i, prot in enumerate(prots):
        ss += f'MODEL {i}\n'
        prot = protein.to_pdb(prot)
        ss += '\n'.join(prot.split('\n')[2:-3])
        ss += '\nENDMDL\n'
    return ss

