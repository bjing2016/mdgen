import torch
import numpy as np

from .rigid_utils import Rigid, Rotation
from . import residue_constants as rc
from .tensor_utils import batched_gather


def atom14_to_atom37(atom14: np.ndarray, aatype, atom14_mask=None):
    atom37 = batched_gather(
        atom14,
        rc.RESTYPE_ATOM37_TO_ATOM14[aatype],
        dim=-2,
        no_batch_dims=len(atom14.shape[:-2]),
    )
    atom37 *= rc.RESTYPE_ATOM37_MASK[aatype, :, None]
    if atom14_mask is not None:
        atom37_mask = batched_gather(
            atom14_mask,
            rc.RESTYPE_ATOM37_TO_ATOM14[aatype],
            dim=-1,
            no_batch_dims=len(atom14.shape[:-2]),
        )
        atom37_mask *= rc.RESTYPE_ATOM37_MASK[aatype]
        return atom37, atom37_mask
    else:
        return atom37


def atom37_to_atom14(atom37: np.ndarray, aatype, atom37_mask=None):
    atom14 = batched_gather(
        atom37,
        rc.RESTYPE_ATOM14_TO_ATOM37[aatype],
        dim=-2,
        no_batch_dims=len(atom37.shape[:-2]),
    )
    atom14 *= rc.RESTYPE_ATOM14_MASK[aatype, :, None]
    if atom37_mask is not None:
        atom14_mask = batched_gather(
            atom37_mask,
            rc.RESTYPE_ATOM14_TO_ATOM37[aatype],
            dim=-1,
            no_batch_dims=len(atom37.shape[:-2]),
        )
        atom14_mask *= rc.RESTYPE_ATOM14_MASK[aatype]
        return atom14, atom14_mask
    else:
        return atom14



def frames_torsions_to_atom37(
    frames: Rigid,
    torsions: torch.Tensor,
    aatype: torch.Tensor,
):
    atom14 = frames_torsions_to_atom14(frames, torsions, aatype)
    return atom14_to_atom37(atom14, aatype)


def frames_torsions_to_atom14(
    frames: Rigid, torsions: torch.Tensor, aatype: torch.Tensor
):
    if type(torsions) is np.ndarray:
        torsions = torch.from_numpy(torsions)
    if type(aatype) is np.ndarray:
        aatype = torch.from_numpy(aatype)
    default_frames = torch.from_numpy(rc.restype_rigid_group_default_frame).to(
        aatype.device
    )
    lit_positions = torch.from_numpy(rc.restype_atom14_rigid_group_positions).to(
        aatype.device
    )
    group_idx = torch.from_numpy(rc.restype_atom14_to_rigid_group).to(aatype.device)
    atom_mask = torch.from_numpy(rc.restype_atom14_mask).to(aatype.device)
    frames_out = torsion_angles_to_frames(frames, torsions, aatype, default_frames)
    return frames_and_literature_positions_to_atom14_pos(
        frames_out, aatype, default_frames, group_idx, atom_mask, lit_positions
    )


def atom37_to_torsions(all_atom_positions, aatype, all_atom_mask=None):

    if type(all_atom_positions) is np.ndarray:
        all_atom_positions = torch.from_numpy(all_atom_positions)
    if type(aatype) is np.ndarray:
        aatype = torch.from_numpy(aatype)
    if all_atom_mask is None:
        all_atom_mask = torch.from_numpy(rc.RESTYPE_ATOM37_MASK[aatype]).to(
            aatype.device
        )
    if type(all_atom_mask) is np.ndarray:
        all_atom_mask = torch.from_numpy(all_atom_mask)

    pad = all_atom_positions.new_zeros([*all_atom_positions.shape[:-3], 1, 37, 3])
    prev_all_atom_positions = torch.cat(
        [pad, all_atom_positions[..., :-1, :, :]], dim=-3
    )

    pad = all_atom_mask.new_zeros([*all_atom_mask.shape[:-2], 1, 37])
    prev_all_atom_mask = torch.cat([pad, all_atom_mask[..., :-1, :]], dim=-2)

    pre_omega_atom_pos = torch.cat(
        [prev_all_atom_positions[..., 1:3, :], all_atom_positions[..., :2, :]],
        dim=-2,
    )
    phi_atom_pos = torch.cat(
        [prev_all_atom_positions[..., 2:3, :], all_atom_positions[..., :3, :]],
        dim=-2,
    )
    psi_atom_pos = torch.cat(
        [all_atom_positions[..., :3, :], all_atom_positions[..., 4:5, :]],
        dim=-2,
    )

    pre_omega_mask = torch.prod(prev_all_atom_mask[..., 1:3], dim=-1) * torch.prod(
        all_atom_mask[..., :2], dim=-1
    )
    phi_mask = prev_all_atom_mask[..., 2] * torch.prod(
        all_atom_mask[..., :3], dim=-1, dtype=all_atom_mask.dtype
    )
    psi_mask = (
        torch.prod(all_atom_mask[..., :3], dim=-1, dtype=all_atom_mask.dtype)
        * all_atom_mask[..., 4]
    )

    chi_atom_indices = torch.as_tensor(get_chi_atom_indices(), device=aatype.device)

    atom_indices = chi_atom_indices[..., aatype, :, :]
    chis_atom_pos = batched_gather(
        all_atom_positions, atom_indices, -2, len(atom_indices.shape[:-2])
    )

    chi_angles_mask = list(rc.chi_angles_mask)
    chi_angles_mask.append([0.0, 0.0, 0.0, 0.0])
    chi_angles_mask = all_atom_mask.new_tensor(chi_angles_mask)

    chis_mask = chi_angles_mask[aatype, :]

    chi_angle_atoms_mask = batched_gather(
        all_atom_mask,
        atom_indices,
        dim=-1,
        no_batch_dims=len(atom_indices.shape[:-2]),
    )
    chi_angle_atoms_mask = torch.prod(
        chi_angle_atoms_mask, dim=-1, dtype=chi_angle_atoms_mask.dtype
    )
    chis_mask = chis_mask * chi_angle_atoms_mask

    torsions_atom_pos = torch.cat(
        [
            pre_omega_atom_pos[..., None, :, :],
            phi_atom_pos[..., None, :, :],
            psi_atom_pos[..., None, :, :],
            chis_atom_pos,
        ],
        dim=-3,
    )

    torsion_angles_mask = torch.cat(
        [
            pre_omega_mask[..., None],
            phi_mask[..., None],
            psi_mask[..., None],
            chis_mask,
        ],
        dim=-1,
    )

    torsion_frames = Rigid.from_3_points(
        torsions_atom_pos[..., 1, :],
        torsions_atom_pos[..., 2, :],
        torsions_atom_pos[..., 0, :],
        eps=1e-8,
    )

    fourth_atom_rel_pos = torsion_frames.invert().apply(torsions_atom_pos[..., 3, :])

    torsion_angles_sin_cos = torch.stack(
        [fourth_atom_rel_pos[..., 2], fourth_atom_rel_pos[..., 1]], dim=-1
    )

    denom = torch.sqrt(
        torch.sum(
            torch.square(torsion_angles_sin_cos),
            dim=-1,
            dtype=torsion_angles_sin_cos.dtype,
            keepdims=True,
        )
        + 1e-8
    )
    torsion_angles_sin_cos = torsion_angles_sin_cos / denom

    torsion_angles_sin_cos = (
        torsion_angles_sin_cos
        * all_atom_mask.new_tensor(
            [1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0],
        )[((None,) * len(torsion_angles_sin_cos.shape[:-2])) + (slice(None), None)]
    )

    return torsion_angles_sin_cos, torsion_angles_mask



def prot_to_frames(ca_coords, c_coords, n_coords):
    prot_frames = Rigid.from_3_points(
        torch.from_numpy(c_coords),
        torch.from_numpy(ca_coords),
        torch.from_numpy(n_coords),
    )
    rots = torch.eye(3)
    rots[0, 0] = -1
    rots[2, 2] = -1
    rots = Rotation(rot_mats=rots)
    return prot_frames.compose(Rigid(rots, None))

def atom14_to_frames(atom14):
    n_coords = atom14[:,:,rc.atom_order['N']]
    ca_coords = atom14[:,:,rc.atom_order['CA']]
    c_coords = atom14[:,:,rc.atom_order['C']]
    prot_frames = Rigid.from_3_points(
        c_coords,
        ca_coords,
        n_coords,
    )
    rots = torch.eye(3, device=atom14.device)[None,None].repeat(atom14.shape[0],atom14.shape[1], 1, 1)
    rots[:,:, 0, 0] = -1
    rots[:,:, 2, 2] = -1
    rots = Rotation(rot_mats=rots)
    return prot_frames.compose(Rigid(rots, None))
    



def frames_and_literature_positions_to_atom14_pos(
    r: Rigid,
    aatype: torch.Tensor,
    default_frames,
    group_idx,
    atom_mask,
    lit_positions,
):
    # [*, N, 14, 4, 4]
    default_4x4 = default_frames[aatype, ...]

    # [*, N, 14]
    group_mask = group_idx[aatype, ...]

    # [*, N, 14, 8]
    group_mask = torch.nn.functional.one_hot(
        group_mask,
        num_classes=default_frames.shape[-3],
    )

    # [*, N, 14, 8]
    t_atoms_to_global = r[..., None, :] * group_mask

    # [*, N, 14]
    t_atoms_to_global = t_atoms_to_global.map_tensor_fn(lambda x: torch.sum(x, dim=-1))

    # [*, N, 14, 1]
    atom_mask = atom_mask[aatype, ...].unsqueeze(-1)

    # [*, N, 14, 3]
    lit_positions = lit_positions[aatype, ...]
    pred_positions = t_atoms_to_global.apply(lit_positions)
    pred_positions = pred_positions * atom_mask

    return pred_positions


def torsion_angles_to_frames(
    r: Rigid,
    alpha: torch.Tensor,
    aatype: torch.Tensor,
    rrgdf: torch.Tensor,
):
    # [*, N, 8, 4, 4]
    default_4x4 = rrgdf[aatype, ...]

    # [*, N, 8] transformations, i.e.
    #   One [*, N, 8, 3, 3] rotation matrix and
    #   One [*, N, 8, 3]    translation matrix
    default_r = r.from_tensor_4x4(default_4x4)

    bb_rot = alpha.new_zeros((*((1,) * len(alpha.shape[:-1])), 2))
    bb_rot[..., 1] = 1

    # [*, N, 8, 2]
    alpha = torch.cat([bb_rot.expand(*alpha.shape[:-2], -1, -1), alpha], dim=-2)

    # [*, N, 8, 3, 3]
    # Produces rotation matrices of the form:
    # [
    #   [1, 0  , 0  ],
    #   [0, a_2,-a_1],
    #   [0, a_1, a_2]
    # ]
    # This follows the original code rather than the supplement, which uses
    # different indices.

    all_rots = alpha.new_zeros(default_r.get_rots().get_rot_mats().shape)
    all_rots[..., 0, 0] = 1
    all_rots[..., 1, 1] = alpha[..., 1]
    all_rots[..., 1, 2] = -alpha[..., 0]
    all_rots[..., 2, 1:] = alpha

    all_rots = Rigid(Rotation(rot_mats=all_rots), None)

    all_frames = default_r.compose(all_rots)

    chi2_frame_to_frame = all_frames[..., 5]
    chi3_frame_to_frame = all_frames[..., 6]
    chi4_frame_to_frame = all_frames[..., 7]

    chi1_frame_to_bb = all_frames[..., 4]
    chi2_frame_to_bb = chi1_frame_to_bb.compose(chi2_frame_to_frame)
    chi3_frame_to_bb = chi2_frame_to_bb.compose(chi3_frame_to_frame)
    chi4_frame_to_bb = chi3_frame_to_bb.compose(chi4_frame_to_frame)

    all_frames_to_bb = Rigid.cat(
        [
            all_frames[..., :5],
            chi2_frame_to_bb.unsqueeze(-1),
            chi3_frame_to_bb.unsqueeze(-1),
            chi4_frame_to_bb.unsqueeze(-1),
        ],
        dim=-1,
    )

    all_frames_to_global = r[..., None].compose(all_frames_to_bb)

    return all_frames_to_global


def get_chi_atom_indices():
    """Returns atom indices needed to compute chi angles for all residue types.

    Returns:
      A tensor of shape [residue_types=21, chis=4, atoms=4]. The residue types are
      in the order specified in rc.restypes + unknown residue type
      at the end. For chi angles which are not defined on the residue, the
      positions indices are by default set to 0.
    """
    chi_atom_indices = []
    for residue_name in rc.restypes:
        residue_name = rc.restype_1to3[residue_name]
        residue_chi_angles = rc.chi_angles_atoms[residue_name]
        atom_indices = []
        for chi_angle in residue_chi_angles:
            atom_indices.append([rc.atom_order[atom] for atom in chi_angle])
        for _ in range(4 - len(atom_indices)):
            atom_indices.append([0, 0, 0, 0])  # For chi angles not defined on the AA.
        chi_atom_indices.append(atom_indices)

    chi_atom_indices.append([[0, 0, 0, 0]] * 4)  # For UNKNOWN residue.

    return chi_atom_indices