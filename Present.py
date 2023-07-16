import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import MessagePassing

# from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims

# from ogb.utils.mol import bond_to_feature_vector
## 根据原子、键特征建立embedding模型，最底层
allowable_features = {
    'possible_atomic_num_list' : list(range(1, 119)) + ['misc'],
    'possible_chirality_list' : [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER'
    ],
    'possible_degree_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
    'possible_formal_charge_list' : [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 'misc'],
    'possible_numH_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
    'possible_number_radical_e_list': [0, 1, 2, 3, 4, 'misc'],
    'possible_hybridization_list' : [
        'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'misc'
        ],
    'possible_is_aromatic_list': [False, True],
    'possible_is_in_ring_list': [False, True],
    'possible_bond_type_list' : [
        'SINGLE',
        'DOUBLE',
        'TRIPLE',
        'AROMATIC',
        'misc'
    ],
    'possible_bond_stereo_list': [
        'STEREONONE',
        'STEREOZ',
        'STEREOE',
        'STEREOCIS',
        'STEREOTRANS',
        'STEREOANY',
    ],
    'possible_is_conjugated_list': [False, True],
    'possible_distance*10': list(range(1,25))+['misc'],
}

def get_atom_feature_dims():
    return list(map(len, [
        allowable_features['possible_atomic_num_list'],
        # allowable_features['possible_chirality_list'],
        # allowable_features['possible_degree_list'],
        # allowable_features['possible_formal_charge_list'],
        # allowable_features['possible_numH_list'],
        # allowable_features['possible_number_radical_e_list'],
        # allowable_features['possible_hybridization_list'],
        # allowable_features['possible_is_aromatic_list'],
        # allowable_features['possible_is_in_ring_list']
        ]))


def get_bond_feature_dims():
    return list(map(len, [
        # allowable_features['possible_distance*10'],
        allowable_features['possible_bond_type_list'],
        # allowable_features['possible_bond_stereo_list'],
        # allowable_features['possible_is_conjugated_list']
        ]))




def safe_index(l, e):
    """
    Return index of element e in list l. If e is not present, return the last index
    """
    try:
        return l.index(e)
    except:
        return len(l) - 1

def atom_to_feature_vector(i,struct):
    """
    Converts rdkit atom object to feature list of indices
    :param mol: rdkit atom object
    :return: list
    """
    atom_feature = [
            safe_index(allowable_features['possible_atomic_num_list'], struct.get_atomic_numbers()[i]),
            # allowable_features['possible_chirality_list'].index(str(atom.GetChiralTag())),
            # safe_index(allowable_features['possible_degree_list'], atom.GetTotalDegree()),
            # safe_index(allowable_features['possible_formal_charge_list'], atom.GetFormalCharge()),
            # safe_index(allowable_features['possible_numH_list'], atom.GetTotalNumHs()),
            # safe_index(allowable_features['possible_number_radical_e_list'], atom.GetNumRadicalElectrons()),
            # safe_index(allowable_features['possible_hybridization_list'], str(atom.GetHybridization())),
            # allowable_features['possible_is_aromatic_list'].index(atom.GetIsAromatic()),
            # allowable_features['possible_is_in_ring_list'].index(atom.IsInRing()),
            ]
    return atom_feature
