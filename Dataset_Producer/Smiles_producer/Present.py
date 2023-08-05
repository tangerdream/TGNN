import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import MessagePassing

# from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims

# from ogb.utils.mol import bond_to_feature_vector
## 根据原子、键特征建立embedding模型，最底层
allowable_features = {
    'possible_atomic_num_list' : list(range(0, 119)),
    'possible_chirality_list' : [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER'
    ],
    'possible_degree_list' : ['misc',0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ],
    'possible_formal_charge_list' : [ 'misc',-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    'possible_numH_list' : ['misc',0, 1, 2, 3, 4, 5, 6, 7, 8],
    'possible_number_radical_e_list': ['misc',0, 1, 2, 3, 4, ],
    'possible_hybridization_list' : [
        'misc','SP', 'SP2', 'SP3', 'SP3D', 'SP3D2'
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
}


def safe_index(l, e):
    """
    Return index of element e in list l. If e is not present, return the last index
    """
    try:
        return l.index(e)
    except:
        return len(l) - 1
# # miscellaneous case
# i = safe_index(allowable_features['possible_atomic_num_list'], 'asdf')
# assert allowable_features['possible_atomic_num_list'][i] == 'misc'
# # normal case
# i = safe_index(allowable_features['possible_atomic_num_list'], 2)
# assert allowable_features['possible_atomic_num_list'][i] == 2

def atom_to_feature_vector(atom):
    """
    Converts rdkit atom object to feature list of indices
    :param mol: rdkit atom object
    :return: list
    """
    atom_feature = [
            safe_index(allowable_features['possible_atomic_num_list'], atom.GetAtomicNum()),
            safe_index(allowable_features['possible_chirality_list'], str(atom.GetChiralTag())),
            safe_index(allowable_features['possible_degree_list'], atom.GetTotalDegree()),
            safe_index(allowable_features['possible_formal_charge_list'], atom.GetFormalCharge()),
            safe_index(allowable_features['possible_numH_list'], atom.GetTotalNumHs()),
            safe_index(allowable_features['possible_number_radical_e_list'], atom.GetNumRadicalElectrons()),
            safe_index(allowable_features['possible_hybridization_list'], str(atom.GetHybridization())),
            allowable_features['possible_is_aromatic_list'].index(atom.GetIsAromatic()),
            allowable_features['possible_is_in_ring_list'].index(atom.IsInRing()),
            ]
    return atom_feature
# from rdkit import Chem
# mol = Chem.MolFromSmiles('Cl[C@H](/C=C/C)Br')
# atom = mol.GetAtomWithIdx(1)  # chiral carbon
# atom_feature = atom_to_feature_vector(atom)
# assert atom_feature == [5, 2, 4, 5, 1, 0, 2, 0, 0]


def get_atom_feature_dims():
    return list(map(len, [
        allowable_features['possible_atomic_num_list'],
        allowable_features['possible_chirality_list'],
        allowable_features['possible_degree_list'],
        allowable_features['possible_formal_charge_list'],
        allowable_features['possible_numH_list'],
        allowable_features['possible_number_radical_e_list'],
        allowable_features['possible_hybridization_list'],
        allowable_features['possible_is_aromatic_list'],
        allowable_features['possible_is_in_ring_list']
        ]))

def bond_to_feature_vector(bond):
    """
    Converts rdkit bond object to feature list of indices
    :param mol: rdkit bond object
    :return: list
    """
    bond_feature = [
                safe_index(allowable_features['possible_bond_type_list'], str(bond.GetBondType())),
                allowable_features['possible_bond_stereo_list'].index(str(bond.GetStereo())),
                allowable_features['possible_is_conjugated_list'].index(bond.GetIsConjugated()),
            ]
    return bond_feature
# uses same molecule as atom_to_feature_vector test
# bond = mol.GetBondWithIdx(2)  # double bond with stereochem
# bond_feature = bond_to_feature_vector(bond)
# assert bond_feature == [1, 2, 0]

def get_bond_feature_dims():
    return list(map(len, [
        allowable_features['possible_bond_type_list'],
        allowable_features['possible_bond_stereo_list'],
        allowable_features['possible_is_conjugated_list']
        ]))

def atom_feature_vector_to_dict(atom_feature):
    [atomic_num_idx,
    chirality_idx,
    degree_idx,
    formal_charge_idx,
    num_h_idx,
    number_radical_e_idx,
    hybridization_idx,
    is_aromatic_idx,
    is_in_ring_idx] = atom_feature

    feature_dict = {
        'atomic_num': allowable_features['possible_atomic_num_list'][atomic_num_idx],
        'chirality': allowable_features['possible_chirality_list'][chirality_idx],
        'degree': allowable_features['possible_degree_list'][degree_idx],
        'formal_charge': allowable_features['possible_formal_charge_list'][formal_charge_idx],
        'num_h': allowable_features['possible_numH_list'][num_h_idx],
        'num_rad_e': allowable_features['possible_number_radical_e_list'][number_radical_e_idx],
        'hybridization': allowable_features['possible_hybridization_list'][hybridization_idx],
        'is_aromatic': allowable_features['possible_is_aromatic_list'][is_aromatic_idx],
        'is_in_ring': allowable_features['possible_is_in_ring_list'][is_in_ring_idx]
    }

def bond_feature_vector_to_dict(bond_feature):
    [bond_type_idx,
    bond_stereo_idx,
    is_conjugated_idx] = bond_feature

    feature_dict = {
        'bond_type': allowable_features['possible_bond_type_list'][bond_type_idx],
        'bond_stereo': allowable_features['possible_bond_stereo_list'][bond_stereo_idx],
        'is_conjugated': allowable_features['possible_is_conjugated_list'][is_conjugated_idx]
    }

    return feature_dict
# # uses same bond as bond_to_feature_vector test
# bond_feature_dict = bond_feature_vector_to_dict(bond_feature)
# assert bond_feature_dict['bond_type'] == 'DOUBLE'
# assert bond_feature_dict['bond_stereo'] == 'STEREOE'
# assert bond_feature_dict['is_conjugated'] == False


def get_ELECTRONEGATIVITY(id):
    # print(id)
    ele=[0,2.2,0,0.98,1.57,2.04,2.55,3.04,3.44,3.98,0,0.93,1.31,1.61,1.9,2.19,2.58,3.16,0,0.82,1,1.36,1.54,1.63,1.66,1.55,1.83,1.88,1.91,1.9,1.65,1.81,2.01,2.18,2.55,2.96,3,0.82,0.95,1.22,1.33,1.6,2.16,1.9,2.2,2.28,2.2,1.93,1.69,1.78,1.96,2.05,2.1,2.66,2.6,0.79,0.89,1.1,1.12,1.13,1.14,1.13,1.17,1.2,1.2,1.22,1.23,1.24,1.24,1.25,1.1,1.27,1.3,1.5,2.36,1.9,2.2,2.2,2.28,2.54,2,1.62,2.33,2.02,2,2.2,0,0.7,0.89,1.1,1.3,1.5,1.38,1.36,1.28,1.3,1.3,1.3,1.3,1.3,1.3,1.3]
    return ele[id]