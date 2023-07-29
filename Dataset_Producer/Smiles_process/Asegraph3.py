import numpy as np
from Bondconnect import find_bonds_molecules
from ase.symbols import chemical_symbols

from Present import (allowable_features, atom_to_feature_vector,bond_to_feature_vector, atom_feature_vector_to_dict, bond_feature_vector_to_dict,get_ELECTRONEGATIVITY)
from ase.data import atomic_masses, atomic_numbers
from ogb.utils.torch_util import replace_numpy_with_torchtensor

#from ogb.utils.features import (allowable_features, atom_to_feature_vector,bond_to_feature_vector, atom_feature_vector_to_dict, bond_feature_vector_to_dict)
from rdkit import Chem


def cacul_distance(atom1,atom2,a,b,c):
    x1=atom1.position[0]
    y1=atom1.position[1]
    z1=atom1.position[2]
    x2=atom2.position[0]
    y2=atom2.position[1]
    z2=atom2.position[2]
    dd=[]
    for xx in [x2,x2-a,x2+a]:
        for yy in [y2,y2-b,y2+b]:
            for zz in [z2,z2-c,z2+c]:
                d=np.sqrt((x1-xx)**2+(y1-yy)**2+(z1-zz)**2)
                dd.append(d)


    distance=min(dd)
    return distance

#|||||||||||||||||||||||for new pos
def get_id(id):
    return id
def get_mass(id):
    return atomic_masses[id]
def get_geometry(id):
    return 1
def get_electronegativity(id):
    return get_ELECTRONEGATIVITY(id)
#
def get_properties(x,properties_list):
    properties=[]
    for pro in properties_list:
        property=[]
        for id in x:
            property.append(eval('get_'+pro)(id[0]))
        properties.append(property)
    return properties

def cores(positions,x,properties_list):
    x=x[0:len(positions[:,0]),:]
    core_dic={}
    properties=get_properties(x,properties_list)
    for i,pro in enumerate(properties_list):
        prop=np.array(properties[i]).reshape(1, -1)
        try:
            assert not np.sum(prop)==0
            core=prop@ positions/np.sum(prop)
            core_dic['core_'+pro]=core
        except:
            prop=prop+1e-5
            core=prop@ positions/np.sum(prop)
            core_dic['core_'+pro]=core

    return core_dic

def Dis(positions,core_dic:dict,maxnodes,properties_list,p=2):
    length=len(positions[:,0])
    clist=np.array([core_dic['core_'+pro] for pro in properties_list])
    wid=len(clist)
    D=np.zeros([maxnodes,wid],dtype=float)


    for i,c in enumerate(clist):
        dis2 = np.sum(np.abs(positions - c) ** p, axis=1)  # 计算
        dis = np.power(dis2, 1 / p)
        D[0:length,i]=dis
    return D


def new_position(positions,x,maxnodes,properties_list,type='Dis'):
    #中心
    core_dic=cores(positions,x,properties_list)
    #距离
    if type=='Dis':
       new_pos=Dis(positions,core_dic,maxnodes,properties_list)
    return new_pos,core_dic
#|||||||||||||||||||||||
def get_edge_index(structure):
    bondsdic, molsdic = find_bonds_molecules(structure)
    edgelist=[]
    for key in bondsdic.keys():
        if bondsdic[key]!=[]:
            for bond in bondsdic[key]:
                edgelist.append([key,bond])
    edge_index = np.array(edgelist, dtype = np.int64).T
    return edge_index

def get_bond_length(struct,edge_index):
    bond_length = []
    a, b, c, alpha, beta, gamma = np.round(struct.cell.cellpar(), 2)
    for i in range(len(edge_index[0])):
        distance = cacul_distance(struct[edge_index[0][i]],struct[edge_index[1][i]],a, b, c)
        bond_length.append([distance])
    bond_length = np.array(bond_length, dtype=np.float)
    return bond_length


def get_AtomSymbol(struct,maxnodes):
    symbols = []
    for i in range(maxnodes):
        try:
            symbols.append(chemical_symbols[struct[i].number])
        except:
            symbols.append('None')
    return symbols


def struct2graph(structure, jobname, y_atoms=None, maxnodes=50, AtomSymbol=False,
                 properties_list=['geometry', 'mass', 'id', 'electronegativity']):
    struct = structure.copy()
    y_atoms_state = np.zeros((1, maxnodes), dtype=np.float32)
    atom_features_list = np.zeros((maxnodes, 1), dtype=np.int64)
    y_marking = np.zeros((1, maxnodes), dtype=np.float32)
    for i, atom in enumerate(struct):
        atom_features_list[i] = struct.get_atomic_numbers()[i]
        if y_atoms != None:
            y_atoms_state[0][i] = y_atoms[i]
            y_marking[0][i] = 1
    x = atom_features_list
    y = y_atoms_state
    y_mark = y_marking

    positions = np.array(struct.get_positions(), dtype=np.float32)
    pos_zeros = np.zeros((maxnodes - len(positions[:, 0]), 3), dtype=np.float32)

    graph = dict()

    if jobname == 'smiles':
        graph['positions'] = np.r_[positions, pos_zeros]
        graph['new_pos'], graph['core_dic'] = new_position(positions, x, maxnodes, properties_list)
        graph['AtomNum'] = len(struct)
        if AtomSymbol:
            graph['AtomSymbol'] = get_AtomSymbol(struct, maxnodes)
        if y_atoms != None:
            graph['y'] = y
            graph['y_mark'] = y_mark
    elif jobname == 'crystal':
        graph['edge_index'] = get_edge_index(struct)
        graph['bond_length'] = get_bond_length(struct, graph['edge_index'])
        graph['node_feat'] = x
        graph['num_nodes'] = len(x)
        graph['positions'] = np.r_[positions, pos_zeros]
        graph['new_pos'], graph['core_dic'] = new_position(positions, x, maxnodes, properties_list)
        graph['AtomNum'] = len(struct)
        if AtomSymbol:
            graph['AtomSymbol'] = get_AtomSymbol(struct, maxnodes)
        if y_atoms != None:
            graph['y'] = y
            graph['y_mark'] = y_mark
    return graph






def smiles2graph(smiles_string,maxnodes):
    """
    Converts SMILES string to graph Data object
    :input: SMILES string (str)
    :return: graph object
    """

    mol = Chem.MolFromSmiles(smiles_string)
    mol = Chem.AddHs(mol)

    # atoms
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_features_list.append(atom_to_feature_vector(atom))
    x = np.array(atom_features_list, dtype = np.int64)

    x_zeros = np.zeros((maxnodes - len(x[:, 0]), len(x[0, :])), dtype=np.float32)

    # bonds
    num_bond_features = 3  # bond type, bond stereo, is_conjugated
    if len(mol.GetBonds()) > 0: # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            edge_feature = bond_to_feature_vector(bond)

            # add edges in both directions
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = np.array(edges_list, dtype = np.int64).T

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = np.array(edge_features_list, dtype = np.int64)

    else:   # mol has no bonds
        edge_index = np.empty((2, 0), dtype = np.int64)
        edge_attr = np.empty((0, num_bond_features), dtype = np.int64)

    return np.r_[x, x_zeros],edge_index,edge_attr
