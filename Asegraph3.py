import numpy as np
from bondconnect import find_bonds_molecules
from Present import atom_to_feature_vector
from ase.symbols import chemical_symbols
from rdkit.Chem import rdPartialCharges
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


def cal_partial_charge(mol):
    charges = []
    rdPartialCharges.ComputeGasteigerCharges(mol)  # charge stored in atoms
    atoms = mol.GetAtoms()
    for atom in atoms:
        charge = atom.GetProp("_GasteigerCharge")
        charges.append(float(charge))
    return np.array(charges)

def cores(positions,mol):
    mass=[]
    id=[]
    for atom in mol.GetAtoms():
        mass.append(atom.GetMass())
        id.append(atom.GetAtomicNum())
    mass = np.array(mass).reshape(1, -1)
    id = np.array(id).reshape(1, -1)
    charge=cal_partial_charge(mol).reshape(1,-1)

    core_mass=mass @ positions/np.sum(mass)#质心
    core_id=id @ positions/np.sum(id)#序号中心


    if np.sum(np.abs(charge))==0:
        charge += 1e-5
    core_charge = np.abs(charge) @ positions / np.sum(np.abs(charge))  # 电荷中心（伪）
    core_geometry = np.mean(positions, axis=0).reshape(1,-1)
    core_dic = {
        'core_mass': core_mass,
        'core_id': core_id,
        'core_charge': core_charge,
        'core_geometry': core_geometry
    }
    return core_dic

def Dis(positions,core_dic:dict,maxnodes,p=2):
    length=len(positions[:,0])
    clist=np.array([core_dic['core_geometry'],core_dic['core_mass'],core_dic['core_charge'],core_dic['core_id']])
    wid=len(clist)
    D=np.zeros([maxnodes,wid],dtype=float)


    for i,c in enumerate(clist):
        dis2 = np.sum(np.abs(positions - c) ** p, axis=1)  # 计算
        dis = np.power(dis2, 1 / p)
        D[0:length,i]=dis
    return D


def new_position(positions,mol,maxnodes):
    #中心
    core_dic=cores(positions,mol)
    #距离
    new_pos=Dis(positions,core_dic,maxnodes)
    return new_pos,core_dic


def struct2graph(structure,mol,y_atoms=None,maxnodes=50,AtomSymbol=False):
    struct=structure.copy()
    y_atoms_state = np.zeros((1,maxnodes), dtype = np.float32)
    atom_features_list = np.zeros((maxnodes,1), dtype = np.int64)
    y_marking = np.zeros((1,maxnodes), dtype = np.float32)
    for i,atom in enumerate(struct):
        atom_features_list[i]=atom_to_feature_vector(i,struct)

        if y_atoms!=None:

            y_atoms_state[0][i] = y_atoms[i]
            y_marking[0][i] = 1


    x = atom_features_list
    y = y_atoms_state
    y_mark=y_marking

    a, b, c, alpha, beta, gamma = np.round(struct.cell.cellpar(), 2)
    bondsdic, molsdic = find_bonds_molecules(structure)
    edgelist=[]
    for key in bondsdic.keys():
        if bondsdic[key]!=[]:
            for bond in bondsdic[key]:
                edgelist.append([key,bond])
    edge_index = np.array(edgelist, dtype = np.int64).T
    bond_length = []
    # edge_attremb = []
    for i in range(len(edge_index[0])):
        # distance = struct.get_distance(edge_index[0][i],edge_index[1][i])
        distance = cacul_distance(struct[edge_index[0][i]],struct[edge_index[1][i]],a, b, c)
        bond_length.append([distance])
        # edge_attremb.append([np.round(distance,1)*10])

    # edge_attremb = np.array(edge_attremb, dtype = np.int64)
    bond_length = np.array(bond_length, dtype=np.float)

    positions = np.array(struct.get_positions(), dtype = np.float32)
    pos_zeros = np.zeros((maxnodes-len(positions[:,0]),3), dtype = np.float32)



    graph = dict()
    graph['edge_index'] = edge_index
    # graph['edge_featemb'] = edge_attremb
    graph['bond_length'] = bond_length
    graph['node_feat'] = x
    graph['num_nodes'] = len(x)
    graph['positions'] = np.r_[positions,pos_zeros]
    graph['new_pos'] ,graph['core_dic']= new_position(positions, mol, maxnodes)
    graph['AtomNum'] = len(struct)
    if AtomSymbol:
        symbols=[]
        for i in range(maxnodes):
            try:
                symbols.append(chemical_symbols[struct[i].number])
            except:
                symbols.append('None')
        graph['AtomSymbol']=symbols
    if y_atoms!=None:
        graph['y'] = y
        graph['y_mark'] = y_mark
    return graph










