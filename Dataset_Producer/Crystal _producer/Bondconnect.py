"""
@todo: assign atom types
"""
import os, sys
import numpy as np
import subprocess
import ase
import ase.io
from ase.neighborlist import NeighborList
from ase.neighborlist import neighbor_list
from ase.data import covalent_radii, atomic_numbers
import argparse

class Params():
    def __init__(self,):

        self.fname = ''

        self.cutoff_file = ''
        self.cutoff_table = {}
        self.nl = []
        self.molecules = []
        self.tolerance = 1.2
        self.ignore_element = []

# def reax_bond_order(a, b, r, ffield):
#     bo = math.exp(pbo1*math.power(r/r0, pbo2))
#     return bo

def build_cutoff_table(atps, p):
    for i in range(len(atps)):
        for j in range(i, len(atps)):
            if atps[i] in p.ignore_element or atps[j] in p.ignore_element:
                r0 = 0.1
            else:
                ri = covalent_radii[atomic_numbers[atps[i]]]
                rj = covalent_radii[atomic_numbers[atps[j]]]
                r0 = p.tolerance*(ri+rj)
            p.cutoff_table[(atps[i], atps[j])] = r0

def get_lists(atoms, p):

    # build neighbor list
    for i in range(len(atoms)):
        p.nl.append([])

    tokens_i, tokens_j = neighbor_list('ij', atoms, p.cutoff_table)
    for i in range(len(tokens_i)):
        p.nl[tokens_i[i]].append(tokens_j[i])

def get_molecule_id(atom_id, mol_id, atom_visits, p):
    if atom_visits[atom_id][2] < 0:
        atom_visits[atom_id][2] = mol_id
        p.molecules[mol_id].append(atom_id)
        for partner in atom_visits[atom_id][1]:
            get_molecule_id(partner, mol_id, atom_visits, p)

def find_molecules(atoms, p):
    for i in range(len(atoms)):
        p.molecules.append([])

    atom_visits = []
    for i in range(len(atoms)):
        atom_visits.append([atoms[i].symbol, p.nl[i], -1])

    for i in range(len(atoms)):
        get_molecule_id(i,i, atom_visits, p)

    p.molecules = [i for i in p.molecules if i]

def output_nl(p):
    o={}
    for i in range(len(p.nl)):
        o[i]=p.nl[i]
    return o

def get_mol_name(atom_lists, atoms):
    mol_name = ''
    elements = atoms.get_chemical_symbols()
    tokens = []
    for i in atom_lists:
        tokens.append(elements[i])
    tokens.sort()
    visit = []
    for i in tokens:
        if not i in visit:
            mol_name += '%s%d'%(i, tokens.count(i))
            visit.append(i)
    return mol_name

def output_molecules(atoms, p):
    o={}
    molid = 0
    for i in p.molecules:
        molname = get_mol_name(i, atoms)
        key= '%d %s'%(molid, molname)
        o[key]=i
        molid += 1
    return o

def update_cutoff_table(p):
    if os.path.exists(p.cutoff_file):
        f = open(p.cutoff_file, 'r')
        for i in f:
            tokens = i.strip().split()
            if len(tokens) == 3:
                p.cutoff_table[(tokens[0], tokens[1])] = float(tokens[2])
                p.cutoff_table[(tokens[1], tokens[0])] = float(tokens[2])
        f.close()



def find_bonds_molecules(structure:ase.atoms.Atoms):
    p = Params()
    p.output_fname = 'mol.dat'
    p.ignore_element = ['Li']



    atoms = structure.copy()

    atps = list(set(atoms.get_chemical_symbols()))
    build_cutoff_table(atps, p)
    update_cutoff_table(p)#更新原有的cutoff表，这里没啥用
    get_lists(atoms, p)
    bondsdic=output_nl(p)
    find_molecules(atoms, p)
    molsdic=output_molecules(atoms, p)


    return bondsdic,molsdic

#----------------------------------------
if __name__=='__main__':
    from ase.io import read
    structure = read('..\..\..\..\\for-gnn-contcar\dme-lhce4002-contcar.vasp')
    bondsdic,molsdic=find_bonds_molecules(structure)
    print(bondsdic)
