import argparse
import os
from multiprocessing import Pool

import pandas as pd
import torch
from ase import Atom
from ase import Atoms
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem
from torch_geometric.data import Data, Dataset
from tqdm import tqdm

from Asegraph3 import struct2graph, smiles2graph, get_bond_length

# from ogb.utils.mol import smiles2graph


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
RDLogger.DisableLog('rdApp.*')


class SmilesProcess_init(argparse.Namespace):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.root = './data.csv.gz'  # csv file
        self.use_new_pos = True
        self.y_name = 'homolumogap'
        self.save = False
        self.cover = False
        self.outputdir = './PTs/'
        self.ptname = 'my_dataset.pt'
        self.seed = 20
        self.num_jobs = 5
        self.divid = True
        self.maxnodes = 50
        self.maxAttempts = 100
        self.length = 100
        self.begin = 1


class SmilesProcess(Dataset):
    r"""Dataset base class for creating graph datasets.
    Args:
        root (string, optional): Root directory where the dataset should be
            saved. (optional: :obj:`None`)


    """

    def __init__(self, root, y_name, use_new_pos=False, save=False, cover=False, outputdir='./', ptname='my_dataset.pt',
                 seed=20, num_jobs=5, divid=True, maxnodes=50, maxAttempts=100, length=None, begin=1):
        super(SmilesProcess, self).__init__(root)

        data_df = pd.read_csv(root)
        self.process_name = 'SmilesProcess'
        self.smiles_list = data_df['smiles']
        self.y_list = data_df[y_name]
        # print(self.smiles_list[0],self.y_list[0])
        self.cover = cover
        self.ptname = ptname
        self.seed = seed
        self.divid = divid
        self.maxnodes = maxnodes
        self.maxAttempts = maxAttempts
        self.use_new_pos = use_new_pos
        self.outputdir = outputdir
        if length == None:
            length = len(self.smiles_list)
        if save:
            print('beginning save from %i' % (begin) + ' to %i' % (begin + length - 1) + ' whit num_jobs=%i' % (
                num_jobs) + ' ptname=%s' % (outputdir + ptname) + ' divid=%s' % (str(divid)))
            self.savept(length, begin, num_jobs)

    def len(self):
        return len(self.smiles_list)

    def get(self, idx):
        smiles, y = self.smiles_list[idx], self.y_list[idx]

        x, edge_index_np, edge_attr = smiles2graph(smiles, self.maxnodes)
        x = torch.from_numpy(x).to(torch.int64)
        assert (len(x) == self.maxnodes)
        edge_index = torch.from_numpy(edge_index_np).to(torch.int64)
        edge_attr = torch.tensor(edge_attr)
        y = torch.Tensor([y])
        num_nodes = int(self.maxnodes)

        if self.use_new_pos:
            mol = self.gen_pos_by_Rdkit(smiles)
            struct = self.mol_to_asestruct(mol)
            graph = struct2graph(struct, jobname='smiles', maxnodes=self.maxnodes)

            AtomNum = int(graph['AtomNum'])
            bond_length = torch.from_numpy(get_bond_length(struct, edge_index_np)).to(torch.float)
            assert (len(bond_length) == edge_index.shape[1])
            pos = torch.from_numpy(graph['positions']).to(torch.float)
            new_pos = torch.from_numpy(graph['new_pos']).to(torch.float)
            core_dic = graph['core_dic']
            outdata = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, new_pos=new_pos,
                        num_nodes=num_nodes, bond_length=bond_length,)
        else:
            outdata = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y,)
        return outdata

    def gen_pos_by_Rdkit(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        try:
            AllChem.EmbedMolecule(mol, randomSeed=self.seed)
            AllChem.MMFFOptimizeMolecule(mol)
        except:
            mol = Chem.MolFromSmiles(smiles)
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, maxAttempts=self.maxAttempts)
            AllChem.MMFFOptimizeMolecule(mol)
        return mol

    def mol_to_asestruct(self, mol):
        molecule = Atoms()
        for j in range(0, len(mol.GetAtoms())):
            x, y, z = mol.GetConformer().GetAtomPosition(j)
            sybol = mol.GetAtoms()[j].GetSymbol()
            molecule.append(Atom(sybol, position=(x, y, z)))
        return molecule

    # 保存数据集
    def savept(self, length, begin, num_jobs):
        if os.path.exists(self.processed_file_names()) and self.cover == False:
            pass
        else:
            data_list = []
            # 模拟一个需要显示进度的迭代过程
            idx = [i for i in range(begin - 1, length + begin - 1)]
            len_divid = round(len(idx) / num_jobs) + 1
            dict = [idx[i:i + len_divid] for i in range(0, len(idx), len_divid)]

            pool = Pool(processes=num_jobs)
            data = []
            for n in range(num_jobs):
                if n == 0:
                    data.append(pool.apply_async(self.catch_data_tqdm, args=(dict[n], length, n,)))
                else:
                    data.append(pool.apply_async(self.catch_data, args=(dict[n], n,)))
            pool.close()
            pool.join()
            if not self.divid:
                pbar = tqdm(total=len(data))
                pbar.set_description('cutting')
                for res in data:
                    data_list.extend(res.get())
                    pbar.update(1)
                print('cutting finished whit the length of', len(data_list))
                torch.save(data_list, self.processed_file_names())
                print('saved in ', os.path.abspath(self.processed_file_names()))

    def catch_data(self, dic, n):
        data = []
        for idx in range(0, len(dic)):
            try:
                data.append(self.get(idx))
                assert not torch.any(torch.isnan(data[-1].new_pos))
            except:
                pass
            # if idx in timetable:
            #     pbar.update(1)
        if self.divid:
            torch.save(data, self.processed_file_names() + '_%i' % (n))
            print('saved in ', os.path.abspath(self.processed_file_names() + '_%i' % (n)))
        else:
            return data

    def catch_data_tqdm(self, dic, num, n):
        data = []

        pbar = tqdm(total=num)
        pbar.set_description('processing')
        step = num / len(dic)
        # pbar.set_description('processing %i' % dic[0]+': %i' % dic[-1])
        # timetable = [ii for ii in range(0, num, round(num / 100))]
        for idx in range(0, len(dic)):
            try:
                data.append(self.get(idx))
            except:
                pass
            pbar.update(step)
        if self.divid:
            torch.save(data, self.processed_file_names() + '_%i' % (n))
            print('saved in ', os.path.abspath(self.processed_file_names() + '_%i' % (n)))
        else:
            return data

    def processed_file_names(self):
        if not os.path.exists(self.outputdir):
            os.makedirs(self.outputdir)
        path = os.path.join(self.outputdir, self.ptname)
        return path


if __name__ == '__main__':
    args = SmilesProcess_init()
    nn_params = {
        'root': args.root,
        'y_name': args.y_name,
        'use_new_pos': args.use_new_pos,
        'save': args.save,
        'cover': args.cover,
        'outputdir': args.outputdir,
        'ptname': args.ptname,
        'seed': args.seed,
        'num_jobs': args.num_jobs,
        'divid': args.divid,
        'maxnodes': args.maxnodes,
        'maxAttempts': args.maxAttempts,
        'length': args.length,
        'begin': args.begin
    }

    dataset1 = SmilesProcess(**nn_params)
    print(dataset1[1])
