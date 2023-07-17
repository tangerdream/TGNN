import os
import os.path as osp
import shutil
import numpy as np
import pandas as pd
import torch
# from ogb.utils.mol import smiles2graph
from ogb.utils.torch_util import replace_numpy_with_torchtensor
from ogb.utils.url import download_url, extract_zip
from rdkit import RDLogger
from torch_geometric.data import Data, Dataset
from ase import Atoms
from ase import Atom
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
from Asegraph3 import struct2graph
from multiprocessing import Pool





os.environ['KMP_DUPLICATE_LIB_OK']='True'
RDLogger.DisableLog('rdApp.*')

class MyPCQM4MDataset(Dataset):

    def __init__(self, root,save=False,cover=False,ptname='my_dataset.pt',length=1000,seed=20,begin=1,num_jobs=5,divid=True,maxnodes=50,maxAttempts=100):
        self.url = 'http://dgl-data.s3-accelerate.amazonaws.com/dataset/OGB-LSC/pcqm4m_kddcup2021.zip'
        super(MyPCQM4MDataset, self).__init__(root)

        filepath = osp.join(root, 'raw/data.csv.gz')
        data_df = pd.read_csv(filepath)
        self.smiles_list = data_df['smiles']
        self.homolumogap_list = data_df['homolumogap']
        self.cover=cover
        self.ptname=ptname
        self.seed=seed
        self.divid=divid
        self.maxnodes=maxnodes
        self.maxAttempts=maxAttempts
        if save:
            print('beginning save from %i'%(begin)+' to %i'%(begin+length-1)+' whit num_jobs=%i'%(num_jobs)+' ptname=%s'%(ptname)+' divid=%s'%(str(divid)))
            self.savept(length,begin,num_jobs)

    @property
    def raw_file_names(self):
        return 'data.csv.gz'

    def download(self):
        path = download_url(self.url, self.root)
        extract_zip(path, self.root)
        os.unlink(path)
        shutil.move(osp.join(self.root, 'pcqm4m_kddcup2021/raw/data.csv.gz'), osp.join(self.root, 'raw/data.csv.gz'))

    def len(self):
        return len(self.smiles_list)

    def get(self, idx):
        smiles, homolumogap = self.smiles_list[idx], self.homolumogap_list[idx]
        # print(smiles)
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
        struct = self.mol_to_asestruct(mol)

        graph = struct2graph(struct,mol,maxnodes=self.maxnodes)
        assert(len(graph['bond_length']) == graph['edge_index'].shape[1])
        assert(len(graph['node_feat']) == graph['num_nodes'])

        x = torch.from_numpy(graph['node_feat']).to(torch.int64)
        edge_index = torch.from_numpy(graph['edge_index']).to(torch.int64)
        bond_length = torch.from_numpy(graph['bond_length']).to(torch.float)
        edge_attr = None
        y = torch.Tensor([homolumogap])
        num_nodes = int(graph['num_nodes'])
        AtomNum = int(graph['AtomNum'])
        # AtomSymbol = graph['AtomSymbol']
        pos = torch.from_numpy(graph['positions']).to(torch.float)
        new_pos = torch.from_numpy(graph['new_pos']).to(torch.float)
        core_dic = graph['core_dic']
        data = Data(x=x, edge_index=edge_index,edge_attr =edge_attr, y=y, new_pos=new_pos, bond_length=bond_length,num_nodes=num_nodes,)
        return data

    def mol_to_asestruct(self,mol):
        molecule = Atoms()
        for j in range(0, len(mol.GetAtoms())):
            x, y, z = mol.GetConformer().GetAtomPosition(j)
            sybol = mol.GetAtoms()[j].GetSymbol()
            molecule.append(Atom(sybol, position=(x, y, z)))
        return molecule

    # 获取数据集划分
    def get_idx_split(self):
        split_dict = replace_numpy_with_torchtensor(torch.load(osp.join(self.root, 'pcqm4m_kddcup2021/split_dict.pt')))
        return split_dict

    # 保存数据集
    def savept(self,length,begin,num_jobs):
        if os.path.exists(self.processed_file_names()) and self.cover==False:
            pass
        else:
            data_list = []
            # 模拟一个需要显示进度的迭代过程
            idx = [i for i in range(begin-1, length + begin-1)]
            len_divid = round(len(idx) / num_jobs) + 1
            dict = [idx[i:i + len_divid] for i in range(0, len(idx), len_divid)]

            pool=Pool(processes=num_jobs)
            data = []
            for n in range(num_jobs):
                if n==0:
                    data.append(pool.apply_async(self.catch_data_tqdm, args=(dict[n],length,n,)))
                else:
                    data.append(pool.apply_async(self.catch_data, args=(dict[n],n,)))
            pool.close()
            pool.join()
            if not self.divid:
                pbar = tqdm(total=len(data))
                pbar.set_description('cutting')
                for res in data:
                    data_list.extend(res.get())
                    pbar.update(1)
                print('cutting finished whit the length of',len(data_list))
                torch.save(data_list, self.processed_file_names())
                print('saved in ',os.path.abspath(self.processed_file_names()))

    def catch_data(self,dic,n):
        data= []
        for idx in range(0,len(dic)):
            try:
                data.append(self.get(idx))
                assert not torch.any(torch.isnan(data[-1].new_pos))
            except:
                pass
            # if idx in timetable:
            #     pbar.update(1)
        if self.divid:
            torch.save(data, self.processed_file_names()+'_%i'%(n))
            print('saved in ', os.path.abspath(self.processed_file_names()+'_%i'%(n)))
        else:
            return data

    def catch_data_tqdm(self,dic,num,n):
        data= []

        pbar = tqdm(total=num)
        pbar.set_description('processing')
        step=num/len(dic)
        # pbar.set_description('processing %i' % dic[0]+': %i' % dic[-1])
        # timetable = [ii for ii in range(0, num, round(num / 100))]
        for idx in range(0,len(dic)):
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
        path=self.ptname
        return path

    # 获取数据集划分
    # def get_idx_split2(self):
    #     # split_dict = replace_numpy_with_torchtensor(torch.load(osp.join(self.root, 'pcqm4m_kddcup2021/split_dict.pt')))
    #     first_list, second_list, third_list = self.random_split(len(self.smiles_list))
    #     split_dict = {
    #         'train':first_list,
    #         'valid':second_list,
    #         'test':third_list
    #     }
    #     return split_dict
    #
    # def random_split(self,length):
    #     # 生成随机数列
    #     random_list = np.random.permutation(list(range(0,length,1)))
    #
    #     num_first_list = round(len(random_list) * 0.9)
    #     num_second_list = round(len(random_list) * 0.05)
    #     num_third_list = len(random_list) - num_first_list - num_second_list
    #
    #     # 分配元素到每个数列
    #     first_list = random_list[:num_first_list]
    #     second_list = random_list[num_first_list:num_first_list + num_second_list]
    #     third_list = random_list[num_first_list + num_second_list:]
    #     first_list = list(first_list)
    #     second_list = list(second_list)
    #     third_list = list(third_list)
    #
    #
    #     return first_list,second_list,third_list




if __name__ == "__main__":
    dataset = MyPCQM4MDataset('dataset')



