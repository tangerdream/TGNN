# TGNN
先装好anaconda和cuda（这里推荐cuda 11.8），按照如下步骤创建虚拟环境(win环境)：

conda create -n py38 python=3.8

activate py38

pip install panda

pip install ogb

pip install ase

pip install rdkit

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install torch_geometric

pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cpu.html

有准备好的数据集理论上，直接python run.py即可开始训练
