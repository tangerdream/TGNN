from datacreate import MyPCQM4MDataset
if '__main__' == __name__:
    param={
        'root':'./dataset',
        'ptname':'./PTs/dataset_100-1100000_new.pt',
        'length':100000,
        'begin':1000000,
        'num_jobs':5,
        'seed':20,
        'save':True,
        'cover':True,
        'divid':True
    }
    print('check input',param)
    if int(input('确认无误 1，取消 0： ')):
        data=MyPCQM4MDataset(param['root'],save=param['save'],ptname=param['ptname'],length=param['length'],begin=param['begin'],num_jobs=param['num_jobs'],cover=param['cover'],divid=param['divid'],seed=param['seed'])
    else:
        pass