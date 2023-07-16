from datacreate import MyPCQM4MDataset
if '__main__' == __name__:
    param={
        'root':'./dataset',
        'ptname':'./PTs/dataset_32-3600000_new.pt',
        'length':1000,
        'begin':3200000,
        'num_jobs':3,
        'seed':20,
        'save':True,
        'cover':True,
        'divid':False
    }
    print('check input',param)
    if int(input('确认无误 1，取消 0： ')):
        data=MyPCQM4MDataset(param['root'],save=param['save'],ptname=param['ptname'],length=param['length'],begin=param['begin'],num_jobs=param['num_jobs'],cover=param['cover'],divid=param['divid'],seed=param['seed'])
    else:
        pass