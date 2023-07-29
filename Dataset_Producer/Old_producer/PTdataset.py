from datacreate import MyPCQM4MDataset
if '__main__' == __name__:
    param={
        'root':'./dataset',
        'ptname':'./PTdata/5-10w/dataset_105-1600000_new.pt',
        'length':550000,
        'begin':1050001,
        'num_jobs':15,
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