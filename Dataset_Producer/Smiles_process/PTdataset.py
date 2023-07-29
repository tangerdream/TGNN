from Datasetmain import SmilesProcess, SmilesProcess_init
import argparse

if '__main__' == __name__:
    class SmilesProcess_PT(argparse.Namespace):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

            self.root = './data.csv.gz'  # csv file
            self.use_new_pos = True
            self.y_name = 'homolumogap'
            self.save = True
            self.cover = True
            self.outputdir = 'F:/OnlinePacket/programfiles/Python/TangerGNN/Transform_rebuild/TGNN/PTs'
            self.ptname = 'my_dataset.pt'
            self.seed = 20
            self.num_jobs = 10
            self.divid = True
            self.maxnodes = 50
            self.maxAttempts = 50
            self.length = 10000
            self.begin = 1


    args = SmilesProcess_PT()

    params = {
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
    print('check input',params)
    if int(input('确认无误 1，取消 0： ')):
        data=SmilesProcess(**params)
    else:
        pass