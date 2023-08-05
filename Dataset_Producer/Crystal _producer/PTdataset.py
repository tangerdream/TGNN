from Datasetmain import CrystalProcess, CrystalProcess_init
import argparse

if '__main__' == __name__:



    class CrystalProcess_test(argparse.Namespace):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

            self.root_samp = './for-gnn-contcar'  # structure files
            self.root_y = './label_csv'  # csv files
            self.y_name = 'core_state'
            self.save = True
            self.cover = True
            self.outputdir = 'F:\\OnlinePacket\\programfiles\\Python\\TangerGNN\\Transform_rebuild\\TGNN\\PTs\\crystal'
            self.ptname = 'Crystall_dataset.pt'
            self.seed = 20
            self.num_jobs = 1
            self.divid = False
            self.maxnodes = 200
            self.maxAttempts = 100
            self.length = 'all'
            self.begin = 1
            self.y_norm = False
            self.y_norm_range = [0, 1000]


    args = CrystalProcess_test()

    nn_params = {
        'root_samp': args.root_samp,
        'root_y': args.root_y,
        'y_name': args.y_name,
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
        'begin': args.begin,
        'y_norm': args.y_norm,
        'y_norm_range': args.y_norm_range,
    }

    dataset1 = CrystalProcess(**nn_params)
    # print(dataset1[1])