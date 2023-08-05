import os
import argparse
from torch.utils.tensorboard import SummaryWriter
import time
import torch
from GEvaluator import Evaluator
from Mid import GINGraphPooling
from Module import load_data,train,evaluate,test,prepartion,continue_train
print('torch version:',torch.__version__)


#参数输入
class MyNamespace(argparse.Namespace):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.batch_size = 20
        self.device=0
        self.drop_ratio=0.1
        self.early_stop=30
        self.early_stop_open = True
        self.emb_dim=256
        self.epochs=200
        self.graph_pooling='sum'
        self.num_layers=3
        self.n_head=3
        self.num_workers=5
        self.num_tasks=1
        self.save_test=True
        self.task_name='GINGraph_crystal_tes-805-v100'
        self.weight_decay=0.1e-05
        self.learning_rate=0.00001
        self.data_type='crystal'  #'smiles','crystal'
        self.dataset_pt = './PTs/crystal_norm'
        self.dataset_split=[0.8,0.19,0.01]
        self.begin=0
        self.evaluate_epoch=1
        self.continue_train=True
        self.checkpoint_path='/home/ml/hctang/TGNN/saves/GINGraph_crystal_test=9/checkpoint.pt'
        self.job_level='node' #'graph','node'
        self.attention=True #是否启用Multi-head self-attention层





def main(args):
    prepartion(args)
    nn_params = {
        'num_layers': args.num_layers,
        'emb_dim': args.emb_dim,
        'n_head':args.n_head,
        'drop_ratio': args.drop_ratio,
        'graph_pooling': args.graph_pooling,
        'num_tasks':args.num_tasks,
        'data_type':args.data_type,
        'job_level':args.job_level,
        'attention':args.attention,

    }

    # automatic dataloading and splitting
    train_loader,valid_loader,test_loader=load_data(args)

    # automatic evaluator. takes dataset name as input
    evaluator = Evaluator()
    criterion_fn = torch.nn.MSELoss()

    device = args.device

    model = GINGraphPooling(**nn_params).to(device)
    optimizer =  torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)
    if args.continue_train:
        continue_train(args,model,optimizer)

    num_params = sum(p.numel() for p in model.parameters())
    print('train data:', len(train_loader), 'valid data:', len(valid_loader), file=args.output_file, flush=True)
    print(f'#Params: {num_params}', file=args.output_file, flush=True)
    print(model, file=args.output_file, flush=True)


    writer = SummaryWriter(log_dir=args.save_dir)

    not_improved = 0
    eva=1
    best_valid_mae = 9999
    valid_mae=10000

    for epoch in range(1, args.epochs + 1):

        print('=====epoch:', epoch,time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) )

        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),"=====Epoch {}".format(epoch), file=args.output_file, flush=True)
        print('Training...', file=args.output_file, flush=True)
        train_mae,maxP,minN,avgP,avgN = train(model, device, train_loader, optimizer, criterion_fn,epoch,args.epochs)
        print(train_mae,maxP,minN,avgP,avgN)
        print('Evaluating...', file=args.output_file, flush=True)
        if epoch==eva:
            valid_mae = evaluate(model, device, valid_loader, evaluator)
            eva += args.evaluate_epoch

        print({'Train': train_mae, 'Validation': valid_mae}, file=args.output_file, flush=True)

        writer.add_scalar('valid/mae', valid_mae, epoch)
        writer.add_scalar('train/mae', train_mae, epoch)
        writer.add_scalar('train/maxP', maxP, epoch)
        writer.add_scalar('train/minN', minN, epoch)
        writer.add_scalar('train/avgP', avgP, epoch)
        writer.add_scalar('train/avgN', avgN, epoch)
        print('valid_mae:',valid_mae,'best_valid_mae:',best_valid_mae)


        if valid_mae < best_valid_mae:
            print('valid_mae:',valid_mae,'Saving checkpoint...')
            best_valid_mae = valid_mae
            if args.save_test:
                print('Saving checkpoint...', file=args.output_file, flush=True)
                checkpoint = {
                    'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(), 'best_val_mae': best_valid_mae, 'num_params': num_params
                }
                torch.save(checkpoint, os.path.join(args.save_dir, 'checkpoint.pt'))
                print('Predicting on test data...', file=args.output_file, flush=True)
                y_pred = test(model, device, test_loader)
                print('Saving test submission file...', file=args.output_file, flush=True)
                evaluator.save_test_submission({'y_pred': y_pred}, args.save_dir)

            not_improved = 0
        else:
            not_improved += 1
            if not_improved == args.early_stop:
                print(f"Have not improved for {not_improved} epoches.", file=args.output_file, flush=True)
                break

        scheduler.step()
        print(f'Best validation MAE so far: {best_valid_mae}', file=args.output_file, flush=True)

    # writer.add_graph(model,train_loader)
    writer.close()
    args.output_file.close()

if __name__ == '__main__':
    # args=p_args()
    args=MyNamespace()
    main(args)
    print('finish')

