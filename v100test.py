import os
import argparse
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
import time
import torch
from datacreate import MyPCQM4MDataset
from GEvaluator import Evaluator
from tqdm import tqdm

from Mid import GINGraphPooling
print('torch version:',torch.__version__)

#参数输入
class MyNamespace(argparse.Namespace):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.batch_size = 10
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
        self.task_name='GINGraph-con-v100'
        self.weight_decay=0.5e-05
        self.learning_rate=0.0001
        self.root='./dataset'
        self.dataset_use_pt=True
        self.dataset_pt = './newPTs/'
        self.dataset_split=[0.8,0.19,0.01]
        self.begin=0
        self.evaluate_epoch=3
        self.continue_train=True
        self.checkpoint_path='./saves/GINGraph-test-v100_/checkpoint.pt'



#数据载入
def load_data(args):
    if  args.dataset_use_pt:
        if os.path.isdir(args.dataset_pt):
            print('data loading in dir:',args.dataset_pt)
            dataset=[]
            for filename in tqdm(os.listdir(args.dataset_pt)):
                dataset.extend(torch.load(os.path.join(args.dataset_pt,filename)))
                # print(len(dataset))
        else:
            print('data loading in .pt:', args.dataset_pt)
            dataset=torch.load(args.dataset_pt)
    else:
        print('MyPCQM4MDataset: data loading from',args.begin,'to',args.begin+args.dataset_length)
        dataset = MyPCQM4MDataset(args.root,save=False,begin=args.begin,length=args.dataset_length)

    length=len(dataset)
    train_idx=round(length*args.dataset_split[0])
    valid_idx=round(length*(args.dataset_split[0]+args.dataset_split[1]))
    test_idx = round(length * (args.dataset_split[0]+args.dataset_split[1] + args.dataset_split[2]))

    train_data=dataset[0:train_idx]
    valid_data=dataset[train_idx:valid_idx]
    test_data=dataset[valid_idx:test_idx]

    print('train data:',len(train_data),'valid data:',len(valid_data))

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return train_loader,valid_loader,test_loader

#trainer
def train(model, device, loader, optimizer, criterion_fn,epoch,epochs):
    print('on training:')
    model.train()
    loss_accum = 0
    maxP = 0
    minN = 0
    avgP = 0
    avgN = 0

    pbar=tqdm(total = len(loader), desc=f'Epoch {epoch}/{epochs}', unit='it')
    for step, batch in enumerate(loader):  # 枚举所有批次的数据
        # print(step)
        # print(type(batch))
        batch = batch.to(device)  # 将数据移动到指定的设备
        # print(batch.x)
        pred = model(batch).view(-1, )  # 前向传播，计算预测值
        try:
            assert not torch.any(torch.isnan(pred))
        except:
            print(batch.new_pos)
            break
        # print(pred.shape)
        optimizer.zero_grad()  # 清空梯度
        deltayy=pred-batch.y.view(pred.shape)
        try:
            if maxP<torch.max(deltayy[deltayy>0]):
                maxP=max(deltayy[deltayy>0])
        except:
            pass

        try:
            if minN>torch.min(deltayy[deltayy<0]):
                minN=min(deltayy[deltayy<0])
        except:
            pass
        avgP+=torch.mean(deltayy[deltayy>0])
        avgN+=torch.mean(deltayy[deltayy<0])
        # print(deltayy)
        # print(pred.shape,batch.y.shape)
        loss = criterion_fn(pred, batch.y.view(pred.shape))  # 计算损失
        assert not torch.any(torch.isnan(loss))

        pbar.set_postfix({'loss' : '{0:1.5f}'.format(loss)}) #在进度条后显示当前batch的损失
        pbar.update(1) #更当前进度，1表示完成了一个batch的训练

        loss.backward()  # 反向传播，计算梯度
        optimizer.step()  # 更新模型参数
        loss_accum += loss.detach().cpu().item()  # 累加损失值

    return loss_accum / (step + 1),maxP,minN,avgP / (step + 1),avgN / (step + 1)

#evaluator
def eval(model, device, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []
    pbar = tqdm(total=len(loader), desc=f'Evaluating:', unit='it')
    with torch.no_grad():  # 禁用梯度计算，加速模型运算
        for _, batch in enumerate(loader):  # 枚举所有批次的数据
            # print('test',type(batch))
            batch = batch.to(device)  # 将数据移动到指定的设备
            pred = model(batch).view(-1, )  # 前向传播，计算预测值
            try:
                assert not torch.any(torch.isnan(pred))
                y_true.append(batch.y.view(pred.shape).detach().cpu())  # 将真实值添加到列表中
                y_pred.append(pred.detach().cpu())  # 将预测值添加到列表中
            except:
                print(batch.new_pos,batch.x)
                pass
            pbar.update(1)  # 更当前进度，1表示完成了一个batch的训练
    print('Evaluate finish')


    y_true = torch.cat(y_true, dim=0)  # 拼接真实值列表成一个张量
    y_pred = torch.cat(y_pred, dim=0)  # 拼接预测值列表成一个张量
    input_dict = {"y_true": y_true, "y_pred": y_pred}  # 构造输入字典
    return evaluator.eval(input_dict)["mae"]

#tester
def test(model, device, loader):
    model.eval()
    y_pred = []
    pbar = tqdm(total=len(loader), desc=f'Testing:', unit='it')

    with torch.no_grad():  # 禁用梯度计算，加速模型运算
        for _, batch in enumerate(loader):  # 枚举所有批次的数据
            batch = batch.to(device)  # 将数据移动到指定的设备
            pred = model(batch).view(-1, )  # 前向传播，计算预测值
            # print(pred)
            try:
                assert not torch.any(torch.isnan(pred))
                y_pred.append(pred.detach().cpu())  # 将预测值添加到列表中
            except:
                print(batch.new_pos,batch.x)
                pass
            pbar.update(1)

    # print('y_pred:',y_pred)
    y_pred = torch.cat(y_pred, dim=0)  # 拼接预测值列表成一个张量
    print('Test finish')
    return y_pred


#log_save
def prepartion(args):
    save_dir = os.path.join('saves', args.task_name+'_')
    if os.path.exists(save_dir):
        for idx in range(1000):
            if not os.path.exists(save_dir + '=' + str(idx)):
                save_dir = save_dir + '=' + str(idx)
                break

    args.save_dir = save_dir
    os.makedirs(args.save_dir, exist_ok=True)
    args.device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    args.output_file = open(os.path.join(args.save_dir, 'output'), 'a')
    print(args, file=args.output_file, flush=True)

def continue_train(args,model,optimizer):
    print('loading modle from',args.checkpoint_path)
    checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print('Load finish')


#main
def main(args):
    prepartion(args)
    nn_params = {
        'num_layers': args.num_layers,
        'emb_dim': args.emb_dim,
        'n_head':args.n_head,
        'drop_ratio': args.drop_ratio,
        'graph_pooling': args.graph_pooling,
        'num_tasks':args.num_tasks,
        'batchsize':args.batch_size

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
    evaluate=args.evaluate_epoch
    best_valid_mae = 9999

    for epoch in range(1, args.epochs + 1):

        print('=====epoch:', epoch,time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) )

        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),"=====Epoch {}".format(epoch), file=args.output_file, flush=True)
        print('Training...', file=args.output_file, flush=True)
        train_mae,maxP,minN,avgP,avgN = train(model, device, train_loader, optimizer, criterion_fn,epoch,args.epochs)
        print(train_mae,maxP,minN,avgP,avgN)
        print('Evaluating...', file=args.output_file, flush=True)
        if epoch==evaluate:
            valid_mae = eval(model, device, valid_loader, evaluator)
            evaluate += args.evaluate_epoch

        print({'Train': train_mae, 'Validation': valid_mae}, file=args.output_file, flush=True)

        writer.add_scalar('valid/mae', valid_mae, epoch)
        writer.add_scalar('train/mae', train_mae, epoch)
        writer.add_scalar('train/maxP', maxP, epoch)
        writer.add_scalar('train/minN', minN, epoch)
        writer.add_scalar('train/avgP', avgP, epoch)
        writer.add_scalar('train/avgN', avgN, epoch)



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

