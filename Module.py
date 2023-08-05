import os
from torch_geometric.loader import DataLoader
import torch
from tqdm import tqdm




#数据载入
def load_data(args):

    if os.path.isdir(args.dataset_pt):
        print('data loading in dir:',args.dataset_pt)
        dataset=[]
        for filename in tqdm(os.listdir(args.dataset_pt)):
            dataset.extend(torch.load(os.path.join(args.dataset_pt,filename)))
            # print(len(dataset))
    else:
        print('data loading in .pt:', args.dataset_pt)
        dataset=torch.load(args.dataset_pt)
    # else:
    #     print('Dataset: data loading from',args.begin,'to',args.begin+args.dataset_length)
    #     dataset = eval(args.producer)(args.root,args.y_name,save=False,begin=args.begin,length=args.dataset_length)

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
        pred = model(batch) # 前向传播，计算预测值
        try:
            assert not torch.any(torch.isnan(pred))
        except:
            print(batch.new_pos)
            break
        # print(pred.shape)
        optimizer.zero_grad()  # 清空梯度
        deltayy=pred-batch.y
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
def evaluate(model, device, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []
    pbar = tqdm(total=len(loader), desc=f'Evaluating:', unit='it')
    with torch.no_grad():  # 禁用梯度计算，加速模型运算
        for _, batch in enumerate(loader):  # 枚举所有批次的数据
            # print('test',type(batch))
            batch = batch.to(device)  # 将数据移动到指定的设备
            pred = model(batch)  # 前向传播，计算预测值
            try:
                assert not torch.any(torch.isnan(pred))
                y_true.append(batch.y.detach().cpu())  # 将真实值添加到列表中
                y_pred.append(pred.detach().cpu())  # 将预测值添加到列表中
            except:
                print(batch.new_pos,batch.x)
                pass
            pbar.update(1)  # 更当前进度，1表示完成了一个batch的训练
    print('Evaluate finish')


    y_true = torch.cat(y_true, dim=0)  # 拼接真实值列表成一个张量
    y_pred = torch.cat(y_pred, dim=0)  # 拼接预测值列表成一个张量
    y_true = y_true.view(y_pred.shape)  # 将真实值张量的形状调整为与预测值张量相同

    assert y_true.shape == y_pred.shape
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
            pred = model(batch)  # 前向传播，计算预测值
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
    save_dir = os.path.join('saves', args.task_name)
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
