import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import argparse,os,logging
import time
from PreResNet import *
from DataProcessing import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='CIFAR10')
parser.add_argument('--dataset_official', action='store_true')
parser.add_argument('--model', type=str, default='PreResnet')
parser.add_argument('--Time', type=float, default=10.0)
parser.add_argument('--block_number', type=int, default=4)
parser.add_argument('--model_official', action='store_true')
parser.add_argument('--optimizer',type=str,default='Adam')
parser.add_argument('--scheduler_expression',type=str,
                    default='optim.lr_scheduler.CosineAnnealingLR(optimizer,args.epochs,eta_min=1e-4)')
parser.add_argument('--criterion', type=str, default='CE', help='loss function')
parser.add_argument('--criterion_expression', type=str, default='nn.CrossEntropyLoss()')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--bs', type=int, default=128)
parser.add_argument('--b2s', type=int, default=None)
parser.add_argument('--init_distribution_expression', type=str, default=None) #'nn.init.normal_(param, mean=0, std=0.07**0.5) for param in model.parameters()'
parser.add_argument('--outf', type=str, default='/home/...')
parser.add_argument('--mylog_dir', type=str, default='/home/...')
parser.add_argument('--num_workers', type=int, default=3)
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--gpu_id', type=int, default=0)
args = parser.parse_args()
#Step: 'optim.lr_scheduler.StepLR(optimizer,step_size=40,gamma=0.2)'
#CosAnn: 'optim.lr_scheduler.CosineAnnealingLR(optimizer,args.epoch,eta_min=1e-4)'
#默认lr不变化的写法有如下多种：ExponentialLR(optimizer,gamma=1),StepLR(optimizer,step_size=1,gamma=1),
#LambdaLR(optimizer,lr_lambda=lambda epoch:1),ConstantLR(optimizer,factor=1,total_iters=args.epoch)

if torch.cuda.is_available():
    torch.cuda.set_device('cuda:'+str(args.gpu_id)) 
    device='cuda'
    print('######## device:',device,torch.cuda.current_device(),':',torch.cuda.get_device_name(),'########')
else:
    device='cpu'
    print('######## device:',device,'########')

#设置随机种子
if args.seed is not None:
    seed = args.seed
    # random.seed(seed)
    # np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#保存模型及log
model_dir = os.path.join(args.outf, 'seed%4d_%s_lr_%.4f_BNo_%d_%3d' % (args.seed, args.model, args.lr, args.block_number, args.epochs))
if not os.path.exists(model_dir):
    os.mkdir(model_dir)
log_file_name1 = "/home/.../%s//train_seed%4d_Log_%s_lr_%.4f_BNo_%d_epoch%3d.txt" % (
     args.mylog_dir, args.seed, args.model, args.lr, args.block_number, args.epochs)
log_file_name2 = "/home/.../%s//val_seed%4d_Log_%s_lr_%.4f_BNo_%d_epoch%3d.txt" % (
     args.mylog_dir, args.seed, args.model, args.lr, args.block_number, args.epochs)


#################### 输入区 ########################
batch_size=args.bs
batch_size2=args.bs if args.b2s is None else args.b2s
num_epochs=args.epochs #10
criterion_expression=args.criterion_expression
if args.optimizer=='Adam':
    optimizer_expression='optim.'+args.optimizer+'(model.parameters(),lr=args.lr)'
elif args.optimizer=='SGD':
    optimizer_expression='optim.'+args.optimizer+'(model.parameters(),lr=args.lr,momentum=0.9,weight_decay=5e-4)'
else:
    optimizer_expression='optim.'+args.optimizer+'(model.parameters(),lr=args.lr)'
# optimizer_expression='optim.SGD(model.parameters(),lr=args.lr,momentum=0.9,weight_decay=5e-4)'
scheduler_expression=args.scheduler_expression
# scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=40,gamma=0.2)
# scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[60,120,160],gamma=0.2)
# scheduler_expression='optim.lr_scheduler.CosineAnnealingLR(optimizer,args.epoch,eta_min=1e-4)'
# scheduler=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=10,T_mult=2)
###################################################


def loss_and_acc(t_loader, model, criterion, batch_size, num_classes=10):
    with torch.no_grad():
        t_total_loss = 0
        t_right = 0
        for (inputs, labels) in t_loader:
            inputs = inputs.to(device);labels = labels.to(device)
            y = model(inputs).float()
            labels = torch.nn.functional.one_hot(labels, num_classes=num_classes).float()
            loss = criterion(y, labels)
            lossdata = loss.cpu().detach().numpy()
            t_total_loss += lossdata
            predicted = torch.argmax(y.data, dim=1)  # torch.max(y.data, 1)[1]
            labeled = torch.argmax(labels.data, dim=1)
            t_right += (predicted == labeled).sum()
        t_acc = t_right / len(t_loader.dataset)
        t_loss = t_total_loss / (len(t_loader.dataset) / batch_size)
    return t_loss, t_acc
###################################################################


if __name__=='__main__':

    # dataset
    data_train,data_test=data_preprocess(args.dataset,official=args.dataset_official)
    if args.dataset=='CIFAR10':
        input_size=32*32*3;in_channels=3;num_classes=10
    elif args.dataset=='CIFAR100':
        input_size=32*32*3;in_channels=3;num_classes=100
    elif args.dataset=='FashionMNIST':
        input_size=28*28*1;in_channels=1;num_classes=10
    elif args.dataset=='Iris':
        input_size=4;in_channels=1;num_classes=3
    elif args.dataset=='MNIST':
        input_size=28*28*1;in_channels=1;num_classes=10
    elif args.dataset=='SVHN':
        input_size=32*32*3;in_channels=3;num_classes=10
    else:
        exit('Check the name of dataset: '+args.dataset)
    train_loader = DataLoader(dataset=data_train, batch_size=batch_size, shuffle=True, drop_last=True,num_workers=args.num_workers)
    test_loader = DataLoader(dataset=data_test, batch_size=batch_size2, shuffle=True, drop_last=True,num_workers=args.num_workers)

    # 选取模型，损失函数，优化器
    step_size = args.Time / args.block_number
    model = PreResNet(step_size=step_size, time=args.Time, in_ch=in_channels, num_classes=num_classes).to(device)
    print('PreResNet: (block_number, step_size) = :', args.block_number, step_size)

    if args.init_distribution_expression is not None:
        exec(args.init_distribution_expression)

    optimizer = eval(optimizer_expression) #给optimizer赋表达式的值，表达式不带等号
    scheduler = eval(scheduler_expression)
    criterion = eval(criterion_expression)
    # exec(optimizer_expression) #执行给optimizer赋值的语句，表达式带等号

    print('training process............')
    for epoch in range(num_epochs):
        # training
        print('epoch=', epoch + 1)
        model.train()
        train_loss = 0
        correct = 0
        total_no = 0
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            if args.criterion!='CE':
                labels = nn.functional.one_hot(labels,num_classes=num_classes).float()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total_no += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            print(batch_idx + 1, '/', len(train_loader), 'epoch: %d' % epoch, '| Loss: %.3f | Acc: %.3f%% (%d/%d)'
                  % (train_loss / (batch_idx + 1), 100. * correct / total_no, correct, total_no))
        print("------------------------------------------------------------------")
        print("Epoch: {}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, train_loss, scheduler.get_lr()[0]))
        with open(log_file_name1, 'a') as f:
            f.write("Epoch: {}\tLoss: {:.4f}\t Acc: {:.4f}\t LearningRate {:.6f}".format(epoch, train_loss,
                                                                                         100. * correct / total_no,
                                                                                         scheduler.get_lr()[
                                                                                             0]) + '\n')
        scheduler.step()

        # testing
        model.eval()
        test_loss = 0.0
        correct = 0.0
        total = 0.0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                # targets = targets.type(torch.int64)
                loss = criterion(outputs, targets)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                print(batch_idx + 1, '/', len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                      % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

        # Save checkpoint.
        acc = 100. * correct / total
        print('Saving..')
        output_test = "[epoch %d] loss: %.4f | Acc: %.3f%% (%d/%d) \n" % (
            epoch, test_loss / len(test_loader), 100. * correct / total, correct, total)
        output_file = open(log_file_name2, 'a')
        output_file.write(output_test)
        output_file.close()
        torch.save(net.state_dict(), os.path.join(model_dir, 'epoch%3d_%2d.pth' % (args.epochs, epoch)))

