import torch
from torch.utils.data import Dataset,TensorDataset,Subset
from torchvision.transforms import transforms
from torchvision import datasets
import os

download=True
ServerUsingRoot="/home/huangjinshu/data/" #在服务器上运行时，下载或读取数据集的路径前缀
LocalUsingRoot="E:/Projects/Datasets/" #在本地运行时使用

if os.path.exists(ServerUsingRoot):
    UsingRoot=ServerUsingRoot
elif os.path.exists(LocalUsingRoot):
    UsingRoot=LocalUsingRoot
else:
    exit('数据集所在的根目录不存在')

def data_preprocess(dataset_name,official=False,subset_proportion=1.0):
    assert subset_proportion>0 and subset_proportion<=1.0,"subset_proportion must be in the range (0, 1.0]"
    #目前包含CIFAR10,CICAR100,MNIST,Iris
    mean_official=(0.485,0.456,0.406);std_official=(0.229,0.224,0.225) #imagenet的mean,std，适用于一般的自然图像
    if dataset_name=='CIFAR10': #32*32的彩色图片
        if official==False:
            mean=(0.4914,0.4822,0.4465); std=(0.2023,0.1994,0.2010)
        else:
            mean=mean_official; std=std_official
        trainTransforms=transforms.Compose([
            transforms.Resize((224,224)),
            transforms.RandomHorizontalFlip(p=0.5), #随机水平镜像
            transforms.RandomResizedCrop((224,224),scale=(0.8,1.0),ratio=(1.0,1.0)),
            # transforms.RandomErasing(scale=(0.04,0.2),ratio=(0.5,2)), #随机遮挡
            # transforms.RandomCrop(224,padding=4), #随机裁剪
            transforms.ToTensor(),
            transforms.Normalize(mean=mean,std=std)
        ])
        testTransforms=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        data_train=datasets.CIFAR10(UsingRoot+'cifar',True,transform=trainTransforms,download=download)#需要下载时改成download=True
        data_test=datasets.CIFAR10(UsingRoot+'cifar',False,transform=testTransforms,download=download)#此处写False是防止py文件在其他地方运行时突然下载cifar数据集
    elif dataset_name=='CIFAR100': #32*32的彩色图片
        if official==False:
            mean=(0.5071,0.4867,0.4408);std=(0.2675,0.2565,0.2761)
        else:
            mean=mean_official;std=std_official
        trainTransforms=transforms.Compose([
            transforms.Resize((224,224)),
            transforms.RandomHorizontalFlip(p=0.5), #随机水平镜像
            transforms.RandomResizedCrop((224,224),scale=(0.8,1.0),ratio=(1.0,1.0)),
            # transforms.RandomErasing(scale=(0.04,0.2),ratio=(0.5,2)), #随机遮挡
            # transforms.RandomCrop(224,padding=4), #随机裁剪
            transforms.ToTensor(),
            transforms.Normalize(mean=mean,std=std)
        ])
        testTransforms=transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean,std=std)
        ])
        data_train=datasets.CIFAR100(UsingRoot+'cifar',True,transform=trainTransforms,download=download)#需要下载时改成download=True
        data_test=datasets.CIFAR100(UsingRoot+'cifar',False,transform=testTransforms,download=download)#此处写False是防止py文件在其他地方运行时突然下载cifar数据集
    elif dataset_name=='FashionMNIST':
        # mean=[0.5];std=[0.5]
        mean=[0.2860];std=[0.3529]
        trainTransforms = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.RandomHorizontalFlip(p=0.5),
            # transforms.RandomRotation((-15,15),expand=False,fill=0,center=None),
            # transforms.RandomAffine(degrees=0,translate=(0.1,0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean,std=std)
        ])
        testTransforms = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean,std=std)
        ])
        data_train=datasets.FashionMNIST(UsingRoot+'FashionMNIST',train=True,transform=trainTransforms,download=download)
        data_test=datasets.FashionMNIST(UsingRoot+'FashionMNIST',train=False,transform=testTransforms,download=download)
    elif dataset_name=='Iris':
        import pandas as pd
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split
        df=pd.read_csv(UsingRoot+"Iris/iris.csv")
        df=df.iloc[:,1:] #df.drop(df.columns[0],axis=1)#首列是行序号，不能作为特征，要删去
        X=df.drop('Species',axis=1).values;X=StandardScaler().fit_transform(X) #特征标准化
        y=df['Species'].map({'setosa':0,'versicolor':1,'virginica': 2}).values #标签从字符串转换为整数
        X=torch.tensor(X,dtype=torch.float32);y=torch.tensor(y,dtype=torch.int64)
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=68)
        data_train=TensorDataset(X_train,y_train);data_test=TensorDataset(X_test,y_test)
    elif dataset_name=='MNIST': #28*28的黑白图片
        # mean=[0.5];std=[0.5]
        mean=[0.1307];std=[0.3081]
        trainTransforms = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.RandomRotation((-15,15),expand=False,fill=0,center=None), #随机旋转-10度到10度
            # transforms.RandomAffine(degrees=0,translate=(0.1,0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean,std=std)
        ])
        testTransforms = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean,std=std)
        ])
        data_train=datasets.MNIST(UsingRoot+'MNIST',train=True,transform=trainTransforms,download=download)
        data_test=datasets.MNIST(UsingRoot+'MNIST',train=False,transform=testTransforms,download=download)
    elif dataset_name=='SVHN': #32*32的彩色图片
        trainTransforms = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor()
        ])
        testTransforms = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor()
        ])
        data_train=datasets.SVHN(UsingRoot+'SVHN',split='train',transform=trainTransforms,download=download)
        data_test=datasets.SVHN(UsingRoot+'SVHN',split='test',transform=testTransforms,download=download)
    else:
        raise ValueError(f'data_preprocess_no_such_data{dataset_name}')
    if subset_proportion<1.0:
        subset_size=int(len(data_train)*subset_proportion)
        subset_indices=torch.randperm(len(data_train))[:subset_size]
        data_train=Subset(data_train,subset_indices)
    # 打印数据集信息
    if isinstance(data_train,Subset):
        if isinstance(data_train.dataset,TensorDataset):
            print(dataset_name+'training set:',type(data_train),data_train.dataset.tensors[0][data_train.indices].shape)
        else:
            print(dataset_name+'training set:',type(data_train),data_train.dataset.data[data_train.indices].shape)
    elif isinstance(data_train,Dataset):  # Dataset包含TensorDataset,CIFAR10,MNIST等子类
        if isinstance(data_train,TensorDataset):
            print(dataset_name+'training set:',type(data_train),data_train.tensors[0].shape)
        else:
            print(dataset_name+'training set:',type(data_train),data_train.data.shape)
    else:
        raise TypeError('Dataset class or Subset class')
    if isinstance(data_test,Subset):
        if isinstance(data_test.dataset,TensorDataset):
            print(dataset_name+'training set:',type(data_test),data_test.dataset.data[data_test.indices].shape)
        else:
            print(dataset_name+'training set:',type(data_test),data_test.dataset.tensors[0][data_test.indices].shape)
    elif isinstance(data_test,Dataset):  # Dataset包含TensorDataset,CIFAR10,MNIST等子类
        if isinstance(data_test,TensorDataset):
            print(dataset_name+'training set:',type(data_test),data_test.tensors[0].shape)
        else:
            print(dataset_name+'training set:',type(data_test),data_test.data.shape)
    else:
        raise TypeError('Dataset class or Subset class')
    print("---------------------------------------------------")
    return data_train, data_test

