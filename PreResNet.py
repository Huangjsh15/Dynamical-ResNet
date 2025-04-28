import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    def __init__(self,in_ch,block_ch,step_size=1.0):
        super(BasicBlock,self).__init__()
        self.step_size=step_size
        self.bn1=nn.BatchNorm2d(block_ch) #(num_features,eps=1e-5,momentum=0.1,affine=True,trach_running_stats=True)
        self.relu1=nn.ReLU()
        self.conv1=nn.Conv2d(in_ch,block_ch,kernel_size=3,padding=1,stride=1,bias=False)
        self.bn2=nn.BatchNorm2d(block_ch)
        self.relu2=nn.ReLU()
        self.conv2=nn.Conv2d(block_ch,block_ch,kernel_size=3,padding=1,stride=1,bias=False)
    def forward(self,x):
        out=self.conv1(self.relu1(self.bn1(x)))
        out=self.conv2(self.relu2(self.bn2(out)))
        out=out*self.step_size+x
        return out

class DownsampleBlock(nn.Module):
    def __init__(self,in_ch,block_ch,kernel_size=1,padding=0,stride=2,bias=False):
        super(DownsampleBlock,self).__init__()
        self.conv0=nn.Conv2d(in_ch,block_ch,kernel_size=kernel_size,padding=padding,stride=stride,bias=bias) #用1x1卷积核实现下采样
        self.bn0=nn.BatchNorm2d(block_ch)
    def forward(self, x):
        out=self.bn0(self.conv0(x))
        return out

class PreResNet(nn.Module):
    def __init__(self,in_ch=3,num_classes=10, step_size=1.0, time=1.0, basic_block=BasicBlock, down_sample_block=DownsampleBlock):
        super(PreResNet, self).__init__()
        # preprocess 输入图像(3,224,224)
        self.conv1 = nn.Conv2d(in_ch,64,kernel_size=7,padding=3,stride=2,bias=False) #(64,112,112)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3,padding=1,stride=2) #(64,56,56)

        #Residual
        #self.layer1_1 = basic_block(in_ch=64, block_ch=64, step_size=step_size)
        self.residual_layer_1 = nn.ModuleList()
        for i in range(int(time/step_size)):
            self.residual_layer_1.append(
                basic_block(in_ch=64, block_ch=64, step_size=step_size)
            )
        self.downsample2 = down_sample_block(in_ch=64, block_ch=128)


        #self.layer2_1=basic_block(in_ch=128,block_ch=128,step_size=step_size)
        self.residual_layer_2 = nn.ModuleList()
        for i in range(int(time / step_size)):
            self.residual_layer_2.append(
                basic_block(in_ch=128, block_ch=128, step_size=step_size)
            )
        self.downsample3 = down_sample_block(in_ch=128, block_ch=256)

        #self.layer3_1 = basic_block(in_ch=256,block_ch=256,step_size=step_size)
        self.residual_layer_3 = nn.ModuleList()
        for i in range(int(time / step_size)):
            self.residual_layer_3.append(
                basic_block(in_ch=256, block_ch=256, step_size=step_size)
            )
        self.downsample4 = down_sample_block(in_ch=256, block_ch=512)

        #self.layer4_1=basic_block(in_ch=512,block_ch=512,step_size=step_size)
        self.residual_layer_4 = nn.ModuleList()
        for i in range(int(time / step_size)):
            self.residual_layer_4.append(
                basic_block(in_ch=512, block_ch=512, step_size=step_size)
            )

        #Linear
        self.avgpool1 = nn.AvgPool2d(kernel_size=7, padding=0, stride=7)
        self.fc_layer = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.maxpool1(self.bn1(self.conv1(x))) #(,3,224,224)-->(,64,56,56)
        for block in self.residual_layer_1: #(,64,56,56)-->(,64,56,56)
            out = block(out)

        out = self.downsample2(out)
        for block in self.residual_layer_2: #(,64,56,56)-->(,128,28,28)-->(,128,28,28)
            out = block(out)

        out = self.downsample3(out)
        for block in self.residual_layer_3:  #(,128,28,28)-->(,256,14,14)-->(,256,14,14)
            out = block(out)

        out = self.downsample4(out)
        for block in self.residual_layer_4:  #(,256,14,14)-->(,512,7,7)-->(,512,7,7)
            out = block(out)

        out = torch.flatten(self.avgpool1(out), start_dim=1) #(,512,7,7)-->(,512,1,1)-->(,512)
        out = self.fc_layer(out)
        return out


def PreResNet_4():
    return PreResNet(step_size=1.0, time=1)

def PreResNet_8():
    return PreResNet(step_size=0.5, time=1)

def PreResNet_16():
    return PreResNet(step_size=0.25, time=1)

def PreResNet_32():
    return PreResNet(step_size=0.125, time=1)

if __name__ == '__main__':
    input = torch.ones([1, 3, 224, 224])

    net1 = PreResNet_4()
    net2 = PreResNet_8()

    output1 = net1(input)
    output2 = net2(input)
    print(' Net1 + Number of params: {}'.format(
        sum([p.data.nelement() for p in net1.parameters()])))
    print(' Net2 + Number of params: {}'.format(
        sum([p.data.nelement() for p in net2.parameters()])))