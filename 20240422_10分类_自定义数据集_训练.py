from matplotlib import pyplot as plt
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import time
from PIL import Image



# 数据集根目录, 更改此目录
dataset_path = r'C:\Users\ff\Desktop\my_ten_classes_dataset'
label_list = ['猫', '马', '车', '手机', '美女', '飞机', '矿泉水', '玫瑰花', '狮子', '狗']


class MyDataset10(Dataset):
    def __init__(self, root_dir, label_dir, train=False, transform=None):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.img_name_list = os.listdir(self.path)
        if train:
            # 9:1   train:test比例
            self.img_name_list = self.img_name_list[:int(len(self.img_name_list)*9/10)]
        else:
            self.img_name_list = self.img_name_list[int(len(self.img_name_list)*9/10):]
        self.transform = transform

    def __getitem__(self, index):
        img_name = self.img_name_list[index]
        img_path = os.path.join(self.path, img_name)
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)
        # 为简单, 把label直接设为了文件夹名
        label = int(self.label_dir)
        return img, label
    
    def __len__(self):
        return len(self.img_name_list) 


transform = torchvision.transforms.Compose([torchvision.transforms.Resize(224),
                                            torchvision.transforms.CenterCrop(224),
                                            torchvision.transforms.ToTensor()])
# train_dataset
dataset_list = []
for i in range(10):
    label_dir = str(i)
    ds = MyDataset10(dataset_path, label_dir, train=True, transform=transform)
    dataset_list.append(ds)
train_dataset = torch.utils.data.ConcatDataset(dataset_list)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

# test_dataset
dataset_list2 = []
for i in range(10):
    label_dir = str(i)
    ds = MyDataset10(dataset_path, label_dir, train=False, transform=transform)
    dataset_list2.append(ds)
test_dataset = torch.utils.data.ConcatDataset(dataset_list2)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=True)
print(len(train_dataset), len(test_dataset))

# 训练部分
net = torchvision.models.mobilenet_v3_small()
net.classifier[-1] = nn.Linear(1024, 10)
net = net.to('cuda')

lr = 0.01
opt = torch.optim.SGD(net.parameters(), lr)
loss_f = nn.CrossEntropyLoss().to('cuda')
epochs = 200
train_acc_list = []
test_acc_list = []

for epoch in range(epochs):
    print(f'第{epoch+1}轮训练开始...')
    train_acc = 0.0
    total_samples = len(train_dataset)
    time0 = time.time()

    # train
    for inputs, labels in train_loader:
        inputs = inputs.to('cuda')
        labels = labels.to('cuda')
        
        outputs = net(inputs)
        loss = loss_f(outputs, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        batch_acc = (torch.argmax(outputs, dim=-1)==labels).sum().item()
        train_acc += batch_acc
    
    train_acc /= total_samples
    print('train_acc:', train_acc, 'time:', time.time()-time0)
    train_acc_list.append(train_acc)
    
    # test
    test_acc = 0.0
    if epoch % 5 == 0:
        for inputs_t, labels_t in test_loader:
            inputs_t = inputs_t.to('cuda')
            labels_t = labels_t.to('cuda')

            outputs_t = net(inputs_t)

            test_batch_acc = (torch.argmax(outputs_t, dim=-1) == labels_t).sum().item()
            test_acc += test_batch_acc
        
        test_acc /= len(test_dataset)
        test_acc_list.append(test_acc)
        print('--test_acc:', test_acc)

            
        

torch.save(net, 'mydataset10_mobliev3_small_200th.pth')   
plt.plot(train_acc_list)
plt.plot(range(0, epochs, 5), test_acc_list, color='red', linestyle='--')
plt.xlabel('epoch')
plt.ylabel('acc')
plt.savefig('mydataset10_mobliev3_small_200th.png')
plt.show()




       
