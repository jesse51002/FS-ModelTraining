import os

import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from dataset import HairTypeDataset, Split

from mobilenetv3 import mobilenet_v3_large

NUM_CLASSES = 6
NUM_EPOCHS = 60
BATCH_SIZE = 20

LOG_INTERVAL = 200

DATA_ROOT = "data/hair_only"
WEIGHTS_ROOT = "weights/mobilenetv3/"

LOSS = CrossEntropyLoss()



def create_dataloader(num_workers=2, shuffle=True, split= Split.ALL):
    dataset = HairTypeDataset(DATA_ROOT, split=split)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=num_workers, shuffle=shuffle)
    return dataloader
    

def train(weight_pth=None):
    train_dataloader = create_dataloader(split=Split.TRAIN)
    test_dataloader = create_dataloader(split=Split.TEST)
    
    net = mobilenet_v3_large(num_classes=NUM_CLASSES).cuda()
    
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    
    writer = SummaryWriter()
    
    for epoch in range(NUM_EPOCHS):
        total_correct = 0
        total = 0
        
        for batch_idx, (train_features, train_labels) in enumerate(train_dataloader):
            train_features = train_features.cuda()
            train_labels = train_labels.cuda()
            
            inference = net(train_features)
            
            predicted = inference.argmax(dim=1)
        
            for i in range(predicted.shape[0]):
                if predicted[i] == train_labels[i]:
                    total_correct += 1
                total += 1
            
            loss = LOSS(inference, train_labels)
            loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()
                        
            if batch_idx % LOG_INTERVAL == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * BATCH_SIZE, len(train_dataloader.dataset),
                    100. * batch_idx / len(train_dataloader), loss.item()))

        train_set_accuracy = total_correct / total
        print("Train set accuracy: {} on epoch {}".format(train_set_accuracy, epoch))
        
        test_set_accuracy = get_test_accuracy(net, test_dataloader, epoch)
        
        writer.add_scalar("Accuracy/train", train_set_accuracy, epoch)
        writer.add_scalar("Accuracy/test", test_set_accuracy, epoch)
        writer.flush()
        
        print("Saving model at epoch {}".format(epoch))
        
        os.makedirs(WEIGHTS_ROOT, exist_ok=True)
        epoch_save_dir = os.path.join(WEIGHTS_ROOT, f"model-{epoch}.pth")
        torch.save(net.state_dict(), epoch_save_dir)
        
@torch.no_grad()
def get_test_accuracy(net, test_dataloader, epoch):
    total_correct = 0
    total = 0
    
    for batch_idx, (test_features, test_labels) in enumerate(test_dataloader):
        test_features = test_features.cuda()
        test_labels = test_labels.cuda()
            
        inference: torch.Tensor = net(test_features)
            
        predicted = inference.argmax(dim=1)
        
        for i in range(predicted.shape[0]):
            if predicted[i] == test_labels[i]:
                total_correct += 1
            total += 1
            
        if batch_idx % LOG_INTERVAL == 0:
            print('Valid Idx: [{}/{} ({:.0f}%)]'.format(
                batch_idx * BATCH_SIZE, len(test_dataloader.dataset),
                100. * batch_idx / len(test_dataloader)))
    
    accuracy = total_correct / total
    print("Test set accuracy: {} on epoch {}".format(accuracy, epoch))
    
    return accuracy
        

if __name__ == "__main__":
    train() 
    
        