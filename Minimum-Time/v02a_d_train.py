import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from torch.autograd.variable import Variable
import matplotlib.pyplot as plt
import time
from torchsummary import summary
import torch.optim as optim
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))
    #y_pred_tag = torch.sigmoid(y_pred)
    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = torch.round(acc * 100)
    return acc

class GanDataset(Dataset):

  def __init__(self, data_path) -> None:
    super().__init__()


    data = pd.read_csv(data_path,delimiter = ",")
    self.Y = torch.tensor(data["target"].values, dtype = torch.float32, requires_grad = True).to(device)

    self.X = torch.tensor(data.iloc[:,1:].values,dtype = torch.float32, requires_grad = True).to(device)



    self.n_samples, self.num_features = self.X.shape



  def __getitem__(self, index):
    return self.X[index],self.Y[index]

  def __len__(self):
    return self.n_samples

class Discriminator(nn.Module):


    def __init__(self):
        super().__init__()

        self.layer1 = nn.Linear(175, 128) # input layer
        self.relu = nn.ReLU()
        #self.dropout = nn.Dropout(p=0.1)
        self.layer2 = nn.Linear(128,128)  # hidden layer 1
        self.layer3 = nn.Linear(128, 128)  # hidden layer 2
        # self.layer4 = nn.Linear(128, 64)  # hidden layer 2
        self.layer4 = nn.Linear(128,64)  # hidden layer 3
        self.layer5 = nn.Linear(64, 1)
        #self.sigmoid = nn.Sigmoid()


    def forward(self, inputs):

        x = self.relu(self.layer1(inputs))
        #x = self.dropout(x)

        x = self.relu(self.layer2(x))
        #x = self.dropout(x)

        x = self.relu(self.layer3(x))
        #x = self.dropout(x)
        x = self.relu(self.layer4(x))
        #x = self.dropout(x)
        x = self.layer5(x)
        #x = self.layer4(x)
        #x = self.sigmoid(x)

        return x

def main():

    # DIFFERENCE FROM V01
    # This version has more hidden layers
    print(f'Using device: {device}')
    start_clock = time.time()
    n_features = 175
    n_discretization = 25






    train_data_path = "C:/Users/nubapat/Data/trial7.csv"
    # data_mintime = np.array(pd.read_csv('C:/Users/nubapat/Data/train7.csv'))
    # print('data',data_mintime.shape)
    # eval_data_path = "C:/Users/nubapat/gan-data/eval.csv"
    # test_data_path = "C:/Users/nubapat/gan-data/test.csv"

    EPOCHS = 50
    BATCH_SIZE = 64
    LEARNING_RATE = 0.0001
    gan_train_dataset = GanDataset(train_data_path)

    # gan__test_dataset = GanDataset(test_data_path)
    # gan_eval_dataset = GanDataset(eval_data_path)
    train_loader = DataLoader(dataset=gan_train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # test_loader = DataLoader(dataset=gan__test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    # eval_loader = DataLoader(dataset=gan_eval_dataset, batch_size=BATCH_SIZE, shuffle=True)

    modelD = Discriminator().to(device)
    print(summary(modelD, [(gan_train_dataset.num_features,)]))
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(modelD.parameters(), lr=LEARNING_RATE)
    modelD.train()

    for e in range(1, EPOCHS + 1):
        epoch_loss = 0
        epoch_acc = 0
        # eval_acc = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            # print(X_batch.shape)
            y_pred = modelD(X_batch)
            # print(y_pred)
            loss = criterion(y_pred, y_batch.unsqueeze(1))

            acc = binary_acc(y_pred, y_batch.unsqueeze(1))

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()


        # with torch.no_grad():
        #     for input,labels in eval_loader:
        #         input, labels = input.to(device),labels.to(device)
        #         eval_output = theDiscriminator(input)
        #         e_acc = binary_acc(eval_output, labels.unsqueeze(1))
        #         eval_acc += e_acc.item()

        #print(f'Epoch {e + 0:03}: | Training Loss: {epoch_loss / len(train_loader):.5f}  |Train Accuracy  Acc: {epoch_acc / len(train_loader):.3f} |Eval  Acc: {eval_acc / len(eval_loader):.3f}')

        print(f'Epoch {e + 0:03}: | Training Loss: {epoch_loss / len(train_loader):.5f}  |Train Accuracy  Acc: {epoch_acc / len(train_loader):.3f}' )
        torch.save(modelD.state_dict(), 'discriminator_model2.pt')

    return









if __name__ == "__main__":
    main()
