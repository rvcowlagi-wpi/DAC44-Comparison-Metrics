# Load the trained discriminator model
import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from torch.autograd.variable import Variable
import matplotlib.pyplot as plt
import time
from baseline_gan_v02 import Discriminator
from sklearn. metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import seaborn as sns
import matplotlib.pyplot as plt

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # data = pd.read_csv("C:/Users/nubapat/Data/test7close.csv", delimiter=',')
    # num_samples, num_features = data.shape
    # num_features = num_features - 1
    # print("num_features", num_features)
    num_classes = 2
    dimD_1 = 900
    dimD_2 = 625
    dimD_3 = 400
    dimD_4 = 225
    dimD_5 = 100
    dimD_6 = 25

    modelD = Discriminator()
    modelD.to(device)

    modelD.load_state_dict(torch.load('discriminator_model.pt'))



    # Prepare the test dataset

    test_data = np.array(pd.read_csv('C:/Users/nubapat/Data/test7close.csv', dtype=np.float32))

    testX = test_data[:, 1:]
    testY = test_data[:, 0]
    # print(testY)
    # print(testX[0].shape)

    # Pass the test dataset through the discriminator model
    testX = torch.tensor(testX, dtype=torch.float32).to(device)
    # print(testX)
    discriminator_output = torch.sigmoid(modelD(testX))
    # discriminator_output = theDiscriminator(testX)
    # print(discriminator_output)
    # print(testX.shape)
    # for op in discriminator_output:
    #      print(op.item())
    #
    # return
    #
    #discriminator_output = (theDiscriminator(testX))
    #print(discriminator_output)
    #print(discriminator_output.shape)
    diff = np.empty(len(discriminator_output))
    mismatch_count = 0

    for i in range(len(discriminator_output)):
         print(f'{(discriminator_output[i].item())}, {testY[i].item()}')
         diff[i] = abs(discriminator_output[i].item() - testY[i].item())

         if round(discriminator_output[i].item()) != testY[i].item():
             mismatch_count += 1
             # print(f"Mismatch at index {i}: discriminator_output = {discriminator_output[i].item()}, testY = {testY[i].item()}")
             # print(f'{(discriminator_output[i].item())}, {testY[i].item()}')
    print(f"Number of mismatches: {mismatch_count}")



    #print('diff = ',diff)

    counts, bins, _ = plt.hist(diff, bins=50, density=False, alpha=0.5)

    # calculate the sum of counts
    sum_counts = np.sum(counts)

    # normalize the histogram to a probability density function
    pdf = counts / (sum_counts * np.diff(bins))

    # plot the normalized histogram
    plt.plot(bins[:-1], pdf, color='r', alpha=0.8)

    plt.show()

    print("Pred  GT")
    accuracy, y_test, y_pred_tag = binary_acc(discriminator_output, testY)
    print(f"Final accuracy: {accuracy}")


    cf_matrix = confusion_matrix(y_test,y_pred_tag, normalize= "all")
    print(cf_matrix)

    sns.heatmap(cf_matrix/np.sum(cf_matrix), annot = True, fmt ="g", cmap ="Blues")
    plt.show()



def binary_acc(y_pred, y_test):


    #y_pred_tag = torch.round(torch.sigmoid(y_pred))
    y_pred_tag = torch.round(y_pred)
    y_pred_tag = y_pred_tag.cpu().detach().numpy()
    y_test = y_test.reshape(y_test.shape[0], 1)
    y_test = y_test.astype(np.uint8)
    y_pred_tag = y_pred_tag.astype(np.uint8)



    N = y_test.shape[0]
    accuracy = (y_test == y_pred_tag).sum() / N

    return accuracy, y_test, y_pred_tag

if __name__ == "__main__":
    main()