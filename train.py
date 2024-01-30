import torch as t
from data import ChallengeDataset
from trainer import Trainer
from matplotlib import pyplot as plt
import numpy as np
import model
import pandas as pd
from sklearn.model_selection import train_test_split


# load the data from the csv file and perform a train-test-split
# this can be accomplished using the already imported pandas and sklearn.model_selection modules
data = pd.read_csv('data.csv', sep=';')
# set up data loading for the training and validation set each using t.utils.data.DataLoader and ChallengeDataset objects
train_df, test_df = train_test_split(data, test_size=0.1, random_state = 42)

dataset_train = ChallengeDataset(train_df, mode = 'train')
dataset_test = ChallengeDataset(train_df, mode = 'val')


# create an instance of our ResNet model
resnet_model = model.ResNet()

# set up a suitable loss criterion (you can find a pre-implemented loss functions in t.nn)

loss = t.nn.BCELoss()
# set up the optimizer (see t.optim)
optim = t.optim.SGD(resnet_model.parameters(), lr= 0.001) #TODO add momentum
# create an object of type Trainer and set its early stopping criterion

trainer = Trainer(model = resnet_model, crit = loss, optim = optim, train_dl = dataset_train, val_test_dl =dataset_test, cuda = True, early_stopping_patience = 5)
# go, go, go... call fit on trainer
res = trainer.fit(epochs = 100)

print(res[0][-1])
print(res[1][-1])
print(res[2][-1])

# plot the results
plt.plot(np.arange(len(res[0])), res[0], label='train loss')
plt.plot(np.arange(len(res[1])), res[1], label='val loss')
plt.yscale('log')
plt.legend()
plt.savefig('losses.png')