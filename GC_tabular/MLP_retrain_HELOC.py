from locale import normalize
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from models import model
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
import sys

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def cuda_available():
    use_cuda = torch.cuda.is_available()
    return use_cuda

def load_data(data_path,TEXT=None):

    org_data=pd.read_csv(data_path)
    org_data_x=org_data.drop([org_data.columns[0]], axis=1)
    org_data_y=label = pd.get_dummies(org_data["RiskPerformance"], drop_first=True)
    org_data_x_tmp=np.squeeze(org_data_x.to_numpy())
    org_data_y_tmp=np.squeeze(org_data_y.to_numpy())

    org_data_x_tensor=torch.from_numpy(org_data_x_tmp).float()
    org_data_y_tensor=torch.from_numpy(org_data_y_tmp).float()

    if cuda_available():
        org_data_tensor=org_data_x_tensor.cuda()
        org_data_tensor=org_data_y_tensor.cuda()

    return org_data, org_data_x_tensor, org_data_y_tensor

def evaluate_model(clf, X_test, y_test, sklearn = False):
    if sklearn == False:
        clf.eval()
        outputs = clf(X_test)
        _, predicted = torch.max(outputs.data, 1)
        print('test_acc', (predicted == y_test).sum()/y_test.shape[0])
    
    elif sklearn == True:
        clf.eval()
        predicted = np.round(clf.predict(X_test))
        correct = (np.round(predicted) == y_test.cpu().numpy().squeeze()).sum()
        print('test_acc', correct/X_test.shape[0])

#hyperparameter

num_epochs = 50
learning_rate = 0.03

#data load
path_heloc = '/home/heedong/Documents/Counterfactual-Explanation-Based-on-Gradual-Construction-for-Deep-Networks/example/HELOC/HELOC_dataset.csv'

X, X_tensor, y_tensor = load_data(path_heloc)
y_tensor = y_tensor[X_tensor[:,0] > -9].long()
X_tensor = X_tensor[X_tensor[:,0] > -9]
X_tensor = X_tensor[:,1:]

# tmp_y = y_tensor.reshape(y_tensor.shape[0],1)
# y_one_hot_encoded = (tmp_y == torch.arange(2).reshape(1,2)).float()
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.25, random_state=0)

X_train_unscaled = torch.tensor(X_train, device=device)
X_test_unscaled = torch.tensor(X_test, device=device)

X_train = torch.tensor(X_train, device=device)
y_train = torch.tensor(y_train, device=device)
X_test = torch.tensor(X_test, device=device)
y_test = torch.tensor(y_test, device=device)



max_list = []
min_list = []

for i in range(X_train.shape[1]):
    X_col_min = X_train[:,i].min()
    X_col_max = X_train[:,i].max()
    X_train[:,i] = (X_train[:,i] - X_col_min)/(X_col_max - X_col_min)
    X_test[:,i] = (X_test[:,i] - X_col_min)/(X_col_max - X_col_min)
    max_list.append(X_col_max)
    min_list.append(X_col_min)


#model
clf = model.MLP(input_size=22, output_size=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(clf.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    clf.train()
    optimizer.zero_grad()
    output = clf(X_train)
    loss = criterion(output, y_train) 
    if epoch%10 == 0:
        print('train_loss', loss.item())
        evaluate_model(clf, X_test, y_test, sklearn=False)

    loss.backward()
    optimizer.step()
    
print()
evaluate_model(clf, X_test, y_test, sklearn=False)


# torch.save(clf, "models/saved/HELOC_retrained.pt")
# clf = torch.load("models/saved/HELOC_retrained.pt")
# evaluate_model(clf, X_test, y_test)


# data generate for running ours 
# for i in range(X_test.shape[0]):
#     pd.DataFrame(X_test[i,:].cpu().numpy()).T.to_csv('/home/heedong/Documents/Counterfactual-Explanation-Based-on-Gradual-Construction-for-Deep-Networks/example/HELOC/heloc_test_set/test'+str(i)+'.csv', index = False)


# data backup
# pd.DataFrame(X_test.cpu().numpy()).to_csv('/home/heedong/Documents/Counterfactual-Explanation-Based-on-Gradual-Construction-for-Deep-Networks/example/HELOC/heloc_dataset_backup/X_test.csv', index = False)
# pd.DataFrame(X_train.cpu().numpy()).to_csv('/home/heedong/Documents/Counterfactual-Explanation-Based-on-Gradual-Construction-for-Deep-Networks/example/HELOC/heloc_dataset_backup/X_train.csv', index = False)

# pd.DataFrame(X_test_unscaled.cpu().numpy()).to_csv('/home/heedong/Documents/Counterfactual-Explanation-Based-on-Gradual-Construction-for-Deep-Networks/example/HELOC/heloc_dataset_backup/X_test_unscaled.csv', index = False)
# pd.DataFrame(X_train_unscaled.cpu().numpy()).to_csv('/home/heedong/Documents/Counterfactual-Explanation-Based-on-Gradual-Construction-for-Deep-Networks/example/HELOC/heloc_dataset_backup/X_train_unscaled.csv', index = False)

# pd.DataFrame(y_train.cpu().numpy()).to_csv('/home/heedong/Documents/Counterfactual-Explanation-Based-on-Gradual-Construction-for-Deep-Networks/example/HELOC/heloc_dataset_backup/y_train.csv', index = False)
# pd.DataFrame(y_test.cpu().numpy()).to_csv('/home/heedong/Documents/Counterfactual-Explanation-Based-on-Gradual-Construction-for-Deep-Networks/example/HELOC/heloc_dataset_backup/y_test.csv', index = False)

# max_int = []
# min_int = []

# for i in range(len(max_list)) :
#     max_int.append(int(max_list[i]))
#     min_int.append(int(min_list[i]))

# pd.DataFrame(np.array(max_int)).T.to_csv('/home/heedong/Documents/Counterfactual-Explanation-Based-on-Gradual-Construction-for-Deep-Networks/example/HELOC/heloc_dataset_backup/X_train_scale_max.csv', index = False)
# pd.DataFrame(np.array(min_int)).T.to_csv('/home/heedong/Documents/Counterfactual-Explanation-Based-on-Gradual-Construction-for-Deep-Networks/example/HELOC/heloc_dataset_backup/X_train_scale_min.csv', index = False)

