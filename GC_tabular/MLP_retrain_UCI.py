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
    org_data_x=org_data.drop([org_data.columns[0]], axis=1) # remove ID
    org_data_x=org_data_x.drop([org_data.columns[24]], axis=1) # remove label
    org_data_y=label = pd.get_dummies(org_data["default.payment.next.month"], drop_first=True)
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


def minmax_scaler(X):
    max_list = pd.read_csv('./example/UCI/UCI_dataset_backup/X_train_scale_max.csv')
    min_list = pd.read_csv('./example/UCI/UCI_dataset_backup/X_train_scale_min.csv')
    max_list = np.array(max_list).squeeze()
    min_list = np.array(min_list).squeeze()
    rtn = torch.tensor(X)
    for col in range(X.shape[1]):
        X_col_max = max_list[col]
        X_col_min = min_list[col]
        rtn[:,col] = (rtn[:,col] - X_col_min)/(X_col_max - X_col_min)
    return rtn


def df_to_cuda(df):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    numpy_array = np.array(df)
    list_tensor = torch.from_numpy(numpy_array)
    list_cuda = list_tensor.to(device).float()
    list_cuda_t = list_cuda.T
    return list_cuda

#hyperparameter

num_epochs = 40
learning_rate = 0.03

# #data load
# path_heloc = '/home/heedong/Documents/Counterfactual-Explanation-Based-on-Gradual-Construction-for-Deep-Networks_UCI/example/UCI/UCI_Credit_Card.csv'

# X, X_tensor, y_tensor = load_data(path_heloc)
# y_tensor = y_tensor.long()

# # tmp_y = y_tensor.reshape(y_tensor.shape[0],1)
# # y_one_hot_encoded = (tmp_y == torch.arange(2).reshape(1,2)).float()
# X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.25, random_state=0)

X_train_df = pd.read_csv('./example/UCI/UCI_dataset_backup/X_train_unscaled.csv')
y_train_df = pd.read_csv('./example/UCI/UCI_dataset_backup/y_train.csv')

X_test_df = pd.read_csv('./example/UCI/UCI_dataset_backup/X_test_unscaled.csv')
y_test_df = pd.read_csv('./example/UCI/UCI_dataset_backup/y_test.csv')

X_train = minmax_scaler(np.array(X_train_df)).to(device).float()
X_test = minmax_scaler(np.array(X_test_df)).to(device).float()

# Check
for i in range(X_train.shape[1]):
    print(X_train[:,i].min())
for i in range(X_train.shape[1]):
    print(X_train[:,i].max())

y_train = df_to_cuda(y_train_df).squeeze()
y_test = df_to_cuda(y_test_df).squeeze()
y_train = torch.tensor(y_train, dtype=torch.long, device = device)
y_test = torch.tensor(y_test, dtype=torch.long, device = device)

# max_list = []
# min_list = []

# for i in range(X_train.shape[1]):
#     X_col_min = X_train[:,i].min()
#     X_col_max = X_train[:,i].max()
#     X_train[:,i] = (X_train[:,i] - X_col_min)/(X_col_max - X_col_min)
#     X_test[:,i] = (X_test[:,i] - X_col_min)/(X_col_max - X_col_min)
#     max_list.append(X_col_max)
#     min_list.append(X_col_min)


#model
clf = model.MLP(input_size=23, output_size=2).to(device)
nbr_class1 = y_train.sum().int()
nbr_class0 = y_train.shape[0] - nbr_class1
nbr_total = y_train.shape[0]
imbalanced_ratio = torch.tensor([nbr_class1/nbr_total, nbr_class0/nbr_total])
criterion = nn.CrossEntropyLoss(weight = imbalanced_ratio.to(device))
optimizer = torch.optim.Adam(clf.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    clf.train()
    optimizer.zero_grad()
    output = clf(X_train)
    loss = criterion(torch.softmax(output, dim = 1) , y_train)

    if epoch%3 == 0:
        print('train_loss', loss.item())
        evaluate_model(clf, X_test, y_test, sklearn=False)

    loss.backward()
    optimizer.step()
    

print()
evaluate_model(clf, X_test, y_test, sklearn=False)


# torch.save(clf, "./models/saved/UCI_MLP.pt")
# clf = torch.load("./models/saved/UCI_MLP.pt")
# evaluate_model(clf, X_test, y_test)


# # data generate for running ours 
# for i in range(X_test.shape[0]):
#     pd.DataFrame(X_test[i,:].cpu().numpy()).T.to_csv('./example/UCI/UCI_test_set/test'+str(i)+'.csv', index = False)


# # data backup
# pd.DataFrame(X_test.cpu().numpy()).to_csv('./example/UCI/UCI_dataset_backup/X_test.csv', index = False)
# pd.DataFrame(X_train.cpu().numpy()).to_csv('./example/UCI/UCI_dataset_backup/X_train.csv', index = False)

# pd.DataFrame(X_test_unscaled.cpu().numpy()).to_csv('./example/UCI/UCI_dataset_backup/X_test_unscaled.csv', index = False)
# pd.DataFrame(X_train_unscaled.cpu().numpy()).to_csv('./example/UCI/UCI_dataset_backup/X_train_unscaled.csv', index = False)

# pd.DataFrame(y_train.cpu().numpy()).to_csv('./example/UCI/UCI_dataset_backup/y_train.csv', index = False)
# pd.DataFrame(y_test.cpu().numpy()).to_csv('./example/UCI/UCI_dataset_backup/y_test.csv', index = False)

# max_int = []
# min_int = []

# for i in range(len(max_list)) :
#     max_int.append(int(max_list[i]))
#     min_int.append(int(min_list[i]))

# pd.DataFrame(np.array(max_int)).T.to_csv('./example/UCI/UCI_dataset_backup/X_train_scale_max.csv', index = False)
# pd.DataFrame(np.array(min_int)).T.to_csv('./example/UCI/UCI_dataset_backup/X_train_scale_min.csv', index = False)



# ## data recover

# X_test_load = pd.read_csv('./example/UCI/UCI_dataset_backup/X_test.csv')
# X_test_load = np.array(X_test_load)
# # data generate for running ours 
# for i in range(X_test_load.shape[0]):
#     pd.DataFrame(X_test_load[i,:]).T.to_csv('./example/UCI/UCI_test_set/test'+str(i)+'.csv', index = False)



# # # ref generate



# def list_to_cuda(list):
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     list_numpy = np.array(list)
#     list_tensor = torch.from_numpy(list_numpy)
#     list_cuda = list_tensor.to(device).float()
#     return list_cuda


# def df_to_cuda(df):
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     numpy_array = np.array(df)
#     list_tensor = torch.from_numpy(numpy_array)
#     list_cuda = list_tensor.to(device).float()
#     list_cuda_t = list_cuda.T
#     return list_cuda


# def cuda_to_numpy(cuda):
#     narray = np.array(cuda.cpu())
#     return narray


# def calculate_lipschitz_factor(x, x1):
#     # norm of x, x1 (both are cuda and scaled respectively)   

#     x = cuda_to_numpy(x)
#     x1 = cuda_to_numpy(x1)
#     norm = LA.norm(x - x1)
#     # scaler = MinMaxScaler()
#     # x_s = scaler.fit_transform(x.ravel().reshape(-1, 1))
#     # x1_s = scaler.fit_transform(x1.ravel().reshape(-1, 1))
#     # norm = LA.norm(x_s - x1_s)


#     return norm



# def minmax_scaler(X):
#     max_list = pd.read_csv('./example/UCI/UCI_dataset_backup/X_train_scale_max.csv')
#     min_list = pd.read_csv('./example/UCI/UCI_dataset_backup/X_train_scale_min.csv')
#     max_list = np.array(max_list).squeeze()
#     min_list = np.array(min_list).squeeze()
#     rtn = torch.tensor(X)
#     for col in range(X.shape[1]):
#         X_col_max = max_list[col]
#         X_col_min = min_list[col]
#         rtn[:,col] = (rtn[:,col] - X_col_min)/(X_col_max - X_col_min)
#     return rtn


# X_tensor_sca = minmax_scaler(X_tensor).to(device)

# idx = np.where((torch.max(clf(X_tensor_sca), 1).indices ==1).cpu())

# for i in range(100):
#     pd.DataFrame(X_tensor_sca[idx][i].cpu().numpy()).T.to_csv('./ref_data/UCI_ref/class0/sample0_'+str(i)+'.csv', index = False)

# idx = np.where((torch.max(clf(X_tensor_sca), 1).indices ==0).cpu())

# for i in range(100):
#     pd.DataFrame(X_tensor_sca[idx][i].cpu().numpy()).T.to_csv('./ref_data/UCI_ref/class1/sample1_'+str(i)+'.csv', index = False)

