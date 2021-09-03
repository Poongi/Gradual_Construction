import numpy as np
import pandas as pd
import torch.nn.functional as F
import os
import sys
from sklearn import datasets, ensemble
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from GradualConstruction.core import Expl_tabular
import main
from models import model
PATH = ''

import numpy as np
import pandas as pd
from sklearn import datasets, ensemble, tree
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import matplotlib
from matplotlib import pyplot as plt
import torch

from numpy import linalg as LA
from sklearn.preprocessing import MinMaxScaler

def list_to_cuda(list):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    list_numpy = np.array(list)
    list_tensor = torch.from_numpy(list_numpy)
    list_cuda = list_tensor.to(device).float()
    return list_cuda


def df_to_cuda(df):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    numpy_array = np.array(df)
    list_tensor = torch.from_numpy(numpy_array)
    list_cuda = list_tensor.to(device).float()
    list_cuda_t = list_cuda.T
    return list_cuda


def minmax_scaler(X):
    max_list = pd.read_csv('./example/HELOC/heloc_dataset_backup/X_train_scale_max.csv')
    min_list = pd.read_csv('./example/HELOC/heloc_dataset_backup/X_train_scale_min.csv')
    max_list = np.array(max_list).squeeze()
    min_list = np.array(min_list).squeeze()
    rtn = torch.tensor(X)
    for col in range(X.shape[1]):
        X_col_max = max_list[col]
        X_col_min = min_list[col]
        rtn[:,col] = (rtn[:,col] - X_col_min)/(X_col_max - X_col_min)
    return rtn


def generate_random_noised_example(example, clf, nbr_samples=20, max_attempts=100000, pad = 0.4):
    # example = a instance(cuda) , clf = classifier(MLP), nbr_samples : number of noised examples, max_attemps : number of try
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X = []
    attempts = 0
    predicted = torch.max(clf(example).data, 0)[1]
    while len(X) < nbr_samples and attempts < max_attempts:
        attempts += 1
        copied_example = torch.tensor(example)
        noise = torch.randn(example.shape, device=device)*pad
        noised_example = copied_example + noise
        predicted_by_noised_example = torch.max(clf(noised_example).data, 0)[1]
        if predicted == predicted_by_noised_example:
            X.append(noised_example)

    X = torch.cat(X, dim=-1)
    X = X.reshape(-1, example.shape[0])
    return X


def cuda_to_numpy(cuda):
    narray = np.array(cuda.cpu())
    return narray


def calculate_lipschitz_factor(x, x1):
    # norm of x, x1 (both are cuda and scaled respectively)   

    x = cuda_to_numpy(x)
    x1 = cuda_to_numpy(x1)
    norm = LA.norm(x - x1)
    # scaler = MinMaxScaler()
    # x_s = scaler.fit_transform(x.ravel().reshape(-1, 1))
    # x1_s = scaler.fit_transform(x1.ravel().reshape(-1, 1))
    # norm = LA.norm(x_s - x1_s)


    return norm


def evaluate_l2_metric(X_test, cf_list):
    # input type : cuda
    diff = X_test.cpu() - cf_list.cpu()
    l2_norm = np.linalg.norm(diff, axis=1, ord=2)
    l2_norm_mean = np.mean(l2_norm)
    l2_norm_std = np.std(l2_norm)
    return l2_norm_mean, l2_norm_std



def evaluate_l1_metric(X_test, cf_list):
    # input type : cuda
    diff = X_test.cpu() - cf_list.cpu()
    ls = []
    for i in range(diff.shape[0]):
        number_of_nonzero = (torch.abs(diff[i]) >= 0.001).sum()
        ls.append(number_of_nonzero)
    l1_mean = np.mean(ls)
    l1_std = np.std(ls)
    return l1_mean, l1_std


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





device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

######## check

# data load
result_original = pd.DataFrame()
result_counterfactual = pd.DataFrame()


for i in range(7500):
    if os.path.isdir("./result/HELOC/test"+str(i)) == False:
        continue
    result_o_tmp = pd.read_csv("./result/HELOC/test"+str(i)+"/org.csv")
    result_c_tmp = pd.read_csv("./result/HELOC/test"+str(i)+"/per.csv")

    result_original = result_original.append(result_o_tmp)
    result_counterfactual = result_counterfactual.append(result_c_tmp)

X_original_tmp = df_to_cuda(result_original)
counterfactual_tmp = df_to_cuda(result_counterfactual)


# selecting changed class examples
clf = torch.load("./models/saved/HELOC_retrained.pt")
output_original = clf(X_original_tmp)
output_cf = clf(counterfactual_tmp)

_, y_pred_from_X_tmp = torch.max(output_original.data, 1)
_, y_pred_form_cf_tmp = torch.max(output_cf.data, 1)

idx_diff_class = []
for i in range(X_original_tmp.shape[0]):
    if ((y_pred_from_X_tmp[i] + y_pred_form_cf_tmp[i]) == 1)  :
        idx_diff_class.append(i)


X_original = X_original_tmp[idx_diff_class]
counterfactual = counterfactual_tmp[idx_diff_class]
y_pred_from_X = y_pred_from_X_tmp[idx_diff_class]
y_pred_from_cf = y_pred_form_cf_tmp[idx_diff_class]


# L2 : distance (distance of total perturbed feature)
l2_mean, l2_std = evaluate_l2_metric(X_original, counterfactual)

# L1 : sparsity (number of perturbed feature)
l1_mean, l1_std = evaluate_l1_metric(X_original, counterfactual)


# check
ls = []
for i in idx_diff_class:
    diff = X_original.cpu() - counterfactual.cpu()
    
    for i in range(diff.shape[0]):
        number_of_nonzero = (torch.abs(diff[i]) >= 0.001).sum()
        ls.append(int(number_of_nonzero.cpu()))
# print(ls)

print("idx print ", idx_diff_class)
print("number of different class:", len(idx_diff_class))


print("Ours L1 mean", l1_mean)
print("Ours L1 std", l1_std)

print("Ours L2 mean", l2_mean)
print("Ours L2 std", l2_std)

