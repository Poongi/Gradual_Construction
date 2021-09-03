import numpy as np
import pandas as pd
import torch.nn.functional as F
import os
import sys
sys.path.append('/home/heedong/Documents/growingspheres/CounterfactualExplanationBasedonGradualConstructionforDeepNetworks')
sys.path.append('/home/heedong/Documents/growingspheres')
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
from scipy.spatial.distance import cdist, pdist

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


def minmax_scaler(X):
    max_list = pd.read_csv('/home/heedong/Documents/Counterfactual-Explanation-Based-on-Gradual-Construction-for-Deep-Networks/example/HELOC/heloc_dataset_backup/X_train_scale_max.csv')
    min_list = pd.read_csv('/home/heedong/Documents/Counterfactual-Explanation-Based-on-Gradual-Construction-for-Deep-Networks/example/HELOC/heloc_dataset_backup/X_train_scale_min.csv')
    max_list = np.array(max_list).squeeze()
    min_list = np.array(min_list).squeeze()
    rtn = torch.tensor(X)
    for col in range(X.shape[1]):
        X_col_max = max_list[col]
        X_col_min = min_list[col]
        rtn[:,col] = (rtn[:,col] - X_col_min)/(X_col_max - X_col_min)
    return rtn



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# coherent calculating



# model load
clf = torch.load("models/saved/HELOC_retrained.pt")

# data load
X_test_raw = pd.read_csv('/home/heedong/Documents/Counterfactual-Explanation-Based-on-Gradual-Construction-for-Deep-Networks/example/HELOC/heloc_dataset_backup/X_test_unscaled.csv')
X_test_raw = df_to_cuda(X_test_raw)

X_train_raw = pd.read_csv('/home/heedong/Documents/Counterfactual-Explanation-Based-on-Gradual-Construction-for-Deep-Networks/example/HELOC/heloc_dataset_backup/X_train_unscaled.csv')
X_train_raw = df_to_cuda(X_train_raw)

y_train = pd.read_csv('/home/heedong/Documents/Counterfactual-Explanation-Based-on-Gradual-Construction-for-Deep-Networks/example/HELOC/heloc_dataset_backup/y_train.csv')
y_train = df_to_cuda(y_train).squeeze()
y_test = pd.read_csv('/home/heedong/Documents/Counterfactual-Explanation-Based-on-Gradual-Construction-for-Deep-Networks/example/HELOC/heloc_dataset_backup/y_test.csv')
y_test = df_to_cuda(y_test).squeeze()

evaluate_model(clf, minmax_scaler(X_test_raw), y_test)



result_original = pd.DataFrame()
result_counterfactual = pd.DataFrame()

for i in range(3000):
    if os.path.isdir("/home/heedong/Documents/Counterfactual-Explanation-Based-on-Gradual-Construction-for-Deep-Networks/result/HELOC/test"+str(i)) == False:
        continue
    result_o_tmp = pd.read_csv("/home/heedong/Documents/Counterfactual-Explanation-Based-on-Gradual-Construction-for-Deep-Networks/result/HELOC/test"+str(i)+"/org.csv")
    result_c_tmp = pd.read_csv("/home/heedong/Documents/Counterfactual-Explanation-Based-on-Gradual-Construction-for-Deep-Networks/result/HELOC/test"+str(i)+"/per.csv")

    result_original = result_original.append(result_o_tmp)
    result_counterfactual = result_counterfactual.append(result_c_tmp)

X_original = df_to_cuda(result_original)
counterfactual = df_to_cuda(result_counterfactual)


output_original = clf(X_original)
output_cf = clf(counterfactual)

_, y_pred_from_X = torch.max(output_original.data, 1)
_, y_pred_form_cf = torch.max(output_cf.data, 1)


nbr_experiments = 1000

X_original_comp = X_original[:nbr_experiments]
y_pred_comp = y_pred_from_X[:nbr_experiments]


distance_metric = 'mahalanobis'

if distance_metric == 'mahalanobis':

    lipschitz_list = []

    for target in range(nbr_experiments):
        target_X = X_original_comp[target]
        target_label = torch.max(clf(target_X).data, 0).indices
        X_idx = np.where(y_pred_from_X.cpu() == target_label.cpu())[0]
        dist = cdist(cuda_to_numpy(target_X).reshape(1,-1), cuda_to_numpy(X_original[X_idx]),  metric = 'mahalanobis')[0]
        eps = np.percentile(dist, 5)
        
        X_idx_eps = X_idx[np.where((dist<=eps))]

        target_cf = counterfactual[X_idx_eps]

        for under_eps in X_idx_eps:
            if under_eps == target:
                continue
            norm_x = calculate_lipschitz_factor(target_X, X_original[under_eps])        
            norm_cf = calculate_lipschitz_factor(counterfactual[target], counterfactual[under_eps])
            lipschitz_list.append(norm_cf / norm_x)

    print("Ours cohenrence mean", np.mean(lipschitz_list))
    print("Ours cohenrence std", np.std(lipschitz_list))

elif distance_metric == 'manhatan':

    lipschitz_list = []

    for target in range(nbr_experiments):
        target_X = X_original_comp[target]
        target_label = torch.max(clf(target_X).data, 0).indices
        X_idx = np.where(y_pred_from_X.cpu() == target_label.cpu())[0]
        dist = []

        for i in range(20):
            individual_distance = target_X - X_original[i]
            individual_distance_norm = LA.norm(individual_distance.cpu())
            dist.append(individual_distance_norm)

        dist = np.array(dist)
        eps = np.percentile(dist, 5)
        
        X_idx_eps = X_idx[np.where((dist<=eps))]

        target_cf = counterfactual[X_idx_eps]

        for under_eps in X_idx_eps:
            if under_eps == target:
                continue
            norm_x = calculate_lipschitz_factor(target_X, X_original[under_eps])        
            norm_cf = calculate_lipschitz_factor(counterfactual[target], counterfactual[under_eps])
            lipschitz_list.append(norm_cf / norm_x)

    print("Ours cohenrence mean", np.mean(lipschitz_list))
    print("Ours cohenrence std", np.std(lipschitz_list))






print()
print()
print()

