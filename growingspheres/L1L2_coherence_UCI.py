import pandas as pd
import numpy as np
from numpy import linalg as LA
from scipy.spatial.distance import cdist, pdist
import pickle
import torch
from models import model

def evaluate_l2_metric(X_test, cf_list):
    diff = X_test - cf_list
    l2_norm = np.linalg.norm(diff, axis=1, ord=2)
    l2_norm_mean = np.mean(l2_norm)
    l2_norm_std = np.std(l2_norm)
    return l2_norm_mean, l2_norm_std


def evaluate_l1_metric(X_test, cf_list):
    diff = X_test - cf_list
    ls = []
    for i in range(diff.shape[0]):
        number_of_nonzero = (np.abs(diff[i]) >= 0.001).sum()
        ls.append(number_of_nonzero)
    l1_mean = np.mean(ls)
    l1_std = np.std(ls)
    return l1_mean, l1_std

def l1_l2_evaluation(x_test_, cf_):
    cf = cf_[np.where(~np.isnan(cf_[:, 0]))]
    x_test = x_test_[np.where(~np.isnan(cf_[:, 0]))]

    l1_mean, l1_std = evaluate_l1_metric(x_test, cf)
    l2_mean, l2_std = evaluate_l2_metric(x_test, cf)

    return l1_mean, l1_std, l2_mean, l2_std

def calculate_lipschitz_factor(x, x1):
    norm = LA.norm(x - x1)
    return norm


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


def num_inequality_finder(string):
    if '<=' in string:
        value = float(string.replace('<=', ''))
        return '<=', value
    elif '>=' in string:
        value = float(string.replace('>=', ''))
        return '>=', value
    elif '<' in string :
        value = float(string.replace('<', ''))
        return '<', value
    elif '>' in  string:
        value = float(string.replace('>', ''))
        return '>', value


def find_cf_value(inequality, original_value, cf_value, eps=1):
    if '<=' == inequality:
        if original_value <= cf_value:
            return original_value
        else :
            return cf_value
    elif '>=' == inequality:
        if original_value >= cf_value:
            return original_value
        else :
            return cf_value
    elif '<' == inequality :
        if original_value < cf_value:
            return original_value - eps
        else :
            return cf_value
    elif '>' ==  inequality:
        if original_value > cf_value:
            return original_value
        else :
            return cf_value + eps




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


counterfactual_tmp = pd.read_csv('./example/GS_example/UCI_cf.csv')
X_original_tmp = pd.read_csv('./example/GS_example/UCI_X.csv')

counterfactual_tmp = df_to_cuda(counterfactual_tmp)
X_original_tmp = df_to_cuda(X_original_tmp)


# selecting changed class examples
clf = torch.load("./models/saved/UCI_retrained.pt")
output_original = clf(X_original_tmp)
output_cf = clf(counterfactual_tmp)

_, y_pred_from_X_tmp = torch.max(output_original.data, 1)
_, y_pred_form_cf_tmp = torch.max(output_cf.data, 1)


# data copy
test_num = 1000
target = np.array(X_original_tmp.cpu())[:test_num]
cf = np.array(counterfactual_tmp.cpu())[:test_num]



# load model
blackbox = torch.load('./models/saved/UCI_retrained.pt')

# load y
print("predict of target match ratio")
evaluate_model(blackbox, target, torch.tensor(blackbox.predict(target)),  sklearn=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# numpy to torch
X_original_tmp = torch.tensor(target).to(device).float()
counterfactual_tmp = torch.tensor(cf).to(device).float()

# selecting changed class examples
_, output_original = torch.max(blackbox(X_original_tmp),1)
_, output_cf = torch.max(blackbox(counterfactual_tmp),1)
print("target, predict of cf match ratio")
evaluate_model(blackbox, X_original_tmp, output_cf)


## selecting
idx_diff_class = []
for i in range(X_original_tmp[:test_num].shape[0]):
    if ((y_pred_from_X_tmp[i] + y_pred_form_cf_tmp[i]) == 1)  :
        idx_diff_class.append(i)


X_original = X_original_tmp[idx_diff_class]
counterfactual = counterfactual_tmp[idx_diff_class]
y_pred_from_X = output_original[idx_diff_class]
y_pred_from_cf = output_cf[idx_diff_class]




'''
Model load
'''
lr = torch.load('./models/saved/UCI_retrained.pt')

'''
Data load
'''
test_x = X_original_tmp # Min-max scaled
test_cf = np.array(counterfactual_tmp.cpu()) # Min-max scaled

# Check lr.prediction(test_x) and (test_cf) are different, as we consider counterfactual ones to x_

predicted_test_x = lr(test_x).argmax(axis=1).cpu()


nbr_experiments = 1000
x = test_x[:nbr_experiments].cpu()
test_x = np.array(test_x.cpu())

lipschitz_list = []
for target in range(nbr_experiments):
    # target = 0
    currentX = x[target].reshape((1,) + x[target].shape).to(device)
    predicted = lr(currentX).argmax(axis=1)
    currentX = np.array(currentX.cpu())
    predicted = predicted.cpu()

    # Input data whose predicted class is same as that of target x
    filteredIdx = np.where(predicted == predicted_test_x)[0]

    '''
    Calculating distances (higher values indicates more similarity)
    '''
    eps = 1e-3 # Possible candidates 1e-2, 1e-3, 1e-4
    dist = []
    for i in range(test_x[filteredIdx].shape[0]):
        if np.sum(currentX == test_x[filteredIdx][i]) == currentX.shape[1]:
            dist.append(-99)
        else:
            nSatisfied = np.sum(np.abs(currentX - test_x[filteredIdx][i]) < eps)
            dist.append(nSatisfied)
    distDescending, distIdx_ = np.sort(dist)[::-1], np.argsort(dist)[::-1]
    distIdx = distIdx_[distDescending > 15] # 2/3 * number of features
    if distIdx.size == 0:
        continue
    '''
    End
    '''
    filteredIdx_under_eps = filteredIdx[distIdx]
    target_cf = test_cf[filteredIdx_under_eps]

    lipschitz_list_for_all_candidates = []
    for under_eps in filteredIdx_under_eps:
        if under_eps == target:
            continue
        norm_x = calculate_lipschitz_factor(currentX, test_x[under_eps])
        norm_cf = calculate_lipschitz_factor(test_cf[target], test_cf[under_eps])
        lipschitz_list_for_all_candidates.append(norm_cf / norm_x)
    lipschitz_list.append(np.max(lipschitz_list_for_all_candidates))

print("Number of instances produced by the distance measure: {}".format(np.sum(lipschitz_list)))
print("Cohenrence mean: {:.2f}".format(np.mean(lipschitz_list)))
print("Cohenrence std: {:.2f}".format(np.std(lipschitz_list)))
print("=====================================================================")

l1_mean, l1_std, l2_mean, l2_std = l1_l2_evaluation(test_x, test_cf)
print("l1_mean: {:.2f} \t\t l2_mean: {:.2f}".format(l1_mean, l2_mean))
print("l1_std: {:.2f} \t\t l2_std: {:.2f}".format(l1_std, l2_std))