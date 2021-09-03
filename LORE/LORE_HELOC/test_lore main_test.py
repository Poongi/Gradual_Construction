import lore

from prepare_dataset import *
from neighbor_generator import *

import sys
sys.path.append("/home/heedong/Documents/LORE")

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import torch
from numpy import linalg as LA
from models import model
import pickle


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

def np_to_cuda(array):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    list_tensor = torch.from_numpy(array)
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


def minmax_unscaler(X):
    max_list = pd.read_csv('./datasets/heloc_dataset_backup/X_train_scale_max.csv')
    min_list = pd.read_csv('./datasets/heloc_dataset_backup/X_train_scale_min.csv')
    max_list = np.array(max_list).squeeze()
    min_list = np.array(min_list).squeeze()
    rtn = torch.tensor(X)
    for col in range(X.shape[1]):
        X_col_max = max_list[col]
        X_col_min = min_list[col]
        rtn[:,col] = rtn[:,col]*(X_col_max - X_col_min) + X_col_min
    return rtn

def minmax_scaler(X):
    max_list = pd.read_csv('./datasets/heloc_dataset_backup/X_train_scale_max.csv')
    min_list = pd.read_csv('./datasets/heloc_dataset_backup/X_train_scale_min.csv')
    max_list = np.array(max_list).squeeze()
    min_list = np.array(min_list).squeeze()
    rtn = torch.tensor(X)
    for col in range(X.shape[1]):
        X_col_max = max_list[col]
        X_col_min = min_list[col]
        rtn[:,col] = (rtn[:,col] - X_col_min)/(X_col_max - X_col_min)
    return rtn


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

warnings.filterwarnings("ignore")



# def main():

# dataset_name = 'german_credit.csv'
path_data = '/home/heedong/Documents/LORE/datasets/'
# dataset = prepare_german_dataset(dataset_name, path_data)

# dataset_name = 'compas-scores-two-years.csv'
# dataset = prepare_compass_dataset(dataset_name, path_data)

# dataset_name = 'adult.csv'
# dataset = prepare_adult_dataset(dataset_name, path_data)

dataset_name = 'HELOC_dataset.csv'
dataset = prepare_heloc_dataset(dataset_name, path_data)
print(dataset['label_encoder'][dataset['class_name']].classes_)
print(dataset['possible_outcomes'])

# dataset_name = 'adult.csv'
# dataset = prepare_adult_dataset(dataset_name, path_data)

X, y = dataset['X'], dataset['y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
blackbox = torch.load("./models/HELOC_retrained.pt")
# blackbox = RandomForestClassifier(n_estimators=20)
# blackbox.fit(X_train, y_train)
# y2E = blackbox.predict(X2E)

target_list = []
explain_list = []
cf_list = []

nbr_evaluates = 1000
for i in range(1000) : 

    X2E = X_test[:nbr_evaluates]
    target_list.append(X2E[i])

    max_list = pd.read_csv('./datasets/heloc_dataset_backup/X_train_scale_max.csv')
    min_list = pd.read_csv('./datasets/heloc_dataset_backup/X_train_scale_min.csv')
    max_list = np.array(max_list).squeeze()
    min_list = np.array(min_list).squeeze()

    # for i, max in enumerate(max_list):
    #     print((X2E[:,i]/max).max())
    # for i, max in enumerate(max_list):
    #     print((X2E[:,i]/max).min())

    X2E_scaled_cuda = minmax_scaler(X2E.astype('float32')).to(device)
    y2E_cuda = torch.max(blackbox(X2E_scaled_cuda), 1).indices

    y2E = blackbox.predict(X2E)

    print("predicted = predict")
    evaluate_model(blackbox, X2E_scaled_cuda, y2E_cuda)
    print("predict = class")
    evaluate_model(blackbox, np.array(X2E,dtype = float), np_to_cuda(y_test[:nbr_evaluates]), sklearn=True)
    y2E = np.asarray([dataset['possible_outcomes'][i] for i in y2E])

    idx_record2explain = 0

    explanation, infos = lore.explain(idx_record2explain, X2E, dataset, blackbox,
                                        ng_function=genetic_neighborhood,
                                        discrete_use_probabilities=True,
                                        continuous_function_estimation=False,
                                        returns_infos=True,
                                        path=path_data, sep=';', log=False)

    dfX2E = build_df2explain(blackbox, X2E, dataset).to_dict('records')
    dfx = dfX2E[idx_record2explain]
    # x = build_df2explain(blackbox, X2E[idx_record2explain].reshape(1, -1), dataset).to_dict('records')[0]

    print('x = %s' % dfx)
    print('r = %s --> %s' % (explanation[0][1], explanation[0][0]))
    for delta in explanation[1]:
        print('delta', delta)

    explain_list.append(explanation[0][1])
    cf_list.append(explanation[1])




array_target = np.array(target_list)
array_cf = np.array(cf_list)
array_explain = np.array(explain_list)
array_features_name = np.array(list(dfx.keys()))


test_num = 100
target = np.array(target_list)[:test_num]
cf = np.array(cf_list)[:test_num]
explain = np.array(explain_list)[:test_num]
features_name = np.array(list(dfx.keys()))[:test_num]


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
            return cf_value
        else :
            return original_value
    elif '>=' == inequality:
        if original_value >= cf_value:
            return original_value
        else :
            return cf_value
    elif '<' == inequality :
        if original_value < cf_value:
            return cf_value - eps
        else :
            return original_value
    elif '>' ==  inequality:
        if original_value > cf_value:
            return original_value
        else :
            return cf_value + eps


eps = 1

for experiment in range(target.shape[0]) : 
    X2val = target[experiment]
    cf_array = np.zeros((10,10,23))
    for case in range(len(cf[experiment])) :
        for key in cf[experiment][case] :
            idx2change = np.where(features_name == key)[0]
            condition = cf[experiment][case][key]
            if len(condition.split()) >= 2 :
                left_condition = condition.split()[0]
                riught_condition = condition.split()[2]
                condition_name = condition.split()[1]
                left_inequality, left_value = num_inequality_finder(left_condition)
                right_inequality, right_value = num_inequality_finder(riught_condition)
                # print(left_inequality, left_value, condition_name, right_inequality, right_value)
                cf_array[experiment][case][idx2change] = find_cf_value(left_inequality, cf_array[idx2change], left_value)
                cf_array[experiment][case][idx2change] = find_cf_value(right_inequality, cf_array[idx2change], right_value)
            else :
                inequality, cf_value = num_inequality_finder(condition)
                # print(inequality, value)
                cf_array[experiment][case][idx2change] = find_cf_value(inequality, cf_array[experiment][case][idx2change], cf_value, eps)






    
# with open('./datasets/LORE_heloc_result/target_X.pickle', 'wb') as f:
#     pickle.dump(array_target, f)

# with open('./datasets/LORE_heloc_result/target_cf.pickle', 'wb') as f:
#     pickle.dump(array_cf, f)

# with open('./datasets/LORE_heloc_result/target_explanation.pickle', 'wb') as f:
#     pickle.dump(array_explain, f)

# with open('./datasets/LORE_heloc_result/heloc_features_name', 'wb') as f:
#     pickle.dump(array_features_name, f)


# with open('./datasets/LORE_heloc_result/target_X.pickle', 'rb') as f:
#     load_X = pickle.load(f)

# with open('./datasets/LORE_heloc_result/target_cf.pickle', 'rb') as f:
#     load_cf = pickle.load(f)

# with open('./datasets/LORE_heloc_result/target_explanation.pickle', 'rb') as f:
#     load_explanation = pickle.load(f)

# with open('./datasets/LORE_heloc_result/heloc_features_name', 'rb') as f:
#     load_features_name = pickle.load(f)







columns = dataset['columns'][1:]







covered = lore.get_covered(explanation[0][1], dfX2E, dataset)
print(len(covered))
print(covered)

print(explanation[0][0][dataset['class_name']], '<<<<')

def eval(x, y):
    return 1 if x == y else 0

precision = [1-eval(v, explanation[0][0][dataset['class_name']]) for v in y2E[covered]]
print(precision)
print(np.mean(precision), np.std(precision))




# if __name__ == "__main__":
#     main()
