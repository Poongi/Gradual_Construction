import numpy as np
import pandas as pd
from sklearn import datasets, ensemble
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import growingspheres.counterfactuals as cf
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
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import cdist, pdist
from numpy import linalg as LA

# path_heloc = 'dataset/heloc/heloc_dataset.csv'
# df_heloc = pd.read_csv(path_heloc)
# label = pd.get_dummies(df_heloc["RiskPerformance"], drop_first=True)
# y = np.array(label).reshape(-1)
# X = np.array(df_heloc.drop([df_heloc.columns[0]], axis=1))

# X,y = datasets.make_moons(n_samples = 200, shuffle=True, noise=0.05, random_state=0)
# X = (X.copy() - X.mean(axis=0))/X.std(axis=0)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# y_test

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
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        X_test = torch.tensor(X_test).to(device)
        predicted = np.round(clf.predict(X_test))
        correct = (np.round(predicted) == y_test).sum()
        print('test_acc', correct/X_test.shape[0])


def minmax_scaler(X):
    max_list = pd.read_csv('./example/HELOC/HELOC_dataset_backup/X_train_scale_max.csv')
    min_list = pd.read_csv('./example/HELOC/HELOC_dataset_backup/X_train_scale_min.csv')
    max_list = np.array(max_list).squeeze()
    min_list = np.array(min_list).squeeze()
    rtn = torch.tensor(X)
    for col in range(X.shape[1]):
        X_col_max = max_list[col]
        X_col_min = min_list[col]
        rtn[:,col] = (rtn[:,col] - X_col_min)/(X_col_max - X_col_min)
    return rtn



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# model load
# clf = torch.load("./models/saved/HELOC_retrained.pt")

# data load
X_test_raw = pd.read_csv('./example/HELOC/heloc_dataset_backup/X_test_unscaled.csv')
X_test_raw = df_to_cuda(X_test_raw)

X_train_raw = pd.read_csv('./example/HELOC/heloc_dataset_backup/X_train_unscaled.csv')
X_train_raw = df_to_cuda(X_train_raw)

X_train = pd.read_csv('./example/HELOC/heloc_dataset_backup/X_train.csv')
X_train = df_to_cuda(X_train)

X_train_raw = pd.read_csv('./example/HELOC/heloc_dataset_backup/X_train_unscaled.csv')
X_train_raw = df_to_cuda(X_train_raw)

X_test = pd.read_csv('./example/HELOC/heloc_dataset_backup/X_test.csv')
X_test = df_to_cuda(X_test)

y_train = pd.read_csv('./example/HELOC/heloc_dataset_backup/y_train.csv')
y_train = df_to_cuda(y_train).squeeze().long()
y_test = pd.read_csv('./example/HELOC/heloc_dataset_backup/y_test.csv')
y_test = df_to_cuda(y_test).squeeze().long()

# hyperparameter
num_epochs = 70
learning_rate = 0.03

# # model training
# clf = model.MLP(input_size=22, output_size=2).to(device)
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(clf.parameters(), lr=learning_rate)

# for epoch in range(num_epochs):
#     clf.train()
#     optimizer.zero_grad()
#     output = clf(X_train)
#     loss = criterion(output, y_train) 
#     if epoch%10 == 0:
#         print('train_loss', loss.item())
#         evaluate_model(clf, X_test, y_test, sklearn=False)

#     loss.backward()
#     optimizer.step()
    
# print()
# evaluate_model(clf, X_test, y_test, sklearn=False)

# # torch.save(clf, "models/saved/HELOC_retrained_GS.pt")
# # clf = torch.load("models/saved/HELOC_retrained_GS.pt")
# # evaluate_model(clf, X_test, y_test)


# clf = ensemble.RandomForestClassifier(n_estimators=200, max_depth=3)
# clf = SVC(gamma=1, kernel = 'rbf', C = 0.1, probability=True)
# clf = tree.DecisionTreeClassifier(max_depth=6)
# clf = clf.fit(X_train, y_train)
# print(' ### Accuracy:', sum(clf.predict(X_test) == y_test)/y_test.shape[0])


y_test = cuda_to_numpy(y_test)
X_test = cuda_to_numpy(X_test)
clf = torch.load("models/saved/HELOC_retrained.pt")
evaluate_model(clf, X_test, y_test, sklearn=True)

cf_list = []
cnt = 0
nbr_experiments = 1000
# X_test_class0 = X_test[np.where(y_test == 0)]
for obs in X_test[:nbr_experiments]:
    print('====================================================', cnt)
    CF = cf.CounterfactualExplanation(obs, clf.predict, method='GS')
    CF.fit(n_in_layer=2000, first_radius=0.1, dicrease_radius=10, sparse=True, verbose=True)
    cf_list.append(CF.enemy)
    cnt += 1
cf_list = np.array(cf_list) 

evaluate_model(clf, X_test, y_test, sklearn=True)

print()
print("qualify : ", (clf.predict(cf_list) + clf.predict(X_test[:nbr_experiments])).sum(),"=", nbr_experiments)


# pd.DataFrame(cf_list).to_csv('./example/GS_example/HELOC_cf.csv', index = False)
# pd.DataFrame(X_test).to_csv('./example/GS_example/HELOC_X.csv', index = False)




# def plot_classification_contour(X, clf, ax=[0,1]):
#     ## Inspired by scikit-learn documentation
#     h = .02  # step size in the mesh
#     cm = plt.cm.RdBu
#     x_min, x_max = X[:, ax[0]].min() - .5, X[:, ax[0]].max() + .5
#     y_min, y_max = X[:, ax[1]].min() - .5, X[:, ax[1]].max() + .5
#     xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
#     Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
#     Z = Z.reshape(xx.shape)
#     #plt.sca(ax)
#     plt.contourf(xx, yy, Z, alpha=.5, cmap=cm)


# plot_classification_contour(X_test, clf)
# plt.scatter(X_test_class0[:, 0], X_test_class0[:, 1], marker='o', edgecolors='k', alpha=0.9, color='red')
# plt.scatter(cf_list[:, 0], cf_list[:, 1], marker='o', edgecolors='k', alpha=0.9, color='green')
# plt.title('Test instances (red) and their generated counterfactuals (green)')
# plt.tight_layout()
# plt.show()
   

# import numpy as np
# import pandas as pd
# from sklearn import datasets, ensemble, tree
# from sklearn.model_selection import train_test_split
# from sklearn.svm import SVC
# from matplotlib import pyplot as plt
# import xgboost as xgb

# #X,y = datasets.make_moons(n_samples = 200, shuffle=True, noise=0.05, random_state=0)

# '''ONLINE NEWS POPULARITY'''
# # df = pd.read_csv('datasets/newspopularity.csv', header=0, nrows=10000)
# df = datasets.fetch_openml(data_id=4545)
# data = df.data[:10000, :]
# y = df.target[:10000]
# y = np.array([int(x>=1400) for x in y])
# print(df.feature_names[2:-1])
# X = np.array(data[:, 2:-1])

# X = (X.copy() - X.mean(axis=0))/X.std(axis=0)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

# print('X_test shape:', X_test.shape)



# clf = ensemble.RandomForestClassifier(n_estimators=200, max_depth=None, n_jobs=-1)
# clf = clf.fit(X_train, y_train)
# print(' ### Accuracy:', sum(clf.predict(X_test) == y_test)/y_test.shape[0])

# def plot_classification_contour(X, clf, ax=[0,1]):
#     ## Inspired by scikit-learn documentation
#     h = .02  # step size in the mesh
#     cm = plt.cm.RdBu
#     x_min, x_max = X[:, ax[0]].min() - .5, X[:, ax[0]].max() + .5
#     y_min, y_max = X[:, ax[1]].min() - .5, X[:, ax[1]].max() + .5
#     xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
#     Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
#     Z = Z.reshape(xx.shape)
#     #plt.sca(ax)
#     plt.contourf(xx, yy, Z, alpha=.5, cmap=cm)

# def get_CF_distances(obs, n_in_layer=10000, first_radius=0.1, dicrease_radius=10, sparse=True):
#     CF = cf.CounterfactualExplanation(obs, clf.predict, method='GS')
#     CF.fit(n_in_layer=n_in_layer, first_radius=first_radius, dicrease_radius=dicrease_radius, sparse=sparse,
#            verbose=False)
#     out = CF.distances()
#     l2, l0 = out['euclidean'], out['sparsity']
#     return l2, l0

# def get_CF(obs, n_in_layer=2000, first_radius=0.1, dicrease_radius=10, sparse=True):
#     CF = cf.CounterfactualExplanation(obs, clf.predict, method='GS')
#     CF.fit(n_in_layer=n_in_layer, first_radius=first_radius, dicrease_radius=dicrease_radius, sparse=sparse,
#            verbose=False)
#     e_tilde = CF.e_star
#     e_f = CF.enemy
#     return obs, e_tilde, e_f
    
    
# def iterate_gs_dataset(n_in_layer=2000, first_radius=0.1, dicrease_radius=10, sparse=True):
#     l2_list, l0_list = [], []
#     cnt = 0
#     for obs in X_test[:10, :]:
#         print('====================================================', cnt)
#         l2, l0 = get_CF(obs, n_in_layer=n_in_layer, 
#                                first_radius=first_radius, 
#                                dicrease_radius=dicrease_radius, 
#                                sparse=sparse)
#         l2_list.append(l2)
#         l0_list.append(l0)
#         cnt += 1
#     return l2_list, l0_list