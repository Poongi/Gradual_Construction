from util import *
import torch

def prepare_german_dataset(filename, path_data):

    # Read Dataset
    df = pd.read_csv(path_data + filename, delimiter=',')

    # Features Categorization
    columns = df.columns
    class_name = 'default'
    possible_outcomes = list(df[class_name].unique())

    type_features, features_type = recognize_features_type(df, class_name)

    discrete = ['installment_as_income_perc', 'present_res_since', 'credits_this_bank', 'people_under_maintenance']
    discrete, continuous = set_discrete_continuous(columns, type_features, class_name, discrete, continuous=None)

    columns_tmp = list(columns)
    columns_tmp.remove(class_name)
    idx_features = {i: col for i, col in enumerate(columns_tmp)}

    # Dataset Preparation for Scikit Alorithms
    df_le, label_encoder = label_encode(df, discrete)
    X = df_le.loc[:, df_le.columns != class_name].values
    y = df_le[class_name].values

    dataset = {
        'name': filename.replace('.csv', ''),
        'df': df,
        'columns': list(columns),
        'class_name': class_name,
        'possible_outcomes': possible_outcomes,
        'type_features': type_features,
        'features_type': features_type,
        'discrete': discrete,
        'continuous': continuous,
        'idx_features': idx_features,
        'label_encoder': label_encoder,
        'X': X,
        'y': y,
    }

    return dataset

def minmax_scaler(X):
    max_list = pd.read_csv('./datasets/heloc_dataset_backup/X_train_scale_max.csv')
    min_list = pd.read_csv('./datasets/heloc_dataset_backup/X_train_scale_min.csv')
    max_list = np.array(max_list).squeeze()
    min_list = np.array(min_list).squeeze()
    rtn = torch.tensor(X).float()
    for col in range(X.shape[1]):
        X_col_max = max_list[col]
        X_col_min = min_list[col]
        rtn[:,col] = (rtn[:,col] - X_col_min)/(X_col_max - X_col_min)
    return rtn


def prepare_adult_dataset(filename, path_data):

    # Read Dataset
    df = pd.read_csv(path_data + filename, delimiter=',', skipinitialspace=True)

    # Remove useless columns
    del df['fnlwgt']
    del df['education-num']

    # Remove Missing Values
    for col in df.columns:
        if '?' in df[col].unique():
            df[col][df[col] == '?'] = df[col].value_counts().index[0]

    # Features Categorization
    columns = df.columns.tolist()
    columns = columns[-1:] + columns[:-1]
    df = df[columns]
    class_name = 'class'
    possible_outcomes = list(df[class_name].unique())

    type_features, features_type = recognize_features_type(df, class_name)

    discrete, continuous = set_discrete_continuous(columns, type_features, class_name, discrete=None, continuous=None)

    columns_tmp = list(columns)
    columns_tmp.remove(class_name)
    idx_features = {i: col for i, col in enumerate(columns_tmp)}

    # Dataset Preparation for Scikit Alorithms
    df_le, label_encoder = label_encode(df, discrete)
    X = df_le.loc[:, df_le.columns != class_name].values
    y = df_le[class_name].values

    dataset = {
        'name': filename.replace('.csv', ''),
        'df': df,
        'columns': list(columns),
        'class_name': class_name,
        'possible_outcomes': possible_outcomes,
        'type_features': type_features,
        'features_type': features_type,
        'discrete': discrete,
        'continuous': continuous,
        'idx_features': idx_features,
        'label_encoder': label_encoder,
        'X': X,
        'y': y,
    }

    return dataset


def prepare_heloc_dataset(filename, path_data):

    # Read Dataset
    df = pd.read_csv(path_data + filename, delimiter=',', skipinitialspace=True)

    # Remove useless columns
    del df['ExternalRiskEstimate']

    # Remove Missing Values
    for col in df.columns:
        if '?' in df[col].unique():
            df[col][df[col] == '?'] = df[col].value_counts().index[0]
    
    # Remove noisy data
    idx_to_remove = df[df['MSinceMostRecentTradeOpen'] == -9].index
    df = df.drop(idx_to_remove)

    # Features Categorization
    columns = df.columns.tolist()
    # columns = columns[-1:] + columns[:-1]
    # df = df[columns]
    class_name = 'RiskPerformance'
    possible_outcomes = list(df[class_name].unique())

    type_features, features_type = recognize_features_type(df, class_name)

    discrete, continuous = set_discrete_continuous(columns, type_features, class_name, discrete=None, continuous=None)

    columns_tmp = list(columns)
    columns_tmp.remove(class_name)
    idx_features = {i: col for i, col in enumerate(columns_tmp)}

    # Dataset Preparation for Scikit Alorithms
    df_le, label_encoder = label_encode(df, discrete)
    X_tmp = df_le.loc[:, df_le.columns != class_name].values
    y = df_le[class_name].values
    X = X_tmp
    # X = minmax_scaler(X_tmp)
    
    X= np.array(X)


    dataset = {
        'name': filename.replace('.csv', ''),
        'df': df,
        'columns': list(columns),
        'class_name': class_name,
        'possible_outcomes': possible_outcomes,
        'type_features': type_features,
        'features_type': features_type,
        'discrete': discrete,
        'continuous': continuous,
        'idx_features': idx_features,
        'label_encoder': label_encoder,
        'X': X,
        'y': y,
    }

    return dataset


def prepare_compass_dataset(filename, path_data):

    # Read Dataset
    df = pd.read_csv(path_data + filename, delimiter=',', skipinitialspace=True)

    columns = ['age', 'age_cat', 'sex', 'race',  'priors_count', 'days_b_screening_arrest', 'c_jail_in', 'c_jail_out',
               'c_charge_degree', 'is_recid', 'is_violent_recid', 'two_year_recid', 'decile_score', 'score_text']

    df = df[columns]

    df['days_b_screening_arrest'] = np.abs(df['days_b_screening_arrest'])

    df['c_jail_out'] = pd.to_datetime(df['c_jail_out'])
    df['c_jail_in'] = pd.to_datetime(df['c_jail_in'])
    df['length_of_stay'] = (df['c_jail_out'] - df['c_jail_in']).dt.days
    df['length_of_stay'] = np.abs(df['length_of_stay'])

    df['length_of_stay'].fillna(df['length_of_stay'].value_counts().index[0], inplace=True)
    df['days_b_screening_arrest'].fillna(df['days_b_screening_arrest'].value_counts().index[0], inplace=True)

    df['length_of_stay'] = df['length_of_stay'].astype(int)
    df['days_b_screening_arrest'] = df['days_b_screening_arrest'].astype(int)

    def get_class(x):
        if x < 7:
            return 'Medium-Low'
        else:
            return 'High'
    df['class'] = df['decile_score'].apply(get_class)

    del df['c_jail_in']
    del df['c_jail_out']
    del df['decile_score']
    del df['score_text']

    columns = df.columns.tolist()
    columns = columns[-1:] + columns[:-1]
    df = df[columns]
    class_name = 'class'
    possible_outcomes = list(df[class_name].unique())

    type_features, features_type = recognize_features_type(df, class_name)

    discrete = ['is_recid', 'is_violent_recid', 'two_year_recid']
    discrete, continuous = set_discrete_continuous(columns, type_features, class_name, discrete=discrete,
                                                   continuous=None)

    columns_tmp = list(columns)
    columns_tmp.remove(class_name)
    idx_features = {i: col for i, col in enumerate(columns_tmp)}

    # Dataset Preparation for Scikit Alorithms
    df_le, label_encoder = label_encode(df, discrete)
    X = df_le.loc[:, df_le.columns != class_name].values
    y = df_le[class_name].values

    dataset = {
        'name': filename.replace('.csv', ''),
        'df': df,
        'columns': list(columns),
        'class_name': class_name,
        'possible_outcomes': possible_outcomes,
        'type_features': type_features,
        'features_type': features_type,
        'discrete': discrete,
        'continuous': continuous,
        'idx_features': idx_features,
        'label_encoder': label_encoder,
        'X': X,
        'y': y,
    }

    return dataset