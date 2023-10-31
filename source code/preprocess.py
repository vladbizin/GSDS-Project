import pandas as pd
import numpy as np
from collections import Counter
from sklearn.decomposition import PCA
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split, StratifiedKFold


def drop_bands(X, bands, drop_noisy = True, drop_water = True):
    """Drop Noisy and/or Water vapor bands.
    
    Parameters
    ----------
    X : pandas DataFrame
        DataFrame with training data

    bands : pandas DataFrame
        DataFrame with info on bands

    drop_noisy : bool
        Drop noisy bands if True

    drop_water : bool
        Drop water vapor bands if True
    
    Returns
    ----------
    modified train_bands : pandas DataFrame
        same DataFrame but with dropped columns representing 'bad' bands
    """

    noisy_bands = bands.loc[bands['Noise_flag']==1]['Band_Number'].tolist() \
        if drop_noisy else bands['Band_Number'].tolist()
    
    water_vapor_bands = bands.loc[(0.37 <= bands['Band_nanometers']) &
                             (bands['Band_nanometers'] <= 0.50)]['Band_Number'].tolist() \
                             if drop_water else bands['Band_Number'].tolist()
    
    drop_bands = np.unique(noisy_bands + water_vapor_bands).tolist()
    return X.drop(columns = drop_bands)


def drop_outliers(X, y, thresh):
    """Drop outliers
    Drops rows that after whitened SPCA have values beyond +-sigma*thresh for any PC.
    In other words leaves only rows that after PCA \
    have all values within +-sigma*thresh.

    Parameters
    ----------
    train_bands : pandas DataFrame
        DataFrame with training data

    thresh : float
        defines the interval of kept data

    Returnds
    ----------
    modified train_bands : pandas DataFrame
    """

    pca = PCA(n_components = 20)
    train_bands_pca = pd.DataFrame(data = pca.fit_transform(np.array(X.iloc[:, 1:])))
    sample_mask = (train_bands_pca.abs() <= thresh).all(axis = 1)
    return X.loc[sample_mask.values], y.loc[sample_mask.values]


def tansform(X, n_components):
    """Transform the data via PCA.
    Leaves first n principal componetns.

    Parameters
    ----------
    X : pandas DataFrame
        DataFrame with training data

    n_components : int
        Number of principal components to keep
    
    Returns
    ----------
    modified train_bands : pandas DataFrame
        transformed train DataFrame
    """

    pca_new = PCA(n_components)
    return pca_new.fit_transform(X.iloc[:, 1:])


def split_id(train_id, calib_size, test_size):
    """Split the tree crowns into traininig, calibration\
    and testing

    Parameters
    ----------
    train_id : pandas DataFrames
        contains crown IDs and species IDs

    calib_size : float
        desired portion of calibration data
    
    test_size : float
        desired portion of test data
    
    N.B. the rest data is training data

    Returns
    ----------
    train_id, calib_id, test_id : pandas DataFrames
        speaks for itself
    """

    train_size = 1 - calib_size - test_size
    
    if train_size == 1: return train_id

    train_id, tmp_id = train_test_split(
        train_id,
        shuffle = True, random_state = 42, train_size = train_size, stratify=train_id.iloc[:, 1]
    )

    if (calib_size == 0 or test_size == 0): return train_id, tmp_id
    calib_id, test_id = train_test_split(
        tmp_id,
        shuffle = True, random_state = 42, test_size=test_size/(test_size + calib_size), stratify=tmp_id.iloc[:, 1]
    )
    
    return train_id, calib_id, test_id


def split(X, y, train_id, calib_size, test_size):
    """Splits the data into traininig, calibration\
    and testing based on crown id splitting.
    
    Parameters
    ----------
    X : pandas DataFrame
        DataFrame with training data

    y : pandas DataFrame
        DataFrame with training data labels

    calib_size : float
        desired portion of calibration data
    
    test_size : float
        desired portion of test data
    
    N.B. the rest data is training data

    Returns
    ----------
    train_df, calib_df, test_df : pandas DataFrames
        merged dataframes of data and labels
    train_id, calib_id, test_id : pandas DataFrames
    """

    train_id, calib_id, test_id = split_id(train_id, calib_size, test_size)

    # train, calibrate, test
    Xs, ys, dfs = [], [], []

    for id in [train_id, calib_id, test_id]:
        Xs.append(
            X.loc[X['crown_id'].isin(id['crown_id'].unique())]
        )
        ys.append(
            y.loc[y['crown_id'].isin(id['crown_id'].unique())]
        )
        dfs.append(
            pd.merge(ys[-1], Xs[-1], on='crown_id', how='inner')
        )
        dfs[-1].columns = dfs[-1].columns.astype(str)
    
    return tuple(dfs) + (train_id, calib_id, test_id)


def resample_df(df, n_samples):
    """Resamples the data.
    Reduces the number of oversampled classes objects and \
    increases the number of undersampled classes objects \
    by randomly sampling with replacement the current available samples.

    Parameters
    ----------
    df : pandas DataFrame
        DataFrame with training data

    n_samples : int
        Desired number of samples in each class

    Returns
    ----------
    TRAIN_res : pandas DataFrame
        resampled data
    """

    #oversample
    oversample_dict = {k:n_samples for k in df['species_id'].unique() if Counter(df['species_id'])[k] < n_samples}
    ros = RandomOverSampler(sampling_strategy = oversample_dict, random_state=2001)
    df_tmp, y_tmp = ros.fit_resample(df, df['species_id'])

    #undersample
    undersample_dict = {k:n_samples for k in y_tmp.unique() if Counter(y_tmp)[k] > n_samples}
    rus = RandomUnderSampler(sampling_strategy = undersample_dict, random_state=2001)
    df_res, y_res = rus.fit_resample(df_tmp, y_tmp)
    
    df_res['species_id'] = y_res
    return df_res


def resample(train_df, calib_df, test_df, n_samples, calib_size, test_size):
    """Resample the data.
    Reduces the number of oversampled classes objects and \
    increases the number of undersampled classes objects \
    by randomly sampling with replacement the current available samples.

    Parameters
    ----------
    train_df, calib_df, test_df : pandas DataFrames
        DataFrame with training, calibration and\
        testing data

    n_samples : int
        Desired number of samples in each class\
        over all data
    
    calib_size : float
        portion of calibration data
    
    test_size : float
        portion of testing data

    Returns
    ----------
    X_train, y_train, X_calib, y_calib, X_test, y_test : array-like of shape (n_samples, n_features)
        resampled data
    """

    train_size = 1 - test_size - calib_size
    sizes = [train_size, calib_size, test_size]
    dfs = [train_df, calib_df, test_df]
    out = []

    for i, size in enumerate(sizes):
        dfs[i] = resample_df(dfs[i], int(n_samples*size))
        dfs[i].reset_index(inplace = True, drop = True)

        out.append(dfs[i].drop(columns = ['species_id']))
        out.append(dfs[i][['crown_id', 'species_id']])

    return tuple(out)
    

def cv_split(X, id, n_splits):
    """Generate indices of stratified folds.\
    Shuffled by default.
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Training data, where n_samples is the number of samples\
        and n_features is the number of features.

    n_splits : int
        Number of folds. Must be at least 2.

    Returns
    ----------
    cv : ndarray
        CV split indices for that split in a shape\
        (train__indices, test_indices) * n_splits
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2001)
    skf.get_n_splits(id, id[['species_id']])
    cv = list(skf.split(id, id[['species_id']]))

    for (i,split) in enumerate(cv):
        split = list(split)
        split[0] = X.loc[X['crown_id'].isin(id.iloc[split[0], 0])].index.values
        split[1] = X.loc[X['crown_id'].isin(id.iloc[split[1], 0])].index.values
        cv[i] = tuple(split)

    return cv


def prepare_training_data(train_bands, train_id, bands,
                          thresh = 3,
                          n_components = 20,
                          calib_size = 0.25, test_size = 0.25,
                          n_samples = 400):
    """Prepares the given data to be fed to classifier.
    1. Drops Noisy and/or Water vapor bands.
    2. Drops outliers.
    3. Transform the data via PCA.
    4. Split the data into traininig, calibration and testing
    5. Resamples the data.
    6. Generates indices of stratified folds.

    Parameters
    ----------
    
    train_bands : array-like of shape (n_samples, n_features)
        Training data, where n_samples is the number of samples and n_features is the number of features.
    
    train_id : array-like of shape (n_samples, n_labels)
        Extra training data and target variable.

    bands : array-like
        Information on spectral bands.

    thresh : float
        defines the interval of kept data

    n_components : int
        Number of principal components to keep

    calib_size : float
        desired portion of calibration data
    
    test_size : float
        desired portion of test data

    n_samples : int
        Desired number of samples in each class for over all data

    Returns
    ----------
    X_train, y_train, X_calib, y_calib, X_test, y_test : array-like
        Prepared train-calibrate-test data

    test_cr : array-like
        crowns ids of corresponding testing data

    test_id : array-like
        species id of crown id sorted by ascending crows ids
        
    train_cv : array-like
        CV split indices for that split in a shape\
        (train__indices, test_indices) * n_splits

    """

    train_id = train_id[['crown_id', 'species_id']]
    X, y = train_bands.drop(columns = ['chm']),\
           pd.merge(train_bands[['crown_id']],
                    train_id,
                    on='crown_id', how='inner')
    
    
    # drop bad bands
    X = drop_bands(X, bands)

    # drop outliers
    X, y = drop_outliers(X, y, thresh)

    # PCA transform
    X = tansform(X, n_components)
    X = pd.concat(
        [y[['crown_id']], pd.DataFrame(index = y.index, data = X)], axis = 1
    ) 
    X.reset_index(inplace=True, drop = True)

    # train-calibrare-test split based on crown IDs
    train_df, calib_df, test_df,\
        train_id, calib_id, test_id = split(X, y, train_id, calib_size, test_size)
    
    # resample
    X_train, y_train, X_calib,\
        y_calib, X_test, y_test = resample(train_df, calib_df, test_df, n_samples, calib_size, test_size)
    
    # prepare splits for cross-validation
    train_cv = cv_split(X_train, train_id, 5)
    
    test_id = y_test.groupby('crown_id').first().iloc[:, 0].values
    
    X_train, y_train, = X_train.iloc[:, 1:].values, y_train.iloc[:, 1].values
    X_calib, y_calib, = X_calib.iloc[:, 1:].values, y_calib.iloc[:, 1].values
    X_test, y_test, test_cr =\
        X_test.iloc[:, 1:].values, y_test.iloc[:, 1].values, X_test.iloc[:, 0].values,
    

    return X_train, y_train, X_calib, y_calib, X_test, y_test,\
        test_cr, test_id, train_cv

def prepare_testing_data(test_bands, test_id, bands, n_components = 20):
    """Prepares the given data to be fed to classifier.
    1. Drops Noisy and/or Water vapor bands.
    3. Transform the data via PCA.

    Parameters
    ----------
    train_bands : array-like of shape (n_samples, n_features)
        Training data, where n_samples is the number of samples and n_features is the number of features.
    
    train_id : array-like of shape (n_samples, n_labels)
        Extra training data and target variable.

    bands : array-like
        Information on spectral bands.

    n_components : int
        Number of principal components to keep

    Returns
    ----------
    X, y, cr, id array-like
        Prepared train-calibrate-test data
    """
    # drop bad bands
    test_bands = drop_bands(test_bands.drop(columns = ['chm']), bands)

    # PCA transformation
    X = tansform(test_bands, n_components = n_components)

    test_id.rename(columns = {'itcID': 'crown_id', 'SpeciesID': 'species_id'}, inplace = True)
    test_id = test_id[['crown_id', 'species_id']]
    y = pd.merge(test_bands[['crown_id']], test_id, on='crown_id', how='inner')
    # TEST = pd.concat([y, pd.DataFrame(data = features)], axis = 1)
    # TEST.columns = TEST.columns.astype(str)

    cr, y = y.iloc[:, 0].values, y.iloc[:, 1].values
    id = test_id.sort_values(by='crown_id').iloc[:,1].values

    return X, y, cr, id 