import numpy as np
import pandas as pd
from preprocess import split
from sklearn.preprocessing import LabelEncoder

def generate_test_data(train_bands, train_id, like_the_competition = True):
    if like_the_competition:
        ids = ['ACRU', 'LIST', 'OTHER', 'PIEL', 'PIPA', 'PITA', 'QUGE', 'QULA', 'QUNI']
        cnt = [2, 1, 3, 2, 83, 6, 4, 23, 1]
        test_cnts = tuple(zip(ids,cnt))

        test_id_mask = np.repeat(False, train_id.shape[0])
        test_bands_mask = np.repeat(False, train_bands.shape[0])

        for id, cnt in test_cnts:
            test_crown_id = train_id.loc[(train_id['species_id'] == id) & (train_id['crown_id'] != 494)].sample(n=cnt)['crown_id'].values
            test_id_mask |= train_id['crown_id'].isin(test_crown_id)

            test_bands_mask |= train_bands['crown_id'].isin(test_crown_id)

        test_id = train_id.loc[test_id_mask]
        train_id = train_id.loc[~test_id_mask]

        test_bands = train_bands.loc[test_bands_mask]
        train_bands = train_bands.loc[~test_bands_mask]
        
        y_train = pd.merge(train_bands[['crown_id']], train_id[['crown_id', 'species_id']], on='crown_id', how='inner')
        y_test = pd.merge(test_bands[['crown_id']], test_id[['crown_id', 'species_id']], on='crown_id', how='inner')

        return train_bands, y_train, test_bands, y_test
    else:
        TRAIN = pd.merge(train_bands, train_id[['crown_id', 'species_id']], on='crown_id', how='inner')
        X_train, y_train, X_test, y_test = split(TRAIN, test_size = 1313/4729, calib_size = 0)
        return X_train, y_train, X_test, y_test
    
def read(train_path, train_id_path, bands_path):
    id_df = pd.read_csv(train_id_path)
    le = LabelEncoder()
    le.fit(id_df['species_id'])
    id_df['species_id'] = le.transform(id_df['species_id'])

    return pd.read_csv(train_path), id_df, pd.read_csv(bands_path), le

def read_test(test_path, test_id_path, le):
    id_df = pd.read_csv(test_id_path)
    id_df['SpeciesID'] = le.transform(id_df['SpeciesID'])
    return pd.read_csv(test_path), id_df