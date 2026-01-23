import os
import sys
import pandas as pd
import numpy as np
import pickle
 
# Machine learning - only what we need
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.pipeline import Pipeline 
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier


if __name__ == "__main__":

    import argparse
    import time
    import datetime

    # Argument Parsing
    # -----------------------------------------------
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--input_directory', type = str,
        required = True, 
        help = 'directory path of kilosort files')
    args = vars(ap.parse_args())

    DATA_FOLDER = args['input_directory']
    assert os.path.exists(DATA_FOLDER), "Input data folder does not exist"

    attrib_loc = os.path.join(DATA_FOLDER, 'cluster_attribute_data.tsv')
    cluster_group = os.path.join(DATA_FOLDER, 'cluster_group.tsv')
    cluster_KSLabel = os.path.join(DATA_FOLDER, 'cluster_KSLabel.tsv')

    c_KSLabel = pd.read_csv(cluster_KSLabel, sep='\t', index_col=0)
    cluster = pd.read_csv(attrib_loc, sep='\t', index_col=0)
    group = pd.read_csv(cluster_group, sep='\t', index_col=0)


    print(group.head())

    model_loc = '../../kilosort model/'
    feature_loc = os.path.join(model_loc, 'features.npy')
    model_loc = os.path.join(model_loc, 'classification_model.npy')
    
    with open(model_loc, 'rb') as inp:
        model = pickle.load(inp)
    full_features = np.load(feature_loc)
    print('Classifying waveforms based on the following pipeline:\n')
    print(model)
    print('These are the features used: ')
    print(full_features)

    # Create full feature datasets
    X_full = cluster[full_features].fillna(0)

    y_predict = model.predict(X_full)
    y_prob = model.predict_proba(X_full)

    definite = 0.8
    maybe = 0.4

    maybes = list(np.where((y_prob[:,0]>maybe)&(y_prob[:,0]<definite))[0])
    print('Number of waveforms that are undetermined: ', len(maybes))
    # maybes = []

    for index, row in group.iterrows():
        if index in maybes:
            group.loc[index, 'group'] = np.nan
        else:
            if y_predict[index]==0:
                group.loc[index, 'group'] = 'good'
            elif y_predict[index]==1:
                group.loc[index, 'group'] = 'mua'
            elif y_predict[index]==2:
                group.loc[index, 'group'] = 'noise'

    same = np.sum(c_KSLabel['KSLabel']==group['group'])
    n_clusters = len(group)
    percent =np.round(same/n_clusters*100,2)
    print('\n{0} of {1} ({2} %) were the same as KS labeling'.format(same, n_clusters, percent))

    if 'KSLabel' in group.columns:
        group.drop('KSLabel', axis=1, inplace=True)
    if 'Unnamed: 0' in group.columns:
        group.drop('Unnamed: 0', axis=1, inplace=True)

    savefile = os.path.join(DATA_FOLDER, 'cluster_group_new.tsv')
    group.to_csv(savefile, sep='\t')


