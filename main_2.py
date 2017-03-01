
import pandas as pd
import sys
from sklearn import metrics
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, train_test_split
import numpy as np
from IPython import embed
from sklearn import metrics, svm
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from time import time
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler, SMOTE




if '__main__':


    file = "../categories/no_GT/no_border/trump/trump_cleaned.csv"
    sar = "../../Sarcasm/sarcasm_cleaned.csv"
    raw = pd.read_csv(file).drop_duplicates()

    sarcasm = pd.read_csv(sar)
    sarcasm = sarcasm.drop_duplicates()
    sarcasm.columns = ['contents',
                     'Bernie',
                     'Trump',
                     'Clinton',
                     'Cruz',
                     'Num of Candidate',
                     'Sarcasm_s1',
                     'Sarcasm_s2',
                     'Sarcasm_s3',
                     'Sarcasm_s4',
                     'Sarcasm_s5',
                     'Sarcasm_s6',
                     'Sarcasm_s7',
                     'Number of Sarc',
                     'Short and link']

    res = pd.merge(raw, sarcasm, how='left', on=['contents']).dropna()
    res = res.drop_duplicates()

    del res['Bernie'], res['Trump'], res['Clinton'], res['Cruz']

    del raw, sarcasm

    raw = res

    print(raw.head())

    print("Total # of samples: ", len(raw))


    text = raw['contents']
    features = raw['Num of Candidate',
                     'Sarcasm_s1',
                     'Sarcasm_s2',
                     'Sarcasm_s3',
                     'Sarcasm_s4',
                     'Sarcasm_s5',
                     'Sarcasm_s6',
                     'Sarcasm_s7',
                     'Number of Sarc',
                     'Short and link']
    y = np.array(raw['label'], dtype='int64')

    vectorizer = CountVectorizer(min_df=0.005, ngram_range=(1,5), stop_words='english')
    weighting = TfidfTransformer(norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=False)
    text_token = weighting.fit_transform(vectorizer.fit_transform(text))

    text_df = pd.DataFrame(text_token.todense(), columns=list(vectorizer.vocabulary_))

    X = pd.concat([text_df, features], axis=1)
    embed()



    # '''
    # concat text_token and other features then over-sample the training set(leave out one test size)
    # '''
    # sm = SMOTE(kind='svm', random_state=2, ratio=0.102)
    # sm.fit_sample(X,y)
    # print("Shape of over-sampled data",sm.X_shape_)




