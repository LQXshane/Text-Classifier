from trainer import tuner
import pandas as pd
import sys
from sklearn import metrics
from sklearn.model_selection import train_test_split
import numpy as np
from IPython import embed


if '__main__':
    # file1 = "../categories/GT/border_2/cruz/cruz_cleaned.csv"
    # file2 = "../categories/GT/border_2/sanders/sanders_cleaned.csv"
    # file3 = "../categories/GT/border_2/trump/trump_cleaned.csv"
    # file4 = "../categories/GT/border_2/clinton/clinton_cleaned.csv"
    # raw1 = pd.read_csv(file1)
    # raw2 = pd.read_csv(file2)
    # raw3 = pd.read_csv(file3)
    # raw4 = pd.read_csv(file4)
    #
    # raw = pd.concat([raw1, raw2, raw3, raw4])

    file = "../categories/no_GT/border/trump/trump_cleaned.csv"
    # sar = "../../Sarcasm/sarcasm_cleaned.csv"
    raw = pd.read_csv(file).drop_duplicates()

    # sarcasm = pd.read_csv(sar)
    # sarcasm = sarcasm.drop_duplicates()
    # sarcasm.columns = ['contents',
    #                  'Bernie',
    #                  'Trump',
    #                  'Clinton',
    #                  'Cruz',
    #                  'Num of Candidate',
    #                  'Sarcasm_s1',
    #                  'Sarcasm_s2',
    #                  'Sarcasm_s3',
    #                  'Sarcasm_s4',
    #                  'Sarcasm_s5',
    #                  'Sarcasm_s6',
    #                  'Sarcasm_s7',
    #                  'Number of Sarc',
    #                  'Short and link']
    #
    # res = pd.merge(raw, sarcasm, how='left', on=['contents']).dropna()
    # res = res.drop_duplicates()
    # # embed()
    # del res['Bernie'], res['Trump'], res['Clinton'], res['Cruz']
    #
    # del raw, sarcasm
    #
    # raw = res
    # # embed()
    # print(raw.head())
    #
    # # del raw1, raw2, raw3, raw4


    X = raw['contents']
    y = np.array(raw['label'], dtype='int64')
    # del raw['label']

    # X = raw

    parameters = {
        # 'vect__min_df': (0.005) ,    #(0.003, 0.007, 0.005,0.001),
        # 'vect__max_features': (None, 5000, 10000, 50000),
        # 'vect__ngram_range': ((1, 1), (1, 2), (1, 5), (1, float("inf"))),  # unigrams or bigrams
        # 'tfidf__use_idf': (True, False),
        # 'tfidf__norm': ('l1', 'l2'),
        'clf__C': (0.9, 0.8, 0.7, 0.6, 0.5, 0.4), # 0.3, 0.2),
        # 'clf__kernel': ('linear', 'rbf'),#, 'precomputed'),, 'poly', 'sigmoid'
        # 'clf__class_weight': (None, 'balanced'),
        # 'clf__tol': (1e-3, 1e-4, 1e-5)
        # 'clf__probability':(False, True),
    }

    svm_models, eval_scores = tuner(X, y, 7, parameters, 5)

    print ("Using precision as CV evaluation: ", eval_scores)

    # embed()

    bestie = svm_models.best_estimator_

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    clf = bestie.fit(X_train, y_train)

    predicted = clf.predict(X_test)

    # embed()
    print("Precision: %0.2f"%(metrics.precision_score(y_test, predicted)))
    print("TPR(recall): %0.2f" % (metrics.recall_score(y_test, predicted)))
    print("F1_score: %0.2f" % (metrics.f1_score(y_test, predicted)))

    print(metrics.classification_report(y_test, predicted))

    cmat = metrics.confusion_matrix(y_test, predicted)
    print(cmat)

    print("CCR %0.2f"%((float(cmat[0][0]+cmat[1][1]))* 100/sum(sum(cmat))))
    print("Percentage of predicted positives %0.2f" % (float((cmat[0][1] + cmat[1][1])) * 100 / sum(sum(cmat))))
    print(svm_models.best_params_)