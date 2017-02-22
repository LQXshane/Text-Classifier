from trainer import tuner
import pandas as pd
import sys
from sklearn import metrics
from sklearn.model_selection import train_test_split
import numpy as np
from IPython import embed

if '__main__':
    raw = pd.read_csv(sys.argv[1])

    print(raw.head())


    X = raw['contents']
    y = np.array(raw['label'], dtype='int64')

    parameters = {
        # 'vect__min_df': (0.005) ,    #(0.003, 0.007, 0.005,0.001),
        # 'vect__max_features': (None, 5000, 10000, 50000),
        # 'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
        # 'tfidf__use_idf': (True, False),
        'tfidf__norm': ('l1', 'l2'),
        'clf__C': (0.5, 0.4, 0.3, 0.2),
        'clf__kernel': ('linear', 'rbf') ,# poly #'sigmoid'),#, 'precomputed'),
        'clf__class_weight': (None, 'balanced'),
        'clf__tol': (1e-3, 1e-4, 1e-5)
        # 'clf__probability':(False, True),
    }


    svm_models, eval_scores = tuner(X, y, 8, parameters, 3)

    print ("Using precision as CV evaluation: ", eval_scores)

    # embed()

    bestie = svm_models.best_estimator_

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    clf = bestie.fit(X_train, y_train)

    predicted = clf.predict(X_test)

    # embed()

    print(metrics.classification_report(y_test, predicted))

    cmat = metrics.confusion_matrix(y_test, predicted)
    print(cmat)

    print("CCR %0.2f"%((float(cmat[0][0]+cmat[1][1]))* 100/sum(sum(cmat))))
    print("Percentage of predicted positives %0.2f" % (float((cmat[0][1] + cmat[1][1])) * 100 / sum(sum(cmat))))
    print(svm_models.best_params_)