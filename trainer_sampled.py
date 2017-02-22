import pandas as pd
from sklearn import metrics, svm
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import KFold, GridSearchCV, train_test_split
from time import time
import numpy as np
from IPython import embed
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import RandomOverSampler, SMOTE




def tuner(X, y, NUM_TRIALS, param, K):
    '''

    :param param: dict containing parameters to tune
    :param k: KFold Cross-Validation
    :return:
    '''




    piper = Pipeline([
        ('clf', svm.SVC())
    ])
    parameters = param
    scores = np.zeros(NUM_TRIALS)
    CCR = np.zeros(NUM_TRIALS)
    t0 = time()
    for i in range(NUM_TRIALS):

        print(i)

        inner_cv = KFold(n_splits = K, shuffle=True, random_state=i)

        model = GridSearchCV(piper, param_grid=parameters, cv=inner_cv, n_jobs=-1, scoring='f1') # accuracy no good, TPR/recall no good
        model.fit(X, y)

        scores[i] = model.best_score_

    print("done in %0.3fs" % (time() - t0))

    return (model, scores)


raw = pd.read_csv('../categories/GT/no_border/trump/trump_cleaned.csv')

print(raw.head())


X_raw = raw['contents']
y_raw = np.array(raw['label'], dtype='int64')

vect = TfidfVectorizer(min_df=0.005)
# ros = RandomOverSampler()
sm = SMOTE(kind='svm')

X = vect.fit_transform(X_raw).toarray()
X, y = sm.fit_sample(X, y_raw)

embed()

parameters = {
    # 'vect__min_df': (0.005) ,    #(0.003, 0.007, 0.005,0.001),
    # 'vect__max_features': (None, 5000, 10000, 50000),
    # 'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
    # 'tfidf__use_idf': (True, False),
    # 'vect__norm': ('l1', 'l2'),
    'clf__C': (0.6, 0.55, 0.5, 0.4, 0.3),
    'clf__kernel': ('linear', 'rbf', 'poly' ),#'sigmoid'),#, 'precomputed'),
    'clf__class_weight': (None, 'balanced'),
    'clf__tol': (1e-3, 1e-4, 1e-5)
    # 'clf__probability':(False, True),
}


svm_models, eval_scores = tuner(X, y, 7, parameters, 5)

print ("Using F1 as CV evaluation: ", eval_scores)

# embed()

bestie = svm_models.best_estimator_

del X,y


'''
careful with the sampler thing don't get it in test set
'''
X_raw = raw['contents']
y = np.array(raw['label'], dtype='int64')

vect = TfidfVectorizer(min_df=0.005)
X = vect.fit_transform(X_raw).toarray()


X_train_raw, X_test, y_train_raw, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

X_train, y_train = sm.fit_sample(X_train_raw, y_train_raw)

clf = bestie.fit(X_train, y_train)

predicted = clf.predict(X_test)

# embed()

print(metrics.classification_report(y_test, predicted))

cmat = metrics.confusion_matrix(y_test, predicted)
print(cmat)

print("CCR %0.2f"%((cmat[0][0]+cmat[1][1])* 100/sum(sum(cmat))))

print(svm_models.best_params_)

embed()