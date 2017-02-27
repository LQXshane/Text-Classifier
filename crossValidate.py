import pandas as pd
from sklearn import metrics, svm
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import KFold, GridSearchCV, train_test_split
from time import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print(__doc__)

almost_black = '#262626'
palette = sns.color_palette()

file = "../categories/GT/border_2/trump/trump_cleaned.csv"
raw = pd.read_csv(file)
print(raw.head())
X = raw['contents']
y = np.array(raw['label'], dtype='int64')

precision = np.zeros(10)
recall = np.zeros(10)
CCR = np.zeros(10)

positives = np.zeros(10)

for i in range(10):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=i)

    piper = Pipeline([
        ('vect', CountVectorizer(min_df=0.005)),
        ('tfidf', TfidfTransformer(use_idf=True, norm='l2')),
        ('clf', svm.SVC(kernel='linear', class_weight='balanced', C=0.6, random_state=i)),
    ])

    clf = piper.fit(X_train, y_train)

    predicted = clf.predict(X_test)
    precision[i] = metrics.precision_score(y_test, predicted)
    recall[i] = metrics.recall_score(y_test, predicted)
    # print("F1_score: %0.2f" % (metrics.f1_score(y_test, predicted)))

    # print(metrics.classification_report(y_test, predicted))

    cmat = metrics.confusion_matrix(y_test, predicted)
    print(cmat)

    CCR[i] = (float(cmat[0][0] + cmat[1][1]))  / sum(sum(cmat))
    positives[i] = float((cmat[0][1] + cmat[1][1]))  / sum(sum(cmat))



print("Precision ", precision)
print("Recall ", recall)
print("CCR ", CCR)
print("Predicted positives", positives)

plt.plot(range(10), precision, '--', label = "precision", color = palette[0])
plt.plot(range(10), recall, 's', label = "recall", color = palette[1])
plt.plot(range(10), CCR, '^', label="CCR", color = palette[2])
plt.title("Performance over different test sets", color = palette[3])
plt.legend(loc=4)
plt.show()