from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
import seaborn as sns
# from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
import pandas as pd
import sys
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from IPython import embed
from imblearn.over_sampling import SMOTE

# print(__doc__)
#
# # Define some color for the plotting
# almost_black = '#262626'
# palette = sns.color_palette()
#
#
# # # Generate the dataset
# # X, y = make_classification(n_classes=2, class_sep=2, weights=[0.1, 0.9],
# #                            n_informative=3, n_redundant=1, flip_y=0,
# #                            n_features=20, n_clusters_per_class=1,
# #                            n_samples=5000, random_state=10)
#
# raw = pd.read_csv(sys.argv[1])
#
# print(raw.head())
#
# raw_tweets = raw['contents']
# y = np.array(raw['label'], dtype='int64')
# vectorizer = TfidfVectorizer(min_df = 0.005, norm='l2')
#
# X = vectorizer.fit_transform(raw_tweets).toarray()
# embed()
#
#
# # Instanciate a PCA object for the sake of easy visualisation
# pca = PCA(n_components=2)
# # Fit and transform x to visualise inside a 2D feature space
# X_vis = pca.fit_transform(X)
#
# # Apply the random over-sampling
# ros = RandomOverSampler()
# X_resampled, y_resampled = ros.fit_sample(X, y)
# X_res_vis = pca.transform(X_resampled)
# #
# # # Apply SMOTE
# # sm = SMOTE(kind='svm')
# # X_resampled, y_resampled = sm.fit_sample(X, y)
# # X_res_vis = pca.transform(X_resampled)
#
# # Two subplots, unpack the axes array immediately
# f, (ax1, ax2) = plt.subplots(1, 2)
#
# ax1.scatter(X_vis[y == 0, 0], X_vis[y == 0, 1], label="Class #0", alpha=0.5,
#             edgecolor=almost_black, facecolor=palette[0], linewidth=0.15)
# ax1.scatter(X_vis[y == 1, 0], X_vis[y == 1, 1], label="Class #1", alpha=0.5,
#             edgecolor=almost_black, facecolor=palette[2], linewidth=0.15)
# ax1.set_title('Original set')
#
# ax2.scatter(X_res_vis[y_resampled == 0, 0], X_res_vis[y_resampled == 0, 1],
#             label="Class #0", alpha=.5, edgecolor=almost_black,
#             facecolor=palette[0], linewidth=0.15)
# ax2.scatter(X_res_vis[y_resampled == 1, 0], X_res_vis[y_resampled == 1, 1],
#             label="Class #1", alpha=.5, edgecolor=almost_black,
#             facecolor=palette[2], linewidth=0.15)
# ax2.set_title('Random over-sampling')
#
# plt.show()


# def ROS(X, y):
# # Apply the random over-sampling
#     ros = RandomOverSampler()
#     X_resampled, y_resampled = ros.fit_sample(X, y)
#
#     return X_resampled, y_resampled
#
# def SMOTE(X,y):
# # Apply SMOTE
#     sm = SMOTE(kind='svm')
#     X_resampled, y_resampled = sm.fit_sample(X, y)
#
#     return X_resampled, y_resampled

