import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl

import pandas as pd
import scipy.stats as ss
from scipy.stats import dirichlet
from sklearn.model_selection import train_test_split
import seaborn as sns
#import plotly
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import lime
from lime import lime_tabular
import external_exp_metric_function as eemf
from sklearn.metrics import balanced_accuracy_score, precision_score, f1_score, recall_score
import shap

with open('practice_data_set.pkl', 'rb') as handle:
    data = pkl.load(handle)

# fig_1, ax_1 = plt.subplots()
# ax_1 = sns.scatterplot(x=data[:, 0],  y=data[:, 1],  hue=data[:, 2], size=1)
# ax_1.axis('equal')
# plt.legend([],[], frameon=False)
# plt.show()

train_set, test_set = train_test_split(data, test_size=0.3, train_size=0.7)
classifier = DecisionTreeClassifier()
classifier.fit(train_set[:, 0:2], train_set[:, 2])

explainer = shap.Explainer(classifier.predict, test_set[:, 0:2])
shap_values = explainer(test_set[:, 0:2])

#shap.plots.bar(shap_values)
shap.plots.beeswarm(shap_values)
shap.plots.waterfall(shap_values[12])
shap.summary_plot(shap_values, plot_type='violin')