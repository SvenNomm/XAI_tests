from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB


def apply_classifier(data_set,)
train_set, test_set = train_test_split(data_set_uf_sm_l, test_size=0.3, train_size=0.7)
classifier = DecisionTreeClassifier()
classifier.fit(train_set[:, 0:2], train_set[:, 2])