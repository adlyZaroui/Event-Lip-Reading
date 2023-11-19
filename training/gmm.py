'''  This file contains code to train a GMM on the data, in order to learn the distribution of the data.  '''

from sklearn.mixture import GaussianMixture
from utils import load_data

# Load the training and test data
x_train, x_test, y_train = load_data()
del x_test

# Fit a Gaussian Mixture Model to your data
gmm = GaussianMixture(n_components=10)
gmm.fit(x_train)

# Now you can use the fitted model to learn the distribution of the data