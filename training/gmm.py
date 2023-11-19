'''  This file contains code to train a GMM on the data, in order to learn the distribution of the data.  '''

from sklearn.mixture import GaussianMixture

# Prepare your data
X = test_event_df[['x', 'y']].values

# Fit a Gaussian Mixture Model to your data
gmm = GaussianMixture(n_components=10)
gmm.fit(X)

# Now you can use the fitted model to learn the distribution of the data