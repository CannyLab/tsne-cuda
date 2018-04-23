# Generate a uniform random distributuion of points
import numpy as np
import scipy
from sklearn.decomposition import PCA
import pyctsne


print('Loading data...')
import tflearn
(X,Y),_ = tflearn.datasets.cifar10.load_data()


# randomly sample some points
r_points = X[np.random.choice(np.arange(0,X.shape[0]), size=20000,replace=False)].reshape(20000,-1)
# r_points = np.random.rand(5000,768)
# print(X.shape)
# r_points = X.reshape(X.shape[0],-1)

print('Computing distances...')
# Compute the pairwise distances between the points
# hd_distances = scipy.spatial.distance.pdist(r_points)
hd_distances = pyctsne.TSNE.e_pw_dist(r_points)

print('Projecting...')
# Project the points using PCA
proj_points = PCA(n_components=30).fit_transform(r_points)

print('Computing LD distances...')
# Compute the pairwise distances between the points
ld_distances = pyctsne.TSNE.e_pw_dist(proj_points) + 1e-4*np.ones(shape=(20000, 20000))

print('Computing ratios...')
# Compute for each pair of points the ratio between the point distances
point_ratios = hd_distances / ld_distances

print(np.percentile(point_ratios.reshape(-1), 5))
print(np.percentile(point_ratios.reshape(-1), 25))
print(np.percentile(point_ratios.reshape(-1), 50))
print(np.percentile(point_ratios.reshape(-1), 75))
print(np.percentile(point_ratios.reshape(-1), 95))

# Display the histogram
import matplotlib
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
# the histogram of the data
ax.hist(point_ratios.reshape(-1), bins=np.arange(0,5,0.001))

# add a 'best fit' line
ax.set_xlabel('Ratio')
ax.set_ylabel('Density')
ax.set_title('Histogram of Ratios between low and high dim distances under PCA')
# Tweak spacing to prevent clipping of ylabel
fig.tight_layout()
plt.show()

