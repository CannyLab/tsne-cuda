from keras.datasets import mnist
from tsnecuda import TSNE
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(y_train.shape)
print(x_train.shape)

tsne = TSNE(n_iter=1000, verbose=1, num_neighbors=64)
tsne_results = tsne.fit_transform(x_train.reshape(60000,-1))


print(tsne_results.shape)

# Create the figure
fig = plt.figure( figsize=(8,8) )
ax = fig.add_subplot(1, 1, 1, title='TSNE' )

# Create the scatter
ax.scatter(
    x=tsne_results[:,0],
    y=tsne_results[:,1],
    c=y_train,
    cmap=plt.cm.get_cmap('Paired'),
    alpha=0.4,
    s=0.5)
plt.show()
