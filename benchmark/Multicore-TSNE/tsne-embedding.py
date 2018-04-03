from MulticoreTSNE import MulticoreTSNE as TSNE
import skimage.io
import argparse
import glob
from scipy.spatial import KDTree
import numpy as np


def imscatter(images, positions):
    '''
        Creates a scatter plot, where each plot is shown by corresponding image
    '''
    positions = np.array(positions)

    bottoms = positions[:, 1] - np.array([im.shape[1] / 2.0 for im in images])
    tops = bottoms + np.array([im.shape[1] for im in images])

    lefts = positions[:, 0] - np.array([im.shape[0] / 2.0 for im in images])
    rigths = lefts + np.array([im.shape[0] for im in images])

    most_bottom = int(np.floor(bottoms.min()))
    most_top = int(np.ceil(tops.max()))

    most_left = int(np.floor(lefts.min()))
    most_right = int(np.ceil(rigths.max()))

    scatter_image = np.zeros(
        [most_right - most_left, most_top - most_bottom, 3], dtype=imgs[0].dtype)

    # shift, now all from zero
    positions -= [most_left, most_bottom]

    for im, pos in zip(images, positions):

        xl = int(pos[0] - im.shape[0] / 2)
        xr = xl + im.shape[0]

        yb = int(pos[1] - im.shape[1] / 2)
        yt = yb + im.shape[1]

        scatter_image[xl:xr, yb:yt, :] = im
    return scatter_image


if __name__ == '__main__':
    '''
    Takes a set of images and returns their T-SNE embedding
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_glob")
    parser.add_argument("--out_path", default='embedding.png')
    args = parser.parse_args()

    files = glob.glob(args.in_glob)

    print('Reading images')
    feats, imgs = [], []
    for f in files:
        im = skimage.io.imread(f)

        feats.append(im.ravel())

        if im.ndim == 2:
            im = im[:, :, None]
        imgs.append(im)

    feats = np.vstack(feats).astype(np.float64)

    print('Running T-SNE')
    tsne = TSNE(n_jobs=1)
    embedding = tsne.fit_transform(feats)

    # Find an appropriate scaling, so that the images not overlap much
    kdt = KDTree(embedding)
    dists = kdt.query(embedding, k=2)[0][:, 1]
    c = (imgs[0].shape[0] + imgs[0].shape[1]) / 2 / np.percentile(dists, 30)

    print('Creating an image scatter')
    img = imscatter(imgs, embedding * c)

    print('Saving result')
    skimage.io.imsave(args.out_path, img)
