import numpy as np
import matplotlib.pyplot as plt 
import imageio as imageio
from sklearn.decomposition import PCA


mars_bar = imageio.imread('/media/sf_shared/marsbar.png')
mean_bar = np.mean(mars_bar,axis=2)
comps=1
pca = PCA(n_components=comps)
pca.fit(mean_bar)

transformed = pca.inverse_transform(pca.transform(mean_bar))

plt.subplot(1,2,1);plt.imshow(transformed); plt.title('components='+str(comps))
plt.subplot(1,2,2);plt.imshow(mean_bar); plt.title('original')

plt.plot(transformed.flatten()[0::100],mean_bar.flatten()[0::100],'.');
plt.xlabel('compressed'); plt.ylabel('original')





