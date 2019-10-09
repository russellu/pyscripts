import numpy as np
from nilearn.decomposition import DictLearning
from nilearn.decomposition import CanICA
import nibabel as nib
import matplotlib.pyplot as plt
from scipy import signal
from scipy.cluster.vq import vq, kmeans, whiten
import scipy as scipy

path = '/media/sf_shared/badger_subjects/badger_valerie/'

nifti = nib.load(path+'reg_topup_mc_retino_gamma_01.nii.gz')
niftidata = nifti.get_data()

mask = np.double(niftidata[:,:,:,0] > np.mean(niftidata))
nmask = nib.Nifti1Image(mask,nifti.affine);


#dlearn = DictLearning(n_components=75,memory="nilearn_cache",memory_level=2,verbose=1,alpha=0.1,
                            # random_state=0, n_epochs=1,mask_strategy="epi",reduction_ratio=1)
#dlearn.fit(nifti)

dlearn = CanICA(n_components=75,smoothing_fwhm=3,threshold=None,standardize=True,
                mask=nmask)
dlearn.fit(nifti);

comps = dlearn.transform([nifti])
compsimg = dlearn.components_img_.get_data()
f = np.abs(np.fft.fft(comps[0],axis=0))

for i in np.arange(0,75):
    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(scipy.ndimage.rotate(np.max(compsimg[:,:,:,i],axis=0),90))
    plt.subplot(1,3,2)
    plt.imshow(scipy.ndimage.rotate(np.max(compsimg[:,:,:,i],axis=1),90))
    plt.subplot(1,3,3)
    plt.imshow(scipy.ndimage.rotate(np.max(compsimg[:,:,:,i],axis=2),270))
    plt.title(i)
    plt.show()

#ps = np.abs(np.fft.fft(comps[0][:,:],axis=0))**2 
#f, pxx = signal.welch(np.transpose(comps[0]), 1/0.693, nperseg=100)
#f = np.abs(np.fft.fft(comps[0],axis=0))
#whitened = whiten(pxx)
#dist, k = kmeans(np.log(np.transpose(f[:,:])),4)
#idx,_ = vq(np.log(np.transpose(f[:,:])),dist)

goodinds = [74,72,71,66]; 

indrange = np.zeros([75])
indrange[goodinds] = 1
allbades = np.where(indrange==0)

freshcomps = dlearn.transform([nifti])
freshcomps[0][:,allbades] = 0

inv = dlearn.inverse_transform(freshcomps)
nib.save(inv[0],'inv_gamma_01.nii.gz')






#canica = CanICA(n_components=35,smoothing_fwhm=0,threshold=None)
#canica.fit(nifti);

#dlearn.fit(nifti); 
imgs = [nifti]
#comps = [canica.components_.T]
#mask = canica.mask_img_.get_data()
#resmask = np.reshape(mask,[np.product(mask.shape)])
#maskinds = np.where(resmask==1)
#res_nii = np.reshape(niftidata,[np.product(niftidata[:,:,:,0].shape),niftidata.shape[3]])

#mask_vals = res_nii[maskinds,:]

#inved = canica.inverse_transform(comps);



a = canica.transform([nifti])
compdat = a[0]



# violate a
#a[0][:,:] = 0; 
inved = canica.inverse_transform(a);

#nib.save(inved[0],"inved.nii.gz")
#nib.save(canica.components_img_,"canica_components.nii.gz")


"""
compdata = dlearn.components_img_.get_data()
compdata = compdata[:,:,:,np.where(idx==0)]
new_img = nib.Nifti1Image(compdata,dlearn.components_img_.affine);
nib.save(new_img,'catall_dlearn_0.nii.gz')

compdata = dlearn.components_img_.get_data()
compdata = compdata[:,:,:,np.where(idx==1)]
new_img = nib.Nifti1Image(compdata,dlearn.components_img_.affine);
nib.save(new_img,'catall_dlearn_1.nii.gz')

compdata = dlearn.components_img_.get_data()
compdata = compdata[:,:,:,np.where(idx==2)]
new_img = nib.Nifti1Image(compdata,dlearn.components_img_.affine);
nib.save(new_img,'catall_dlearn_2.nii.gz')

compdata = dlearn.components_img_.get_data()
compdata = compdata[:,:,:,np.where(idx==3)]
new_img = nib.Nifti1Image(compdata,dlearn.components_img_.affine);
nib.save(new_img,'catall_dlearn_3.nii.gz')
"""




