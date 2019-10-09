import matplotlib.pyplot as plt 
from nilearn.decomposition import CanICA, DictLearning
import nibabel as nib 
import numpy as np 
from scipy.cluster.vq import vq, kmeans, whiten
import scipy as scipy 

subs = ['sub_amal','sub_angelina','sub_cesar','sub_esteban','sub_felix','sub_francis',
        'sub_greg','sub_lyes','sub_lyndis','sub_pascal','sub_raphael','sub_reihaneh',
        'sub_russell','sub_samuel','sub_valerie'];

#path = '/media/sf_E_DRIVE/fmris/badger_russell/';
path = '/media/sf_E_DRIVE/orientation_retinotopy/sub_lyndis/';
nifti = nib.load(path + 'topup_mc_orientation_1.nii.gz')
niftidata = nifti.get_data()

mask = np.double(niftidata[:,:,:,0] > np.mean(niftidata))
nmask = nib.Nifti1Image(mask,nifti.affine);

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
    
    

goodinds = [72,68,57,56,51,47,43,28,22,20,8,6,3,1]; 

indrange = np.zeros([75])
indrange[goodinds] = 1
allbades = np.where(indrange==0)

freshcomps = dlearn.transform([nifti])
freshcomps[0][:,allbades] = 0

inv = dlearn.inverse_transform(freshcomps)

invdata = inv[0].get_data()
invdata = invdata - np.mean(invdata,axis=3,keepdims=True)
new_nifti = nib.Nifti1Image(invdata,nifti.affine)

nib.save(new_nifti,path+'inv_orientation.nii.gz')    
    










