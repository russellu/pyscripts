import numpy as np
import nibabel as nib 
from scipy import io 
import matplotlib.pyplot as plt 

# get the FMRI dataset
fmri = nib.load('/media/sf_shared/pyfmri/fmri.nii.gz') 
# get the ndarray from the data
fmri_data = fmri.get_data() 
 # load the stimulus
stim_times = io.loadmat('/media/sf_shared/pyfmri/conved.mat')['conved']
# tile the stimulus (to make it same size as FMRI)
tiled_times = np.tile(stim_times,[66,67,45,1]) 

# subtract mean from FMRI data
meansub_data = fmri_data - \
 np.tile(np.expand_dims(np.mean(fmri_data,axis=3),axis=3),[1,1,1,245])
 
# subtract mean from tiled stimulus data
meansub_times = tiled_times - \
np.tile(np.expand_dims(np.mean(tiled_times,axis=3),axis=3),[1,1,1,245])

# compute pearson's correlation in each voxel
r = np.sum(meansub_data*meansub_times,axis=3)                 \
    / np.sqrt((np.sum(meansub_data*meansub_data,axis=3)       \
               *np.sum(meansub_times*meansub_times,axis=3)))

# save the correlation dataset
img = nib.Nifti1Image(r, fmri.affine)
nib.save(img,'/media/sf_shared/pyfmri/r_brain.nii.gz')


