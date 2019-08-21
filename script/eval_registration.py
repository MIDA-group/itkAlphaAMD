
import numpy as np
import scipy as sp
import scipy.ndimage.morphology
import matplotlib.pyplot as plt

folder = '../../itkAlphaAMD-deform-liver-build/'

#before_files = ['Merle_maskapplied_pre.raw', 'Merle_maskapplied_p1.raw', 'Merle_late.raw']
before_files = ['Merle_pre.raw', 'Merle_p1.raw', 'Merle_late.raw']
after_files = ['pre_registered.nii.gz', 'p1_registered.nii.gz']
mask_file = 'masque_16_12_fixed.raw'

def load_raw_file(path, dtype):
    return np.fromfile(path, dtype=dtype, count=-1)

#shp = [260, 320, 80]
shp = [80, 320, 260]
spacing = [1.25, 1.25, 2.5]

pre_before = load_raw_file(folder + before_files[0], dtype='uint16').reshape(shp)
p1_before = load_raw_file(folder + before_files[1], dtype='uint16').reshape(shp)
late_before = load_raw_file(folder + before_files[2], dtype='uint16').reshape(shp)

pre_before = pre_before.astype('float64')
p1_before = p1_before.astype('float64')
late_before = late_before.astype('float64')

mask = load_raw_file(folder + mask_file, dtype='uint8').reshape(shp)
mask = (mask > 0)
print(np.sum(mask))
#plt.imshow(mask[50, :, :])
#plt.show()

def eval(im, msk, edge=False):
    if edge==True:
        SE = np.ones([3,3,3])
        msk_edge = np.logical_xor(msk, scipy.ndimage.morphology.binary_erosion(msk, structure=SE))
        # scipy.ndimage.morphology.morphological_gradient(msk, structure=SE)
        msk_inds = np.nonzero(msk_edge)    
    else:
        msk_inds = np.nonzero(msk)

    print(str(msk_inds[0].size))
    im_masked = im[msk_inds]

    return (np.mean(im_masked[:]), np.std(im_masked[:]))

#pre_before_masked = pre_before[mask_inds]
#p1_before_masked = p1_before[mask_inds]
#late_before_masked = late_before[mask_inds]

#means = [np.mean(pre_before_masked[:]), np.mean(p1_before_masked[:]), np.mean(late_before_masked[:])]
#ddof = 1
#std = [np.std(pre_before_masked[:], ddof=ddof), np.std(p1_before_masked[:], ddof=ddof), np.std(late_before_masked[:], ddof=ddof)]

do_edge = False
stats = [eval(pre_before, mask, do_edge), eval(p1_before, mask, do_edge), eval(late_before, mask, do_edge)]

print(str(stats))






