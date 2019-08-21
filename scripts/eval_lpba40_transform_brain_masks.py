
import sys
import numpy as np
import script_tools as st

mri_path = "/home/johof680/work/LPBA40_for_ANTs/mri/"
labels_path = "/home/johof680/work/LPBA40_for_ANTs/labels/"
masks_path = "/home/johof680/work/LPBA40_for_ANTs/masks/"
atlas_path = "/home/johof680/work/LPBA40_for_ANTs/templates/MI/"

#in_path = "/home/johof680/work/VironovaRegistrationImages/1Original/"
out_path = "/home/johof680/work/LPBA40_for_ANTs/eval/ALPHA_SMD_4/"
exe_path = "/home/johof680/work/itkAlphaAMD-build2/ACTransform"

# NOTE: CHANGE THE FOLLOWING TO RUN THIS FOR EACH OF THE 2 PARTS OF THE CROSS VALIDATION
if(len(sys.argv) < 2):
  task = 0
else:
  task = np.int(sys.argv[1])

tasks = [("21to40", 1, 8), ("21to40", 8, 16), ("21to40", 16, 21), ("01to20", 21, 28), ("01to20", 28, 36), ("01to20", 36, 41)]
(subset, start_index, end_index) = tasks[task]

#if first_half:
#  subset = "21to40"
#  start_index = 1
#  end_index = 21
#else:
#  subset = "01to20"
#  start_index = 21
#  end_index = 41

param = {}

param["interpolation"] = "nearest"
param["spacing_mode"] = "default"
param["label_mode"] = "binary"
param["landmark_mode"] = "0"
param["16bit"] = "1"
param["bg"] = "0"
param["dim"] = "3"

def ind_str(prefix, index, postfix):
  return prefix + ("%.2d" % index) + postfix

def make_dir(path, index):
  return st.makedir_string(ind_str(path + "S", index, "/"))

def make_transform(index, atlas_id, params):
  # atlas image
  #label_atlas_str = atlas_path + "MI" + atlas_id + "labels.nii.gz"#ind_str("S", index, ".labels.nii.gz")
  mask_atlas_str = atlas_path + "MI" + atlas_id + "mask.nii.gz"
  #mri_atlas_str = atlas_path + "MI" + atlas_id + "template.nii.gz"

  # ref image
  #label_ref_str = labels_path + ind_str("S", index, ".labels.nii.gz")
  mask_ref_str = masks_path + ind_str("S", index, ".brainmask.nii.gz")
  #mri_ref_str = mri_path + ind_str("S", index, ".nii.gz")

  out = out_path + ind_str("S", index, "/")
  out_image = out + "mask.nii.gz"
  transform_path = out + "transform_complete.txt"
  overlap_out = out_path + ind_str("S", index, "/") + "brain_mask_overlap.csv"
  #" -mask1 " + mask_ref_str + " -mask2 " + mask_atlas_str +

  return exe_path + " -ref " + mask_ref_str + " -in " + mask_atlas_str + " -out " + out_image + " -label_overlap_out " + overlap_out + " -transform " + transform_path + st.param_dict_to_string(params)

def make_cmd(index, params):
  print("# Image %.3d" % (index))
  print("echo \"Image %.3d\"" % (index))
  print(make_dir(out_path, index))
  print(make_transform(index, subset, params))
  print("")

for i in xrange(start_index, end_index):
  make_cmd(i, param)
  #make_cmd(i, in_path + "im008.tif", in_path + ind_str("im", i, ".tif"), out_path + ind_str("Out", i, "/"), param)

