
import script_tools as st
import registration as reg
import os

if os.name == "nt":
  # Windows
  base_path = "C:/dev/data/LPBA40_for_ANTs/"
  exe_path = "C:/cygwin64//home/johan/itkAlphaCut-release/Release/"
else:
  # Linux
  base_path = "/home/johof680/work/LPBA40_for_ANTs/"
  exe_path = "/home/johof680/work/itkAlphaCut-4j/build-release/"

mri_path = base_path + "mri/"
labels_path = base_path + "labels/"
masks_path = base_path + "masks/"
atlas_path = base_path + "templates/MI/"

ENABLE_PYRAMIDS = False

if ENABLE_PYRAMIDS:
    out_path = base_path + "eval/ALPHA_SMD_PYRAMIDS_2/"
else:
    out_path = base_path + "eval/ALPHA_SMD_2/"

cnt = 20

r = reg.Registration(2, exe_path, dim = 3, enable_logging = True)

param = r.get_register_param_defaults()

#param.pop("weights1", None)
#param.pop("weights2", None)
param["weights1"] = "hann"
param["weights2"] = "hann"
param["learning_rate"] = "0.5"
param["relaxation_factor"] = "0.99"
param["alpha_levels"] = "7"
param["alpha_max_distance"] = "128"
param["seed"] = "1000"
param["metric"] = "alpha_smd"
param["iterations"] = "3000"
param["sampling_fraction"] = "0.05"
param["spacing_mode"] = "default"
param["normalization"] = "0.05"
param["alpha_outlier_rejection"] = "0.0"
if ENABLE_PYRAMIDS:
  param["multiscale_sampling_factors"] = "4x2x1"
  param["multiscale_smoothing_sigmas"] = "4.0x2.0x0.0"
else:
  param["multiscale_sampling_factors"] = "1"
  param["multiscale_smoothing_sigmas"] = "0.0"
param["histogram_matching"] = "0"
param["3d"] = "1"

def ind_str(prefix, index, postfix):
  return prefix + ("%.2d" % index) + postfix

RUN_REGISTRATIONS = True
TRANSFORM_LABELS = True
COMPUTE_OVERLAP = True

if RUN_REGISTRATIONS:
  for i in xrange(0, cnt):
    atlas_metric = "MI"
    atlas_id = "21to40"
    template = atlas_path + atlas_metric + atlas_id + "template.nii.gz"
    ref = mri_path + ind_str("S", i+1, ".nii.gz")

    r.register_affine(ref, template, out_path + ind_str("Out", i+1, "/"), param)

  for i in xrange(20, 20+cnt):
    atlas_metric = "MI"
    atlas_id = "01to20"
    template = atlas_path + atlas_metric + atlas_id + "template.nii.gz"
    ref = mri_path + ind_str("S", i+1, ".nii.gz")

    r.register_affine(ref, template, out_path + ind_str("Out", i+1, "/"), param)

  r.run("Affine registration")

# Transform 

if TRANSFORM_LABELS:
  for i in xrange(0, cnt):
    atlas_metric = "MI"
    atlas_id = "21to40"
    template = atlas_path + atlas_metric + atlas_id + "labels.nii.gz"
    ref = labels_path + ind_str("S", i+1, ".labels.nii.gz")
    out_path_1 = out_path + ind_str("Out", i+1, "/")

    param = r.get_transform_param_defaults()
    param["interpolation"] = "nearest"
    param["16bit"] = "1"
    param["divide_factor"] = "65535"

    r.transform(ref, template, out_path_1, "transformed_labels.nii.gz", out_path_1 + "/transform_complete.txt", param)

  for i in xrange(20, 20+cnt):
    atlas_metric = "MI"
    atlas_id = "01to20"
    template = atlas_path + atlas_metric + atlas_id + "labels.nii.gz"
    ref = labels_path + ind_str("S", i+1, ".labels.nii.gz")
    out_path_1 = out_path + ind_str("Out", i+1, "/")

    param = r.get_transform_param_defaults()
    param["interpolation"] = "nearest"
    param["16bit"] = "1"
    param["divide_factor"] = "65535"

    r.transform(ref, template, out_path_1, "transformed_labels.nii.gz", out_path_1 + "/transform_complete.txt", param)
  
  for i in xrange(0, cnt):
    atlas_metric = "MI"
    atlas_id = "21to40"
    template = atlas_path + atlas_metric + atlas_id + "mask.nii.gz"
    ref = masks_path + ind_str("S", i+1, ".brainmask.nii.gz")
    out_path_1 = out_path + ind_str("Out", i+1, "/")

    param = r.get_transform_param_defaults()
    param["interpolation"] = "nearest"
    param["16bit"] = "0"

    r.transform(ref, template, out_path_1, "transformed_brainmask.nii.gz", out_path_1 + "/transform_complete.txt", param)

  for i in xrange(20, 20+cnt):
    atlas_metric = "MI"
    atlas_id = "01to20"
    template = atlas_path + atlas_metric + atlas_id + "mask.nii.gz"
    ref = masks_path + ind_str("S", i+1, ".brainmask.nii.gz")
    out_path_1 = out_path + ind_str("Out", i+1, "/")

    param = r.get_transform_param_defaults()
    param["interpolation"] = "nearest"
    param["16bit"] = "0"

    r.transform(ref, template, out_path_1, "transformed_brainmask.nii.gz", out_path_1 + "/transform_complete.txt", param)

  r.run("Transform labels")

if COMPUTE_OVERLAP:
  
  for i in xrange(0, cnt):
      param = r.get_label_overlap_param_defaults()
      param["label_mode"] = "multi"
      #param["max_label"] = "182"

      atlas_metric = "MI"
      out_path_1 = out_path + ind_str("Out", i+1, "/")
      
      ref = labels_path + ind_str("S", i+1, ".labels.nii.gz")
      out_path_1 = out_path + ind_str("Out", i+1, "/")
      
      r.label_overlap(ref, out_path_1 + "transformed_labels.nii.gz", out_path_1, "label_overlap", param)

  for i in xrange(20, 20+cnt):
      param = r.get_label_overlap_param_defaults()
      param["label_mode"] = "multi"
      #param["max_label"] = "182"

      atlas_metric = "MI"
      out_path_1 = out_path + ind_str("Out", i+1, "/")
      
      ref = labels_path + ind_str("S", i+1, ".labels.nii.gz")
      out_path_1 = out_path + ind_str("Out", i+1, "/")
      
      r.label_overlap(ref, out_path_1 + "transformed_labels.nii.gz", out_path_1, "label_overlap", param)

  for i in xrange(0, cnt):
      param = r.get_label_overlap_param_defaults()
      param["label_mode"] = "binary"

      atlas_metric = "MI"
      out_path_1 = out_path + ind_str("Out", i+1, "/")
      
      ref = masks_path + ind_str("S", i+1, ".brainmask.nii.gz")
      out_path_1 = out_path + ind_str("Out", i+1, "/")

      r.label_overlap(ref, out_path_1 + "transformed_brainmask.nii.gz", out_path_1, "brainmask_overlap", param)

  for i in xrange(20, 20+cnt):
      param = r.get_label_overlap_param_defaults()
      param["label_mode"] = "binary"

      atlas_metric = "MI"
      out_path_1 = out_path + ind_str("Out", i+1, "/")
      
      ref = masks_path + ind_str("S", i+1, ".brainmask.nii.gz")
      out_path_1 = out_path + ind_str("Out", i+1, "/")

      r.label_overlap(ref, out_path_1 + "transformed_brainmask.nii.gz", out_path_1, "brainmask_overlap", param)

  r.run("Compute label overlap")

#transformation

#par = r.get_transform_param_defaults()
#par["16bit"] = "1"

#for i in xrange(9, N):
#  r.transform(in_path + "im008.tif", in_path + ind_str("im", i, ".tif"), dataset_path + ind_str("Out", i, "/"), "transformed.png", dataset_path + ind_str("Out", i, "/transform_complete.txt"), par)

#r.run("Transformations")
  #make_cmd(i, in_path + "im008.tif", in_path + ind_str("im", i, ".tif"), out_path + ind_str("Out", i, "/"), param)

#mkdir ../../../VironovaRegistrationImages/Out009/
#../../build6/ACRegister -in1 ../../../VironovaRegistrationImages/1Original/im008.tif -in2 ../../../VironovaRegistrationImages/1Original/im009.tif -weights1 ../../../VironovaRegistrationImages/hann_mask.tif -weights2 ../../../VironovaRegistrationImages/hann_mask.tif -out ../../../VironovaRegistrationImages/Out009/ -learning_rate 0.5 -relaxation_factor 0.99 -alpha_levels 7 -alpha_max_distance 128 -seed 1000 -metric alpha_smd -iterations 1000 -sampling_fraction 0.01 -spacing_mode remove -normalization 0.005 -rigid 1 -smoothing 2.0
