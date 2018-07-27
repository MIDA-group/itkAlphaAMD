
import script_tools as st
import registration as reg
import numpy as np
import landmarks
import dist_table as dt

# Windows
#dataset_path = "C:/cygwin64/home/johan/itkAlphaCut/assets/01.png"
#bin_path = "C:/cygwin64/home/johan/itkAlphaCut-release/Release/"

# Linux
dataset_path = "/home/johof680/work/itkAlphaAMD/images/cilia_public_cut_out.png"
bin_path = "/home/johof680/work/itkAlphaAMD-build/"

def main(USE_PYRAMIDS, RANDOMIZE_TRANSFORMS, DO_AFFINE_REGISTRATION, DO_TRANSFORM_LANDMARKS, EVAL_LANDMARKS, metric_name, transformation_size, noise_level, reverse):

  out_path = dataset_path

  # image extension
  im_ext = "tif"

  # number of registrations to perform
  count = 1000

  # create registration object
  parallel_count = 6
  image_dimensions = 2
  r = reg.Registration(parallel_count, bin_path, dim = image_dimensions, enable_logging = True)

  out_path1 = "/home/johof680/work/itkAlphaAMD-build/cilia/" + transformation_size + "/" + noise_level + "/"
  
  # Generate random transforms

  if RANDOMIZE_TRANSFORMS:

    rnd_param = r.get_random_transforms_param_defaults()
    if noise_level == "none":
      rnd_param["noise"] = "0.0"
    elif noise_level == "large":
      rnd_param["noise"] = "0.1"
    else:
      raise "Illegal noise level."
    
    if transformation_size == "small":
      rnd_param["rotation"] = "10"
      rnd_param["translation"] = "10"
      rnd_param["min_rotation"] = "0"
      rnd_param["min_translation"] = "0"
    elif transformation_size == "medium":
      rnd_param["rotation"] = "20"
      rnd_param["translation"] = "20"
      rnd_param["min_rotation"] = "10"
      rnd_param["min_translation"] = "10"
    elif transformation_size == "large":
      rnd_param["rotation"] = "30"
      rnd_param["translation"] = "30"
      rnd_param["min_rotation"] = "20"
      rnd_param["min_translation"] = "20"
    elif transformation_size == "all":
      rnd_param["rotation"] = "30"
      rnd_param["translation"] = "30"
      rnd_param["min_rotation"] = "0"
      rnd_param["min_translation"] = "0"
    else:
      raise "Illegal transformation size."

    r.random_transforms(dataset_path, out_path1, count, rnd_param)

    st.create_directory(out_path1)

    r.run("Random transformations")

  in_path = out_path1
  if reverse:
    out_path1 = out_path1 + metric_name + "_reverse/"
  else:
    out_path1 = out_path1 + metric_name + "/"
  st.create_directory(out_path1)

  # Do the registrations

  def register(rp, input1_path, input2_path, output_path, metric, cnt):
      for i in xrange(cnt):
          pth = output_path + ("registration_%d/" % (i+1))
          rpar = rp.get_register_param_defaults()
          rpar.pop("weights1", None)
          rpar.pop("weights2", None)
          msk2 = input2_path + "transformed_mask_%d.%s" % (i+1, im_ext)
          if reverse:
              rpar["mask2"] = "circle"
              rpar["mask1"] = msk2           
              in2 = input1_path + "ref_image_%d.%s" %(i+1, im_ext)
              in1 = input2_path + "transformed_image_%d.%s" % (i+1, im_ext)
          else:
              rpar["mask1"] = "circle"
              rpar["mask2"] = msk2           
              in1 = input1_path + "ref_image_%d.%s" %(i+1, im_ext)
              in2 = input2_path + "transformed_image_%d.%s" % (i+1, im_ext)           
          if USE_PYRAMIDS:
            rpar["multiscale_sampling_factors"] = "4x2x1"
            rpar["multiscale_smoothing_sigmas"] = "5x3x0"
          else:
            rpar["multiscale_sampling_factors"] = "1"
            rpar["multiscale_smoothing_sigmas"] = "0"
          rpar["metric"] = metric
          rpar["learning_rate"] = "0.5"
          rpar["alpha_outlier_rejection"] = "0.0"
          rpar["sampling_fraction"] = "0.1"
          rpar["normalization"] = "0.05"

          rp.register_affine(in1, in2, pth, rpar)

    #def landmark_transform(self, landmarks_in_path, out_path, out_name, transform_path):
  def transform_landmarks(rp, landmark_path, landmark_prefix, transform_path_base, cnt):
      for i in xrange(cnt):
          transform_path = transform_path_base + ("registration_%d/transform_complete.txt" % (i+1))
          if reverse:
            input_path = landmark_path + landmark_prefix + ".csv"
          else:
            input_path = landmark_path + landmark_prefix + "_%d.csv" % (i + 1)
          out_name = "registered_landmarks_%d.csv" % (i + 1)       

          rp.landmark_transform(input_path, transform_path_base, out_name, transform_path)

  if DO_AFFINE_REGISTRATION:
      register(r, in_path, in_path, out_path1, metric_name, count)
      r.run("Affine Registration")

  if DO_TRANSFORM_LANDMARKS:
      if reverse:
          transform_landmarks(r, in_path, "ref_landmarks", out_path1, count)
      else:
          transform_landmarks(r, in_path, "transformed_landmarks", out_path1, count)
      r.run("Transform Landmarks")


  def eval_landmarks(ref_landmark_path, output_path, cnt):
      if reverse:
          dists = np.zeros(cnt)
          for i in xrange(cnt):
              ref_lm = landmarks.read_csv(in_path + "transformed_landmarks_%d.csv" % (i+1), False) #csv_2_np.read_csv(out1_path + "ref_landmarks.csv", False)
              tra_lm = landmarks.read_csv(output_path + "registered_landmarks_%d.csv" % (i+1))
              dists[i] = landmarks.mean_euclidean(ref_lm, tra_lm)
          return (np.mean(dists), np.std(dists), dists, dt.make_distribution(dists))
      else:
          ref_lm = landmarks.read_csv(ref_landmark_path + "ref_landmarks.csv", False) #csv_2_np.read_csv(out1_path + "ref_landmarks.csv", False)
          dists = np.zeros(cnt)
          for i in xrange(cnt):
              tra_lm = landmarks.read_csv(output_path + "registered_landmarks_%d.csv" % (i+1))
              dists[i] = landmarks.mean_euclidean(ref_lm, tra_lm)
          return (np.mean(dists), np.std(dists), dists, dt.make_distribution(dists))

  if EVAL_LANDMARKS:
    (mn, stddev, full_distri, distri) = eval_landmarks(in_path, out_path1, count)
    np.set_printoptions(precision=5)
    np.set_printoptions(suppress=True)
    print("")
    #print(full_distri)
    print(distri)
    print(mn)
    print(stddev)
    print("Metric:" + metric_name)
    print("Transformation: " + transformation_size)
    print("Noise level: " + noise_level)
    print("IsReverse: " + str(reverse))
    np.savetxt(out_path1 + metric_name + ".csv", full_distri, delimiter=",")

  def ind_str(prefix, index, postfix):
    return prefix + ("%.3d" % index) + postfix

if __name__ == "__main__":
  RUN_MODE = True
  PYRAMID_MODE = True
  metric_list = ["alpha_smd", "ssd", "ncc", "mi"]
  transformation_list = ["small", "medium", "large", "all"]
  noise_list = ["none", "large"]

  combs = [(m, t, n) for m in metric_list for t in transformation_list for n in noise_list]

  for (m, t, n) in combs:
    main(PYRAMID_MODE, RUN_MODE, RUN_MODE, True, True, m, t, n, False)
    main(PYRAMID_MODE, RUN_MODE, RUN_MODE, True, True, m, t, n, True)


