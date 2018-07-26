
import script_tools as st
import registration as reg
import numpy as np
import landmarks
import dist_table as dt
#import matplotlib
import matplotlib.pyplot as plt
from matplotlib import mlab

def main(USE_PYRAMIDS, RANDOMIZE_TRANSFORMS, DO_AFFINE_REGISTRATION, DO_TRANSFORM_LANDMARKS, EVAL_LANDMARKS, metric_name, transformation_size, noise_level):

  # Windows
  #dataset_path = "C:/cygwin64/home/johan/itkAlphaCut/assets/01.png"
  #bin_path = "C:/cygwin64/home/johan/itkAlphaCut-release/Release/"

  # Linux
  dataset_path = "/home/johof680/work/itkAlphaCut-4j/itkAlphaCut/assets/01.png"
  bin_path = "/home/johof680/work/itkAlphaCut-4j/build-release/"

  #weight_path = "/home/johof680/work/VironovaRegistrationImages/"
  out_path = dataset_path#"/home/johof680/work/VironovaRegistrationImages/"

  # image extension
  im_ext = "tif"

  # number of registrations to perform
  count = 1000

  # create registration object
  parallel_count = 6
  image_dimensions = 2
  r = reg.Registration(parallel_count, bin_path, dim = image_dimensions, enable_logging = True)

  out_path1 = "/home/johof680/work/itkAlphaCut-4j/cilia_param_search7/" + transformation_size + "/" + noise_level + "/"
  #out_path1 = "C:/cygwin64/home/johan/cilia_random/large4/"

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
  out_path1 = out_path1 + metric_name + "/"
  st.create_directory(out_path1)

  #learning_rates_list = [0.001, 0.01, 0.1, 0.5, 1.0]
  #sampling_fractions_list = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
  #normalizations_list = [0.0, 0.01, 0.025, 0.05]
  #sampling_fractions_list = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]

  exponents = [-3.75, -3.5, -3.25, -3, -2.75, -2.5, -2.25, -2.0, -1.75, -1.5, -1.25, -1.0, -0.75, -0.5, -0.25, 0.0]
  #sampling_fractions_list = [np.power(10.0, i) for i in exponents]
  #sampling_fractions_list = [0.1]
  #learning_rates_list = [0.5]
  #normalizations_list = [0.0]
  #alpha_levels_list = [7]

  sampling_fractions_list = [0.1]
  learning_rates_list = [0.5]
  normalizations_list = [0.0]
  alpha_levels_list = range(1, 256)

  param_comb = [(a, b, c, d) for a in learning_rates_list for b in sampling_fractions_list for c in normalizations_list for d in alpha_levels_list]

  print("Parameter format (learning_rate, sampling_fraction, normalization percentile)")
  print(param_comb)

  # Do the registrations

  def register(rp, input1_path, input2_path, output_path, metric, cnt):
      for i in xrange(cnt):
          for j in xrange(len(param_comb)):
            (learning_rate, sampling_fraction, normalization, alpha_levels) = param_comb[j]

            pth = output_path + ("registration_%d_%d/" % (j+1, i+1))
            rpar = rp.get_register_param_defaults()
            rpar.pop("weights1", None)
            rpar.pop("weights2", None)
            in1 = input1_path + "ref_image_%d.%s" %(i+1, im_ext)
            in2 = input2_path + "transformed_image_%d.%s" % (i+1, im_ext)
            msk2 = input2_path + "transformed_mask_%d.%s" % (i+1, im_ext)
            if USE_PYRAMIDS:
              rpar["multiscale_sampling_factors"] = "4x2x1"
              rpar["multiscale_smoothing_sigmas"] = "5x3x0"
            else:
              rpar["multiscale_sampling_factors"] = "1"
              rpar["multiscale_smoothing_sigmas"] = "0"
            rpar["metric"] = metric
            rpar["learning_rate"] = str(learning_rate)
            rpar["alpha_outlier_rejection"] = "0.0"
            rpar["sampling_fraction"] = str(sampling_fraction)
            rpar["normalization"] = str(normalization)
            rpar["alpha_levels"] = str(alpha_levels)

            rpar["mask1"] = "circle"
            rpar["mask2"] = msk2
            rp.register_affine(in1, in2, pth, rpar)

    #def landmark_transform(self, landmarks_in_path, out_path, out_name, transform_path):
  def transform_landmarks(rp, landmark_path, transform_path_base, cnt):
      for i in xrange(cnt):
          for j in xrange(len(param_comb)):
            transform_path = transform_path_base + ("registration_%d_%d/transform_complete.txt" % (j+1, i+1))
            input_path = landmark_path + "transformed_landmarks_%d.csv" % (i + 1)
            out_name = "registered_landmarks_%d_%d.csv" % (j + 1, i + 1)       

            rp.landmark_transform(input_path, transform_path_base, out_name, transform_path)

  if DO_AFFINE_REGISTRATION:
      register(r, in_path, in_path, out_path1, metric_name, count)
      r.run("Affine Registration")

  if DO_TRANSFORM_LANDMARKS:
      transform_landmarks(r, in_path, out_path1, count)
      r.run("Transform Landmarks")


  def eval_landmarks(ref_landmark_path, output_path, cnt):
      ref_lm = landmarks.read_csv(ref_landmark_path + "ref_landmarks.csv", False) #csv_2_np.read_csv(out1_path + "ref_landmarks.csv", False)
      out_succ_freq = []
      out_succ_means = []
      out_means = []
      out_stddevs = []
      out_dists = []
      out_dist_summary = []

      for j in xrange(len(param_comb)):
        dists = np.zeros(cnt)
        for i in xrange(cnt):
            tra_lm = landmarks.read_csv(output_path + "registered_landmarks_%d_%d.csv" % (j + 1, i+1))
            dists[i] = landmarks.mean_euclidean(ref_lm, tra_lm)
        succ_freq = np.count_nonzero(np.where(dists <= 1.0))/np.float(cnt)
        succ_means = np.mean(dists[np.where(dists <= 1.0)])

        out_succ_freq.append(succ_freq)
        out_succ_means.append(succ_means)
        out_means.append(np.mean(dists))
        out_stddevs.append(np.std(dists))
        out_dists.append(dists)
        out_dist_summary.append(dt.make_distribution(dists))
          #np.sort(dists)
      return (out_succ_freq, out_succ_means, out_means, out_stddevs, out_dists, out_dist_summary)
          #print("%.4d: %f" % (i+1, dist))


          #print(eval_landmarks(out_path1, out_path1, count))

  def filter_set(values, tup, tup_index, tup_value):
    vals = []
    tups = []
    for i in xrange(len(tup)):
      if tup[i][tup_index] == tup_value:
        tups.append(tup[i])
        vals.append(values[i])
    return (np.array(vals), tups)

  if EVAL_LANDMARKS:
    (succ_freq, succ_means, mn, stddev, full_distri, distri) = eval_landmarks(in_path, out_path1, count)
    np.set_printoptions(precision=5)
    np.set_printoptions(suppress=True)
    print("")
    #print(full_distri)
    #print(distri)
    print("Succ freq: ")
    print(succ_freq)
    print("Succ means: ")
    print(succ_means)
    print("Means: ")
    print(mn)
    print(stddev)

    M = 9
    if M > len(param_comb):
      M = len(param_comb)

    argsort = np.argsort(mn)

    best_indices = argsort[0:M]
    best_params = [param_comb[ind] for ind in best_indices]
    best_vals = [mn[ind] for ind in best_indices]

    worst_indices = argsort[len(argsort)-M:]
    worst_params = [param_comb[ind] for ind in worst_indices]
    worst_vals = [mn[ind] for ind in worst_indices]

    print("Best param: " + str(best_params))
    print("Best vals: " + str(best_vals))

    print("Worst param: " + str(worst_params))
    print("Worst vals: " + str(worst_vals))   

    for i in xrange(len(full_distri)):
      np.savetxt(out_path1 + metric_name + "%d.csv" % (i+1), full_distri[i], delimiter=",")

    print("For learning rate 0.5")
    (mn_lr_0_5, tup_lr_0_5) = filter_set(mn, param_comb, 0, 0.5)
    print(mn_lr_0_5)
    print(tup_lr_0_5)

    #print(str(param_comb))
    #make_plot(distri)

  # Eval



  # param = r.get_register_param_defaults()

  # param["weights1"] = "hann"
  # param["weights2"] = "hann"
  # param["learning_rate"] = "0.05"
  # param["relaxation_factor"] = "0.99"
  # param["alpha_levels"] = "7"
  # param["alpha_max_distance"] = "128"
  # param["seed"] = "1000"
  # param["metric"] = "alpha_smd"
  # param["iterations"] = "1000"
  # param["sampling_fraction"] = "0.001"
  # param["spacing_mode"] = "default"
  # param["normalization"] = "0.005"
  # param["multiscale_sampling_factors"] = "4x2x1"
  # param["multiscale_smoothing_sigmas"] = "4.0x2.0x1.0"

  #def make_dir(path, index):
  #  return st.makedir_string(path + "Out%.3d" % (index))

  #def make_register(index, in1, in2, out, params):
  #  return exe_path + " -in1 " + in1 + " -in2 " + in2 + " -out " + out + st.param_dict_to_string(params)

  def ind_str(prefix, index, postfix):
    return prefix + ("%.3d" % index) + postfix

  #def make_cmd(index, in1, in2, out, params):
  #  print("# Image %.3d" % (index))
  #  print(make_dir(out_path, index))
  #  print(make_register(index, in1, in2, out, params))
  #  print("")

  #registration
  # N = 18
  # DO_REGISTRATION = 1

  # if DO_REGISTRATION == 1:
  #   for i in xrange(9, N):
  #     r.register_rigid(in_path + "im008.tif", in_path + ind_str("im", i, ".tif"), dataset_path + ind_str("Out", i, "/"), param)

  #   r.run("Rigid registration")

  #transformation

  # par = r.get_transform_param_defaults()
  # par["16bit"] = "1"

  # for i in xrange(9, N):
  #   r.transform(in_path + "im008.tif", in_path + ind_str("im", i, ".tif"), dataset_path + ind_str("Out", i, "/"), "transformed.png", dataset_path + ind_str("Out", i, "/transform_complete.txt"), par)

  # r.run("Transformations")



    #make_cmd(i, in_path + "im008.tif", in_path + ind_str("im", i, ".tif"), out_path + ind_str("Out", i, "/"), param)

  #mkdir ../../../VironovaRegistrationImages/Out009/
  #../../build6/ACRegister -in1 ../../../VironovaRegistrationImages/1Original/im008.tif -in2 ../../../VironovaRegistrationImages/1Original/im009.tif -weights1 ../../../VironovaRegistrationImages/hann_mask.tif -weights2 ../../../VironovaRegistrationImages/hann_mask.tif -out ../../../VironovaRegistrationImages/Out009/ -learning_rate 0.5 -relaxation_factor 0.99 -alpha_levels 7 -alpha_max_distance 128 -seed 1000 -metric alpha_smd -iterations 1000 -sampling_fraction 0.01 -spacing_mode remove -normalization 0.005 -rigid 1 -smoothing 2.0

if __name__ == "__main__":
  RUN_MODE = True
  PYRAMID_MODE = True
  #main(PYRAMID_MODE, RUN_MODE, RUN_MODE, RUN_MODE, True, "alpha_smd", "small", "large")
#  main(PYRAMID_MODE, False, RUN_MODE, RUN_MODE, True, "ssd", "small", "large")
#  main(PYRAMID_MODE, False, RUN_MODE, RUN_MODE, True, "ncc", "small", "large")
#  main(PYRAMID_MODE, False, RUN_MODE, RUN_MODE, True, "mi", "small", "large")

  main(PYRAMID_MODE, RUN_MODE, RUN_MODE, RUN_MODE, True, "alpha_smd", "all", "large")
#  main(PYRAMID_MODE, False, RUN_MODE, RUN_MODE, True, "ssd", "medium", "large")
#  main(PYRAMID_MODE, False, RUN_MODE, RUN_MODE, True, "ncc", "medium", "large")
#  main(PYRAMID_MODE, False, RUN_MODE, RUN_MODE, True, "mi", "medium", "large")

  #main(PYRAMID_MODE, RUN_MODE, RUN_MODE, RUN_MODE, True, "alpha_smd", "large", "large")
#  main(PYRAMID_MODE, False, RUN_MODE, RUN_MODE, True, "ssd", "large", "large")
#  main(PYRAMID_MODE, False, RUN_MODE, RUN_MODE, True, "ncc", "large", "large")
#  main(PYRAMID_MODE, False, RUN_MODE, RUN_MODE, True, "mi", "large", "large")

  #main(PYRAMID_MODE, RUN_MODE, RUN_MODE, RUN_MODE, True, "alpha_smd", "small", "none")
#  main(PYRAMID_MODE, False, RUN_MODE, RUN_MODE, True, "ssd", "small", "none")
#  main(PYRAMID_MODE, False, RUN_MODE, RUN_MODE, True, "ncc", "small", "none")
#  main(PYRAMID_MODE, False, RUN_MODE, RUN_MODE, True, "mi", "small", "none")

  #main(PYRAMID_MODE, RUN_MODE, RUN_MODE, RUN_MODE, True, "alpha_smd", "medium", "none")
#  main(PYRAMID_MODE, False, RUN_MODE, RUN_MODE, True, "ssd", "medium", "none")
#  main(PYRAMID_MODE, False, RUN_MODE, RUN_MODE, True, "ncc", "medium", "none")
#  main(PYRAMID_MODE, False, RUN_MODE, RUN_MODE, True, "mi", "medium", "none")

  #main(PYRAMID_MODE, RUN_MODE, RUN_MODE, RUN_MODE, True, "alpha_smd", "large", "none")
  #main(PYRAMID_MODE, False, RUN_MODE, RUN_MODE, True, "ssd", "large", "none")
  #main(PYRAMID_MODE, False, RUN_MODE, RUN_MODE, True, "ncc", "large", "none")
  #main(PYRAMID_MODE, False, RUN_MODE, RUN_MODE, True, "mi", "large", "none")
