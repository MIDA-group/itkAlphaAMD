
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
  #dataset_path = "c:/dev/data/VironovaRegistrationImages/"
  #bin_path = "C:/cygwin64/home/johan/itkAlphaCut-release/Release/"

  # Linux
  dataset_path = "/home/johof680/work/LPBA40_for_ANTs/mri/S01.nii.gz"
  bin_path = "/home/johof680/work/itkAlphaCut-4j/build-release/"

  # image extension
  im_ext = "nii.gz"

  # number of registrations to perform
  count = 200

  # create registration object
  parallel_count = 6
  image_dimensions = 3
  r = reg.Registration(parallel_count, bin_path, dim = image_dimensions, enable_logging = True)

  out_path1 = "/home/johof680/work/itkAlphaCut-4j/lpba40_random_5/" + transformation_size + "/" + noise_level + "/"

  #metric_name = "ssd"

  #USE_PYRAMIDS = True

  #RANDOMIZE_TRANSFORMS = True
  #DO_AFFINE_REGISTRATION = True
  #DO_TRANSFORM_LANDMARKS = True
  #EVAL_LANDMARKS = True

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
      rnd_param["rotation"] = "15"
      rnd_param["translation"] = "15"
      rnd_param["min_rotation"] = "10"
      rnd_param["min_translation"] = "10"
    elif transformation_size == "large":
      rnd_param["rotation"] = "20"
      rnd_param["translation"] = "20"
      rnd_param["min_rotation"] = "15"
      rnd_param["min_translation"] = "15"
    elif transformation_size == "all":
      rnd_param["rotation"] = "20"
      rnd_param["translation"] = "20"
      rnd_param["min_rotation"] = "0"
      rnd_param["min_translation"] = "0"      
    else:
      raise "Illegal transformation size."
      
    #rnd_param["format_ext"] = im_ext

    r.random_transforms(dataset_path, out_path1, count, rnd_param)

    st.create_directory(out_path1)

    r.run("Random transformations")

  in_path = out_path1
  out_path1 = out_path1 + metric_name + "/"
  if USE_PYRAMIDS:
    out_path1 = out_path1 + "w_pyramid/"
  else:
    out_path1 = out_path1 + "wo_pyramid/"
  st.create_directory(out_path1)

  # Do the registrations

  def register(rp, input1_path, input2_path, output_path, metric, cnt):
      for i in xrange(cnt):
          pth = output_path + ("registration_%d/" % (i+1))
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
          rpar["learning_rate"] = "0.5"
          rpar["alpha_outlier_rejection"] = "0.0"
          rpar["sampling_fraction"] = "0.01"

          rpar["mask1"] = "circle"
          rpar["mask2"] = msk2
          rp.register_affine(in1, in2, pth, rpar)

    #def landmark_transform(self, landmarks_in_path, out_path, out_name, transform_path):
  def transform_landmarks(rp, landmark_path, transform_path_base, cnt):
      for i in xrange(cnt):
          transform_path = transform_path_base + ("registration_%d/transform_complete.txt" % (i+1))
          input_path = landmark_path + "transformed_landmarks_%d.csv" % (i + 1)
          out_name = "registered_landmarks_%d.csv" % (i + 1)       

          rp.landmark_transform(input_path, transform_path_base, out_name, transform_path)

  if DO_AFFINE_REGISTRATION:
      register(r, in_path, in_path, out_path1, metric_name, count)
      r.run("Affine Registration")

  if DO_TRANSFORM_LANDMARKS:
      transform_landmarks(r, in_path, out_path1, count)
      r.run("Transform Landmarks")


  def eval_landmarks(ref_landmark_path, output_path, cnt):
      ref_lm = landmarks.read_csv(ref_landmark_path + "ref_landmarks.csv", False) #csv_2_np.read_csv(out1_path + "ref_landmarks.csv", False)
      dists = np.zeros(cnt)
      for i in xrange(cnt):
          tra_lm = landmarks.read_csv(output_path + "registered_landmarks_%d.csv" % (i+1))
          dists[i] = landmarks.mean_euclidean(ref_lm, tra_lm)
          #np.sort(dists)
      return (np.mean(dists), np.std(dists), dists, dt.make_distribution(dists))
          #print("%.4d: %f" % (i+1, dist))


          #print(eval_landmarks(out_path1, out_path1, count))

  # def count_less_than(sorted_distrib, value):
  #     return np.searchsorted(sorted_distrib, value)

  # def compute_hist(distrib, bins):
  #     yy = np.sort(distrib)
  #     x = np.arange(0, 30.0, 30.0 / bins)
  #     h = np.zeros(bins)
  #     for i in xrange(bins):
  #         h[i] = count_less_than(yy, x[i])
  #     return (x, h)



  # def make_plot(distrib):
  #     fig, ax = plt.subplots(figsize = (8, 4))
  #     n_bins = 500
  #     (x, h) = compute_hist(distrib, n_bins)
  #     #n, bins, patches = ax.hist(distrib, n_bins, normed=1, color=c[i], cumulative=True, label='alpha_smd')

  #     c = ['r', 'b', 'g', 'y', 'k']
  #     ax.fill_between(x, 0, h)
  #     #histtype='step'
  #     ax.grid(True)
  #     ax.legend(loc='right')
  #     ax.set_title('Registration results')
  #     ax.set_xlabel('Registration error (px)')
  #     ax.set_ylabel('Success rate')
  #     ax.set_ylim(0.0, 1.0)
  #     ax.set_xlim(0.0, 1.0)
  #     plt.show()

  if EVAL_LANDMARKS:
    (mn, stddev, full_distri, distri) = eval_landmarks(in_path, out_path1, count)
    np.set_printoptions(precision=5)
    np.set_printoptions(suppress=True)
    print("")
    print(distri)
    print(mn)
    print(stddev)
    np.savetxt(out_path1 + metric_name + ".csv", full_distri, delimiter=",")
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
  def build(tsizes, nsizes):
    for t in tsizes:
      for n in nsizes:
        print("Build " + t + ", " + n)
        main(False, True, False, False, False, "alpha_smd", t, n)

  def run(metrics, tsizes, nsizes):
    for m in metrics:
      for t in tsizes:
        for n in nsizes:
          print("Run " + m + ", " + t + ", " + n)
          main(True, False, True, True, True, m, t, n)

  #build(["large", "all"], ["large"])
  #build(["small", "medium", "large", "all"], ["large"])
  #run(["alpha_smd", "ssd", "ncc", "mi"], ["large", "all"], ["large"])
  run(["alpha_smd", "ssd", "ncc", "mi"], ["small", "medium", "large", "all"], ["large"])
 
