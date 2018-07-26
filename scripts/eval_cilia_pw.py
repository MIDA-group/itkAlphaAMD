
import script_tools as st
import registration as reg
import numpy as np
import landmarks
import dist_table as dt
#import matplotlib
import matplotlib.pyplot as plt
from matplotlib import mlab
import scipy.ndimage

# Windows
#dataset_path = "C:/cygwin64/home/johan/itkAlphaCut/assets/01.png"
#bin_path = "C:/cygwin64/home/johan/itkAlphaCut-release/Release/"

# Linux
dataset_path = "/home/johof680/work/cilia_dataset/"
bin_path = "/home/johof680/work/itkAlphaCut-4j/build-release/"

def register_cilia_small(ref, flo_list, out_list, landmarks_list, rigid, initial_list):
  im_ext = "png"
  
  # create registration object
  parallel_count = 6
  image_dimensions = 2
  r = reg.Registration(parallel_count, bin_path, dim = image_dimensions, enable_logging = True)

  for index in xrange(len(flo_list)):
    flo = flo_list[index]
    pth = out_list[index]
    rpar = r.get_register_param_defaults()
    #rpar.pop("weights1", None)
    #rpar.pop("weights2", None)
    rpar["weights1"] = "hann2"
    rpar["weights2"] = "hann2"
 
    rpar["multiscale_sampling_factors"] = "1"
    rpar["multiscale_smoothing_sigmas"] = "0"

    rpar["metric"] = "alpha_smd"
    rpar["alpha_levels"] = "7"
    rpar["learning_rate"] = "0.1"
    rpar["alpha_max_distance"] = "128"
    rpar["alpha_outlier_rejection"] = "0.0"
    rpar["sampling_fraction"] = "1.0"#0.05
    rpar["normalization"] = "0.0"

    #rpar["mask1"] = dataset_path + "circle_shrunk.png"
    #rpar["mask2"] = dataset_path + "circle_shrunk.png"
    #rpar.pop("mask1", None)
    #rpar.pop("mask2", None)
    rpar["mask1"] = dataset_path + "small_circle2.png"
    rpar["mask2"] = dataset_path + "small_circle2.png"

    rpar["init_transform"] = initial_list[index]

    if rigid:
      r.register_rigid(dataset_path + ref, dataset_path + flo, pth, rpar)
    else:
      r.register_affine(dataset_path + ref, dataset_path + flo, pth, rpar)
  if rigid:
    name = "Rigid"
  else:
    name = "Affine"
  r.run(name)

  for index in xrange(len(flo_list)):
    flo = flo_list[index]
    pth = out_list[index]
    rpar = r.get_transform_param_defaults()

    r.transform(dataset_path + ref, dataset_path + flo, pth, "outimg.png", pth + "transform_complete.txt")
  r.run("Transform")

  for index in xrange(len(landmarks_list)):
    pth = out_list[index]
    r.landmark_transform(landmarks_list[index], pth, "transformed_landmarks.csv", pth + "transform_complete.txt")
  r.run("Landmarks")

def register_cilia_large(ref, flo_list, out_list, landmarks_list, rigid, initial_list):
  im_ext = "png"
  
  # create registration object
  parallel_count = 6
  image_dimensions = 2
  r = reg.Registration(parallel_count, bin_path, dim = image_dimensions, enable_logging = True)

  for index in xrange(len(flo_list)):
    flo = flo_list[index]
    pth = out_list[index]
    rpar = r.get_register_param_defaults()
    rpar.pop("weights1", None)
    rpar.pop("weights2", None)
    #rpar["weights1"] = "hann4"#dataset_path + "hann3.png"
    #rpar["weights2"] = "hann4"#dataset_path + "hann3.png"

    rpar["multiscale_sampling_factors"] = "1"
    rpar["multiscale_smoothing_sigmas"] = "0"

    rpar["metric"] = "alpha_smd"
    rpar["alpha_levels"] = "7"
    rpar["learning_rate"] = "0.5"
    rpar["alpha_max_distance"] = "128"
    rpar["alpha_outlier_rejection"] = "0.0"
    rpar["sampling_fraction"] = "1.0"
    rpar["normalization"] = "0.01"

    rpar["mask1"] = dataset_path + "circle_shrunk.png"
    rpar["mask2"] = dataset_path + "circle_shrunk.png"

    rpar["init_transform"] = initial_list[index]

    if rigid:
      r.register_rigid(dataset_path + ref, dataset_path + flo, pth, rpar)
    else:
      r.register_affine(dataset_path + ref, dataset_path + flo, pth, rpar)
  if rigid:
    name = "Rigid"
  else:
    name = "Affine"
  r.run(name)

  for index in xrange(len(flo_list)):
    flo = flo_list[index]
    pth = out_list[index]
    rpar = r.get_transform_param_defaults()

    r.transform(dataset_path + ref, dataset_path + flo, pth, "outimg.png", pth + "transform_complete.txt")
  r.run("Transform")

  for index in xrange(len(landmarks_list)):
    pth = out_list[index]
    r.landmark_transform(landmarks_list[index], pth, "transformed_landmarks.csv", pth + "transform_complete.txt")
  r.run("Landmarks")

def load_distances(out_list, k, n, output):
  d = np.zeros(len(out_list))
  for (index, out) in enumerate(out_list):
    distance_path = out + "distance.csv"
    v = st.read_csv(distance_path)
    d[index] = v[0]
  d = d.reshape([n, k])
  np.savetxt(output, d, fmt = "%.7f", delimiter = ",")
  best_index = np.argmin(d, axis=1)

  return (d, best_index)

def merge_images(d, out_list, k, n, w, h, output):
  result = np.zeros([h*n,w*k])
  inds_x = [kk * w for nn in xrange(n) for kk in xrange(k)]
  inds_y = [nn * h for nn in xrange(n) for kk in xrange(k)]
  inds_x.append(w*k)
  inds_y.append(h*n)

  for (index, out) in enumerate(out_list):
    image_path = out + "outimg.png"
    img = scipy.ndimage.imread(image_path)
    result[inds_y[index]:inds_y[index]+h, inds_x[index]:inds_x[index]+w] = img

  mn = np.argmin(d, axis=1)
  
  #result[50:150,100:120] = 65335/2
  stroke = 5
  for nn in xrange(n):
    #print(str(mn[nn]))
    start_x = (mn[nn]*w)
    end_x = ((mn[nn]+1)*w)
    start_y = (nn * h)
    end_y = ((nn * h)+stroke)

    result[start_y:end_y, start_x:end_x] = 65335/2
    result[(start_y + h):(end_y + h), start_x:end_x] = 65335/2

    start_x = (mn[nn]*w)
    end_x = start_x + stroke
    start_y = (nn * h)
    end_y = ((nn+1) * h)

    result[start_y:end_y, start_x:end_x] = 65335/2
    result[start_y:end_y, start_x+w:end_x+w] = 65335/2
    #result[(((nn+1) * h)):((nn+1) * h), ((mn[nn]+1)*w-stroke):(((mn[nn]+1))*w)] = 65335/2

  scipy.misc.imsave(output, result)

def min_distance(p, set):
    mind = float('inf')
    mini = -1
    for i in xrange(set.shape[0]):
        setp = set[i, :]
        d = np.linalg.norm(p-setp, ord=2)#np.sqrt(np.sum(np.square(p-setp)))
        if d < mind:
            mind = d
            mini = i
    return (mind, mini)

def smd(ref, target):
    refshape = ref.shape
    acc = 0
    for i in xrange(refshape[0]):
        (d, ind) = min_distance(ref[i, :], target)
        acc = acc + d
    return acc

ttype_first = "rigid"
ttype_second = "affine"

def first(N):
  initial_transforms = [dataset_path + "transforms/transform%d.txt" % (k+1) for n in xrange(N) for k in xrange(9)]
  landmark_paths = [dataset_path + "cilia_landmarks_%d.csv" % (n+1) for n in xrange(1, N+1) for k in xrange(9)]
  out_paths = [dataset_path + "first/" + ttype_first + "_%d/%d/" % (k+1, n+1) for n in xrange(N) for k in xrange(9)]
  register_cilia_small("cilia_1.png", ["cilia_" + str(i+1) + ".png" for i in xrange(1, N+1) for k in xrange(9)], out_paths, landmark_paths, ttype_first == "rigid", initial_transforms)
  (d, best_index) = load_distances(out_paths, 9, N, dataset_path + "first/distances.csv")

  merge_images(d, out_paths, 9, N, 129, 129, dataset_path + "first/collage.png")

  return best_index

def second(N, bind):
  initial_transforms = [dataset_path + "first/" + (ttype_first + "_%d/%d/transform_" % (bind[n]+1, n+1)) + ttype_first + ".txt" for n in xrange(N)]
  landmark_paths = [dataset_path + "cilia_landmarks_%d.csv" % (n+1) for n in xrange(1, N+1)]
  out_paths = [dataset_path + "second/" + ttype_second + "_%d/%d/" % (1, n+1) for n in xrange(N)]
  register_cilia_large("cilia_1.png", ["cilia_" + str(i+1) + ".png" for i in xrange(1, N+1)], out_paths, landmark_paths, ttype_second == "rigid", initial_transforms)
  (d, best_index2) = load_distances(out_paths, 1, N, dataset_path + "second/distances.csv")
  merge_images(d, out_paths, 1, N, 129, 129, dataset_path + "second/collage.png")
  
def split_landmarks(lm):
  cp_lm = lm[0:2, :]
  odd_lm = lm[2::2, :]
  even_lm = lm[3::2, :]
  return (cp_lm, odd_lm, even_lm)

def eval(ref, landmarks):
  ref_landmarks = st.read_csv(ref)
  (ref_cp_lm, ref_odd_lm, ref_even_lm) = split_landmarks(ref_landmarks)
  # create registration object
  parallel_count = 6
  image_dimensions = 2
  r = reg.Registration(parallel_count, bin_path, dim = image_dimensions, enable_logging = True)
  
  all_smd_list = []
  cp_smd_list = []
  outer_smd_list = []

  for landmark_path in landmarks:
    flo_landmarks = st.read_csv(landmark_path)
    (flo_cp_lm, flo_odd_lm, flo_even_lm) = split_landmarks(flo_landmarks)
    
    cp_lm_smd = smd(ref_cp_lm, flo_cp_lm)
    odd_lm_smd = smd(ref_odd_lm, flo_odd_lm)
    even_lm_smd = smd(ref_even_lm, flo_even_lm)
    
    outer_lm_smd = odd_lm_smd + even_lm_smd
    all_lm_smd = cp_lm_smd + outer_lm_smd

    cp_smd_list.append(cp_lm_smd / 2.0)
    outer_smd_list.append(outer_lm_smd / 18.0)
    all_smd_list.append(all_lm_smd / 20.0)
  
  final_smd = np.array([all_smd_list, cp_smd_list, outer_smd_list])
  print(final_smd)

  all_smd = np.array(all_smd_list)
  cp_smd = np.array(cp_smd_list)
  outer_smd = np.array(outer_smd_list)

  print("All:   %.5f +- %.5f" % (np.mean(all_smd), np.std(all_smd)))
  print("CP:    %.5f +- %.5f" % (np.mean(cp_smd), np.std(cp_smd)))
  print("Outer: %.5f +- %.5f" % (np.mean(outer_smd), np.std(outer_smd)))

  









if __name__ == "__main__":
  N = 19
  best_index = first(N)
  #eval_first(N, best_index)
  second(N, best_index)
  eval(dataset_path + "cilia_landmarks_1.csv", [dataset_path + "first/" + (ttype_first + "_%d/%d/transformed_landmarks.csv" % (best_index[n-1]+1, n)) for n in xrange(1, N+1)] )
  eval(dataset_path + "cilia_landmarks_1.csv", [dataset_path + "second/" + (ttype_second + "_1/%d/transformed_landmarks.csv" % n) for n in xrange(1, N+1)] )
  eval(dataset_path + "cilia_landmarks_1.csv", [dataset_path + "cilia_landmarks_%d.csv" % n for n in xrange(2, N+2)])
  #ttype = "affine"
  #ttypeflag = False
  #second_stage = True
  #if second_stage:
  #  initial_transform = #dataset_path + ttype + "_out_1/1/transform_affine.txt"
  #  out_path = dataset_path + ttype + "_out2_%d/" % (k+1)
  #  register_cilia("cilia_1.png", ["cilia_" + str(i+1) + ".png" for i in xrange(1, N+1)], out_path, ttypeflag, initial_transform)
  #else:
  #  for k in xrange(9):
  #    initial_transform = [dataset_path + "transforms/transform%d.txt" % (k+1) for k in xrange(N)]     
  #    out_paths = [dataset_path + ttype + "_out_%d/" % (k+1) for k in xrange(N)]
  #    register_cilia("cilia_1.png", ["cilia_" + str(i+1) + ".png" for i in xrange(1, N+1)], out_path, ttypeflag, initial_transform)

    #if second_stage:
    #  initial_transform = #dataset_path + ttype + "_out_1/1/transform_affine.txt"
    #  out_path = dataset_path + ttype + "_out2_%d/" % (k+1)
    #else:
    #  initial_transform = [dataset_path + "transforms/transform%d.txt" % (k+1), 
    #  out_path = dataset_path + ttype + "_out_%d/" % (k+1)
    #register_cilia("cilia_1.png", ["cilia_" + str(i+1) + ".png" for i in xrange(1, 2)], out_path, ttypeflag, initial_transform)
  