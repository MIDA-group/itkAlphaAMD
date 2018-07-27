
###
### Registration script API for running sets of jobs in parallel
### with a simple python interface, on Windows, MacOSX and Linux.
### Author: Johan Ofverstedt, 2018
###

import script_tools as st
import os
import os.path

if os.name == "nt":
  exe_extension = ".exe"
else:
  exe_extension = ""

class Registration:
  # Constructor
  def __init__(self, cpu_count, bin_path, dim=2, enable_logging=True):
    self.cpu_count = cpu_count
    if cpu_count <= 0:
      self.cpu_count = st.cpu_count()
    self.pool = st.create_pool(cpu_count)
    self.bin_path = bin_path
    self.enable_logging = enable_logging
    self.dim = dim
    self.queue = []

  # Syncronize, by waiting for all the asyncronous tasks to complete,
  # and then start a new thread-pool/task-set, discarding the outputs
  # from the first pool.
  def run(self, name="Unnamed"):
    print("")    
    print("--- " + name + " ---")
    print("Starting %d tasks..." % len(self.queue))

    # start all the tasks
    for i in xrange(0, len(self.queue)):
      cmd = self.queue[i][0]
      log_path = self.queue[i][1]
      st.run(self.pool, cmd, log_path)

    # synchronize
    st.join_pool(self.pool)
    result = self.pool[1]
    # create a new thread-pool and queue
    self.pool = st.create_pool(self.cpu_count)
    self.queue = []
    return result

  def _add_task(self, cmd, log_path):
    self.queue.append((cmd, log_path,))

  def get_register_param_defaults(self):
    param = {}
    
    param["weights1"] = "hannsqrt"
    param["weights2"] = "hannsqrt"
    param["learning_rate"] = "0.5"
    param["relaxation_factor"] = "0.99"
    param["alpha_levels"] = "7"
    param["alpha_max_distance"] = "128"
    param["alpha_outlier_rejection"] = "0.05"
    param["seed"] = "1000"
    param["metric"] = "alpha_smd"
    param["iterations"] = "3000"
    param["sampling_fraction"] = "0.05"
    param["spacing_mode"] = "default"#"default"
    param["normalization"] = "0.05"
    param["histogram_matching"] = "0"
    #param["init_transform"] = "/some_path/"

    return param

  def _make_register_cmd(self, ref_im_path, flo_im_path, out_path, is_rigid, param=None):
    if param is None:
      print("Using default param.")
      param = self.get_register_param_defaults()

    st.check_exists(ref_im_path, "Reference")
    st.check_exists(ref_im_path, "Floating")

    if self.dim == 2:
      three_d_str = " -3d 0"
    else:
      three_d_str = " -3d 1"
    if is_rigid:
      rigid_str = " -rigid 1"
    else:
      rigid_str = " -rigid 0"
    cmd = self.bin_path + "ACRegister" + exe_extension
    st.check_exists(cmd, "Executable")
    cmd = cmd + " -in1 " + ref_im_path + " -in2 " + flo_im_path + " -out " + out_path + rigid_str + three_d_str + st.param_dict_to_string(param)
    return cmd

  def register_affine(self, ref_im_path, flo_im_path, out_path, param=None):
    st.create_directory(out_path)

    cmd = self._make_register_cmd(ref_im_path, flo_im_path, out_path, False, param)
    #print(cmd)
    if self.enable_logging:
      log_path = out_path + "register_affine_"
    else:
      log_path = None
    self._add_task(cmd, log_path)

  def register_rigid(self, ref_im_path, flo_im_path, out_path, param=None):
    st.create_directory(out_path)
    
    cmd = self._make_register_cmd(ref_im_path, flo_im_path, out_path, True, param)

    if self.enable_logging:
      log_path = out_path + "register_rigid_"
    else:
      log_path = None
    self._add_task(cmd, log_path)

  def get_transform_param_defaults(self):
    param = {}
        
    param["interpolation"] = "cubic"
    param["spacing_mode"] = "default"
    param["scaling_factor"] = "1.0"
    param["16bit"] = "1"
    param["bg"] = "0.0"

    return param

  def _make_transform_cmd(self, ref_im_path, flo_im_path, out_path, transform_path, param=None):
    if param is None:
      param = self.get_transform_param_defaults()

    st.check_exists(ref_im_path, "Reference")
    st.check_exists(flo_im_path, "Floating")

    dim_str = " -dim %d" % (self.dim)
    cmd = self.bin_path + "ACTransform" + exe_extension
    st.check_exists(cmd, "Executable")
    cmd = cmd + " -ref " + ref_im_path + " -in " + flo_im_path + " -out " + out_path + " -transform " + transform_path + dim_str + st.param_dict_to_string(param)
    return cmd

  def transform(self, ref_im_path, flo_im_path, out_path, out_name, transform_path, param=None):
    cmd = self._make_transform_cmd(ref_im_path, flo_im_path, out_path + out_name, transform_path, param)

    if self.enable_logging:
      log_path = out_path + "transform_"
    else:
      log_path = None
    self._add_task(cmd, log_path)

  def transform_labels(self, ref_im_path, flo_im_path, out_path, out_name, transform_path, param=None):
    if param is None:
      param = self.get_transform_param_defaults()
    param["interpolation"] = "nearest"

    self.transform(ref_im_path, flo_im_path, out_path, out_name, transform_path, param)

  def get_label_overlap_param_defaults(self):
    param = {}

    param["label_mode"] = "multi"
    param["bit_depth1"] = "16"
    param["bit_depth2"] = "16"

    return param

  def _make_label_overlap_cmd(self, ref_im_path, transformed_im_path, out_path, param=None):
    if param is None:
      param = self.get_label_overlap_param_defaults()

    st.check_exists(ref_im_path, "Reference")
    st.check_exists(transformed_im_path, "Transformed")

    dim_str = " -dim %d" % (self.dim)
    cmd = self.bin_path + "ACLabelOverlap" + exe_extension
    st.check_exists(cmd, "Executable")
    cmd = cmd + " -in1 " + ref_im_path + " -in2 " + transformed_im_path + " -out " + out_path + dim_str + st.param_dict_to_string(param)
    return cmd

  def label_overlap(self, ref_im_path, transformed_im_path, out_path, out_name, param=None):
    #param = {}
    #if binary:
    #  param["label_mode"] = "binary"
    #else:
    #  param["label_mode"] = "multi"

    cmd = self._make_label_overlap_cmd(ref_im_path, transformed_im_path, out_path + out_name + ".csv", param)

    if self.enable_logging:
      log_path = out_path + "label_overlap_" + out_name + "_"
    else:
      log_path = None
    self._add_task(cmd, log_path)

  def _make_landmark_transform_cmd(self, landmarks_in, landmarks_out, transform_path):
    st.check_exists(landmarks_in, "Landmarks")

    dim_str = " -dim %d" % (self.dim)
    cmd = self.bin_path + "ACTransformLandmarks" + exe_extension
    st.check_exists(cmd, "Executable")
    cmd = cmd + " -in " + landmarks_in + " -out " + landmarks_out + " -transform " + transform_path + dim_str
    return cmd

  def landmark_transform(self, landmarks_in_path, out_path, out_name, transform_path):
    cmd = self._make_landmark_transform_cmd(landmarks_in_path, out_path + out_name, transform_path)

    if self.enable_logging:
      log_path = out_path + "landmark_transform_"
    else:
      log_path = None
    self._add_task(cmd, log_path)


    # if(strcmp(mod, "-in") == 0) {
    #   param.inPath = arg;
    # } else if(strcmp(mod, "-out") == 0) {
    #   param.outPath = arg;
    # } else if(strcmp(mod, "-seed") == 0) {
    #   param.seed = atoi(arg);
    # } else if(strcmp(mod, "-count") == 0) {
    #   param.count = atoi(arg);
    #     } else if(strcmp(mod, "-rotation") == 0) {
    #   param.rotation = atof(arg);
    #     } else if(strcmp(mod, "-translation") == 0) {
    #   param.translation = atof(arg);
    #     } else if(strcmp(mod, "-scaling") == 0) {
    #   param.scaling = atof(arg);
    #     } else if(strcmp(mod, "-min_rotation") == 0) {
    #   param.minRotation = atof(arg);
    #     } else if(strcmp(mod, "-min_translation") == 0) {
    #   param.minTranslation = atof(arg);
    # } else if(strcmp(mod, "-noise") == 0) {
    #         param.noiseStdDev = atof(arg);
    # } else if(strcmp(mod, "-dim") == 0) {
    #   unsigned int dim = atoi(arg);
    #   if(dim == 3U) {
    #     vol3DMode = true;
    #   } else if(dim == 2U) {
    #     vol3DMode = false;
    #   } else {
    #     std::cerr << "Illegal dimension '" << dim << "' given. Only '2' or '3' allowed." << std::endl;
    #     return -1;
    #   }
    # } else if(strcmp(mod, "-bit_depth") == 0) {
    #         param.bitDepth = atoi(arg);
    # }

  def get_random_transforms_param_defaults(self):
    param = {}
        
    param["seed"] = "1000"
    param["rotation"] = "30"
    param["translation"] = "30"
    param["scaling"] = "1"
    param["min_rotation"] = "20"
    param["min_translation"] = "20"
    param["noise"] = "0.1"
    param["bit_depth"] = "16"
    if self.dim == 2:
      param["format_ext"] = "tif"
    elif self.dim == 3:
      param["format_ext"] = "nii.gz"

    return param

  def _make_random_transforms_cmd(self, in_path, out_path, count, param=None):
    if param is None:
      param = self.get_random_transforms_param_defaults()

    st.check_exists(in_path, "Input")

    dim_str = " -dim %d" % (self.dim)
    count_str = " -count %d" % (count)
    cmd = self.bin_path + "ACRandomTransforms" + exe_extension
    cmd = cmd + " -in " + in_path + " -out " + out_path + dim_str + count_str + st.param_dict_to_string(param)
    return cmd

  def random_transforms(self, in_path, out_path, count, param=None):
    cmd = self._make_random_transforms_cmd(in_path, out_path, count, param)

    if self.enable_logging:
      log_path = out_path + "random_transforms_"
    else:
      log_path = None
    self._add_task(cmd, log_path)

# Test script
if __name__ == "__main__":
  r = Registration(2, "C:/cygwin64/home/johan/itkAlphaCut-release/Release/", dim = 2, enable_logging = True)

  par = r.get_register_param_defaults()

  N = 10

  for i in xrange(N):
    r.register_affine("c:/dev/registration/01.png", "c:/dev/registration/04.png", "c:/dev/registration/reg%.3d/" % (i+1), par)

  r.run("Affine Registration")

  par = r.get_transform_param_defaults()

  for i in xrange(N):
    r.transform("c:/dev/registration/01.png", "c:/dev/registration/04.png", "c:/dev/registration/reg%.3d/" % (i+1), "transformed.png", "c:/dev/registration/reg%.3d/transform_complete.txt" % (i+1), par)

  r.run("Transformation")

  par  = r.get_transform_param_defaults()
  par["16bit"] = "0"

  for i in xrange(N):
    r.transform_labels("c:/dev/registration/01_labels.png", "c:/dev/registration/04_labels.png", "c:/dev/registration/reg%.3d/" % (i+1), "transformed_labels.png", "c:/dev/registration/reg%.3d/transform_complete.txt" % (i+1), par)

  r.run("Transformation of Labels")

  for i in xrange(N):
    r.label_overlap("c:/dev/registration/01_labels.png", "c:/dev/registration/reg%.3d/transformed_labels.png" % (i+1), "c:/dev/registration/reg%.3d/" % (i+1), "label_overlap1", False)

  r.run("Label overlap Multi")

  for i in xrange(N):
    r.label_overlap("c:/dev/registration/01_labels.png", "c:/dev/registration/reg%.3d/transformed_labels.png" % (i+1), "c:/dev/registration/reg%.3d/" % (i+1), "label_overlap_binary", True)

  r.run("Label overlap Binary")

  transform_01_path = "c:/dev/registration/reg001/transform_complete.txt"
  r.landmark_transform("C:/cygwin64/home/johan/itkAlphaCut/assets/landmarks.csv", "C:/cygwin64/home/johan/itkAlphaCut/assets/", "landmarks_04.csv", transform_01_path)

  r.run("Landmark Transformation")