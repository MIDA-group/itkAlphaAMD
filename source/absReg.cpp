
#include <stdio.h>
#include <string.h>
#include <string>

#include "common/itkImageProcessingTools.h"

#include "itkTimeProbesCollectorBase.h"
#include "itkMemoryProbesCollectorBase.h"

#include "bsplineFunctions.h"

struct ProgramParams {
  std::string ref_im_path;
  std::string flo_im_path;
  std::string ref_mask_path;
  std::string flo_mask_path;
  std::string ref_weights_path;
  std::string flo_weights_path;
  std::string config_path;
  std::string output_path;
};

bool parseParam(ProgramParams& par, std::string key, std::string value) {
  if(key == "-ref")
    par.ref_im_path = value;
  else if(key == "-flo")
    par.flo_im_path = value;
  else if(key == "-ref_mask")
    par.ref_mask_path = value;
  else if(key == "-flo_mask")
    par.flo_mask_path = value;
  else if(key == "-ref_weights")
    par.ref_weights_path = value;
  else if(key == "-flo_weights")
    par.flo_weights_path = value;
  else if(key == "-cfg")
    par.config_path = value;
  else if(key == "-out")
    par.output_path = value;
  else {
    std::cout << "Ignoring invalid key: " << key << ", with value: " << value << std::endl;
    return false;
  }
  return true;
}

bool parseParams(ProgramParams& par, int argc, char** argv) {
  for(unsigned int i = 2; i + 1 < argc; i += 2) {
    std::string key = argv[i];
    std::string value = argv[i+1];
    if(!parseParam(par, key, value))
      return false;
  }
}

template <unsigned int ImageDimension>
class ABSReg {
public:
  typedef itk::Image<double, ImageDimension> ImageType;
  typedef typename ImageType::Pointer ImagePointer;
  
  static void run(ProgramParams& par) {
    typedef itk::IPT<double, ImageDimension> IPT;

    itk::MultiThreader::SetGlobalMaximumNumberOfThreads(1);
    
    ImagePointer refImage = IPT::LoadImage(par.ref_im_path.c_str());
    ImagePointer floImage = IPT::LoadImage(par.flo_im_path.c_str());

    ImagePointer refMask;
    ImagePointer floMask;
    if(par.ref_mask_path.length() == 0) {
      refMask = IPT::ConstantImage(1.0, refImage->GetLargestPossibleRegion().GetSize());
    } else {
      refMask = IPT::LoadImage(par.ref_mask_path.c_str());
    }
    if(par.flo_mask_path.length() == 0) {
      floMask = IPT::ConstantImage(1.0, floImage->GetLargestPossibleRegion().GetSize());
    } else {
      floMask = IPT::LoadImage(par.flo_mask_path.c_str());
    }

    typedef BSplines<ImageDimension> BSplineFunc;
    typedef typename BSplineFunc::TransformType TransformType;
    typedef typename TransformType::Pointer TransformPointer;

    BSplineFunc bsf;

    TransformPointer forwardTransform;
    TransformPointer reverseTransform;

    BSplineRegParamOuter config = readConfig(par.config_path);
    bool verbose = false;

    bsf.bspline_register(refImage, floImage, config, refMask, floMask, verbose, forwardTransform, reverseTransform);

    std::string refToFloTransformPath = par.output_path + "_forward.txt";
    std::string floToRefTransformPath = par.output_path + "_reverse.txt";

    IPT::SaveTransformFile(refToFloTransformPath.c_str(), forwardTransform.GetPointer());
    IPT::SaveTransformFile(floToRefTransformPath.c_str(), reverseTransform.GetPointer());
  }
};

void printInstructions() {
  std::cout << "absreg dim -cfg config_file -ref ref_im -flo flo_im -out output_transform_file_prefix (-ref_mask ref_mask_im -flo_mask flo_mask_im -ref_weights ref_weight_im -flo_weights flo_weight_im)" << std::endl;
}

int main(int argc, char** argv) {
  // Parameters dim config ref_im flo_im output
  if(argc < 2) {
    std::cout << "absreg - Too few parameters." << std::endl;
    printInstructions();
    return -1;
  }
  
  ProgramParams params;
  bool parseStatus = parseParams(params, argc, argv);

  if(parseStatus == false) {
    printInstructions();
    std::cout << "Error. Aborting." << std::endl;
    return -1;
  }
    
  unsigned int dim = atoi(argv[1]);
  if(dim == 2U) {
    ABSReg<2U>::run(params);
  } else if(dim == 3U) {
    ABSReg<3U>::run(params);
  } else {
    std::cout << "Error - absreg only supports 2d and 3d images." << std::endl;
    return -1;
  }

  return 0;
}
