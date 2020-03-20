
#include <thread>

#include "../common/itkImageProcessingTools.h"

#include "itkLabelOverlapMeasuresImageFilter.h"
#include "itkHausdorffDistanceImageFilter.h"
#include "itkMersenneTwisterRandomVariateGenerator.h"
#include "../bsplineFunctions.h"
#include "itkNearestNeighborInterpolateImageFunction.h"
#include "itkBSplineInterpolateImageFunction.h"

#include "itkTimeProbesCollectorBase.h"
#include "itkMemoryProbesCollectorBase.h"

#include <chrono>

#include "itkVersion.h"
#include "itkMultiThreaderBase.h"

#include "itkPNGImageIOFactory.h"
#include "itkNiftiImageIOFactory.h"

#include "../common/progress.h"

void RegisterIOFactories() {
    itk::PNGImageIOFactory::RegisterOneFactory();
    itk::NiftiImageIOFactory::RegisterOneFactory();
}

struct ProgramConfig {
    std::string affineConfigPath;
    std::string configPath;
    unsigned int seed;
    unsigned int workers;
    std::string refImagePath;
    std::string floImagePath;
    std::string refMaskPath;
    std::string floMaskPath;
    std::string outPathForward;
    std::string outPathReverse;
};

/**
 * Command line arguments
 * binpath dim cfgPath refImagePath floImagePath outPathForward outPathReverse (-refmask refMaskPath) (-flomaskpath floMaskPath) (-seed 1337) (-workers 6)
 */

void readKeyValuePairForProgramConfig(int argc, char** argv, int startIndex, ProgramConfig& cfg) {
    //if (startIndex + 1 < argc) {
    assert(startIndex + 1 < argc);

        std::string key = argv[startIndex];
        std::string value = argv[startIndex + 1];

        if (key == "-refmask") {
            cfg.refMaskPath = value;
        } else if (key == "-flomask") {
            cfg.floMaskPath = value;
        } else if (key == "-seed") {
            cfg.seed = atoi(value.c_str());
        } else if (key == "-workers") {
            cfg.seed = atoi(value.c_str());
        }
    //}
}
ProgramConfig readProgramConfigFromArgs(int argc, char** argv) {
    ProgramConfig res;

    res.affineConfigPath = argv[2];
    res.configPath = argv[3];
    res.refImagePath = argv[4];
    res.floImagePath = argv[5];
    res.outPathForward = argv[6];
    res.outPathReverse = argv[7];

    // Defaults for optional parameters
    res.seed = 1337;
    res.workers = 6;
    res.refMaskPath = "";
    res.floMaskPath = "";
    
    for (int i = 8; i+1 < argc; ++i) {
        readKeyValuePairForProgramConfig(argc, argv, i, res);
    }

    return res;
}

template <unsigned int ImageDimension>
class RegisterDeformableProgram
{
    public:

    typedef itk::IPT<double, ImageDimension> IPT;

    typedef itk::Image<double, ImageDimension> ImageType;
    typedef typename ImageType::Pointer ImagePointer;
    typedef itk::Image<bool, ImageDimension> MaskType;
    typedef typename MaskType::Pointer MaskPointer;

    typedef BSplines<ImageDimension> BSplineFunc;

    static void Run(
        ProgramConfig cfg,
        BSplineRegParamOuter& affineParams,
        BSplineRegParamOuter& params)
    {

        BSplineFunc bsf;

        typedef typename BSplineFunc::TransformType TransformType;
        typedef typename BSplineFunc::TransformType::Pointer TransformPointer;
	
        ImagePointer refImage;
        try {
            refImage = IPT::LoadImage(cfg.refImagePath.c_str());
        }
        catch (itk::ExceptionObject & err)
	    {
            std::cerr << "Error loading reference image: " << cfg.refImagePath.c_str() << std::endl;
	        std::cerr << "ExceptionObject caught !" << std::endl;
		    std::cerr << err << std::endl;
		}
            
        ImagePointer floImage;
        try {
            floImage = IPT::LoadImage(cfg.floImagePath.c_str());
        }
        catch (itk::ExceptionObject & err)
		{
            std::cerr << "Error loading floating image: " << cfg.floImagePath.c_str() << std::endl;
		    std::cerr << "ExceptionObject caught !" << std::endl;
		    std::cerr << err << std::endl;
	    }

        ImagePointer refImageMask;
        if (cfg.refMaskPath != "") {
            try {
                refImageMask = IPT::LoadImage(cfg.refMaskPath.c_str());
            }
            catch (itk::ExceptionObject & err)
            {
                std::cerr << "Error loading reference mask: " << cfg.refMaskPath.c_str() << std::endl;
	            std::cerr << "ExceptionObject caught !" << std::endl;
	            std::cerr << err << std::endl;
	        }
        }
        ImagePointer floImageMask;
        if (cfg.floMaskPath != "") {
            try {
                floImageMask = IPT::LoadImage(cfg.floMaskPath.c_str());
            }
            catch (itk::ExceptionObject & err)
            {
                std::cerr << "Error loading reference mask: " << cfg.floMaskPath.c_str() << std::endl;
	            std::cerr << "ExceptionObject caught !" << std::endl;
	            std::cerr << err << std::endl;
	        }
        }

        TransformPointer forwardTransform;
        TransformPointer inverseTransform;

        std::cerr << "Starting registration" << std::endl;

        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

        bsf.bspline_register(
            refImage,
            floImage,
            affineParams,
            params,
            refImageMask,
            floImageMask,
            forwardTransform,
            inverseTransform,
            nullptr);

        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(end - begin).count();

        // Remove the mask images (if they exist)
        refImageMask = nullptr;
        floImageMask = nullptr;

        if(cfg.outPathForward.length() > 260) {
            std::cerr << "Error. Too long forward transform path." << std::endl;
            exit(-1);
        }

        std::cerr << forwardTransform << std::endl;
        try {
            IPT::SaveTransformFile(cfg.outPathForward.c_str(), forwardTransform.GetPointer());
        }
        catch (itk::ExceptionObject & err)
        {
            std::cerr << "Error saving forward transformation file." << std::endl;
	        std::cerr << err << std::endl;
		}

        try {
            IPT::SaveTransformFile(cfg.outPathReverse.c_str(), inverseTransform.GetPointer());
        }
        catch (itk::ExceptionObject & err)
        {
            std::cerr << "Error saving reverse transformation file." << std::endl;
	        std::cerr << err << std::endl;
		}

        std::cout << "(Registration) Time elapsed: " << elapsed << "[s]" << std::endl;
    }

    static int MainFunc(int argc, char** argv) {
        RegisterIOFactories();

        itk::TimeProbesCollectorBase chronometer;
        itk::MemoryProbesCollectorBase memorymeter;

        std::cout << "--- RegisterDeformable ---" << std::endl;

        chronometer.Start("Registration");
        memorymeter.Start("Registration");

        ProgramConfig config = readProgramConfigFromArgs(argc, argv);
        std::cout << "Program config read..." << std::endl;
        BSplineRegParamOuter affineParams = readConfig(config.affineConfigPath);
        std::cout << "Affine registration config read..." << std::endl;
        BSplineRegParamOuter params = readConfig(config.configPath);
        std::cout << "Deformable Registration config read..." << std::endl;

        // Threading
        itk::MultiThreaderBase::SetGlobalMaximumNumberOfThreads(config.workers);
        itk::MultiThreaderBase::SetGlobalDefaultNumberOfThreads(config.workers);

        Run(config, affineParams, params);

        chronometer.Stop("Registration");
        memorymeter.Stop("Registration");

        //chronometer.Report(std::cout);
        //memorymeter.Report(std::cout);

        return 0;
    }
};

int main(int argc, char** argv) {
    if(argc < 2) {
        std::cout << "No arguments..." << std::endl;
    }
    int ndim = atoi(argv[1]);
    if(ndim == 2) {
        RegisterDeformableProgram<2U>::MainFunc(argc, argv);
    } else if(ndim == 3) {
        RegisterDeformableProgram<3U>::MainFunc(argc, argv);
    } else {
        std::cout << "Error: Dimensionality " << ndim << " is not supported." << std::endl;
    }
}
