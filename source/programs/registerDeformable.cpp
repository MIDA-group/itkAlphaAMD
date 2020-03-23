
#include <thread>
#include <fstream>
#include <iostream>

#include "../common/itkImageProcessingTools.h"

#include "itkLabelOverlapMeasuresImageFilter.h"
#include "itkHausdorffDistanceImageFilter.h"
#include "itkMersenneTwisterRandomVariateGenerator.h"
#include "../alphaAMDDeformableRegistration.h"
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
    bool rerun;
    std::string refImagePath;
    std::string floImagePath;
    std::string refMaskPath;
    std::string floMaskPath;
    std::string outPathAffineForward;
    std::string outPathAffineReverse;
    std::string outPathForward;
    std::string outPathReverse;
};

/**
 * Command line arguments
 * binpath dim cfgPath refImagePath floImagePath outPathForward outPathReverse (-refmask refMaskPath) (-flomaskpath floMaskPath) (-seed 1337) (-workers 6)
 */

void readKeyValuePairForProgramConfig(int argc, char** argv, int startIndex, ProgramConfig& cfg) {
    assert(startIndex + 1 < argc);

    std::string key = argv[startIndex];
    std::string value = argv[startIndex + 1];

    if (key == "-ref") {
        cfg.refImagePath = value;
    } else if (key == "-flo") {
        cfg.floImagePath = value;
    } else if (key == "-affine_cfg") {
        cfg.affineConfigPath = value;
    } else if (key == "-deform_cfg") {
        cfg.configPath = value;
    } else if (key == "-ref_mask") {
        cfg.refMaskPath = value;
    } else if (key == "-flo_mask") {
        cfg.floMaskPath = value;
    } else if (key == "-seed") {
        cfg.seed = atoi(value.c_str());
    } else if (key == "-workers") {
        cfg.workers = atoi(value.c_str());
    } else if (key == "-rerun") {
        cfg.rerun = atoi(value.c_str()) != 0;
    } else if (key == "-out_path_affine_forward") {
        cfg.outPathAffineForward = value;
    } else if (key == "-out_path_affine_reverse") {
        cfg.outPathAffineReverse = value;
    } else if (key == "-out_path_deform_forward") {
        cfg.outPathForward = value;
    } else if (key == "-out_path_deform_reverse") {
        cfg.outPathReverse = value;
    }
}

ProgramConfig readProgramConfigFromArgs(int argc, char** argv) {
    ProgramConfig res;

    //res.affineConfigPath = argv[2];
    //res.configPath = argv[3];
    //res.refImagePath = argv[4];
    //res.floImagePath = argv[5];

    // Defaults for optional parameters
    res.seed = 1337;
    res.workers = 6;
    res.rerun = true;
    res.refMaskPath = "";
    res.floMaskPath = "";
    
    for (int i = 2; i+1 < argc; ++i) {
        readKeyValuePairForProgramConfig(argc, argv, i, res);
    }

    return res;
}

bool checkFile(std::string path)
{
    std::ifstream file(path.c_str());
    if (!file)
    {
        return false;
    }
    else
    {
        return true;
    }
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

    static bool Run(
        ProgramConfig cfg,
        BSplineRegParamOuter& affineParams,
        BSplineRegParamOuter& params)
    {
        bool success = true;

        if (!cfg.rerun)
        {
            bool allfine = true;

            if (!allfine || (cfg.outPathAffineForward != "" && !checkFile(cfg.outPathAffineForward)))
            {
                allfine = false;
            }
            if (!allfine || (cfg.outPathAffineReverse != "" && !checkFile(cfg.outPathAffineReverse)))
            {
                allfine = false;
            }
            if (!allfine || (cfg.outPathForward != "" && !checkFile(cfg.outPathForward)))
            {
                allfine = false;
            }
            if (!allfine || (cfg.outPathReverse != "" && !checkFile(cfg.outPathReverse)))
            {
                allfine = false;
            }

            if (allfine)
            {
                return success;
            }
        }

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
            success = false;
            return success;
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
            success = false;
            return success;
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
                success = false;
                return success;
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
                success = false;
                return success;
	        }
        }

        using CompositeTranformType = itk::CompositeTransform<double, ImageDimension>;
        using CompositeTranformPointer = typename CompositeTranformType::Pointer;

        CompositeTranformPointer affineTransform;
        TransformPointer forwardTransform;
        TransformPointer inverseTransform;

        std::cerr << "Starting registration" << std::endl;

        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

        bsf.register_deformable(
            refImage,
            floImage,
            affineParams,
            params,
            refImageMask,
            floImageMask,
            affineTransform,
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
            success = false;
            return success;
        }

        if(cfg.outPathAffineForward != "") {
            try {
                IPT::SaveTransformFile(cfg.outPathAffineForward.c_str(), affineTransform.GetPointer());
            }
            catch (itk::ExceptionObject & err)
            {
                std::cerr << "Error saving forward affine transformation file." << std::endl;
	            std::cerr << err << std::endl;
                success = false;
	    	}
        }

        if(cfg.outPathAffineReverse != "") {
            try {
                auto inverse = affineTransform->GetInverseTransform();
                if (inverse) {
                    IPT::SaveTransformFile(cfg.outPathAffineReverse.c_str(), affineTransform->GetInverseTransform().GetPointer());
                } else {
                    std::cerr << "Error saving reverse affine transformation file. The transformation is not invertible." << std::endl;
                    success = false;
                }
            }
            catch (itk::ExceptionObject & err)
            {
                std::cerr << "Error saving reverse affine transformation file." << std::endl;
	            std::cerr << err << std::endl;
                success = false;
	    	}
        }

        try {
            IPT::SaveTransformFile(cfg.outPathForward.c_str(), forwardTransform.GetPointer());
        }
        catch (itk::ExceptionObject & err)
        {
            std::cerr << "Error saving forward transformation file." << std::endl;
	        std::cerr << err << std::endl;
            success = false;
		}

        try {
            IPT::SaveTransformFile(cfg.outPathReverse.c_str(), inverseTransform.GetPointer());
        }
        catch (itk::ExceptionObject & err)
        {
            std::cerr << "Error saving reverse transformation file." << std::endl;
	        std::cerr << err << std::endl;
            success = false;
		}

        std::cout << "(Registration) Time elapsed: " << elapsed << "[s]" << std::endl;
        
        return success;
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

        bool success = Run(config, affineParams, params);

        chronometer.Stop("Registration");
        memorymeter.Stop("Registration");

        //chronometer.Report(std::cout);
        //memorymeter.Report(std::cout);

        return (success ? 0 : -1);
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
