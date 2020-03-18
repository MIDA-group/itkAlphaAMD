
#include <thread>

#include "./common/itkImageProcessingTools.h"

#include "itkLabelOverlapMeasuresImageFilter.h"
#include "itkHausdorffDistanceImageFilter.h"
#include "itkMersenneTwisterRandomVariateGenerator.h"
#include "bsplineFunctions.h"
#include "itkNearestNeighborInterpolateImageFunction.h"
#include "itkBSplineInterpolateImageFunction.h"

#include "itkTimeProbesCollectorBase.h"
#include "itkMemoryProbesCollectorBase.h"

#include <chrono>

#include "itkVersion.h"
#include "itkMultiThreaderBase.h"

#include "itkPNGImageIOFactory.h"
#include "itkNiftiImageIOFactory.h"

#include "common/progress.h"

void RegisterIOFactories() {
    itk::PNGImageIOFactory::RegisterOneFactory();
    itk::NiftiImageIOFactory::RegisterOneFactory();
}

struct EvaluationConfig {
    unsigned int seed;
    unsigned int threads;
    unsigned int workers;
    std::vector<std::string> images;
    std::vector<std::string> labels;
    unsigned int startIndex;
    unsigned int endIndex;
    std::string transformsPath;
};

EvaluationConfig readEvaluationConfig(std::string path) {
    json jc = readJSON(path);

    EvaluationConfig c;

    c.threads = (unsigned int)jc["threads"];
    c.workers = (unsigned int)jc["workers"];
    
    c.transformsPath = "./transforms/";
    readJSONKey(jc, "transformsPath", &c.transformsPath);
    c.startIndex = (unsigned int)jc["startIndex"];
    c.endIndex = (unsigned int)jc["endIndex"];
    for(unsigned int i = 0; i< jc["images"].size(); ++i)
        c.images.push_back(jc["images"][i]);
    for(unsigned int i = 0; i< jc["labels"].size(); ++i)
        c.labels.push_back(jc["labels"][i]);

    return c;
}

struct PerformanceMetrics {
    unsigned int refIndex;
    unsigned int floIndex;
    double totalOverlap;
    double meanTotalOverlap;
    double absDiff;
};

PerformanceMetrics MakePerformanceMetrics(unsigned int refIndex, unsigned int floIndex) {
    PerformanceMetrics pm;
    pm.refIndex = refIndex;
    pm.floIndex = floIndex;
    return pm;
}

struct EvalThread {
    std::vector<PerformanceMetrics> beforePerf;
    std::vector<PerformanceMetrics> afterPerf;
    unsigned int threadID;
    unsigned int startIndex;
    unsigned int endIndex;
    Progress* progress;
};

template <typename TStream>
void printMetrics(TStream& strm, PerformanceMetrics m, std::string name, bool linebreak=true) {
    strm << name << "(totalOverlap: " << m.totalOverlap << ", meanTotalOverlap: " << m.meanTotalOverlap << ", absdiff: " << m.absDiff << ").";
    if(linebreak)
        strm << std::endl;
}

void SaveMetrics(const char* path, const std::vector<PerformanceMetrics>& m) {
    FILE *f = fopen(path, "wb");

    // Write refIndex
    for (size_t i = 0; i < m.size(); ++i)
    {
        if (i > 0)
            fprintf(f, ",");

        fprintf(f, "%d", m[i].refIndex);
    }
    fprintf(f, "\n");

    // Write floIndex
    for (size_t i = 0; i < m.size(); ++i)
    {
        if (i > 0)
            fprintf(f, ",");

        fprintf(f, "%d", m[i].floIndex);
    }
    fprintf(f, "\n");

    // Write totalOverlap
    for (size_t i = 0; i < m.size(); ++i)
    {
        if (i > 0)
            fprintf(f, ",");

        fprintf(f, "%.7f", m[i].totalOverlap);
    }
    fprintf(f, "\n");

    // Write meanTotalOverlap
    for (size_t i = 0; i < m.size(); ++i)
    {
        if (i > 0)
            fprintf(f, ",");

        fprintf(f, "%.7f", m[i].meanTotalOverlap);
    }
    fprintf(f, "\n");

    // Write absdiff
    for (size_t i = 0; i < m.size(); ++i)
    {
        if (i > 0)
            fprintf(f, ",");

        fprintf(f, "%.7f", m[i].absDiff);
    }
    fprintf(f, "\n");

    fclose(f);
}

    template <typename TImageType, typename TTransformType>
    typename TImageType::Pointer ApplyTransformToImage(
        typename TImageType::Pointer refImage,
        typename TImageType::Pointer floImage,
        typename TTransformType::Pointer transform,
        int interpolator = 1,
        typename TImageType::PixelType defaultValue = 0)
    {
        if(!transform)
        {
            return floImage;
        }

        typedef itk::ResampleImageFilter<
            TImageType,
            TImageType>
            ResampleFilterType;

        typename ResampleFilterType::Pointer resample = ResampleFilterType::New();

        resample->SetTransform(transform);
        resample->SetInput(floImage);

        // Linear interpolator (1) is the default
        if (interpolator == 0)
        {
            auto interp = itk::NearestNeighborInterpolateImageFunction<TImageType, double>::New();
            resample->SetInterpolator(interp);
        }
        else if (interpolator == 2)
        {
            auto interp = itk::BSplineInterpolateImageFunction<TImageType, double>::New();//itk::BSplineInterpolationWeightFunction<double, ImageDimension, 3U>::New();
            resample->SetInterpolator(interp);
        }

        resample->SetSize(refImage->GetLargestPossibleRegion().GetSize());
        resample->SetOutputOrigin(refImage->GetOrigin());
        resample->SetOutputSpacing(refImage->GetSpacing());
        resample->SetOutputDirection(refImage->GetDirection());
        resample->SetDefaultPixelValue(defaultValue);

        resample->UpdateLargestPossibleRegion();

        return resample->GetOutput();
    }

template <typename ImageType>
double MeanAbsDifference(
    typename ImageType::Pointer image1,
    typename ImageType::Pointer image2) {
    
    using ImagePointer = typename ImageType::Pointer;
    using ValueType = typename ImageType::ValueType;
    typedef itk::IPT<ValueType, ImageType::ImageDimension> IPT;

    ImagePointer diffImage = IPT::DifferenceImage(image1, image2);
    typename IPT::ImageStatisticsData stats = IPT::ImageStatistics(diffImage);
    return stats.mean;
}

template <typename TImageType>
void LabelAccuracy(
    typename TImageType::Pointer image1,
    typename TImageType::Pointer image2,
    double& totalOverlap,
    double& meanTotalOverlap) {
        
        typedef itk::LabelOverlapMeasuresImageFilter<TImageType> FilterType;
        typedef typename FilterType::Pointer FilterPointer;

        FilterPointer filter = FilterType::New();

        filter->SetSourceImage(image1);
        filter->SetTargetImage(image2);

        filter->Update();

        totalOverlap = filter->GetTotalOverlap();

        typedef typename FilterType::MapType MapType;
        typedef typename FilterType::MapConstIterator MapConstIterator;
    
        double overlapAcc = 0.0;
        unsigned int overlapCount = 0;

        MapType map = filter->GetLabelSetMeasures();
        for (MapConstIterator mapIt = map.begin(); mapIt != map.end(); ++mapIt) {
            // Do not include the background in the final value.
            if ((*mapIt).first == 0)
            {
                continue;
            }
            double numerator = (double)(*mapIt).second.m_Intersection;
            double denominator = (double)(*mapIt).second.m_Target;
            if(denominator > 0) {
                overlapAcc += (numerator/denominator);
                ++overlapCount;
            }
        }

        meanTotalOverlap = overlapAcc / overlapCount;
}

template <typename TransformType, typename ImageType, typename LabelImageType>
void EvaluateRegistration(
    typename ImageType::Pointer refImage,
    typename ImageType::Pointer floImage,
    typename LabelImageType::Pointer refImageLabel,
    typename LabelImageType::Pointer floImageLabel,
    typename TransformType::Pointer transformForward,
    typename TransformType::Pointer transformReverse,
    PerformanceMetrics& refToFloMetrics,
    PerformanceMetrics& floToRefMetrics
    )
{
    using ImagePointer = typename ImageType::Pointer;
    using LabelImagePointer = typename LabelImageType::Pointer;

    ImagePointer floToRefRegisteredImage = ApplyTransformToImage<ImageType, TransformType>(refImage, floImage, transformForward, 1, 0.0);
    ImagePointer refToFloRegisteredImage = ApplyTransformToImage<ImageType, TransformType>(floImage, refImage, transformReverse, 1, 0.0);

    // Compute the before mean absolute difference
    refToFloMetrics.absDiff = MeanAbsDifference<ImageType>(refImage, floToRefRegisteredImage);
    floToRefMetrics.absDiff = MeanAbsDifference<ImageType>(floImage, refToFloRegisteredImage);

    if(refImageLabel && floImageLabel)
    {       
        LabelImagePointer floToRefRegisteredLabel = ApplyTransformToImage<LabelImageType, TransformType>(refImageLabel, floImageLabel, transformForward, 0, 0);
        LabelImagePointer refToFloRegisteredLabel = ApplyTransformToImage<LabelImageType, TransformType>(floImageLabel, refImageLabel, transformReverse, 0, 0);

        LabelAccuracy<LabelImageType>(floToRefRegisteredLabel, refImageLabel, refToFloMetrics.totalOverlap, refToFloMetrics.meanTotalOverlap);
        LabelAccuracy<LabelImageType>(refToFloRegisteredLabel, floImageLabel, floToRefMetrics.totalOverlap, floToRefMetrics.meanTotalOverlap);
    } else
    {
        refToFloMetrics.totalOverlap = 0.0;
        refToFloMetrics.meanTotalOverlap = 0.0;
        floToRefMetrics.totalOverlap = 0.0;
        floToRefMetrics.meanTotalOverlap = 0.0;
    }
}


template <typename TTransformType, typename TImageType, unsigned int TImageDimension>
class BSplineRegistrationCallbackSegmentationOverlap : public BSplineRegistrationCallback<TTransformType, TImageDimension>
{
    public:

    using Superclass = BSplineRegistrationCallback<TTransformType, TImageDimension>;
    using TransformType = TTransformType;
    using TransformPointer = typename TransformType::Pointer;

    constexpr static unsigned int ImageDimension = TImageDimension;        

    typedef itk::IPT<double, ImageDimension> IPT;

    //typedef itk::Image<double, ImageDimension> ImageType;
    using ImageType = TImageType;
    typedef typename ImageType::Pointer ImagePointer;
    typedef itk::Image<unsigned short, ImageDimension> LabelImageType;
    typedef typename LabelImageType::Pointer LabelImagePointer;
    typedef itk::Image<bool, ImageDimension> MaskType;
    typedef typename MaskType::Pointer MaskPointer;

    std::vector<PerformanceMetrics> refToFloPerformances;
    std::vector<PerformanceMetrics> floToRefPerformances;

    ImagePointer refImage;
    ImagePointer floImage;
    LabelImagePointer refImageLabel;
    LabelImagePointer floImageLabel;

    unsigned int refIndex;
    unsigned int floIndex;

    Progress* progress;

    virtual void Invoke()
    {
        PerformanceMetrics refToFloPM = MakePerformanceMetrics(refIndex, floIndex);
        PerformanceMetrics floToRefPM = MakePerformanceMetrics(floIndex, refIndex);

        EvaluateRegistration<TransformType, ImageType, LabelImageType>(
            refImage,
            floImage,
            refImageLabel,
            floImageLabel,
            Superclass::m_TransformForward,
            Superclass::m_TransformReverse,
            refToFloPM,
            floToRefPM);

        refToFloPerformances.push_back(refToFloPM);
        floToRefPerformances.push_back(floToRefPM);

        progress->StartProgressUpdate();

        char str[512];
        sprintf(str, "R%dF%d: ", refIndex, floIndex);
        printMetrics(std::cout, refToFloPM, str);
        sprintf(str, "R%dF%d: ", floIndex, refIndex);
        printMetrics(std::cout, floToRefPM, str);

        progress->EndProgressUpdate();
    }
};

template <unsigned int ImageDimension>
class PWEvalDeformable
{
    public:

    typedef itk::IPT<double, ImageDimension> IPT;

    typedef itk::Image<double, ImageDimension> ImageType;
    typedef typename ImageType::Pointer ImagePointer;
    typedef itk::Image<unsigned short, ImageDimension> LabelImageType;
    typedef typename LabelImageType::Pointer LabelImagePointer;
    typedef itk::Image<bool, ImageDimension> MaskType;
    typedef typename MaskType::Pointer MaskPointer;

    typedef BSplines<ImageDimension> BSplineFunc;

    static ImagePointer Chessboard(ImagePointer image1, ImagePointer image2, int cells)
    {
        itk::FixedArray<unsigned int, ImageDimension> pattern;
        pattern.Fill(cells);

        typedef itk::CheckerBoardImageFilter<ImageType> CheckerBoardFilterType;
        typename CheckerBoardFilterType::Pointer checkerBoardFilter = CheckerBoardFilterType::New();
        checkerBoardFilter->SetInput1(image1);
        checkerBoardFilter->SetInput2(image2);
        checkerBoardFilter->SetCheckerPattern(pattern);
        checkerBoardFilter->Update();
        return checkerBoardFilter->GetOutput();
    }

    static ImagePointer BlackAndWhiteChessboard(ImagePointer refImage, int cells)
    {
        return Chessboard(IPT::ZeroImage(refImage->GetLargestPossibleRegion().GetSize()), IPT::ConstantImage(1.0, refImage->GetLargestPossibleRegion().GetSize()), cells);
    }

    static void Evaluate(
        const std::vector<std::string>& images,
        const std::vector<std::string>& labels,
        std::vector<unsigned int>& refIndices,
        std::vector<unsigned int>& floIndices,
        std::string transformsPath,
        BSplineRegParamOuter& params,
        EvalThread& t)
    {

        BSplineFunc bsf;

        auto prog = t.progress;

        typedef typename BSplineFunc::TransformType TransformType;
        typedef typename BSplineFunc::TransformType::Pointer TransformPointer;
	
        unsigned int startIndex = t.startIndex;
        unsigned int endIndex = t.endIndex;

        for(unsigned int i = startIndex; i < endIndex; ++i) {
            unsigned int refIndex = refIndices[i];
            unsigned int floIndex = floIndices[i];

            ImagePointer refImage;
            try {
                refImage = IPT::LoadImage(images[refIndex].c_str());
            }
            catch (itk::ExceptionObject & err)
		    {
                std::cerr << "Error loading reference image: " << images[refIndex].c_str() << std::endl;
			    std::cerr << "ExceptionObject caught !" << std::endl;
			    std::cerr << err << std::endl;
		    }
            
            ImagePointer floImage;
            try {
                floImage = IPT::LoadImage(images[floIndex].c_str());
            }
            catch (itk::ExceptionObject & err)
		    {
                std::cerr << "Error loading floating image: " << images[floIndex].c_str() << std::endl;
			    std::cerr << "ExceptionObject caught !" << std::endl;
			    std::cerr << err << std::endl;
		    }

            // Load the label images
            LabelImagePointer refImageLabel = IPT::LoadLabelImage(labels[refIndex].c_str());
            LabelImagePointer floImageLabel = IPT::LoadLabelImage(labels[floIndex].c_str());

            ImagePointer refImageMask;
            ImagePointer floImageMask;
            //ImagePointer refImageMask = IPT::ConstantImage(1.0, refImage->GetLargestPossibleRegion().GetSize());
            //ImagePointer floImageMask = IPT::ConstantImage(1.0, floImage->GetLargestPossibleRegion().GetSize());

            TransformPointer forwardTransform;
            TransformPointer inverseTransform;

            std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

            BSplineRegistrationCallbackSegmentationOverlap<TransformType, ImageType, ImageType::ImageDimension> callback;

            callback.refIndex = refIndex;
            callback.floIndex = floIndex;
            callback.refImage = refImage;
            callback.floImage = floImage;
            callback.refImageLabel = refImageLabel;
            callback.floImageLabel = floImageLabel;
            callback.progress = prog;

            std::cout << "Registration to commence." << std::endl;

            bsf.bspline_register(
                refImage,
                floImage,
                params,
                refImageMask,
                floImageMask,
                //false,
                forwardTransform,
                inverseTransform,
                &callback);

            // Registered image

            {

                auto registeredImage = ApplyTransformToImage<ImageType, TransformType>(
                    refImage,
                    floImage,
                    forwardTransform,
                    1,
                    0);

                auto registeredCBImage = ApplyTransformToImage<ImageType, TransformType>(
                    refImage,
                    BlackAndWhiteChessboard(floImage, 512),
                    forwardTransform,
                    1,
                    0);

                char sbuf[512];
                sprintf(sbuf, "./out_%d_%d.png", refIndex, floIndex);
                IPT::SaveImageU8(sbuf, registeredImage);

                sprintf(sbuf, "./cb_%d_%d.png", refIndex, floIndex);
                IPT::SaveImageU8(sbuf, registeredCBImage);

            }

            std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(end - begin).count();

            // Remove the mask images
            refImageMask = nullptr;
            floImageMask = nullptr;

            if(transformsPath.length() > 260) {
                std::cerr << "Error. Too long transform path." << std::endl;
                exit(-1);
            }

            char fwdTransformPath[512];
            sprintf(fwdTransformPath, "%s/t%d_%d.txt", transformsPath.c_str(), refIndex, floIndex);
            char revTransformPath[512];
            sprintf(revTransformPath, "%s/t%d_%d.txt", transformsPath.c_str(), floIndex, refIndex);
            try {
                IPT::SaveTransformFile(fwdTransformPath, forwardTransform.GetPointer());
                IPT::SaveTransformFile(revTransformPath, inverseTransform.GetPointer());
            }
            catch (itk::ExceptionObject & err)
		    {
                std::cerr << "Error saving transformation files." << std::endl;
			    std::cerr << err << std::endl;
		    }

            // Initialize performance metrics structures
            PerformanceMetrics beforeFwdPM = MakePerformanceMetrics(refIndex, floIndex);
            PerformanceMetrics afterFwdPM = MakePerformanceMetrics(refIndex, floIndex);
            PerformanceMetrics beforeRevPM = MakePerformanceMetrics(floIndex, refIndex);
            PerformanceMetrics afterRevPM = MakePerformanceMetrics(floIndex, refIndex);

            EvaluateRegistration<TransformType, ImageType, LabelImageType>(refImage, floImage, refImageLabel, floImageLabel, nullptr, nullptr, beforeFwdPM, beforeRevPM);
            EvaluateRegistration<TransformType, ImageType, LabelImageType>(refImage, floImage, refImageLabel, floImageLabel, forwardTransform, inverseTransform, afterFwdPM, afterRevPM);
            
            prog->StartProgressUpdate();

            std::cout << "(Registration) Time elapsed: " << elapsed << "[s]" << std::endl;

/*
            // Apply transformations to intensity images and to labels
            ImagePointer floToRefRegisteredImage = ApplyTransform<ImageType, TransformType>(refImage, floImage, forwardTransform, 1, 0.0);
            ImagePointer refToFloRegisteredImage = ApplyTransform<ImageType, TransformType>(floImage, refImage, inverseTransform, 1, 0.0);
            LabelImagePointer floToRefRegisteredLabel = ApplyTransform<LabelImageType, TransformType>(refImageLabel, floImageLabel, forwardTransform, 0, 0);
            LabelImagePointer refToFloRegisteredLabel = ApplyTransform<LabelImageType, TransformType>(floImageLabel, refImageLabel, inverseTransform, 0, 0);

            // Compute the before mean absolute difference
            beforeFwdPM.absDiff = MeanAbsDifference(refImage, floImage);
            beforeRevPM.absDiff = beforeFwdPM.absDiff;

            LabelAccuracy<LabelImageType>(floImageLabel, refImageLabel, beforeFwdPM.totalOverlap, beforeFwdPM.meanTotalOverlap);
            LabelAccuracy<LabelImageType>(refImageLabel, floImageLabel, beforeRevPM.totalOverlap, beforeRevPM.meanTotalOverlap);

            afterFwdPM.absDiff = MeanAbsDifference(refImage, floToRefRegisteredImage);
            afterRevPM.absDiff = MeanAbsDifference(floImage, refToFloRegisteredImage);

            LabelAccuracy<LabelImageType>(floToRefRegisteredLabel, refImageLabel, afterFwdPM.totalOverlap, afterFwdPM.meanTotalOverlap);
            LabelAccuracy<LabelImageType>(refToFloRegisteredLabel, floImageLabel, afterRevPM.totalOverlap, afterRevPM.meanTotalOverlap);
*/
            t.beforePerf.push_back(beforeFwdPM);
            t.afterPerf.push_back(afterFwdPM);
            t.beforePerf.push_back(beforeRevPM);
            t.afterPerf.push_back(afterRevPM);

            char str[512];
            sprintf(str, "R%dF%d: [Before]", refIndex, floIndex);
            printMetrics(std::cout, beforeFwdPM, str);
            sprintf(str, "R%dF%d: [After]", refIndex, floIndex);
            printMetrics(std::cout, afterFwdPM, str);
            sprintf(str, "R%dF%d: [Before]", floIndex, refIndex);
            printMetrics(std::cout, beforeRevPM, str);
            sprintf(str, "R%dF%d: [After]", floIndex, refIndex);
            printMetrics(std::cout, afterRevPM, str);

            prog->ReportProgressUpdate(std::cout, false);

            prog->EndProgressUpdate();
        }
    }

    static int MainFunc(int argc, char** argv) {
        constexpr int threadCount = 32;

        RegisterIOFactories();

        itk::TimeProbesCollectorBase chronometer;
        itk::MemoryProbesCollectorBase memorymeter;

        std::cout << "--- PWEvalDeformableND ---" << std::endl;

        chronometer.Start("Evaluation");
        memorymeter.Start("Evaluation");

        EvaluationConfig config = readEvaluationConfig(argv[2]);
        std::cout << "Evaluation config read..." << std::endl;
        BSplineRegParamOuter params = readConfig(argv[3]);
        std::cout << "Registration config read..." << std::endl;

        // Threading
        itk::MultiThreaderBase::SetGlobalMaximumNumberOfThreads(config.workers);
        itk::MultiThreaderBase::SetGlobalDefaultNumberOfThreads(config.workers);

        std::vector<unsigned int> refIndices;
        std::vector<unsigned int> floIndices;

        for(unsigned int i = 0; i < config.images.size(); ++i) {
            for(unsigned int j = i+1; j < config.images.size(); ++j) {
                refIndices.push_back(i);
                floIndices.push_back(j);
            }
        }

        Progress prg(refIndices.size(), 100);

        BSplineFunc bspline_func;

        std::vector<PerformanceMetrics> beforePerf;
        std::vector<PerformanceMetrics> afterPerf;
        EvalThread threadData[threadCount];
        std::thread threads[threadCount];

        assert(config.threads <= threadCount);
        assert(config.endIndex >= config.startIndex);

        unsigned int count = config.endIndex - config.startIndex;

        for(unsigned int i = 0; i < config.threads; ++i) {
            double spn = count/(double)config.threads;
            unsigned int start = config.startIndex + (int)(i * spn);
            unsigned int end = config.startIndex + (int)((i+1) * spn);
            if(end > config.startIndex+count)
                end = config.startIndex+count;
            std::cout << "(" << start << " - " << end << ")" << std::endl;
            threadData[i].startIndex = start;
            threadData[i].endIndex = end;

            threadData[i].threadID = i;
            threadData[i].progress = &prg;

            auto fn = [&, i]() -> void { Evaluate(config.images, config.labels, refIndices, floIndices, config.transformsPath, params, threadData[i]); };
            threads[i] = std::thread(fn);
        }

        for(unsigned int i = 0; i < config.threads; ++i) {
            threads[i].join();
            for(size_t j = 0; j < threadData[i].beforePerf.size(); ++j) {
                beforePerf.push_back(threadData[i].beforePerf[j]);
                afterPerf.push_back(threadData[i].afterPerf[j]);
            }
        }

        chronometer.Stop("Evaluation");
        memorymeter.Stop("Evaluation");

        chronometer.Report(std::cout);
        memorymeter.Report(std::cout);

        char str[512];
        sprintf(str, "before_metrics_%d_%d.csv", config.startIndex, config.endIndex);
        SaveMetrics(str, beforePerf);
        sprintf(str, "after_metrics_%d_%d.csv", config.startIndex, config.endIndex);
        SaveMetrics(str, afterPerf);

        return 0;
    }
};

