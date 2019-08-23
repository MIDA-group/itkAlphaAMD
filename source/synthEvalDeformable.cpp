
#include <thread>

#include "./common/itkImageProcessingTools.h"

#include "itkLabelOverlapMeasuresImageFilter.h"
#include "itkHausdorffDistanceImageFilter.h"
#include "itkMersenneTwisterRandomVariateGenerator.h"
#include "bsplineFunctions.h"
#include "itkNearestNeighborInterpolateImageFunction.h"
//#include "itkBSplineInterpolationWeightFunction.h"
#include "itkBSplineInterpolateImageFunction.h"

#include "itkTimeProbesCollectorBase.h"
#include "itkMemoryProbesCollectorBase.h"

struct EvaluationConfig {
    unsigned int count;
    double deformationMagnitude;
    unsigned int controlPoints;
    unsigned int seed;
    double noiseSigma;
    std::string metricName;

    std::string imagePath;
    std::string imageMaskPath;
    std::string imageLabelPath;
    std::string methodConfigPath;
};

EvaluationConfig readEvaluationConfig(std::string path) {
    json jc = readJSON(path);

    EvaluationConfig c;

    c.count = (unsigned int)jc["count"];
    c.deformationMagnitude = (double)jc["deformationMagnitude"];
    c.controlPoints = (unsigned int)jc["controlPoints"];
    c.seed = (unsigned int)jc["seed"];
    c.noiseSigma = (double)jc["noiseSigma"];
    c.metricName = jc["metric"];

    c.imagePath = jc["imagePath"];
    c.imageMaskPath = jc["imageMaskPath"];
    c.imageLabelPath = jc["imageLabelPath"];
    c.methodConfigPath = jc["methodConfigPath"];

    return c;
}

struct PerformanceMetrics {
    double accuracy;
    double hausdorff;
    double absDiff;
};

struct EvalThread {
    std::vector<PerformanceMetrics> beforePerf;
    std::vector<PerformanceMetrics> afterPerf;
    unsigned int threadID;
    unsigned int startIndex;
    unsigned int endIndex;
};

void printMetrics(PerformanceMetrics m, std::string name, bool linebreak=true) {
    std::cout << name << "(acc: " << m.accuracy << ", hausdorff: " << m.hausdorff << ", absdiff: " << m.absDiff << ").";
    if(linebreak)
        std::cout << std::endl;
}

PerformanceMetrics meanMetrics(std::vector<PerformanceMetrics>& m) {
    PerformanceMetrics acc;
    acc.accuracy = 0.0;
    acc.hausdorff = 0.0;
    acc.absDiff = 0.0;

    for(size_t i = 0; i < m.size(); ++i) {
        acc.accuracy += m[i].accuracy / m.size();
        acc.hausdorff += m[i].hausdorff / m.size();
        acc.absDiff += m[i].absDiff / m.size();
    }

    return acc;
}

PerformanceMetrics sdMetrics(std::vector<PerformanceMetrics>& m) {
    PerformanceMetrics mn = meanMetrics(m);

    PerformanceMetrics acc;
    acc.accuracy = 0.0;
    acc.hausdorff = 0.0;
    acc.absDiff = 0.0;

    for(size_t i = 0; i < m.size(); ++i) {
        acc.accuracy += pow(m[i].accuracy-mn.accuracy, 2.0);
        acc.hausdorff += pow(m[i].hausdorff-mn.hausdorff, 2.0);
        acc.absDiff += pow(m[i].absDiff-mn.absDiff, 2.0);
    }

    acc.accuracy = sqrt(acc.accuracy / (m.size()-1));
    acc.hausdorff = sqrt(acc.hausdorff / (m.size()-1));
    acc.absDiff = sqrt(acc.absDiff / (m.size()-1));

    return acc;
}

PerformanceMetrics minMetrics(std::vector<PerformanceMetrics>& m) {
    PerformanceMetrics acc;
    acc.accuracy = m[0].accuracy;
    acc.hausdorff = m[0].hausdorff;
    acc.absDiff = m[0].absDiff;

    for(size_t i = 1; i < m.size(); ++i) {
        if(acc.accuracy > m[i].accuracy)
            acc.accuracy = m[i].accuracy;
        if(acc.hausdorff > m[i].hausdorff)
            acc.hausdorff = m[i].hausdorff;
        if(acc.absDiff > m[i].absDiff)
            acc.absDiff = m[i].absDiff;
    }

    return acc;
}

PerformanceMetrics maxMetrics(std::vector<PerformanceMetrics>& m) {
    PerformanceMetrics acc;
    acc.accuracy = 0.0;
    acc.hausdorff = 0.0;
    acc.absDiff = 0.0;

    for(size_t i = 0; i < m.size(); ++i) {
        if(acc.accuracy < m[i].accuracy)
            acc.accuracy = m[i].accuracy;
        if(acc.hausdorff < m[i].hausdorff)
            acc.hausdorff = m[i].hausdorff;
        if(acc.absDiff < m[i].absDiff)
            acc.absDiff = m[i].absDiff;
    }

    return acc;
}

template <unsigned int ImageDimension>
class SynthEvalDeformable
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

    template <typename TransformType>
    static void RandomizeDeformableTransform(typename TransformType::Pointer transform, double magnitude, unsigned int seed)
    {
        auto N = transform->GetNumberOfParameters();
        typedef itk::Statistics::MersenneTwisterRandomVariateGenerator GeneratorType;
        typename GeneratorType::Pointer RNG = GeneratorType::New();
        RNG->SetSeed(seed);

        typename TransformType::DerivativeType delta(N);

        for (unsigned int i = 0; i < N; ++i)
        {
            double x = RNG->GetVariateWithClosedRange();
            double y = (-1.0 + 2.0 * x) * magnitude;
            delta[i] = y;
        }

        transform->UpdateTransformParameters(delta, 1.0);
    }

    template <typename TImageType, typename TTransformType>
    static typename TImageType::Pointer ApplyTransform(
        typename TImageType::Pointer refImage,
        typename TImageType::Pointer floImage,
        typename TTransformType::Pointer transform,
        int interpolator = 1,
        typename TImageType::PixelType defaultValue = 0)
    {
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

    template <typename TImageType>
    static double LabelAccuracy(typename TImageType::Pointer image1, typename TImageType::Pointer image2) {
        typedef itk::LabelOverlapMeasuresImageFilter<TImageType> FilterType;
        typedef typename FilterType::Pointer FilterPointer;

        FilterPointer filter = FilterType::New();

        filter->SetSourceImage(image1);
        filter->SetTargetImage(image2);

        filter->Update();

        return filter->GetTotalOverlap();
    }

    template <typename TImageType>
    static double HausdorffDistance(typename TImageType::Pointer image1, typename TImageType::Pointer image2) {
        typedef itk::HausdorffDistanceImageFilter<TImageType, TImageType> FilterType;
        typedef typename FilterType::Pointer FilterPointer;

        FilterPointer filter = FilterType::New();

        filter->SetInput1(image1);
        filter->SetInput2(image2);
        filter->SetUseImageSpacing(true);
        filter->Update();

        return filter->GetHausdorffDistance();
    }

    static void Evaluate(
        ImagePointer refImage,
        ImagePointer refImageMask,
        LabelImagePointer refImageLabel,
        unsigned int controlPointCount,
        double transformMagnitude,
        const std::vector<unsigned int>& seeds,
        double noiseSigma,
        BSplineRegParamOuter& params,
        unsigned int metricID,
        EvalThread& t)
    {

        BSplineFunc bsf;

        typedef typename BSplineFunc::TransformType TransformType;
        typedef typename BSplineFunc::TransformType::Pointer TransformPointer;
	
        ImagePointer cbImage = BlackAndWhiteChessboard(refImage, 32);

        unsigned int startIndex = t.startIndex;
        unsigned int endIndex = t.endIndex;

        //std::cout << refImage << std::endl;

        for(unsigned int i = startIndex; i < endIndex; ++i) {
            TransformPointer randTransform = bsf.CreateBSplineTransform(refImage, controlPointCount);
            RandomizeDeformableTransform<TransformType>(randTransform, transformMagnitude, seeds[i]);

            ImagePointer floImage = ApplyTransform<ImageType, TransformType>(refImage, refImage, randTransform, 1, 0.0);
            ImagePointer floImageMask = ApplyTransform<ImageType, TransformType>(refImageMask, refImageMask, randTransform, 0, 0.0);
            LabelImagePointer floImageLabel = ApplyTransform<LabelImageType, TransformType>(refImageLabel, refImageLabel, randTransform, 0, 0);

    	    ImagePointer refImageNoisy = IPT::AdditiveNoise(refImage, noiseSigma, 0.0, seeds[i]*19+13, true);
	        ImagePointer floImageNoisy = IPT::AdditiveNoise(floImage, noiseSigma, 0.0, seeds[i]*17+11, true);

            TransformPointer forwardTransform;
            TransformPointer inverseTransform;

            bool verbose = true;
            if(metricID == 0) {
                bsf.bspline_register(refImageNoisy, floImageNoisy, params, refImageMask, floImageMask, verbose, forwardTransform, inverseTransform);
            } else {
                bsf.bspline_register_baseline(refImageNoisy, floImageNoisy, params, refImageMask, floImageMask, verbose, forwardTransform, inverseTransform, metricID-1);
            }

            ImagePointer registeredImage = ApplyTransform<ImageType, TransformType>(refImage, floImage, forwardTransform, 1, 0.0);
            LabelImagePointer registeredLabel = ApplyTransform<LabelImageType, TransformType>(refImageLabel, floImageLabel, forwardTransform, 0, 0);

            PerformanceMetrics beforePM;
            PerformanceMetrics afterPM;

            typename ImageType::Pointer beforeDiff = IPT::DifferenceImage(refImage, floImage);

            typename IPT::ImageStatisticsData beforeStats = IPT::ImageStatistics(beforeDiff);
            beforePM.absDiff = beforeStats.mean;

            beforePM.accuracy = LabelAccuracy<LabelImageType>(floImageLabel, refImageLabel);

            beforePM.hausdorff = HausdorffDistance<LabelImageType>(floImageLabel, refImageLabel);

            typename ImageType::Pointer afterDiff = IPT::DifferenceImage(refImage, registeredImage);

            typename IPT::ImageStatisticsData afterStats = IPT::ImageStatistics(afterDiff);
            afterPM.absDiff = afterStats.mean;

            afterPM.accuracy = LabelAccuracy<LabelImageType>(registeredLabel, refImageLabel);

            afterPM.hausdorff = HausdorffDistance<LabelImageType>(registeredLabel, refImageLabel);

            t.beforePerf.push_back(beforePM);
            t.afterPerf.push_back(afterPM);

            printMetrics(beforePM, "[Before]");
            printMetrics(afterPM, "[After]");

            std::string prefix = "./synth";
            prefix += ('0' + metricID);
            
            if(i == 0) {
                ImagePointer deformedCBImage = ApplyTransform<ImageType, TransformType>(refImage, cbImage, randTransform, 1, 0.0);
                ImagePointer transformedCBImage = ApplyTransform<ImageType, TransformType>(refImage, cbImage, forwardTransform, 1, 0.0);
                
                IPT::SaveImageU8(prefix + "_noisy_ref.png", refImageNoisy);
                IPT::SaveImageU8(prefix + "_deformed.png", floImage);
                IPT::SaveImageU8(prefix + "_noisy_deformed.png", floImageNoisy);
                IPT::SaveImageU8(prefix + "_registered.png", registeredImage);
                IPT::SaveImageU8(prefix + "_chessboard_deformed.png", deformedCBImage);
                IPT::SaveImageU8(prefix + "_chessboard_registered.png", transformedCBImage);
                IPT::SaveLabelImage(prefix + "_label_deformed.png", floImageLabel);
                IPT::SaveLabelImage(prefix + "_label_registered.png", registeredLabel);
            }
        }
    }

    static int MainFunc(int argc, char** argv) {
        itk::MultiThreader::SetGlobalMaximumNumberOfThreads(1);
        
        itk::TimeProbesCollectorBase chronometer;
        itk::MemoryProbesCollectorBase memorymeter;

        chronometer.Start("Evaluation");
        memorymeter.Start("Evaluation");

        EvaluationConfig config = readEvaluationConfig(argv[2]);
        std::cout << "Evaluation config read..." << std::endl;
        BSplineRegParamOuter params = readConfig(config.methodConfigPath);
        std::cout << "Registration config read..." << std::endl;

        std::string refImagePath = config.imagePath;
        std::string refImageMaskPath = config.imageMaskPath;
        std::string refImageLabelPath = config.imageLabelPath;
        //std::string refImagePath = argv[2];
        //std::string refImageMaskPath = argv[3];
        //std::string refImageLabelPath = argv[4];

        unsigned int seed = config.seed;//atoi(argv[5]);
        unsigned int count = config.count;//atoi(argv[6]);
        double noiseSigma = config.noiseSigma;

        std::vector<unsigned int> seeds;
        for(unsigned int i = 0; i < count; ++i) {
            seed = seed * 17 + 13;
            seeds.push_back(seed);
        }

        ImagePointer refImage = IPT::LoadImage(refImagePath.c_str());
        ImagePointer refImageMask = IPT::LoadImage(refImageMaskPath.c_str());
        LabelImagePointer refImageLabel = IPT::LoadLabelImage(refImageLabelPath.c_str());

        BSplineFunc bspline_func;

        const unsigned int deformControlPoints = config.controlPoints;
        const double deformMagnitude = config.deformationMagnitude;
        /*
    void Evaluate(
        ImagePointer refImage,
        ImagePointer refImageMask,
        LabelImagePointer refImageLabel,
        unsigned int controlPointCount,
        double transformMagnitude,
        const std::vector<unsigned int>& seeds,
        unsigned int startIndex,
        unsigned int endIndex,
        std::vector<PerformanceMetrics>& beforePerf,
        std::vector<PerformanceMetrics>& afterPerf)*/
        unsigned int metricID = 0;
        if(config.metricName=="alpha-amd")
            metricID = 0;
        else if(config.metricName == "msd")
            metricID = 1;
        else if(config.metricName == "mi")
            metricID = 2;
        else {
            std::cout << "Unreconized metric name: alpha-amd or msd supported." << std::endl;
            return -1;
        }

        constexpr int threadCount = 6;

            std::vector<PerformanceMetrics> beforePerf;
            std::vector<PerformanceMetrics> afterPerf;
        EvalThread threadData[threadCount];
        std::thread threads[threadCount];


        for(unsigned int i = 0; i < threadCount; ++i) {
            double spn = count/(double)threadCount;
            unsigned int start = (int)(i * spn);
            unsigned int end = (int)((i+1) * spn);
            if(end > count)
                end = count;
            std::cout << "(" << start << " - " << end << ")" << std::endl;
            threadData[i].startIndex = start;
            threadData[i].endIndex = end;

            threadData[i].threadID = i;

            auto fn = [&, i]() -> void { Evaluate(refImage, refImageMask, refImageLabel, deformControlPoints, deformMagnitude, seeds, noiseSigma, params, metricID, threadData[i]); };
            threads[i] = std::thread(fn);
        }

        for(unsigned int i = 0; i < threadCount; ++i) {
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

            printMetrics(meanMetrics(beforePerf), "[Before mean]");
            printMetrics(meanMetrics(afterPerf), "[After mean]");
            printMetrics(sdMetrics(beforePerf), "[Before sd]");
            printMetrics(sdMetrics(afterPerf), "[After sd]");
            printMetrics(minMetrics(beforePerf), "[Before min]");
            printMetrics(minMetrics(afterPerf), "[After min]");
            printMetrics(maxMetrics(beforePerf), "[Before max]");
            printMetrics(maxMetrics(afterPerf), "[After max]");


        return 0;
    }
};

