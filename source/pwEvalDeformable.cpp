
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
    unsigned int seed;
    std::string metricName;
    std::vector<std::string> images;
    std::vector<std::string> labels;
    std::string methodConfigPath;
};

EvaluationConfig readEvaluationConfig(std::string path) {
    json jc = readJSON(path);

    EvaluationConfig c;

    c.count = (unsigned int)jc["count"];
    c.metricName = jc["metric"];
    for(unsigned int i = 0; i< jc["images"].size(); ++i)
        c.images.push_back(jc["images"][i]);
    for(unsigned int i = 0; i< jc["labels"].size(); ++i)
        c.labels.push_back(jc["labels"][i]);

    c.methodConfigPath = jc["methodConfigPath"];

    return c;
}

struct PerformanceMetrics {
    double totalOverlap;
    double meanTotalOverlap;
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
    std::cout << name << "(totalOverlap: " << m.totalOverlap << ", meanTotalOverlap: " << m.meanTotalOverlap << ", absdiff: " << m.absDiff << ").";
    if(linebreak)
        std::cout << std::endl;
}

/*PerformanceMetrics meanMetrics(std::vector<PerformanceMetrics>& m) {
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
}*/

void SaveMetrics(const char* path, const std::vector<PerformanceMetrics>& m) {
    FILE *f = fopen(path, "wb");

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
    static void LabelAccuracy(typename TImageType::Pointer image1, typename TImageType::Pointer image2, double& totalOverlap, double& meanTotalOverlap) {
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

    static void Evaluate(
        const std::vector<std::string>& images,
        const std::vector<std::string>& labels,
        std::vector<unsigned int>& refIndices,
        std::vector<unsigned int>& floIndices,
        BSplineRegParamOuter& params,
        unsigned int metricID,
        EvalThread& t)
    {

        BSplineFunc bsf;

        typedef typename BSplineFunc::TransformType TransformType;
        typedef typename BSplineFunc::TransformType::Pointer TransformPointer;
	
        unsigned int startIndex = t.startIndex;
        unsigned int endIndex = t.endIndex;

        for(unsigned int i = startIndex; i < endIndex; ++i) {
            unsigned int refIndex = refIndices[i];
            unsigned int floIndex = floIndices[i];

            ImagePointer refImage = IPT::LoadImage(images[refIndex].c_str());
            ImagePointer floImage = IPT::LoadImage(images[floIndex].c_str());
            LabelImagePointer refImageLabel = IPT::LoadLabelImage(labels[refIndex].c_str());
            LabelImagePointer floImageLabel = IPT::LoadLabelImage(labels[floIndex].c_str());

            ImagePointer refImageMask = IPT::ConstantImage(1.0, refImage->GetLargestPossibleRegion().GetSize());
            ImagePointer floImageMask = IPT::ConstantImage(1.0, floImage->GetLargestPossibleRegion().GetSize());

            ImagePointer cbImage = BlackAndWhiteChessboard(refImage, 32);

            TransformPointer forwardTransform;
            TransformPointer inverseTransform;

            bool verbose = true;
            if(metricID == 0) {
                bsf.bspline_register(refImage, floImage, params, refImageMask, floImageMask, verbose, forwardTransform, inverseTransform);
            } else {
                bsf.bspline_register_baseline(refImage, floImage, params, refImageMask, floImageMask, verbose, forwardTransform, inverseTransform, metricID-1);
            }

            ImagePointer registeredImage = ApplyTransform<ImageType, TransformType>(refImage, floImage, forwardTransform, 1, 0.0);
            LabelImagePointer registeredLabel = ApplyTransform<LabelImageType, TransformType>(refImageLabel, floImageLabel, forwardTransform, 0, 0);

            PerformanceMetrics beforePM;
            PerformanceMetrics afterPM;

            typename ImageType::Pointer beforeDiff = IPT::DifferenceImage(refImage, floImage);

            typename IPT::ImageStatisticsData beforeStats = IPT::ImageStatistics(beforeDiff);
            beforePM.absDiff = beforeStats.mean;

            double totalOverlap;
            double meanTotalOverlap;
            LabelAccuracy<LabelImageType>(floImageLabel, refImageLabel, totalOverlap, meanTotalOverlap);
            beforePM.totalOverlap = totalOverlap;
            beforePM.meanTotalOverlap = meanTotalOverlap;
            //beforePM.totalOverlap = LabelAccuracy<LabelImageType>(floImageLabel, refImageLabel);

            typename ImageType::Pointer afterDiff = IPT::DifferenceImage(refImage, registeredImage);

            typename IPT::ImageStatisticsData afterStats = IPT::ImageStatistics(afterDiff);
            afterPM.absDiff = afterStats.mean;

            //afterPM.accuracy = LabelAccuracy<LabelImageType>(registeredLabel, refImageLabel);
            LabelAccuracy<LabelImageType>(registeredLabel, refImageLabel, totalOverlap, meanTotalOverlap);
            afterPM.totalOverlap = totalOverlap;
            afterPM.meanTotalOverlap = meanTotalOverlap;

            t.beforePerf.push_back(beforePM);
            t.afterPerf.push_back(afterPM);

            printMetrics(beforePM, "[Before]");
            printMetrics(afterPM, "[After]");

            std::string prefix = "./results/synth";
            prefix += ('0' + metricID);
            
            if(i == 0) {
                ImagePointer transformedCBImage = ApplyTransform<ImageType, TransformType>(refImage, cbImage, forwardTransform, 1, 0.0);
                
                //IPT::SaveImageU16(prefix + "_registered.nii.gz", registeredImage);
                IPT::SaveImageU16(prefix + "_chessboard_registered.nii.gz", transformedCBImage);
                IPT::SaveLabelImage(prefix + "_label_registered.nii.gz", registeredLabel);
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

        std::vector<unsigned int> refIndices;
        std::vector<unsigned int> floIndices;

        for(unsigned int i = 0; i < config.images.size(); ++i) {
            for(unsigned int j = 0; j < config.images.size(); ++j) {
                if(i == j)
                    continue;
                refIndices.push_back(i);
                floIndices.push_back(j);
            }
        }
        if(config.count == 0) {
            config.count = (unsigned int)refIndices.size();
        }

        unsigned int count = config.count;

        BSplineFunc bspline_func;

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
/*
    static void Evaluate(
        const std::vector<std::string>& images,
        const std::vector<std::string>& labels,
        std::vector<unsigned int>& refIndices,
        std::vector<unsigned int>& floIndices,
        BSplineRegParamOuter& params,
        unsigned int metricID,
        EvalThread& t)
    {*/

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

            auto fn = [&, i]() -> void { Evaluate(config.images, config.labels, refIndices, floIndices, params, metricID, threadData[i]); };
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

        SaveMetrics("before_metrics.csv", beforePerf);
        SaveMetrics("after_metrics.csv", afterPerf);

        return 0;
    }
};

