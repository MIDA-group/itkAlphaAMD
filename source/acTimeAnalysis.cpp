
#include <stdio.h>
#include <string.h>
#include <string>
#include <iomanip>

#include "common/itkImageProcessingTools.h"
#include "itkTextOutput.h"

#include "metric/itkAlphaSMDMetric2.h"
#include "itkRealTimeClock.h"
#include "itkNumericTraits.h"

#include "itkMeanSquaresImageToImageMetricv4.h"
#include "itkCorrelationImageToImageMetricv4.h"
#include "itkMattesMutualInformationImageToImageMetricv4.h"

#include "itkImageMaskSpatialObject.h"

struct TimeAnalysisProgramParam
{
    std::string in1;
    std::string in2;

    unsigned int repetitions;
    unsigned int iterations;
    double samplingFraction;

    double normalization;

    bool cubicInterpolation;

    std::string metric;
};

template <typename MetricType>
static double PerformTimeAnalysisForMetric(typename MetricType::Pointer metric, unsigned int repetitions, unsigned int iterations)
{
    typedef typename MetricType::DerivativeType DerivativeType;
    typedef typename MetricType::MeasureType MeasureType;   

    MeasureType measureSum = itk::NumericTraits<MeasureType>::ZeroValue();
    MeasureType measure;
    DerivativeType derivative(metric->GetNumberOfParameters());

	typedef itk::RealTimeClock ClockType;
	typedef typename ClockType::Pointer ClockPointer;
	typedef typename ClockType::TimeStampType TimeStampType;
	ClockPointer timer = itk::RealTimeClock::New();
	TimeStampType beginTime, endTime;
    double elapsedTime;

    std::vector<double> timeSeries;
    timeSeries.reserve(repetitions);

    for(unsigned int j = 0; j < repetitions; ++j)
    {
	    beginTime = timer->GetTimeInSeconds();

        for(unsigned int i = 0; i < iterations; ++i)
        {
            metric->GetValueAndDerivative(measure, derivative);

            measureSum = measureSum + measure;
        }

	    endTime = timer->GetTimeInSeconds();

        elapsedTime = static_cast<double>(endTime - beginTime) / iterations;

        timeSeries.push_back(elapsedTime);
    }

    double mean = VectorMean(timeSeries);
    double stdDev = VectorStdDev(timeSeries, mean);

    for(size_t i = 0; i < timeSeries.size(); ++i)
    {
        if(i > 0)
            std::cout << ",";
        std::cout << timeSeries[i];
    }
    std::cout << std::endl;
    std::cout << "Mean: " << std::fixed << std::setprecision(7) <<  mean << std::endl;
    std::cout << "StdDev: " << std::fixed << std::setprecision(7) << stdDev << std::endl;

    std::cout << "Average measure value: " << (measureSum / (double)iterations) << std::endl;
    
    //elapsedTime = 1000.0 * static_cast<double>(endTime - beginTime) / iterations;

    //std::cout << "Elapsed time: " << elapsedTime << " milliseconds per iteration." << std::endl;

    return mean;
}

template <typename MetricType, unsigned int Dim>
static double PerformTimeAnalysisForMetricITK(typename itk::IPT<double, Dim>::ImagePointer refImage, typename itk::IPT<double, Dim>::ImagePointer floatingImage, TimeAnalysisProgramParam &param)
{
	typedef itk::IPT<double, Dim> IPT;

	typedef typename IPT::ImageType ImageType;
	typedef typename IPT::ImagePointer ImagePointer;
    

	typedef itk::AffineTransform<double, Dim> TransformType;
	typedef typename TransformType::Pointer TransformPointer;

	typedef typename MetricType::Pointer MetricPointer;

	typedef itk::InterpolateImageFunction<typename IPT::ImageType, double> InterpolatorType;

	typename InterpolatorType::Pointer fixedInterpolator = IPT::MakeInterpolator(IPT::kImwarpInterpNearest);
	typename InterpolatorType::Pointer movingInterpolator;
    if(param.cubicInterpolation)
        movingInterpolator = IPT::MakeInterpolator(IPT::kImwarpInterpCubic);
    else
        movingInterpolator = IPT::MakeInterpolator(IPT::kImwarpInterpLinear);
	//typename InterpolatorType::Pointer movingInterpolator = IPT::MakeInterpolator(IPT::kImwarpInterpCubic);

    MetricPointer metric = MetricType::New();

	metric->SetFixedInterpolator(fixedInterpolator);
	metric->SetMovingInterpolator(movingInterpolator);

	itk::Point<double, Dim> fixedCenter = IPT::ComputeImageCenter(refImage, true);

	typename itk::IdentityTransform<double, Dim>::Pointer fixedTransform = itk::IdentityTransform<double, Dim>::New();

	TransformPointer movingTransform = TransformType::New();

	fixedTransform->SetIdentity();
	movingTransform->SetIdentity();

	movingTransform->SetCenter(fixedCenter);

	typedef unsigned char uchar;

	typename itk::ImageMaskSpatialObject<Dim>::Pointer fixedMaskSO = itk::ImageMaskSpatialObject<Dim>::New();
	fixedMaskSO->SetImage(itk::ConvertImageToIntegerFormat<uchar, Dim>(IPT::RectMask(refImage->GetLargestPossibleRegion(), refImage->GetSpacing(), 1.0)));

	metric->SetFixedImageMask(fixedMaskSO);
	typename itk::CompositeTransform<double, Dim>::Pointer compositeMovingTransform = itk::CompositeTransform<double, Dim>::New();

	typename itk::ImageMaskSpatialObject<Dim>::Pointer movingMaskSO = itk::ImageMaskSpatialObject<Dim>::New();
	movingMaskSO->SetImage(itk::ConvertImageToIntegerFormat<uchar, Dim>(IPT::RectMask(floatingImage->GetLargestPossibleRegion(), floatingImage->GetSpacing(), 1.0)));

	metric->SetMovingImageMask(movingMaskSO);

    metric->SetFixedTransform(fixedTransform);
    metric->SetMovingTransform(movingTransform);

    metric->SetFixedImage(refImage);
    metric->SetMovingImage(floatingImage);

    metric->SetVirtualDomainFromImage(refImage);
    metric->SetUseFixedSampledPointSet(false);

    metric->Initialize();

    return PerformTimeAnalysisForMetric<MetricType>(metric, param.repetitions, param.iterations);
}

template <unsigned int Dim>
static void DoTimeAnalysis(TimeAnalysisProgramParam &param)
{
	// Types

	typedef itk::IPT<double, Dim> IPT;

	typedef typename IPT::ImageType ImageType;
	typedef typename IPT::ImagePointer ImagePointer;

	itk::MultiThreader::SetGlobalMaximumNumberOfThreads(1);
	itk::MultiThreader::SetGlobalDefaultNumberOfThreads(1);
    itk::OutputWindow::SetInstance(itk::TextOutput::New());

	ImagePointer refImage;

	// Load reference image if exists
	if (param.in1 != "")
	{
		refImage = IPT::LoadImage(param.in1.c_str());
		itk::PrintStatistics<ImageType>(refImage, "Reference Before Normalization");
	}
	else
	{
		std::cout << "Error: No reference image provided." << std::endl;
		return;
	}

	ImagePointer floatingImage = IPT::LoadImage(param.in2.c_str());
	itk::PrintStatistics<ImageType>(refImage, "Floating Before Normalization");

    if(param.normalization >= 0.0)
    {
        typedef typename IPT::MinMaxSpan MinMaxSpan;
        MinMaxSpan refMinMax = IPT::IntensityMinMax(refImage, param.normalization);
        MinMaxSpan floMinMax = IPT::IntensityMinMax(floatingImage, param.normalization);
        refImage = IPT::NormalizeImage(refImage, refMinMax);
        floatingImage = IPT::NormalizeImage(floatingImage, floMinMax);        
    }

	itk::PrintStatistics<ImageType>(refImage, "Reference After Normalization");
	itk::PrintStatistics<ImageType>(floatingImage, "Floating After Normalization");

    double elapsedTime = 0.0;

    if(param.metric == "ssd")
    {
		typedef itk::MeanSquaresImageToImageMetricv4<ImageType, ImageType, ImageType> MetricType;
        elapsedTime = PerformTimeAnalysisForMetricITK<MetricType, Dim>(refImage, floatingImage, param);
    } else if(param.metric == "ncc")
    {
        typedef itk::CorrelationImageToImageMetricv4<ImageType, ImageType, ImageType> MetricType;
        elapsedTime = PerformTimeAnalysisForMetricITK<MetricType, Dim>(refImage, floatingImage, param);
    } else if(param.metric == "mi")
    {
        typedef itk::MattesMutualInformationImageToImageMetricv4<ImageType, ImageType, ImageType> MetricType;
        elapsedTime = PerformTimeAnalysisForMetricITK<MetricType, Dim>(refImage, floatingImage, param);
    } else if(param.metric == "alpha_smd")
    {
    	typedef itk::AlphaSMDAffineTransform<double, Dim> SymmetricTransformType;

	    typedef itk::AlphaSMDObjectToObjectMetric2v4<typename IPT::ImageType, Dim, double, SymmetricTransformType > MetricType;
	    typedef typename MetricType::Pointer MetricPointer;

	    MetricPointer metric = MetricType::New();
        
        unsigned int alphaLevels = 7;

	    metric->SetAlphaLevels(alphaLevels);
	    metric->SetFixedMask(IPT::ThresholdImage(IPT::RectMask(refImage->GetLargestPossibleRegion(), refImage->GetSpacing(), 1.0), 0.5));
        metric->SetFixedWeightImage(IPT::RectMask(refImage->GetLargestPossibleRegion(), refImage->GetSpacing(), 1.0));
	    metric->SetFixedImage(refImage);
	    metric->SetMovingImage(floatingImage);
	    metric->SetMovingMask(IPT::ThresholdImage(IPT::RectMask(floatingImage->GetLargestPossibleRegion(), floatingImage->GetSpacing(), 1.0), 0.5));
	    metric->SetMovingWeightImage(IPT::RectMask(floatingImage->GetLargestPossibleRegion(), floatingImage->GetSpacing(), 1.0));
	    metric->SetSymmetricMeasure(true);
	    metric->SetSquaredMeasure(false);
	    metric->SetLinearInterpolation(true);
	    metric->SetMaxDistance(128.0);

    	metric->SetFixedSamplingPercentage(1.0);
	    metric->SetMovingSamplingPercentage(1.0);
        metric->SetOutlierRejectionPercentage(0.0);

        metric->Update();

        elapsedTime = PerformTimeAnalysisForMetric<MetricType>(metric, param.repetitions, param.iterations);
    }
}

int main(int argc, char **argv)
{
	TimeAnalysisProgramParam param;

	unsigned int dim = 2U;

	// Defaults
    param.metric = "alpha_smd";
    param.repetitions = 10;
    param.iterations = 100;
    param.samplingFraction = 1.0;
    param.normalization = 0.0;

	// Parameters
	for (int pi = 1; pi + 1 < argc; pi += 2)
	{
		const char* mod = argv[pi];
		const char* arg = argv[pi + 1];

		if (strcmp(mod, "-in1") == 0)
		{
			param.in1 = arg;
		}
		else if (strcmp(mod, "-in2") == 0)
		{
			param.in2 = arg;
		}
		else if (strcmp(mod, "-metric") == 0)
		{
			param.metric = arg;
		}
		else if (strcmp(mod, "-repetitions") == 0)
		{
			param.repetitions = atoi(arg);
		}
		else if (strcmp(mod, "-iterations") == 0)
		{
			param.iterations = atoi(arg);
		}
		else if (strcmp(mod, "-interpolation") == 0)
		{
			if (strcmp(arg, "cubic") == 0)
            {
                param.cubicInterpolation = true;
            } else if(strcmp(arg, "linear") == 0) {
                param.cubicInterpolation = false;
            } else {
                std::cout << "Illegal interpolation mode: Only cubic and linear supported." << std::endl;
                return -1;
            }
		}
		else if (strcmp(mod, "-normalization") == 0)
		{
            if (strcmp(arg, "off") == 0)
            {
    			param.normalization = -1.0;
            } else {
                param.normalization = atof(arg);
            }
		}        
		else if (strcmp(mod, "-dim") == 0)
		{
			dim = (unsigned int)atoi(arg);
			if(dim < 2 || dim > 3) {
				std::cout << "Illegal number of dimensions: '" << dim << "', only 2 or 3 supported." << std::endl;
				return -1;
			}
		}
	}

	if (dim == 3U)
	{
		DoTimeAnalysis<3U>(param);
	}
	else if(dim == 2U)
	{
		DoTimeAnalysis<2U>(param);
	}

	return 0;
}
