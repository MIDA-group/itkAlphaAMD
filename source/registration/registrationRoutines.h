
#ifndef REGISTRATION_ROUTINES_H
#define REGISTRATION_ROUTINES_H

// Registration routines

#include <math.h>

#include "../common/itkImageProcessingTools.h"

#include "itkRealTimeClock.h"
#include "itkImageMaskSpatialObject.h"

#include "itkRegularStepGradientDescentOptimizerv4.h"
#include "itkImageRegistrationMethodv4.h"

#include "itkAffineTransform.h"

// Metrics
#include "itkMeanSquaresImageToImageMetricv4.h"
#include "itkCorrelationImageToImageMetricv4.h"
#include "itkMattesMutualInformationImageToImageMetricv4.h"

#include "../metric/itkAlphaSMDMetric2.h"

struct RegistrationParam {
	unsigned int iterations;
	unsigned int numberOfLevels;
	double       sigmaIncrement;
	std::vector<unsigned int> multiscaleSamplingFactors;
	std::vector<double> multiscaleSmoothingSigmas;
	double       learningRate;
	double       relaxationFactor;
	double       samplingPercentage;
	double       normalizationPercentage;
	bool         normalizationEnabled;
	unsigned int metricSeed;
};

// Make a no-op registration, which only returns transformations relating the image centers
template <unsigned int Dim>
typename itk::Transform<double, Dim, Dim>::Pointer noop_registration(
	typename itk::Image<double, Dim>::Pointer fixedImage,
	typename itk::Image<double, Dim>::Pointer movingImage)
{
	typedef itk::IPT<double, Dim> IPT;

	itk::Point<double, Dim> fixedCenter = IPT::ComputeImageCenter(fixedImage, true);
	itk::Point<double, Dim> movingCenter = IPT::ComputeImageCenter(movingImage, true);

	typedef typename itk::TranslationTransform<double, Dim> TranslationTransformType;

	typename TranslationTransformType::Pointer transform = TranslationTransformType::New();

	typedef typename TranslationTransformType::ParametersType TranslationParametersType;

	TranslationParametersType transParam(Dim);
	for (unsigned int i = 0; i < Dim; ++i)
	{
		transParam[i] = movingCenter[i]-fixedCenter[i];
	}

	transform->SetParameters(transParam);

	return itk::CastSmartPointer<TranslationTransformType, itk::Transform<double, Dim, Dim>>(transform);
}

// Multi level/scale registration based on the built-in measures in ITK.

template <typename TTransform, typename TMetric, unsigned int Dim>
typename itk::Transform<double, Dim, Dim>::Pointer apply_registration(
	typename itk::Image<double, Dim>::Pointer fixedImage,
	typename itk::Image<double, Dim>::Pointer fixedImageMask,
	typename itk::Image<double, Dim>::Pointer movingImage,
	typename itk::Image<double, Dim>::Pointer movingImageMask,
	typename TMetric::Pointer metric,
	const RegistrationParam& param,
	double* distanceOut = nullptr) {

	typedef itk::IPT<double, Dim> IPT;

	typedef TTransform TransformType;
	typedef typename TransformType::Pointer TransformPointer;

	//typedef TMetric                      MetricType;
	//typedef typename MetricType::Pointer MetricPointer;

	typedef itk::ImageRegistrationMethodv4<typename IPT::ImageType, typename IPT::ImageType, TransformType> RegistrationType;

	typedef itk::RegularStepGradientDescentOptimizerv4<double>              OptimizerType;
	typedef itk::InterpolateImageFunction<typename IPT::ImageType, double> InterpolatorType;

	typename OptimizerType::Pointer      optimizer = OptimizerType::New();
	typename RegistrationType::Pointer   registration = RegistrationType::New();

	typename InterpolatorType::Pointer fixedInterpolator = IPT::MakeInterpolator(IPT::kImwarpInterpNearest);
	typename InterpolatorType::Pointer movingInterpolator = IPT::MakeInterpolator(IPT::kImwarpInterpCubic);

	metric->SetFixedInterpolator(fixedInterpolator);
	metric->SetMovingInterpolator(movingInterpolator);

	itk::Point<double, Dim> fixedCenter = IPT::ComputeImageCenter(fixedImage, true);

	typename itk::CompositeTransform<double, Dim>::Pointer compositeMovingTransform = itk::CompositeTransform<double, Dim>::New();

	typename itk::IdentityTransform<double, Dim>::Pointer fixedTransform = itk::IdentityTransform<double, Dim>::New();

	TransformPointer movingTransform = TransformType::New();

	fixedTransform->SetIdentity();
	movingTransform->SetIdentity();

	movingTransform->SetCenter(fixedCenter);

	typename itk::TranslationTransform<double, Dim>::Pointer lastTranslation = itk::TranslationTransform<double, Dim>::New();

	lastTranslation->SetIdentity();

	compositeMovingTransform->AddTransform(lastTranslation);
	compositeMovingTransform->AddTransform(movingTransform);

	compositeMovingTransform->SetNthTransformToOptimize(0, false);
	compositeMovingTransform->SetNthTransformToOptimize(1, true);

	typedef unsigned char uchar;

	typename itk::ImageMaskSpatialObject<Dim>::Pointer fixedMaskSO = itk::ImageMaskSpatialObject<Dim>::New();
	fixedMaskSO->SetImage(itk::ConvertImageToIntegerFormat<uchar, Dim>(fixedImageMask));

	metric->SetFixedImageMask(fixedMaskSO);

	typename itk::ImageMaskSpatialObject<Dim>::Pointer movingMaskSO = itk::ImageMaskSpatialObject<Dim>::New();
	movingMaskSO->SetImage(itk::ConvertImageToIntegerFormat<uchar, Dim>(movingImageMask));

	metric->SetMovingImageMask(movingMaskSO);

	optimizer->SetLearningRate(param.learningRate);
	optimizer->DoEstimateLearningRateAtEachIterationOn();
	optimizer->SetRelaxationFactor(param.relaxationFactor);

	OptimizerType::ScalesType scaling(Dim * (Dim + 1));

	double fixedImageSz = IPT::ComputeImageDiagonalSize(fixedImage, true);
	double movingImageSz = IPT::ComputeImageDiagonalSize(movingImage, true);

	const double scaleMultiplier = 1.5;

	const double diag = scaleMultiplier * 0.5 * (fixedImageSz + movingImageSz);

	const double scale_translation = 1.0 / diag;

	//const double scale_translation = 1.0 / (sqrt(2) * 256.0);

	scaling.Fill(1.0);

	for(unsigned int i = 0; i < Dim; ++i) {
		scaling[Dim * Dim + i] = scale_translation;
	}

	optimizer->SetScales(scaling);

	//General optimizer parameters

	// Set a stopping criterion
	optimizer->SetNumberOfIterations(param.iterations);
	optimizer->SetReturnBestParametersAndValue(true);

	registration->SetMovingInitialTransform(compositeMovingTransform);

	registration->SetFixedImage(fixedImage);
	registration->SetMovingImage(movingImage);

	unsigned int numberOfLevels = (unsigned int)param.multiscaleSamplingFactors.size();

	typename RegistrationType::ShrinkFactorsArrayType shrinkFactorsPerLevel;
	shrinkFactorsPerLevel.SetSize(numberOfLevels);

	typename RegistrationType::SmoothingSigmasArrayType smoothingSigmasPerLevel;
	smoothingSigmasPerLevel.SetSize(numberOfLevels);
	for(unsigned int i = 0; i < numberOfLevels; ++i) {
		shrinkFactorsPerLevel[i] = param.multiscaleSamplingFactors[i];
		smoothingSigmasPerLevel[i] = param.multiscaleSmoothingSigmas[i];
	}
/*
	int factor = 1;
	for (unsigned int i = 0; i < numberOfLevels; ++i) {
		unsigned int lvl = param.numberOfLevels - i - 1;
		shrinkFactorsPerLevel[lvl] = factor;
		factor = factor * 2;

		smoothingSigmasPerLevel[lvl] = i * param.sigmaIncrement;
	}*/

	registration->SetNumberOfLevels(numberOfLevels);
	registration->SetSmoothingSigmasPerLevel(smoothingSigmasPerLevel);
	registration->SetShrinkFactorsPerLevel(shrinkFactorsPerLevel);

	typename RegistrationType::MetricSamplingStrategyType  samplingStrategy =
		RegistrationType::RANDOM;

	registration->SetMetricSamplingStrategy(samplingStrategy);
	registration->SetMetricSamplingPercentage(param.samplingPercentage);
	registration->MetricSamplingReinitializeSeed(param.metricSeed);

	registration->SetMetric(metric);
	registration->SetOptimizer(optimizer);

	optimizer->SetNumberOfThreads(1);
	registration->SetNumberOfThreads(1);

	if(param.iterations > 0) {
		try
		{
			registration->Update();
			std::cout << "Optimizer stop condition: "
				<< registration->GetOptimizer()->GetStopConditionDescription()
				<< std::endl;
		}
		catch (itk::ExceptionObject & err)
		{
			std::cerr << "ExceptionObject caught !" << std::endl;
			std::cerr << err << std::endl;
		}
		std::cout << "Number of iterations elapsed: " << optimizer->GetCurrentIteration() << std::endl;
	}

	if(distanceOut)
	{
		*distanceOut = registration->GetOptimizer()->GetCurrentMetricValue();
	}

	auto modifiableTransform = registration->GetModifiableTransform();
	std::cout << modifiableTransform << std::endl;

	return modifiableTransform;
}




// Register using the alpha cut based measure and a custom registration routine using the gradient
// descent optimizer of ITK.

itk::Transform<double, 2U, 2U>::Pointer apply_registration_alpha_smd_internal_2d_affine(
	itk::Image<double, 2U>::Pointer fixedImage,
	itk::Image<double, 2U>::Pointer fixedImageMask,
	itk::Image<double, 2U>::Pointer fixedImageWeights,
	itk::Image<double, 2U>::Pointer movingImage,
	itk::Image<double, 2U>::Pointer movingImageMask,
	itk::Image<double, 2U>::Pointer movingImageWeights,
	const RegistrationParam& param,
	itk::Transform<double, 2U, 2U>::Pointer initialTransform,
	unsigned int alphaLevels,
	bool useSymmetricMeasure,
	bool useSquaredMeasure,
	itk::CovariantVector<double, 2U> offsetCenter,
	double maxDistance=0.0,
	double outlierRejection=0.05,
	double* distanceOut = nullptr) {
	
	static const unsigned int Dim = 2U;

	typedef itk::IPT<double, Dim> IPT;
	typedef itk::AlphaSMDAffineTransform<double, Dim> SymmetricTransformType;

	typedef itk::AlphaSMDObjectToObjectMetric2v4<typename IPT::ImageType, Dim, double, SymmetricTransformType > MetricType;
	typedef MetricType::Pointer MetricPointer;

	MetricPointer metric = MetricType::New();

	// Create timer

	typedef itk::RealTimeClock ClockType;
	typedef ClockType::Pointer ClockPointer;
	typedef ClockType::TimeStampType TimeStampType;
	ClockPointer timer = itk::RealTimeClock::New();

	typedef itk::RegularStepGradientDescentOptimizerv4<double> OptimizerType;
	typedef OptimizerType::Pointer OptimizerPointer;

	// Initialize transform if given an initial transform
	
	metric->GetSymmetricTransformPointer()->ConvertFromITKTransform(initialTransform);

	OptimizerPointer optimizer = OptimizerType::New();

	double fixedImageSz = IPT::ComputeImageDiagonalSize(fixedImage, true);
	double movingImageSz = IPT::ComputeImageDiagonalSize(movingImage, true);
	const double diag = 0.5 * (fixedImageSz + movingImageSz);

	itk::Image<bool, 2U>::Pointer fixedImageBinaryMask = IPT::ThresholdImage(fixedImageMask, 0.5);
	itk::Image<bool, 2U>::Pointer movingImageBinaryMask = IPT::ThresholdImage(movingImageMask, 0.5);
		
	// Normalization

	if(param.normalizationEnabled)
	{
		if(fixedImageMask)
		{
			fixedImage = IPT::NormalizeImage(fixedImage, IPT::IntensityMinMax(fixedImage, param.normalizationPercentage, fixedImageBinaryMask));
		}
		else
		{
			fixedImage = IPT::NormalizeImage(fixedImage, IPT::IntensityMinMax(fixedImage, param.normalizationPercentage));
		}
		if(movingImageMask)
		{
			movingImage = IPT::NormalizeImage(movingImage, IPT::IntensityMinMax(movingImage, param.normalizationPercentage, movingImageBinaryMask));
		}
		else
		{
			movingImage = IPT::NormalizeImage(movingImage, IPT::IntensityMinMax(movingImage, param.normalizationPercentage));
		}
	}

	metric->SetAlphaLevels(alphaLevels);
	metric->SetFixedMask(fixedImageBinaryMask);
	metric->SetFixedImage(fixedImage);
	metric->SetMovingImage(movingImage);
	metric->SetSymmetricMeasure(useSymmetricMeasure);
	metric->SetSquaredMeasure(useSquaredMeasure);
	metric->SetLinearInterpolation(true);
	metric->SetMovingMask(movingImageBinaryMask);
	if(maxDistance <= 0.0) {
		metric->SetMaxDistance(diag);
	} else {
		metric->SetMaxDistance(maxDistance);
	}

	if(!fixedImageWeights) {
		metric->SetFixedUnitWeightImage();
	} else {
		metric->SetFixedWeightImage(fixedImageWeights);
	}
	if(!movingImageWeights) {
		metric->SetMovingUnitWeightImage();
	} else {
		metric->SetMovingWeightImage(movingImageWeights);
	}

	metric->SetFixedSamplingPercentage(param.samplingPercentage);
	metric->SetMovingSamplingPercentage(param.samplingPercentage);
    metric->SetRandomSeed(param.metricSeed);

	// Enable outlier rejection:
	metric->SetOutlierRejectionPercentage(outlierRejection);

	TimeStampType beginTime = timer->GetTimeInSeconds();

	// Start pre-processing

	metric->Update();

	TimeStampType endTime = timer->GetTimeInSeconds();

    std::cout << "Pre-processing time elapsed: " << static_cast<double>(endTime-beginTime) << std::endl;

	optimizer->SetLearningRate(param.learningRate);
	optimizer->SetRelaxationFactor(param.relaxationFactor);

	OptimizerType::ScalesType scaling(Dim * (Dim + 1));

	const double scale_translation = 1.0 / diag;

	scaling.Fill(1.0);

	for(unsigned int i = 0; i < Dim; ++i) {
		scaling[Dim + (Dim+1) * i] = scale_translation;
	}

	optimizer->SetScales(scaling);

	//General optimizer parameters

	// Set a stopping criterion
	optimizer->SetNumberOfIterations(param.iterations);
	optimizer->SetReturnBestParametersAndValue(false);

	// Set center points
	 
	metric->SetFixedCenterPoint(IPT::ImageGeometricalCenterPoint(fixedImage));
	itk::CovariantVector<double, Dim> zeroVector;
	zeroVector.Fill(0.0);

	metric->SetMovingCenterPoint(IPT::ImageGeometricalCenterPoint(movingImage));

	optimizer->SetMetric(metric.GetPointer());

	optimizer->SetNumberOfThreads(1);

	if(param.iterations > 0) {
		try
		{
			optimizer->StartOptimization();

			std::cout << "Optimizer stop condition: "
				<< optimizer->GetStopConditionDescription()
				<< std::endl;
		}
		catch (itk::ExceptionObject & err)
		{
			std::cerr << "ExceptionObject caught !" << std::endl;
			std::cerr << err << std::endl;
		}
	}

	if(distanceOut)
	{
		*distanceOut = optimizer->GetCurrentMetricValue();
	}

	std::cout << "Number of iterations elapsed: " << optimizer->GetCurrentIteration() << std::endl;

	itk::Transform<double, Dim, Dim>::Pointer at = metric->MakeFinalTransform();

	return at;
}

itk::Transform<double, 3U, 3U>::Pointer apply_registration_alpha_smd_internal_3d_affine(
	itk::Image<double, 3U>::Pointer fixedImage,
	itk::Image<double, 3U>::Pointer fixedImageMask,
	itk::Image<double, 3U>::Pointer fixedImageWeights,
	itk::Image<double, 3U>::Pointer movingImage,
	itk::Image<double, 3U>::Pointer movingImageMask,
	itk::Image<double, 3U>::Pointer movingImageWeights,
	const RegistrationParam& param,
	itk::Transform<double, 3U, 3U>::Pointer initialTransform,
	unsigned int alphaLevels,
	bool useSymmetricMeasure,
	bool useSquaredMeasure,
	itk::CovariantVector<double, 3U> offsetCenter,
	double maxDistance=0.0,
	double outlierRejection=0.05,
	double* distanceOut = nullptr) {
	
	static const unsigned int Dim = 3U;

	typedef itk::IPT<double, Dim> IPT;
	typedef itk::AlphaSMDAffineTransform<double, Dim> SymmetricTransformType;

	typedef itk::AlphaSMDObjectToObjectMetric2v4<typename IPT::ImageType, Dim, double, SymmetricTransformType > MetricType;
	typedef MetricType::Pointer MetricPointer;

	MetricPointer metric = MetricType::New();

	// Create timer

	typedef itk::RealTimeClock ClockType;
	typedef ClockType::Pointer ClockPointer;
	typedef ClockType::TimeStampType TimeStampType;
	ClockPointer timer = itk::RealTimeClock::New();

	typedef itk::RegularStepGradientDescentOptimizerv4<double> OptimizerType;
	typedef OptimizerType::Pointer OptimizerPointer;

	// Initialize transform if given an initial transform

	metric->GetSymmetricTransformPointer()->ConvertFromITKTransform(initialTransform);

	OptimizerPointer optimizer = OptimizerType::New();

	double fixedImageSz = IPT::ComputeImageDiagonalSize(fixedImage, true);
	double movingImageSz = IPT::ComputeImageDiagonalSize(movingImage, true);
	const double diag = 0.5 * (fixedImageSz + movingImageSz);

	itk::Image<bool, 3U>::Pointer fixedImageBinaryMask = IPT::ThresholdImage(fixedImageMask, 0.5);
	itk::Image<bool, 3U>::Pointer movingImageBinaryMask = IPT::ThresholdImage(movingImageMask, 0.5);
		
	// Normalization

	if(param.normalizationEnabled)
	{
		if(fixedImageMask)
		{
			fixedImage = IPT::NormalizeImage(fixedImage, IPT::IntensityMinMax(fixedImage, param.normalizationPercentage, fixedImageBinaryMask));
		}
		else
		{
			fixedImage = IPT::NormalizeImage(fixedImage, IPT::IntensityMinMax(fixedImage, param.normalizationPercentage));
		}
		if(movingImageMask)
		{
			movingImage = IPT::NormalizeImage(movingImage, IPT::IntensityMinMax(movingImage, param.normalizationPercentage, movingImageBinaryMask));
		}
		else
		{
			movingImage = IPT::NormalizeImage(movingImage, IPT::IntensityMinMax(movingImage, param.normalizationPercentage));
		}
	}

	metric->SetAlphaLevels(alphaLevels);
	metric->SetFixedMask(fixedImageBinaryMask);
	metric->SetFixedImage(fixedImage);
	metric->SetMovingImage(movingImage);
	metric->SetSymmetricMeasure(useSymmetricMeasure);
	metric->SetSquaredMeasure(useSquaredMeasure);
	metric->SetLinearInterpolation(true);
	metric->SetMovingMask(movingImageBinaryMask);
	if(maxDistance <= 0.0) {
		metric->SetMaxDistance(diag);
	} else {
		metric->SetMaxDistance(maxDistance);
	}

	if(!fixedImageWeights) {
		metric->SetFixedUnitWeightImage();
	} else {
		metric->SetFixedWeightImage(fixedImageWeights);
	}
	if(!movingImageWeights) {
		metric->SetMovingUnitWeightImage();
	} else {
		metric->SetMovingWeightImage(movingImageWeights);
	}

	metric->SetFixedSamplingPercentage(param.samplingPercentage);
	metric->SetMovingSamplingPercentage(param.samplingPercentage);
    metric->SetRandomSeed(param.metricSeed);

	// Enable outlier rejection:
	metric->SetOutlierRejectionPercentage(outlierRejection);

	TimeStampType beginTime = timer->GetTimeInSeconds();

	// Start pre-processing
	
	metric->Update();

	TimeStampType endTime = timer->GetTimeInSeconds();

    std::cout << "Pre-processing time elapsed: " << static_cast<double>(endTime-beginTime) << std::endl;

	optimizer->SetLearningRate(param.learningRate);
	optimizer->SetRelaxationFactor(param.relaxationFactor);

	OptimizerType::ScalesType scaling(Dim * (Dim + 1));

	const double scale_translation = 1.0 / diag;

	scaling.Fill(1.0);

	for(unsigned int i = 0; i < Dim; ++i) {
		scaling[Dim + (Dim+1) * i] = scale_translation;
	}

	optimizer->SetScales(scaling);

	//General optimizer parameters

	// Set a stopping criterion
	optimizer->SetNumberOfIterations(param.iterations);
	optimizer->SetReturnBestParametersAndValue(false);
  
	// Set center points
	 
	metric->SetFixedCenterPoint(IPT::ImageGeometricalCenterPoint(fixedImage));
	itk::CovariantVector<double, Dim> zeroVector;
	zeroVector.Fill(0.0);

	metric->SetMovingCenterPoint(IPT::ImageGeometricalCenterPoint(movingImage));

	optimizer->SetMetric(metric.GetPointer());

	optimizer->SetNumberOfThreads(1);

	if(param.iterations > 0) {
		try
		{
			optimizer->StartOptimization();

			std::cout << "Optimizer stop condition: "
				<< optimizer->GetStopConditionDescription()
				<< std::endl;
		}
		catch (itk::ExceptionObject & err)
		{
			std::cerr << "ExceptionObject caught !" << std::endl;
			std::cerr << err << std::endl;
		}
	}

	if(distanceOut)
	{
		*distanceOut = optimizer->GetCurrentMetricValue();
	}

	std::cout << "Number of iterations elapsed: " << optimizer->GetCurrentIteration() << std::endl;

	itk::Transform<double, Dim, Dim>::Pointer at = metric->MakeFinalTransform();

	return at;
}

itk::Transform<double, 2U, 2U>::Pointer apply_registration_alpha_smd_internal_2d_rigid(
	itk::Image<double, 2U>::Pointer fixedImage,
	itk::Image<double, 2U>::Pointer fixedImageMask,
	itk::Image<double, 2U>::Pointer fixedImageWeights,
	itk::Image<double, 2U>::Pointer movingImage,
	itk::Image<double, 2U>::Pointer movingImageMask,
	itk::Image<double, 2U>::Pointer movingImageWeights,
	const RegistrationParam& param,
	itk::Transform<double, 2U, 2U>::Pointer initialTransform,
	unsigned int alphaLevels,
	bool useSymmetricMeasure,
	bool useSquaredMeasure,
	itk::CovariantVector<double, 2U> offsetCenter,
	double maxDistance=0.0,
	double outlierRejection=0.05,
	double* distanceOut = nullptr) {
	
	static const unsigned int Dim = 2U;

	typedef itk::IPT<double, Dim> IPT;
	typedef itk::AlphaSMDRigid2DTransform<double> SymmetricTransformType;

	typedef itk::AlphaSMDObjectToObjectMetric2v4<typename IPT::ImageType, Dim, double, SymmetricTransformType > MetricType;
	typedef MetricType::Pointer MetricPointer;

	MetricPointer metric = MetricType::New();

	// Create timer

	typedef itk::RealTimeClock ClockType;
	typedef ClockType::Pointer ClockPointer;
	typedef ClockType::TimeStampType TimeStampType;
	ClockPointer timer = itk::RealTimeClock::New();

	typedef itk::RegularStepGradientDescentOptimizerv4<double> OptimizerType;
	typedef OptimizerType::Pointer OptimizerPointer;

	OptimizerPointer optimizer = OptimizerType::New();

	double fixedImageSz = IPT::ComputeImageDiagonalSize(fixedImage, true);
	double movingImageSz = IPT::ComputeImageDiagonalSize(movingImage, true);
	const double diag = 0.5 * (fixedImageSz + movingImageSz);

	itk::Image<bool, 2U>::Pointer fixedImageBinaryMask = IPT::ThresholdImage(fixedImageMask, 0.5);
	itk::Image<bool, 2U>::Pointer movingImageBinaryMask = IPT::ThresholdImage(movingImageMask, 0.5);
		
	// Normalization

	if(param.normalizationEnabled)
	{
		if(fixedImageMask)
		{
			fixedImage = IPT::NormalizeImage(fixedImage, IPT::IntensityMinMax(fixedImage, param.normalizationPercentage, fixedImageBinaryMask));
		}
		else
		{
			fixedImage = IPT::NormalizeImage(fixedImage, IPT::IntensityMinMax(fixedImage, param.normalizationPercentage));
		}
		if(movingImageMask)
		{
			movingImage = IPT::NormalizeImage(movingImage, IPT::IntensityMinMax(movingImage, param.normalizationPercentage, movingImageBinaryMask));
		}
		else
		{
			movingImage = IPT::NormalizeImage(movingImage, IPT::IntensityMinMax(movingImage, param.normalizationPercentage));
		}
	}

	metric->SetAlphaLevels(alphaLevels);
	metric->SetFixedMask(fixedImageBinaryMask);
	metric->SetFixedImage(fixedImage);
	metric->SetMovingImage(movingImage);
	metric->SetSymmetricMeasure(useSymmetricMeasure);
	metric->SetSquaredMeasure(useSquaredMeasure);
	metric->SetLinearInterpolation(true);
	metric->SetMovingMask(movingImageBinaryMask);
	if(maxDistance <= 0.0) {
		metric->SetMaxDistance(diag);
	} else {
		metric->SetMaxDistance(maxDistance);
	}

	if(!fixedImageWeights) {
		metric->SetFixedUnitWeightImage();
	} else {
		metric->SetFixedWeightImage(fixedImageWeights);
	}
	if(!movingImageWeights) {
		metric->SetMovingUnitWeightImage();
	} else {
		metric->SetMovingWeightImage(movingImageWeights);
	}

	metric->SetFixedSamplingPercentage(param.samplingPercentage);
	metric->SetMovingSamplingPercentage(param.samplingPercentage);
    metric->SetRandomSeed(param.metricSeed);

	// Enable outlier rejection:
	metric->SetOutlierRejectionPercentage(outlierRejection);

	TimeStampType beginTime = timer->GetTimeInSeconds();

	// Start pre-processing
	
	metric->Update();

	TimeStampType endTime = timer->GetTimeInSeconds();

    std::cout << "Pre-processing time elapsed: " << static_cast<double>(endTime-beginTime) << std::endl;

	optimizer->SetLearningRate(param.learningRate);
	optimizer->SetRelaxationFactor(param.relaxationFactor);

	OptimizerType::ScalesType scaling(3U);

	const double scale_translation = 1.0 / diag;

	scaling.Fill(1.0);

	scaling[1] = scale_translation;
	scaling[2] = scale_translation;
	//for(unsigned int i = 0; i < Dim; ++i) {
		//scaling[Dim + (Dim+1) * i] = scale_translation;
	//}

	optimizer->SetScales(scaling);

	//General optimizer parameters

	// Set a stopping criterion
	optimizer->SetNumberOfIterations(param.iterations);
	optimizer->SetReturnBestParametersAndValue(false);

  	// Initialize transform if given an initial transform
	
	metric->GetSymmetricTransformPointer()->ConvertFromITKTransform(initialTransform);

	// Set center points
	 //ImageCenter(fixedImage, offsetCenter, true)
	metric->SetFixedCenterPoint(IPT::ImageGeometricalCenterPoint(fixedImage));
	itk::CovariantVector<double, Dim> zeroVector;
	zeroVector.Fill(0.0);

	metric->SetMovingCenterPoint(IPT::ImageGeometricalCenterPoint(movingImage));//IPT::ComputeImageCenter(movingImage, zeroVector, true));

	optimizer->SetMetric(metric.GetPointer());

	optimizer->SetNumberOfThreads(1);

	std::cout << "Initial rotation: " << metric->GetSymmetricTransformPointer()->GetParam(0) << std::endl;

	if(param.iterations > 0) {
		try
		{
			optimizer->StartOptimization();

			std::cout << "Optimizer stop condition: "
				<< optimizer->GetStopConditionDescription()
				<< std::endl;
		}
		catch (itk::ExceptionObject & err)
		{
			std::cerr << "ExceptionObject caught !" << std::endl;
			std::cerr << err << std::endl;
		}
	}

	if(distanceOut)
	{
		*distanceOut = optimizer->GetCurrentMetricValue();
	}

	std::cout << "Number of iterations elapsed: " << optimizer->GetCurrentIteration() << std::endl;

	itk::Transform<double, Dim, Dim>::Pointer at = metric->MakeFinalTransform();

	return at;
}

template <unsigned int Dim>
typename itk::Transform<double, Dim, Dim>::Pointer apply_registration_alpha_smd(
	typename itk::Image<double, Dim>::Pointer fixedImage,
	typename itk::Image<double, Dim>::Pointer fixedImageMask,
	typename itk::Image<double, Dim>::Pointer fixedImageWeights,
	typename itk::Image<double, Dim>::Pointer movingImage,
	typename itk::Image<double, Dim>::Pointer movingImageMask,
	typename itk::Image<double, Dim>::Pointer movingImageWeights,
	const RegistrationParam& param,
	typename itk::Transform<double, Dim, Dim>::Pointer initialTransform,
	unsigned int alphaLevels,
	bool useSymmetricMeasure,
	bool useSquaredMeasure,
	itk::CovariantVector<double, Dim> offsetCenter,
	double maxDistance=0.0,
	double outlierRejection=0.05,
	bool rigidMode=false,
	double* distanceOut = nullptr);

template <>
typename itk::Transform<double, 2U, 2U>::Pointer apply_registration_alpha_smd<2U>(
	typename itk::Image<double, 2U>::Pointer fixedImage,
	typename itk::Image<double, 2U>::Pointer fixedImageMask,
	typename itk::Image<double, 2U>::Pointer fixedImageWeights,
	typename itk::Image<double, 2U>::Pointer movingImage,
	typename itk::Image<double, 2U>::Pointer movingImageMask,
	typename itk::Image<double, 2U>::Pointer movingImageWeights,
	const RegistrationParam& param,
	typename itk::Transform<double, 2U, 2U>::Pointer initialTransform,
	unsigned int alphaLevels,
	bool useSymmetricMeasure,
	bool useSquaredMeasure,
	itk::CovariantVector<double, 2U> offsetCenter,
	double maxDistance,
	double outlierRejection,
	bool rigidMode,
	double* distanceOut) {
	
	itk::Transform<double, 2U, 2U>::Pointer curTransform = initialTransform;

	unsigned int numberOfLevels = param.multiscaleSamplingFactors.size();
	
	assert(param.multiscaleSamplingFactors.size() == param.multiscaleSmoothingSigmas.size());

	for(unsigned int i = 0; i < numberOfLevels; ++i) {
		unsigned int subsampleFactor = param.multiscaleSamplingFactors[i];
		double curSigma = param.multiscaleSmoothingSigmas[i];

		std::cout << "subsampleFactor: " << subsampleFactor << std::endl;
		std::cout << "sigma: " << curSigma << std::endl;

		itk::Image<double, 2U>::Pointer curFixedImage = fixedImage;
		itk::Image<double, 2U>::Pointer curMovingImage = movingImage;
		itk::Image<double, 2U>::Pointer curFixedImageWeights = fixedImageWeights;
		itk::Image<double, 2U>::Pointer curMovingImageWeights = movingImageWeights;
		itk::Image<double, 2U>::Pointer curFixedImageMask = fixedImageMask;
		itk::Image<double, 2U>::Pointer curMovingImageMask = movingImageMask;

	    curFixedImage = itk::IPT<double, 2U>::SmoothImage(curFixedImage, curSigma);
		curMovingImage = itk::IPT<double, 2U>::SmoothImage(curMovingImage, curSigma);
		curFixedImage = itk::IPT<double, 2U>::SubsampleImage(curFixedImage, subsampleFactor);
		curMovingImage = itk::IPT<double, 2U>::SubsampleImage(curMovingImage, subsampleFactor);
		curFixedImageWeights = itk::IPT<double, 2U>::SubsampleImage(curFixedImageWeights, subsampleFactor);
		curMovingImageWeights = itk::IPT<double, 2U>::SubsampleImage(curMovingImageWeights, subsampleFactor);
		curFixedImageMask = itk::IPT<double, 2U>::SubsampleImage(curFixedImageMask, subsampleFactor);
		curMovingImageMask = itk::IPT<double, 2U>::SubsampleImage(curMovingImageMask, subsampleFactor);

		//char buf[512];
		//sprintf(buf, "fixedImage%d.png", i);
		//itk::IPT<double, 2U>::SaveImageU16(buf, curFixedImage);
		//sprintf(buf, "movingImage%d.png", i);
		//itk::IPT<double, 2U>::SaveImageU16(buf, curMovingImage);

		if(rigidMode == false) {
			curTransform = apply_registration_alpha_smd_internal_2d_affine(curFixedImage,
				curFixedImageMask,
				curFixedImageWeights,
				curMovingImage,
				curMovingImageMask,
				curMovingImageWeights,
				param,
				curTransform,
				alphaLevels,
				useSymmetricMeasure,
				useSquaredMeasure,
				offsetCenter,
				maxDistance,
				outlierRejection,
				distanceOut);
		} else {
			curTransform = apply_registration_alpha_smd_internal_2d_rigid(curFixedImage,
				curFixedImageMask,
				curFixedImageWeights,
				curMovingImage,
				curMovingImageMask,
				curMovingImageWeights,
				param,
				curTransform,
				alphaLevels,
				useSymmetricMeasure,
				useSquaredMeasure,
				offsetCenter,
				maxDistance,
				outlierRejection,
				distanceOut);
		}

		if (i + 1 < numberOfLevels) {
			curTransform = itk::IPT<double, 2U>::GetNthTransform(curTransform, 1);
		}
		std::cout << curTransform << std::endl;
	}

	return curTransform;
}

template <>
typename itk::Transform<double, 3U, 3U>::Pointer apply_registration_alpha_smd<3U>(
	typename itk::Image<double, 3U>::Pointer fixedImage,
	typename itk::Image<double, 3U>::Pointer fixedImageMask,
	typename itk::Image<double, 3U>::Pointer fixedImageWeights,
	typename itk::Image<double, 3U>::Pointer movingImage,
	typename itk::Image<double, 3U>::Pointer movingImageMask,
	typename itk::Image<double, 3U>::Pointer movingImageWeights,
	const RegistrationParam& param,
	typename itk::Transform<double, 3U, 3U>::Pointer initialTransform,
	unsigned int alphaLevels,
	bool useSymmetricMeasure,
	bool useSquaredMeasure,
	itk::CovariantVector<double, 3U> offsetCenter,
	double maxDistance,
	double outlierRejection,
	bool rigidMode,
	double* distanceOut) {

	itk::Transform<double, 3U, 3U>::Pointer curTransform = initialTransform;

	unsigned int numberOfLevels = param.multiscaleSamplingFactors.size();
	
	assert(param.multiscaleSamplingFactors.size() == param.multiscaleSmoothingSigmas.size());

	for(unsigned int i = 0; i < numberOfLevels; ++i) {
		unsigned int subsampleFactor = param.multiscaleSamplingFactors[i];
		double curSigma = param.multiscaleSmoothingSigmas[i];

		std::cout << "subsampleFactor: " << subsampleFactor << std::endl;
		std::cout << "sigma: " << curSigma << std::endl;

		itk::Image<double, 3U>::Pointer curFixedImage = fixedImage;
		itk::Image<double, 3U>::Pointer curMovingImage = movingImage;
		itk::Image<double, 3U>::Pointer curFixedImageWeights = fixedImageWeights;
		itk::Image<double, 3U>::Pointer curMovingImageWeights = movingImageWeights;
		itk::Image<double, 3U>::Pointer curFixedImageMask = fixedImageMask;
		itk::Image<double, 3U>::Pointer curMovingImageMask = movingImageMask;

	    curFixedImage = itk::IPT<double, 3U>::SmoothImage(curFixedImage, curSigma);
		curMovingImage = itk::IPT<double, 3U>::SmoothImage(curMovingImage, curSigma);
		curFixedImage = itk::IPT<double, 3U>::SubsampleImage(curFixedImage, subsampleFactor);
		curMovingImage = itk::IPT<double, 3U>::SubsampleImage(curMovingImage, subsampleFactor);
		curFixedImageWeights = itk::IPT<double, 3U>::SubsampleImage(curFixedImageWeights, subsampleFactor);
		curMovingImageWeights = itk::IPT<double, 3U>::SubsampleImage(curMovingImageWeights, subsampleFactor);
		curFixedImageMask = itk::IPT<double, 3U>::SubsampleImage(curFixedImageMask, subsampleFactor);
		curMovingImageMask = itk::IPT<double, 3U>::SubsampleImage(curMovingImageMask, subsampleFactor);

		//char buf[512];
		//sprintf(buf, "fixedImage%d.png", i);
		//itk::IPT<double, 3U>::SaveImageU16(buf, curFixedImage);
		//sprintf(buf, "movingImage%d.png", i);
		//itk::IPT<double, 3U>::SaveImageU16(buf, curMovingImage);

		if(rigidMode == false) {
			curTransform = apply_registration_alpha_smd_internal_3d_affine(curFixedImage,
				curFixedImageMask,
				curFixedImageWeights,
				curMovingImage,
				curMovingImageMask,
				curMovingImageWeights,
				param,
				curTransform,
				alphaLevels,
				useSymmetricMeasure,
				useSquaredMeasure,
				offsetCenter,
				maxDistance,
				outlierRejection,
				distanceOut);
		} else {
			std::cout << "[INTERNAL ERROR] Rigid not supported in 3D..." << std::endl;
			std::exit(-1);
		}

		if (i + 1 < numberOfLevels) {
			curTransform = itk::IPT<double, 3U>::GetNthTransform(curTransform, 1);
		}
		std::cout << curTransform << std::endl;
	}

	return curTransform;	
}

#endif // REGISTRATION_ROUTINES_H
