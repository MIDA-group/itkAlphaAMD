
#include <stdio.h>
#include <string.h>
#include <string>

#include "common/itkImageProcessingTools.h"
#include "registration/registrationRoutines.h"
#include "itkTextOutput.h"

struct RegistrationProgramParam
{
	std::string inputs[2];
	std::string outputPath;

	std::string masks[2];
	std::string weights[2];

	std::string metric;

	std::string initialTransformPath;

	int iterations;
	double relaxationFactor;
	double learningRate;
	unsigned int alphaLevels;
	double alphaMaxDistance;
	double alphaOutlierRejection;
	double samplingFraction;
	double normalization;
	bool histogramMatching;
	bool alphaSquared;
	bool mode3D;
	bool rigidMode;
	unsigned int seed;

	std::string spacingMode;

	std::vector<unsigned int> multiscaleSamplingFactors;
	std::vector<double> multiscaleSmoothingSigmas;
};

template <unsigned int Dim>
static void DoRegister(RegistrationProgramParam &param)
{
	// Types

	typedef itk::IPT<double, Dim> IPT;

	typedef typename IPT::ImageType ImageType;
	typedef typename IPT::ImagePointer ImagePointer;
	typedef typename itk::AffineTransform<double, Dim> AffineTransformType;

	itk::MultiThreader::SetGlobalMaximumNumberOfThreads(1);
	itk::MultiThreader::SetGlobalDefaultNumberOfThreads(1);
    itk::OutputWindow::SetInstance(itk::TextOutput::New());

	// Timing

	typedef itk::RealTimeClock ClockType;
	typedef ClockType::Pointer ClockPointer;
	typedef ClockType::TimeStampType TimeStampType;
	ClockPointer timer = itk::RealTimeClock::New();
	TimeStampType beginTime, endTime;
	double elapsedTime;

	ImagePointer images[2];
	ImagePointer imageMasks[2];
	ImagePointer imageWeights[2];
	double distance = 0.0;

	typename IPT::SpacingMode spacingMode = IPT::kDefaultSpacingMode;

	if (param.spacingMode == "remove")
	{
		spacingMode = IPT::kRemoveSpacingMode;
	}
	else if (param.spacingMode == "resample")
	{
		spacingMode = IPT::kResampleSpacingMode;
	}
	else if (param.spacingMode != "default")
	{
		std::cout << "Invalid spacing mode: " << param.spacingMode << " (remove, resample, default)" << std::endl;
	}

	beginTime = timer->GetTimeInSeconds();

	for (unsigned int i = 0; i < 2; ++i)
	{
		std::cout << "Loading main image " << (i + 1) << std::endl;

		images[i] = IPT::RemoveSpacing(IPT::LoadImage(param.inputs[i].c_str()), spacingMode);

		itk::PrintStatistics<ImageType>(images[i], "Main");

		if (param.masks[i] == "circle")
		{
			imageMasks[i] = IPT::CircularMask(images[i]->GetLargestPossibleRegion(), images[i]->GetSpacing(), 0.0, 1.0); //IPT::RectMask(images[i]->GetLargestPossibleRegion(), images[i]->GetSpacing(), 1.0);
			itk::PrintStatistics<ImageType>(imageMasks[i], "Mask");
		}
		else if (!param.masks[i].empty())
		{
			std::cout << "Loading mask " << (i + 1) << std::endl;
			imageMasks[i] = IPT::RemoveSpacing(IPT::LoadImage(param.masks[i].c_str()), spacingMode);
			itk::PrintStatistics<ImageType>(imageMasks[i], "Mask");

			//imageMasks[i] = IPT::PercentileRescaleImage(imageMasks[i], 0.0);
			imageMasks[i] = IPT::NormalizeImage(imageMasks[i], IPT::IntensityMinMax(imageMasks[i], 0.0));
		}
		else
		{
			imageMasks[i] = IPT::RectMask(images[i]->GetLargestPossibleRegion(), images[i]->GetSpacing(), 1.0);
			itk::PrintStatistics<ImageType>(imageMasks[i], "Mask");
		}

		if (param.weights[i] == "circle")
		{
			imageWeights[i] = IPT::CircularMask(images[i]->GetLargestPossibleRegion(), images[i]->GetSpacing(), 0.0, 1.0);
			itk::PrintStatistics<ImageType>(imageWeights[i], "Weights");
		}
		else if (param.weights[i] == "hann")
		{
			imageWeights[i] = IPT::HannMask(images[i]->GetLargestPossibleRegion(), images[i]->GetSpacing());
			itk::PrintStatistics<ImageType>(imageWeights[i], "Weights");
		}
		else if (param.weights[i] == "hannsqrt")
		{
			imageWeights[i] = IPT::HannMask(images[i]->GetLargestPossibleRegion(), images[i]->GetSpacing(), 0.5);
			itk::PrintStatistics<ImageType>(imageWeights[i], "Weights");
		}
		else if (param.weights[i] == "hann2")
		{
			imageWeights[i] = IPT::HannMask(images[i]->GetLargestPossibleRegion(), images[i]->GetSpacing(), 2.0);
			itk::PrintStatistics<ImageType>(imageWeights[i], "Weights");
		}
		else if (param.weights[i] == "hann3")
		{
			imageWeights[i] = IPT::HannMask(images[i]->GetLargestPossibleRegion(), images[i]->GetSpacing(), 3.0);
			itk::PrintStatistics<ImageType>(imageWeights[i], "Weights");
		}
		else if (param.weights[i] == "hann4")
		{
			imageWeights[i] = IPT::HannMask(images[i]->GetLargestPossibleRegion(), images[i]->GetSpacing(), 4.0);
			itk::PrintStatistics<ImageType>(imageWeights[i], "Weights");
			//char buf[512];
			//sprintf(buf, "hann4_%d.png", i);
			//itk::IPT<double, Dim>::SaveImageU16(buf, imageWeights[i]);
		}
		else if (param.weights[i] == "hann5")
		{
			imageWeights[i] = IPT::HannMask(images[i]->GetLargestPossibleRegion(), images[i]->GetSpacing(), 5.0);
			itk::PrintStatistics<ImageType>(imageWeights[i], "Weights");
			//char buf[512];
			//sprintf(buf, "hann4_%d.png", i);
			//itk::IPT<double, Dim>::SaveImageU16(buf, imageWeights[i]);
		}
		else if (!param.weights[i].empty())
		{
			std::cout << "Loading weights " << (i + 1) << std::endl;
			imageWeights[i] = IPT::RemoveSpacing(IPT::LoadImage(param.weights[i].c_str()), spacingMode);
			itk::PrintStatistics<ImageType>(imageWeights[i], "Weights");
		}
		else
		{
			imageWeights[i] = IPT::RectMask(images[i]->GetLargestPossibleRegion(), images[i]->GetSpacing(), 1.0); //IPT::CircularMask(images[i]->GetLargestPossibleRegion(), images[i]->GetSpacing(), 0.0, 1.0);
		}
	}

	// Load initial transform

	typename itk::Transform<double, Dim, Dim>::Pointer initTransform;
	if(!param.initialTransformPath.empty()) {
	 	initTransform = IPT::LoadTransformFile(param.initialTransformPath.c_str());
		if(initTransform)
		{
			std::cout << "Initial transform: " << (initTransform) << std::endl;
		} else {
			std::cout << "Error reading transform file." << std::endl;
		}
	} else {
		std::cout << "No initial transform." << std::endl;
	}

	endTime = timer->GetTimeInSeconds();

	elapsedTime = static_cast<double>(endTime - beginTime);

	std::cout << "Loading and preprocessing time: " << elapsedTime << std::endl;

	// Setup registration parameters
	RegistrationParam regParam;

	regParam.iterations = param.iterations;
	regParam.numberOfLevels = 3;
	regParam.sigmaIncrement = 0.5;
	regParam.learningRate = param.learningRate;
	regParam.relaxationFactor = param.relaxationFactor;
	regParam.samplingPercentage = param.samplingFraction;
	regParam.metricSeed = param.seed;
	regParam.normalizationPercentage = param.normalization;
	regParam.normalizationEnabled = param.normalization >= 0.0;
	regParam.multiscaleSamplingFactors = param.multiscaleSamplingFactors;
	regParam.multiscaleSmoothingSigmas = param.multiscaleSmoothingSigmas;

	typename itk::Transform<double, Dim, Dim>::Pointer resultTransform;

	beginTime = timer->GetTimeInSeconds();

	if (param.metric == "ssd")
	{
		typedef itk::MeanSquaresImageToImageMetricv4<ImageType, ImageType, ImageType> MetricType;
		typedef typename MetricType::Pointer MetricPointer;
		MetricPointer metric = MetricType::New();

		regParam.learningRate = param.learningRate;

		resultTransform = apply_registration<AffineTransformType, MetricType, Dim>(
			images[0],
			imageMasks[0],
			images[1],
			imageMasks[1],
			metric,
			regParam,
			&distance);
	}
	else if (param.metric == "ncc")
	{
		typedef itk::CorrelationImageToImageMetricv4<ImageType, ImageType, ImageType> MetricType;
		typedef typename MetricType::Pointer MetricPointer;
		MetricPointer metric = MetricType::New();

		regParam.learningRate = param.learningRate;

		resultTransform = apply_registration<AffineTransformType, MetricType, Dim>(
			images[0],
			imageMasks[0],
			images[1],
			imageMasks[1],
			metric,
			regParam,
			&distance);
	}
	else if (param.metric == "mi")
	{
		typedef itk::MattesMutualInformationImageToImageMetricv4<ImageType, ImageType, ImageType> MetricType;
		typedef typename MetricType::Pointer MetricPointer;
		MetricPointer metric = MetricType::New();

		regParam.learningRate = 0.5 * param.learningRate;
		metric->SetNumberOfHistogramBins(20);

		resultTransform = apply_registration<AffineTransformType, MetricType, Dim>(
			images[0],
			imageMasks[0],
			images[1],
			imageMasks[1],
			metric,
			regParam,
			&distance);
	}
	else if (param.metric == "alpha_asmd" || param.metric == "alpha_smd")
	{
		regParam.learningRate = param.learningRate;
		bool isSymmetric = (param.metric == "alpha_smd");
		itk::CovariantVector<double, Dim> offsetCenter;
		offsetCenter.Fill(0.0);

		resultTransform =
			apply_registration_alpha_smd<Dim>(
				images[0],
				imageMasks[0],
				imageWeights[0],
				images[1],
				imageMasks[1],
				imageWeights[1],
				regParam,
				initTransform,
				param.alphaLevels,
				isSymmetric,
				param.alphaSquared,
				offsetCenter,
				param.alphaMaxDistance,
				param.alphaOutlierRejection,
				param.rigidMode,
			    &distance);
	}
	else
	{
		std::cout << "No registration performed." << std::endl;
		resultTransform = noop_registration<Dim>(images[0], images[1]);
	}

	endTime = timer->GetTimeInSeconds();

	elapsedTime = static_cast<double>(endTime - beginTime);

	std::cout << "Metric: " << param.metric << std::endl;
	std::cout << "Final distance: " << distance << std::endl;

	std::cout << resultTransform << std::endl;

	std::string transformCompletePath = param.outputPath + "transform_complete.txt";

	IPT::SaveTransformFile(transformCompletePath.c_str(), resultTransform);

	// Save transformation
	if (IPT::GetTransformCount(resultTransform) > 1)
	{
		const char* specFileName;

		// Save only the affine or rigid transformation
		if (param.rigidMode)
		{
			specFileName = "transform_rigid.txt";	
		}
		else
		{
			specFileName = "transform_affine.txt";	
		}

		std::string transformSpecificPath = param.outputPath + specFileName;

		IPT::SaveTransformFile(transformSpecificPath.c_str(), IPT::GetNthTransform(resultTransform, 1U));
	}

	std::string distancePath = param.outputPath + "distance.csv";
	std::vector<double> distanceVector; distanceVector.push_back(distance);

	itk::VectorToCSV(distancePath.c_str(), distanceVector);

	std::cout << "Registration time elapsed:  " << elapsedTime << std::endl;
}

int main(int argc, char **argv)
{
	RegistrationProgramParam param;

	bool vol3DMode = false;

	// Defaults

	param.outputPath = "./RegisterOutput/";

	param.learningRate = 0.5;
	param.relaxationFactor = 0.95;
	param.metric = "none";
	param.iterations = 2000;
	param.alphaLevels = 7;
	param.mode3D = false;
	param.seed = 121212;
	param.alphaMaxDistance = 0.0;
	param.alphaOutlierRejection = 0.05;
	param.samplingFraction = 0.25;
	param.alphaSquared = false;
	param.spacingMode = "default";
	param.normalization = 0.02;
	param.histogramMatching = true;

	param.multiscaleSamplingFactors.push_back(1);

	param.multiscaleSmoothingSigmas.push_back(0.0);

	// Parameters
	for (int pi = 1; pi + 1 < argc; pi += 2)
	{
		const char* mod = argv[pi];
		const char* arg = argv[pi + 1];

		if (strcmp(mod, "-in1") == 0)
		{
			param.inputs[0] = arg;
		}
		else if (strcmp(mod, "-in2") == 0)
		{
			param.inputs[1] = arg;
		}
		else if (strcmp(mod, "-out") == 0)
		{
			param.outputPath = arg;
		}
		else if (strcmp(mod, "-mask1") == 0)
		{
			param.masks[0] = arg;
		}
		else if (strcmp(mod, "-mask2") == 0)
		{
			param.masks[1] = arg;
		}
		else if (strcmp(mod, "-weights1") == 0)
		{
			param.weights[0] = arg;
		}
		else if (strcmp(mod, "-weights2") == 0)
		{
			param.weights[1] = arg;
		}
		else if (strcmp(mod, "-iterations") == 0)
		{
			param.iterations = atoi(arg);
		}
		else if (strcmp(mod, "-metric") == 0)
		{
			param.metric = arg;
		}
		else if (strcmp(mod, "-alpha_levels") == 0)
		{
			param.alphaLevels = (unsigned int)atoi(arg);
		}
		else if (strcmp(mod, "-alpha_max_distance") == 0)
		{
			param.alphaMaxDistance = atof(arg);
		}
		else if (strcmp(mod, "-seed") == 0)
		{
			param.seed = atoi(arg);
		}
		else if (strcmp(mod, "-learning_rate") == 0)
		{
			param.learningRate = atof(arg);
		}
		else if (strcmp(mod, "-relaxation_factor") == 0)
		{
			param.relaxationFactor = atof(arg);
		}
		else if (strcmp(mod, "-alpha_outlier_rejection") == 0)
		{
			param.alphaOutlierRejection = atof(arg);
		}
		else if (strcmp(mod, "-sampling_fraction") == 0)
		{
			param.samplingFraction = atof(arg);
		}
		else if (strcmp(mod, "-spacing_mode") == 0)
		{
			param.spacingMode = arg;
		}
		else if (strcmp(mod, "-init_transform") == 0)
		{
			param.initialTransformPath = arg;
		}
		else if (strcmp(mod, "-alpha_squared") == 0)
		{
			param.alphaSquared = (0 != (unsigned int)atoi(arg));
		}
		else if (strcmp(mod, "-histogram_matching") == 0)
		{
			param.histogramMatching = (0 != (unsigned int)atoi(arg));
		}
		else if (strcmp(mod, "-normalization") == 0)
		{
			if (strcmp(arg, "off") == 0)
			{
				param.normalization = -1.0;
			}
			else
			{
				param.normalization = atof(arg);
				if (param.normalization < 0.0)
					param.normalization = 0.0;
				if (param.normalization > 1.0)
					param.normalization = 1.0;
			}
		}
		else if (strcmp(mod, "-3d") == 0)
		{
			vol3DMode = (0 != (unsigned int)atoi(arg));
		}
		else if (strcmp(mod, "-rigid") == 0)
		{
			param.rigidMode = (0 != (unsigned int)atoi(arg));
		}
		else if (strcmp(mod, "-multiscale_sampling_factors") == 0)
		{
			std::string s = arg;

			param.multiscaleSamplingFactors = itk::ParseSamplingFactors(s);
		}
		else if (strcmp(mod, "-multiscale_smoothing_sigmas") == 0)
		{
			std::string s = arg;

			param.multiscaleSmoothingSigmas = itk::ParseSmoothingSigmas(s);
		}
		else {
			std::cerr << "Warning!!! Illegal parameter '" << mod << "' with argument '" << arg << "' found." << std::endl;
		}
	}

	if(param.multiscaleSamplingFactors.size() != param.multiscaleSmoothingSigmas.size())
	{
	  std::cerr << "Error: Multiscale sampling factor (" << param.multiscaleSamplingFactors.size() << ") and smoothing sigmas (" << param.multiscaleSmoothingSigmas.size() << ")  count mismatch." << std::endl;
		return -1;
	}

	if (vol3DMode)
	{
		if (param.rigidMode)
		{
			std::cout << "Rigid registration is only supported in 2D currently." << std::endl;
			return -1;
		}
		DoRegister<3U>(param);
	}
	else
	{
		DoRegister<2U>(param);
	}

	return 0;
}
