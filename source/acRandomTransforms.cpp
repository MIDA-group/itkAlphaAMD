
#include "itkAffineTransform.h"

#include <stdio.h>
#include <string.h>
#include <string>

#include <math.h>
#include <cstdlib>
//#include "itklab/itklab.h"

#include "itkRealTimeClock.h"

#include "common/itkImageProcessingTools.h"

#include "itkMersenneTwisterRandomVariateGenerator.h"

struct RandomTransformsParameters
{
	std::string inPath;
	std::string outPath;
	int seed;
	int count;
	double rotation;
	double translation;
	double scaling;
	double minRotation;
	double minTranslation;
	double noiseStdDev;
	unsigned int bitDepth;
	std::string formatExt;
};

template <unsigned int Dim>
struct RandomAffineTransformation
{
	double a[Dim * Dim];
	itk::CovariantVector<double, Dim> t;
};

template <unsigned int Dim>
typename itk::AffineTransform<double, Dim>::Pointer
MakeRandomTransform(RandomAffineTransformation<Dim> rat)
{
	typedef itk::AffineTransform<double, Dim> AffineTransformType;
	typedef typename AffineTransformType::Pointer AffineTransformPointer;

	AffineTransformPointer transform = AffineTransformType::New();

	typename itk::AffineTransform<double, Dim>::ParametersType param(transform->GetNumberOfParameters());
	param.Fill(0);

	for (unsigned int i = 0; i < Dim * Dim; ++i)
	{
		param[i] = rat.a[i];
	}

	transform->SetParameters(param);

	return transform;
}

template <unsigned int Dim>
RandomAffineTransformation<Dim> MakeRandomTransformParam(
	typename itk::Image<double, Dim>::SizeType imageSize,
	double rotation,
	double translation,
	double scaling,
	double minRotation,
	double minTranslation,
	typename itk::Statistics::MersenneTwisterRandomVariateGenerator::Pointer rng)
{
	typedef itk::IPT<double, Dim> IPT;

	typedef itk::AffineTransform<double, Dim> AffineTransformType;
	typedef typename AffineTransformType::Pointer AffineTransformPointer;

	RandomAffineTransformation<Dim> result;

	const double radians = (rotation / 360) * 2.0 * 3.14159265358979323846;
	const double minRadians = (minRotation / 360) * 2.0 * 3.14159265358979323846;

	bool anyAboveMin = false;
	if(minTranslation <= 0.0 || minRotation <= 0.0)
		anyAboveMin = true;

	if (Dim == 2)
	{
		double theta;
		theta = -radians + 2.0 * radians * rng->GetVariate();

		if(fabs(theta) >= minRadians)
			anyAboveMin = true;

		const double scale = 1.0 - (1 - scaling) + 2.0 * (1 - scaling) * rng->GetVariate();
		const double costheta = scale * cos(theta);
		const double sintheta = scale * sin(theta);

		result.a[0] = costheta;
		result.a[1] = -sintheta;
		result.a[2] = sintheta;
		result.a[3] = costheta;
	}
	else if (Dim == 3)
	{
		AffineTransformPointer transform = AffineTransformType::New();

		transform->SetIdentity();

		typename itk::AffineTransform<double, Dim>::ParametersType param(transform->GetNumberOfParameters());

		param = transform->GetParameters();

		unsigned int axisIndices[3] = {0, 1, 2};

		// Randomly shuffle the axis indices

		unsigned int cnt = 2;
		for (unsigned int i = 0; i < Dim - 1; ++i)
		{
			unsigned int k = rng->GetIntegerVariate(cnt);
			std::swap(axisIndices[k], axisIndices[i]);
			--cnt;
		}

		for (unsigned int i = 0; i < Dim; ++i)
		{
			typename IPT::VectorType axis;
			axis.Fill(0);
			axis[axisIndices[i]] = 1.0;

			const double theta = -radians + 2.0 * radians * rng->GetVariate();
			if(fabs(theta) >= minRadians)
				anyAboveMin = true;

			transform->Rotate3D(axis, theta);
		}

		transform->Scale(1.0 - (1 - scaling) + 2.0 * (1 - scaling) * rng->GetVariate());

		for (unsigned int i = 0; i < Dim * Dim; ++i)
		{
			result.a[i] = transform->GetParameters()[i];
		}
	}

	// Translation

	for (unsigned int i = 0; i < Dim; ++i)
	{
		double tfac = (2.0 * rng->GetVariate() - 1.0);
		double minTrans_i = (minTranslation * 0.01) * imageSize[i];
		double t_i = (translation * 0.01) * imageSize[i] * tfac;
		if(fabs(t_i) >= minTrans_i)
			anyAboveMin = true;
		result.t[i] = t_i;
	}

	if(anyAboveMin)
		return result;
	else {
		std::cout << "Oops - transformation too small - trying again." << std::endl;
		return MakeRandomTransformParam<Dim>(imageSize, rotation, translation, scaling, minRotation, minTranslation,	rng); // Try again
	}
}

template <unsigned int Dim>
void MakeRandomTransforms(RandomTransformsParameters param)
{
	typedef itk::IPT<double, Dim> IPT;

	typedef typename IPT::ImageType ImageType;
	typedef typename IPT::ImagePointer ImagePointer;
	typedef typename IPT::PointSetType PointSetType;

	typedef itk::RealTimeClock ClockType;
	typedef ClockType::Pointer ClockPointer;
	typedef ClockType::TimeStampType TimeStampType;
	ClockPointer timer = itk::RealTimeClock::New();

	typedef itk::AffineTransform<double, Dim> AffineTransformType;
	typedef typename AffineTransformType::Pointer AffineTransformPointer;

	itk::MultiThreader::SetGlobalMaximumNumberOfThreads(1);
	itk::MultiThreader::SetGlobalDefaultNumberOfThreads(1);

	char buf[512];

	ImagePointer image;
	try {
	  image = IPT::LoadImage(param.inPath.c_str());
	} catch(itk::ExceptionObject & err) {
		std::cout << err << std::endl;
		return;
	}

	typename IPT::SizeType imageSize = IPT::GetImageSize(image);

	unsigned int N = param.count;

	std::vector<RandomAffineTransformation<Dim>> transforms;

	// Create transformation random number generator.

	typename itk::Statistics::MersenneTwisterRandomVariateGenerator::Pointer rng = itk::Statistics::MersenneTwisterRandomVariateGenerator::New();
	rng->SetSeed(param.seed + 1337);

	// Create all random transforms
	for (unsigned int kind = 0; kind < N; ++kind)
	{
		RandomAffineTransformation<Dim> rat = MakeRandomTransformParam<Dim>(
			imageSize, param.rotation, param.translation, param.scaling, param.minRotation, param.minTranslation, rng);
		transforms.push_back(rat);
	}

	std::string imExt;

	imExt = "." + param.formatExt;

	double p = 0.005;

	ImagePointer originalImage = IPT::NormalizeImage(image, IPT::IntensityMinMax(image, 0.0));
	ImagePointer originalMask = IPT::CircularMask(originalImage->GetLargestPossibleRegion(), originalImage->GetSpacing(), 0.0, 1.0);

	PointSetType originalLandmarks;
	
	IPT::ComputeImageEdgeVertices(originalImage, originalLandmarks, true);

	sprintf(buf, "%sref_landmarks.csv", param.outPath.c_str());

	IPT::SavePointSet(buf, originalLandmarks);

	for (unsigned int kind = 0; kind < N; ++kind)
	{
		ImagePointer noisyImageUnNormalized = IPT::AdditiveNoise(originalImage, param.noiseStdDev, 0.0, param.seed + 1 + 10 * kind, true);
		ImagePointer noisyImageUnNormalized2 = IPT::AdditiveNoise(originalImage, param.noiseStdDev, 0.0, param.seed + 1 + 10 * kind + 71, true);
		ImagePointer noisyImage = IPT::NormalizeImage(noisyImageUnNormalized, IPT::IntensityMinMax(noisyImageUnNormalized, p));
		ImagePointer noisyImage2 = IPT::NormalizeImage(noisyImageUnNormalized2, IPT::IntensityMinMax(noisyImageUnNormalized2, p));	

		sprintf(buf, "%sref_image_%d%s", param.outPath.c_str(), kind + 1, imExt.c_str());

		if(param.bitDepth == 8U) {
			IPT::SaveImageU8(buf, noisyImage);
		} else if(param.bitDepth == 16U) {
			IPT::SaveImageU16(buf, noisyImage);
		}

		AffineTransformPointer transform = MakeRandomTransform<Dim>(transforms[kind]);

		typename IPT::CovariantVectorType t = transforms[kind].t;

		PointSetType cornersList;

		typedef itk::AffineTransform<double, Dim> AffineTransformType;
		typedef typename IPT::BaseTransformType BaseTransformType;
		typedef typename IPT::BaseTransformPointer BaseTransformPointer;

		BaseTransformPointer baseTransform =  static_cast<BaseTransformPointer>(transform.GetPointer());

		ImagePointer image2 = IPT::TransformImageAutoCrop(noisyImage2, baseTransform, IPT::kImwarpInterpCubic, &cornersList, 0.0);

		cornersList = IPT::TranslatePointSet(IPT::ReLUVector(IPT::RoundVector(IPT::MakeVector(t))), cornersList);

		ImagePointer image3 = IPT::TranslateImage(image2, IPT::RoundCovariantVector(t), 0.0);
		
		sprintf(buf, "%stransformed_image_%d%s", param.outPath.c_str(), kind + 1, imExt.c_str());

		if(param.bitDepth == 8U)
			IPT::SaveImageU8(buf, image3);
		else
			IPT::SaveImageU16(buf, image3);

		ImagePointer transformedMask = IPT::TranslateImage(IPT::TransformImageAutoCrop(originalMask, baseTransform, IPT::kImwarpInterpNearest, 0), IPT::RoundCovariantVector(t), 0.0);

		sprintf(buf, "%stransformed_mask_%d%s", param.outPath.c_str(), kind + 1, imExt.c_str());

		if(param.bitDepth == 8U)
			IPT::SaveImageU8(buf, transformedMask);
		else
			IPT::SaveImageU16(buf, transformedMask);

		sprintf(buf, "%stransformed_landmarks_%d.csv", param.outPath.c_str(), kind + 1);

		IPT::SavePointSet(buf, cornersList);
	}

}

int main(int argc, char** argv) {
	
	RandomTransformsParameters param;
	
	param.seed = 1477;
	param.count = 1;
	param.rotation = 10;
	param.translation = 10;
	param.scaling = 1.0;
	param.minRotation = 0;
	param.minTranslation = 0;
	param.noiseStdDev = 0.0;
	param.bitDepth = 16U;
	param.formatExt = "tif";
    bool vol3DMode = false;

	for(int pi = 1; pi + 1 < argc; pi += 2) {
		const char* mod = argv[pi];
		const char* arg = argv[pi + 1];

		if(strcmp(mod, "-in") == 0) {
			param.inPath = arg;
		} else if(strcmp(mod, "-out") == 0) {
			param.outPath = arg;
		} else if(strcmp(mod, "-seed") == 0) {
			param.seed = atoi(arg);
		} else if(strcmp(mod, "-count") == 0) {
			param.count = atoi(arg);
        } else if(strcmp(mod, "-rotation") == 0) {
			param.rotation = atof(arg);
        } else if(strcmp(mod, "-translation") == 0) {
			param.translation = atof(arg);
        } else if(strcmp(mod, "-scaling") == 0) {
			param.scaling = atof(arg);
        } else if(strcmp(mod, "-min_rotation") == 0) {
			param.minRotation = atof(arg);
        } else if(strcmp(mod, "-min_translation") == 0) {
			param.minTranslation = atof(arg);
		} else if(strcmp(mod, "-noise") == 0) {
            param.noiseStdDev = atof(arg);
		} else if(strcmp(mod, "-format_ext") == 0) {
            param.formatExt = arg;
		} else if(strcmp(mod, "-dim") == 0) {
			unsigned int dim = atoi(arg);
			if(dim == 3U) {
				vol3DMode = true;
			} else if(dim == 2U) {
				vol3DMode = false;
			} else {
				std::cerr << "Illegal dimension '" << dim << "' given. Only '2' or '3' allowed." << std::endl;
				return -1;
			}
		} else if(strcmp(mod, "-bit_depth") == 0) {
            param.bitDepth = atoi(arg);
		}
	}

	std::cout << "Input path: " << param.inPath << std::endl;

	if(vol3DMode) {
		MakeRandomTransforms<3U>(param);
	} else {
		MakeRandomTransforms<2U>(param);
	}

	return 0;
}
