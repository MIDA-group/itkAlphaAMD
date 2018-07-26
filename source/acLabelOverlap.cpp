
#include <stdio.h>
#include <string.h>
#include <string>

#include "common/itkImageProcessingTools.h"
#include "itkTextOutput.h"

struct TransformProgramParam
{	
	std::string inputPath1;
	std::string inputPath2;

	std::string outputPath;

	std::string labelMode;

	unsigned int bitDepth1;
	unsigned int bitDepth2;
	unsigned int maxLabel;
};

template <unsigned int Dim>
static void DoOverlap(TransformProgramParam &param)
{
	// Types

	typedef itk::IPT<double, Dim> IPT;

	typedef typename IPT::ImageType ImageType;
	typedef typename IPT::ImagePointer ImagePointer;
	typedef itk::Image<unsigned short, Dim> LabelImageType;
	typedef typename LabelImageType::Pointer LabelImagePointer;

	itk::MultiThreader::SetGlobalMaximumNumberOfThreads(1);
	itk::MultiThreader::SetGlobalDefaultNumberOfThreads(1);
	itk::OutputWindow::SetInstance(itk::TextOutput::New());

	if(param.inputPath1.empty()) {
		std::cout << "Error: No reference image provided." << std::endl;
		return;
	}
	if(param.inputPath2.empty()) {
		std::cout << "Error: No transformed image provided." << std::endl;
		return;
	}

	if(param.labelMode == "multi") {
		LabelImagePointer refImage;
		LabelImagePointer floImage;

		if(param.maxLabel > 0) {
			ImagePointer referenceImageDouble = IPT::LoadImage(param.inputPath1.c_str());
			ImagePointer floatingImageDouble = IPT::LoadImage(param.inputPath2.c_str());

			typedef typename IPT::ImageStatisticsData ISD;

			ISD refStats = IPT::ImageStatistics(referenceImageDouble);
			ISD floStats = IPT::ImageStatistics(floatingImageDouble);

			referenceImageDouble = IPT::MultiplyImageByConstant(referenceImageDouble, param.maxLabel / refStats.max);
			floatingImageDouble = IPT::MultiplyImageByConstant(floatingImageDouble, param.maxLabel / floStats.max);

			refImage = itk::CastImage<ImageType, LabelImageType>(referenceImageDouble);
			floImage = itk::CastImage<ImageType, LabelImageType>(floatingImageDouble);
			//itk::PrintStatistics<ImageType>(referenceImageDouble, "ReferenceDouble");
			//itk::PrintStatistics<ImageType>(floatingImageDouble, "TransformedDouble");

		} else {
			refImage = IPT::LoadLabelImage(param.inputPath1.c_str());
			floImage = IPT::LoadLabelImage(param.inputPath2.c_str());			
		}

		itk::PrintStatistics<LabelImageType>(refImage, "Reference");
		itk::PrintStatistics<LabelImageType>(floImage, "Transformed");

		std::vector<typename IPT::OverlapScore> labelOverlap = IPT::ComputeLabelOverlap(refImage, floImage);
		
		IPT::SaveLabelOverlapCSV(param.outputPath.c_str(), labelOverlap, true);
	} else if(param.labelMode == "binary") {
		ImagePointer referenceImageDouble = IPT::LoadImage(param.inputPath1.c_str());
		ImagePointer floatingImageDouble = IPT::LoadImage(param.inputPath2.c_str());

		const double threshold = 0.0000001;
		
		LabelImagePointer refImage = IPT::ThresholdToLabelImage(referenceImageDouble, threshold);
		LabelImagePointer floImage = IPT::ThresholdToLabelImage(floatingImageDouble, threshold);

		itk::PrintStatistics<ImageType>(referenceImageDouble, "Reference Double");
		itk::PrintStatistics<ImageType>(floatingImageDouble, "Transformed Double");

		itk::PrintStatistics<LabelImageType>(refImage, "Reference");
		itk::PrintStatistics<LabelImageType>(floImage, "Transformed");

		std::vector<typename IPT::OverlapScore> labelOverlap = IPT::ComputeLabelOverlap(refImage, floImage);
		
		IPT::SaveLabelOverlapCSV(param.outputPath.c_str(), labelOverlap, true);
	}
}

int main(int argc, char **argv)
{
	TransformProgramParam param;

	unsigned int dim = 2U;

	// Defaults
	param.labelMode = "multi";
	param.bitDepth1 = 16U;
	param.bitDepth2 = 16U;
	param.maxLabel = 0U;

	// Parameters
	for (int pi = 1; pi + 1 < argc; pi += 2)
	{
		const char* mod;
		const char* arg;

		mod = argv[pi];
		arg = argv[pi + 1];

		if (strcmp(mod, "-in1") == 0)
		{
			param.inputPath1 = arg;
		}
		else if (strcmp(mod, "-in2") == 0)
		{
			param.inputPath2 = arg;
		}
		else if (strcmp(mod, "-out") == 0)
		{
			param.outputPath = arg;
		}
		else if (strcmp(mod, "-label_mode") == 0)
		{
			std::string mode = arg;
			param.labelMode = arg;

			if(mode == "multi") {
			} else if(mode == "binary") {
			} else {
				std::cout << "Illegal label mode: '" << mode << "', allowed modes are: multi, binary." << std::endl;
			}
		}
		else if (strcmp(mod, "-max_label") == 0)
		{
			param.maxLabel = atoi(arg);
		}
		else if (strcmp(mod, "-bit_depth1") == 0)
		{
			param.bitDepth1 = atoi(arg);
			if(param.bitDepth1 != 8U && param.bitDepth1 != 16U)
			{
				std::cout << "Illegal bit depth for reference image: '" << param.bitDepth1 << "', allowed bit depths are: 8, 16." << std::endl;
			}
		}
		else if (strcmp(mod, "-bit_depth2") == 0)
		{
			param.bitDepth2 = atoi(arg);
			if(param.bitDepth2 != 8U && param.bitDepth2 != 16U)
			{
				std::cout << "Illegal bit depth for transformed image: '" << param.bitDepth2 << "', allowed bit depths are: 8, 16." << std::endl;
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
		DoOverlap<3U>(param);
	}
	else if(dim == 2U)
	{
		DoOverlap<2U>(param);
	}

	return 0;
}
