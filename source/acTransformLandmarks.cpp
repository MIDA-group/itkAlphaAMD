
#include <stdio.h>
#include <string.h>
#include <string>

#include "common/itkImageProcessingTools.h"

struct TransformProgramParam
{
	std::string transformPath;

	std::string inputPath;
	std::string outputPath;
};

template <unsigned int Dim>
static void DoTransform(TransformProgramParam &param)
{
	// Types

	typedef itk::IPT<double, Dim> IPT;

	typedef typename IPT::ImageType ImageType;
	typedef typename IPT::ImagePointer ImagePointer;
	typedef itk::Image<unsigned short, Dim> LabelImageType;
	typedef typename LabelImageType::Pointer LabelImagePointer;

	itk::MultiThreader::SetGlobalMaximumNumberOfThreads(1);
	itk::MultiThreader::SetGlobalDefaultNumberOfThreads(1);

	std::cout << "Loading transform" << std::endl;

	typename itk::Transform<double, Dim, Dim>::Pointer transform = IPT::LoadTransformFile(param.transformPath.c_str());

	if(transform.GetPointer()) {
		std::cout << transform << std::endl;
	} else {
		std::cout << "Failed to load transform." << std::endl;
	}

	typedef typename IPT::PointSetType PointSetType;

	typename itk::Transform<double, Dim, Dim>::Pointer invTransform = transform->GetInverseTransform();

	PointSetType ps;

	std::cout << "Loading landmarks" << std::endl;

	ps = IPT::LoadPointSet(param.inputPath.c_str());

	std::cout << ps.size() << " landmark points loaded." << std::endl;
	for(size_t i = 0; i < ps.size(); ++i) {
		std::cout << ps[i] << std::endl;
	}

	std::cout << "Transforming landmarks" << std::endl;

	PointSetType tps = IPT::TransformPointSet(invTransform, ps);

	std::cout << tps.size() << " landmark points transformed." << std::endl;
	for(size_t i = 0; i < tps.size(); ++i) {
		std::cout << tps[i] << std::endl;
	}

	std::cout << "Saving landmarks" << std::endl;

	IPT::SavePointSet(param.outputPath.c_str(), tps);

	std::cout << "Done" << std::endl;
}

int main(int argc, char **argv)
{
	TransformProgramParam param;

	unsigned int dim = 2U;

	// Parameters
	for (int pi = 1; pi + 1 < argc; pi += 2)
	{
		const char* mod = argv[pi];
		const char* arg = argv[pi + 1];

		if (strcmp(mod, "-transform") == 0)
		{
			param.transformPath = arg;
		}
		else if (strcmp(mod, "-in") == 0)
		{
			param.inputPath = arg;
		}
		else if (strcmp(mod, "-out") == 0)
		{
			param.outputPath = arg;
		}
		else if (strcmp(mod, "-dim") == 0)
		{
			dim = (unsigned int)atoi(arg);
			if (dim < 2 || dim > 3)
			{
				std::cout << "Illegal number of dimensions: '" << dim << "', only 2 or 3 supported." << std::endl;
				return -1;
			}
		}
	}

	if (dim == 3U)
	{
		DoTransform<3U>(param);
	}
	else if (dim == 2U)
	{
		DoTransform<2U>(param);
	}

	return 0;
}
