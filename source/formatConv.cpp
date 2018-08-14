
/**
 * Small program for doing file format conversion.
 **/

#include <stdio.h>
#include <string.h>
#include <string>

#include "Log.h"

#include "common/itkImageProcessingTools.h"

template <unsigned int Dim>
void FormatConv(std::string in, std::string out, bool isLabel, bool is16Bit) {
    typedef itk::IPT<double, Dim> IPT;

    if(isLabel) {
        // is16Bit is ignored here, all label images are 16 bit in our world
        typename IPT::LabelImagePointer image = IPT::LoadLabelImage(in.c_str());

        IPT::SaveLabelImage(out.c_str(), image);
    } else {
        std::cout << "Loading image" << std::endl;
        typename IPT::ImagePointer image = IPT::LoadImage(in.c_str());

        std::cout << "Image loaded" << std::endl;

		itk::PrintStatistics<typename IPT::ImageType>(image, "Image");

        if(is16Bit)
            IPT::SaveImageU16(out.c_str(), image);
        else
            IPT::SaveImageU8(out.c_str(), image);
    }
}

int main(int argc, char **argv)
{
	bool vol3DMode = false;

	// Defaults

    std::string in;
    std::string out;
    bool isLabel = false;
    bool is16Bit = true;

	// Parameters
	for (int pi = 1; pi + 1 < argc; pi += 2)
	{
		char mod[512];
		char arg[512];

		sprintf(mod, "%s", argv[pi]);
		sprintf(arg, "%s", argv[pi + 1]);

		if (strcmp(mod, "-in") == 0)
		{
			in = arg;
		}
		else if (strcmp(mod, "-out") == 0)
		{
			out = arg;
		}
		else if (strcmp(mod, "-3d") == 0)
		{
			vol3DMode = (1 == (unsigned int)atoi(arg));
		}
		else if (strcmp(mod, "-is_label") == 0)
		{
			isLabel = (1 == (unsigned int)atoi(arg));
        } else if (strcmp(mod, "-is_16_bit") == 0)
		{
			is16Bit = (1 == (unsigned int)atoi(arg));
		}
	}

	if(in.length() > 0 && out.length() > 0) {
		if (vol3DMode)
		{
			FormatConv<3U>(in, out, isLabel, is16Bit);
		}
		else
		{
			FormatConv<2U>(in, out, isLabel, is16Bit);
		}
	} else {
		if(in.length() == 0)
			std::cout << "Input file missing." << std::endl;
		if(out.length() == 0)
			std::cout << "Output file missing." << std::endl;
	}

	// Test read/write transforms

	itk::CompositeTransform<double, 2U>::Pointer comp = itk::CompositeTransform<double, 2U>::New();
	itk::AffineTransform<double, 2U>::Pointer affine = itk::AffineTransform<double, 2U>::New();
	itk::TranslationTransform<double, 2U>::Pointer translation = itk::TranslationTransform<double, 2U>::New();
	
	typedef itk::AffineTransform<double, 2U>::ParametersType ParametersType;
	ParametersType affineTransformParam(6U);
	affineTransformParam[0] = 1;
	affineTransformParam[3] = 1;

	itk::AffineTransform<double, 2U>::InputPointType cor;

	affine->SetParameters(affineTransformParam);
  	
	cor.Fill(12);
  	affine->SetCenter(cor);
	
	comp->AddTransform(affine);
	comp->AddTransform(translation);
	
	typedef itk::IPT<double, 2U>::BaseTransformType BaseTransformType;
	typedef itk::IPT<double, 2U>::BaseTransformPointer BaseTransformPointer;
	itk::IPT<double, 2U>::BaseTransformPointer baseTransform = static_cast<BaseTransformType*>(comp.GetPointer());

	itk::IPT<double, 2U>::SaveTransformFile("test.txt", baseTransform);
	BaseTransformPointer loadedTransform = itk::IPT<double, 2U>::LoadTransformFile("test.txt");

	std::cout << (loadedTransform) << std::endl;

	return 0;
}
