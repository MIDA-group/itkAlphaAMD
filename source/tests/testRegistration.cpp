
#include "../registration/alphaBSplineRegistration.h"
#include "../samplers/pointSampler.h"
#include "../metric/mcAlphaCutPointToSetDistance.h"

#include "../common/itkImageProcessingTools.h"
#include "itkTimeProbesCollectorBase.h"

namespace TestRegistration
{
using ImageType = itk::Image<float, 2U>;
using ImagePointer = typename ImageType::Pointer;
using TransformType = itk::BSplineTransform<double, 2U, 3U>;
using TransformPointer = typename TransformType::Pointer;
constexpr unsigned int ImageDimension = 2U;
constexpr unsigned int splineOrder = 3U;

TransformPointer CreateBSplineTransform(ImagePointer image, unsigned int numberOfGridNodes)
{
    typename TransformType::PhysicalDimensionsType fixedPhysicalDimensions;
    typename TransformType::MeshSizeType meshSize;

    TransformPointer transform = TransformType::New();

    for (unsigned int i = 0; i < ImageDimension; i++)
    {
        fixedPhysicalDimensions[i] = image->GetSpacing()[i] *
                                     static_cast<double>(
                                         image->GetLargestPossibleRegion().GetSize()[i] - 1);
    }
    meshSize.Fill(numberOfGridNodes - splineOrder);
    transform->SetTransformDomainOrigin(image->GetOrigin());
    transform->SetTransformDomainPhysicalDimensions(fixedPhysicalDimensions);
    transform->SetTransformDomainMeshSize(meshSize);
    transform->SetTransformDomainDirection(image->GetDirection());

    return transform;
}

typename ImageType::Pointer ApplyTransform(ImagePointer refImage, ImagePointer floImage, TransformPointer transform)
{
    typedef itk::ResampleImageFilter<
        ImageType,
        ImageType>
        ResampleFilterType;

    typedef itk::IPT<double, ImageDimension> IPT;

    typename ResampleFilterType::Pointer resample = ResampleFilterType::New();

    resample->SetTransform(transform);
    resample->SetInput(floImage);

    resample->SetSize(refImage->GetLargestPossibleRegion().GetSize());
    resample->SetOutputOrigin(refImage->GetOrigin());
    resample->SetOutputSpacing(refImage->GetSpacing());
    resample->SetOutputDirection(refImage->GetDirection());
    resample->SetDefaultPixelValue(0.5);

    resample->UpdateLargestPossibleRegion();

    return resample->GetOutput();
}

ImagePointer MakeTestImage(unsigned int xpos, unsigned int ypos, unsigned int xsz, unsigned int ysz) {
    ImageType::RegionType region;
    ImageType::IndexType index;
    ImageType::SizeType size;

    index[0] = 0;
    index[1] = 0;
    size[0] = 64;
    size[1] = 64;

    region.SetIndex(index);
    region.SetSize(size);

    ImagePointer image = ImageType::New();

    image->SetRegions(region);
    image->Allocate();
    image->FillBuffer(0.0f);

    double valacc = 0.0;
    for(unsigned int i = ypos; i < ypos + ysz; ++i) {
        for(unsigned int j = xpos; j < xpos + xsz; ++j) {
            ImageType::IndexType ind;
            ind[0] = j;
            ind[1] = i;
            image->SetPixel(ind, 1.0f);
            valacc += 1.0;
        }
    }

    std::cout << "Mean value [GT]: " << (valacc / (64*64)) << std::endl;

    return image;
}

double MeanAbsDiff(ImagePointer image1, ImagePointer image2)
{
    using IPT = itk::IPT<float, 2U>;

    typename ImageType::Pointer diff = IPT::DifferenceImage(image1, image2);
    typename IPT::ImageStatisticsData stats = IPT::ImageStatistics(diff);

    return stats.mean;
}


    void RunTest()
    {
        using IPT = itk::IPT<float, 2U>;

        itk::MultiThreader::SetGlobalDefaultNumberOfThreads(6U);

        ImagePointer refImage = MakeTestImage(5, 7, 10, 8);
        ImagePointer floImage = MakeTestImage(9, 13, 8, 10);

        //IPT::SaveImageU8("./reftest.png", refImage);
        //IPT::SaveImageU8("./flotest.png", floImage);

        using DistType = MCAlphaCutPointToSetDistance<ImageType, unsigned short>;
        using DistPointer = typename DistType::Pointer;
        
        DistPointer distStructRefImage = DistType::New();
        DistPointer distStructFloImage = DistType::New();

        distStructRefImage->SetSampleCount(20U);
        distStructRefImage->SetImage(refImage);
        distStructRefImage->SetMaxDistance(0);

        distStructFloImage->SetSampleCount(20U);
        distStructFloImage->SetImage(floImage);
        distStructFloImage->SetMaxDistance(0);
        
        distStructRefImage->Initialize();
        distStructFloImage->Initialize();

        using RegistrationType = AlphaBSplineRegistration<ImageType, DistType, 3U>;
        using RegistrationPointer = typename RegistrationType::Pointer;

        RegistrationPointer reg = RegistrationType::New();

        reg->SetDistDataStructRefImage(distStructRefImage);
        reg->SetDistDataStructFloImage(distStructFloImage);

        using PointSamplerType = PointSamplerBase<ImageType, itk::Image<bool, 2U>, ImageType>;
        using PointSamplerPointer = typename PointSamplerType::Pointer;
        PointSamplerPointer sampler1 = QuasiRandomPointSampler<ImageType, itk::Image<bool, 2U>, ImageType>::New().GetPointer();
        sampler1->SetImage(refImage);
        sampler1->SetThreads(32U);
        sampler1->Initialize();
        PointSamplerPointer sampler2 = QuasiRandomPointSampler<ImageType, itk::Image<bool, 2U>, ImageType>::New().GetPointer();
        sampler2->SetImage(floImage);
        sampler2->SetThreads(32U);
        sampler2->Initialize();
        
        reg->SetPointSamplerRefImage(sampler1);
        reg->SetPointSamplerFloImage(sampler2);
        unsigned count = 24;
        reg->SetTransformRefToFlo(CreateBSplineTransform(refImage, count));
        reg->SetTransformFloToRef(CreateBSplineTransform(floImage, count));

        reg->SetSampleCountRefToFlo(8000);
        reg->SetSampleCountFloToRef(8000);
        reg->SetLearningRate(3.0);
        reg->SetIterations(500);
        reg->SetSymmetryLambda(0.02);

        std::cout << "Initializing" << std::endl;
        reg->Initialize();

        std::cout << "Running" << std::endl;

        itk::TimeProbesCollectorBase chronometer;

        chronometer.Start("Registration");

        reg->Run();

        chronometer.Stop("Registration");
        chronometer.Report(std::cout);

        TransformPointer t1 = reg->GetTransformRefToFlo();
/*
        for(unsigned int i = 0; i < t1->GetNumberOfParameters(); ++i)
        {
            if(i > 0 && i%count!=0)
                std::cout << ", ";
            std::cout << t1->GetParameters()[i];
            if(i%count==count-1)
                std::cout << std::endl;
        }
        std::cout << std::endl;
*/
        ImagePointer transformedImage = ApplyTransform(refImage, floImage, t1);

        std::cout << "Before diff: " << MeanAbsDiff(refImage, floImage) << std::endl;
        std::cout << "After diff:  " << MeanAbsDiff(refImage, transformedImage) << std::endl;
/*
        for(unsigned int i = 0; i < 64; ++i)
        {
            for(unsigned int j = 0; j < 64; ++j)
            {
                typename ImageType::IndexType index;
                index[0] = j;
                index[1] = i;
                float value = transformedImage->GetPixel(index);
                std::cout << value << ", ";
            }
        }*/

        IPT::SaveImageU8("./transformedtest.png", transformedImage);
    }
}