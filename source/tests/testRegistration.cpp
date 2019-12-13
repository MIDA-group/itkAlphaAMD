
//#include "../registration/alphaBSplineRegistration.h"
//#include "../registration/alphaBSplineRegistration3.h"
#include "../registration/alphaBSplineRegistration4.h"
//#include "../registration/alphaBSplineRegistration5.h"
#include "../samplers/pointSampler.h"
#include "../metric/mcAlphaCutPointToSetDistance.h"
#include "../samplers/quasiRandomGenerator.h"

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

typename ImageType::Pointer ApplyTransform(ImagePointer refImage, ImagePointer floImage, TransformPointer transform, double bgValue = 0.5)
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
    resample->SetDefaultPixelValue(bgValue);

    resample->UpdateLargestPossibleRegion();

    return resample->GetOutput();
}

typename PointSamplerBase<ImageType, itk::Image<bool, 2U>, ImageType>::Pointer CreateHybridPointSampler(ImagePointer im, double w1 = 0.5)
{
    using PointSamplerType = PointSamplerBase<ImageType, itk::Image<bool, 2U>, ImageType>;
    using PointSamplerPointer = typename PointSamplerType::Pointer;
    PointSamplerPointer sampler1 = QuasiRandomPointSampler<ImageType, itk::Image<bool, 2U>, ImageType>::New().GetPointer();
    GradientWeightedPointSampler<ImageType, itk::Image<bool, 2U>, ImageType>::Pointer sampler2 = GradientWeightedPointSampler<ImageType, itk::Image<bool, 2U>, ImageType>::New().GetPointer();
    typename HybridPointSampler<ImageType, itk::Image<bool, 2U>, ImageType>::Pointer sampler3 = HybridPointSampler<ImageType, itk::Image<bool, 2U>, ImageType>::New();
    sampler2->SetSigma(1.0);
/*    sampler1->SetImage(im);
    sampler1->SetThreads(32U);
    sampler1->Initialize();
    sampler2->SetImage(im);
    sampler2->SetThreads(32U);
    sampler2->Initialize();*/
    sampler3->AddSampler(sampler1, w1);
    sampler3->AddSampler(sampler2.GetPointer(), 1.0-w1);
    sampler3->SetImage(im);
    sampler3->SetSeed(1000U);
    //sampler3->SetDitheringOn();
    sampler3->Initialize();

    return sampler3.GetPointer();
}

typename PointSamplerBase<ImageType, itk::Image<bool, 2U>, ImageType>::Pointer CreateQuasiRandomPointSampler(ImagePointer im)
{
    using PointSamplerType = PointSamplerBase<ImageType, itk::Image<bool, 2U>, ImageType>;
    using PointSamplerPointer = typename PointSamplerType::Pointer;

    PointSamplerPointer sampler1 = QuasiRandomPointSampler<ImageType, itk::Image<bool, 2U>, ImageType>::New().GetPointer();
    sampler1->SetImage(im);
    sampler1->Initialize();

    return sampler1.GetPointer();
}

ImagePointer MakeTestImage(unsigned int xpos, unsigned int ypos, unsigned int xsz, unsigned int ysz) {
    ImageType::RegionType region;
    ImageType::IndexType index;
    ImageType::SizeType size;

    index[0] = 0;
    index[1] = 0;
    size[0] = 256;//64;
    size[1] = 256;//64;

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


    void RunTest(int argc, char** argv)
    {
        using IPT = itk::IPT<float, 2U>;

        unsigned int threads = 6U;
        if(argc >= 2)
        {
            threads = atoi(argv[1]);
        }
        itk::MultiThreader::SetGlobalDefaultNumberOfThreads(threads);

        ImagePointer refImage = MakeTestImage(5, 7, 10, 8);
        ImagePointer floImage = MakeTestImage(9, 13, 8, 10);

        //IPT::SaveImageU8("./reftest.png", refImage);
        //IPT::SaveImageU8("./flotest.png", floImage);

        using DistType = MCAlphaCutPointToSetDistance<ImageType, unsigned short>;
        using DistPointer = typename DistType::Pointer;
        
        DistPointer distStructRefImage = DistType::New();
        DistPointer distStructFloImage = DistType::New();

        distStructRefImage->SetSampleCount(5U);
        distStructRefImage->SetImage(refImage);
        distStructRefImage->SetMaxDistance(0);
        distStructRefImage->SetApproximationThreshold(20.0);
        distStructRefImage->SetApproximationFraction(0.1);

        distStructFloImage->SetSampleCount(5U);
        distStructFloImage->SetImage(floImage);
        distStructFloImage->SetMaxDistance(0);
        distStructFloImage->SetApproximationThreshold(20.0);
        distStructFloImage->SetApproximationFraction(0.1);

        distStructRefImage->Initialize();
        distStructFloImage->Initialize();

        using RegistrationType = AlphaBSplineRegistration<ImageType, DistType, 3U>;
        using RegistrationPointer = typename RegistrationType::Pointer;

        RegistrationPointer reg = RegistrationType::New();

        reg->SetDistDataStructRefImage(distStructRefImage);
        reg->SetDistDataStructFloImage(distStructFloImage);

        using PointSamplerType = PointSamplerBase<ImageType, itk::Image<bool, 2U>, ImageType>;
        using PointSamplerPointer = typename PointSamplerType::Pointer;
        constexpr double w = 0.5;
        PointSamplerPointer sampler1 = CreateHybridPointSampler(refImage, w); //QuasiRandomPointSampler<ImageType, itk::Image<bool, 2U>, ImageType>::New().GetPointer();
        //PointSamplerPointer sampler1 = CreateQuasiRandomPointSampler(refImage);
        PointSamplerPointer sampler2 = CreateHybridPointSampler(floImage, w);
        //PointSamplerPointer sampler2 = CreateQuasiRandomPointSampler(floImage);

        /*sampler1->SetImage(refImage);
        sampler1->Initialize();*/
        //PointSamplerPointer sampler2 = CreateHybridPointSampler(floImage);//QuasiRandomPointSampler<ImageType, itk::Image<bool, 2U>, ImageType>::New().GetPointer();
        /*sampler2->SetImage(floImage);
        sampler2->Initialize();*/
        
        reg->SetPointSamplerRefImage(sampler1);
        reg->SetPointSamplerFloImage(sampler2);
        constexpr unsigned int gridPointCount = 24;
        reg->SetTransformRefToFlo(CreateBSplineTransform(refImage, gridPointCount));
        reg->SetTransformFloToRef(CreateBSplineTransform(floImage, gridPointCount));

        reg->SetSampleCountRefToFlo(4096);
        reg->SetSampleCountFloToRef(4096);
        //reg->SetSampleCountRefToFlo(512);
        //reg->SetSampleCountFloToRef(512);
        reg->SetLearningRate(0.5);
        reg->SetIterations(1000U);
        reg->SetSymmetryLambda(0.025);

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
        ImagePointer transformedImage = ApplyTransform(refImage, floImage, t1, 0.0);

        char buf[64];
        sprintf(buf, "%.15f", MeanAbsDiff(refImage, transformedImage));
        std::cout << "Before diff: " << MeanAbsDiff(refImage, floImage) << std::endl;
        std::cout << "After diff:  " << buf << std::endl;

        std::cout << "Final Distance: " << reg->GetValue() << std::endl;
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
/*
        auto qrand = QuasiRandomGenerator<1U>::New();
        for(unsigned int i = 0; i < 1000; ++i)
        {
            itk::FixedArray<double, 1U> val = qrand->GetConstVariate(i+1);
            std::cout << val[0] << ", ";
        }*/
    }
}