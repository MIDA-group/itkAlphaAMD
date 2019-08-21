
#include <stdio.h>
#include <string.h>
#include <string>

//#include "itkLBFGSOptimizer.h"
#include "itkImageFileWriter.h"
#include "itkResampleImageFilter.h"
#include "itkCastImageFilter.h"
#include "itkSquaredDifferenceImageFilter.h"

#include "itkImageRegistrationMethod.h"
#include "itkMeanSquaresImageToImageMetric.h"
#include "itkTimeProbesCollectorBase.h"
#include "itkSpatialObjectToImageFilter.h"
#include "itkEllipseSpatialObject.h"

#include "common/itkImageProcessingTools.h"
#include "itkTextOutput.h"

#include "metric/itkAlphaSMDMetricDeform.h"

const unsigned int ImageDimension = 3;
typedef double PixelType;

typedef itk::Image<PixelType, ImageDimension> ImageType;
typedef typename ImageType::Pointer ImagePointer;

static void CreateEllipseImage(ImageType::Pointer image);
static void CreateCircleImage(ImageType::Pointer image);

constexpr unsigned int splineOrder = 3;

typedef itk::BSplineTransform<double, ImageDimension, splineOrder> TransformType;
typedef typename TransformType::Pointer TransformPointer;

TransformPointer CreateBSplineTransform(ImagePointer image, unsigned int numberOfGridNodes)
{
    TransformType::PhysicalDimensionsType fixedPhysicalDimensions;
    TransformType::MeshSizeType meshSize;

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

ImageType::Pointer ApplyTransform(ImagePointer refImage, ImagePointer floImage, TransformPointer transform)
{
    typedef itk::ResampleImageFilter<
        ImageType,
        ImageType>
        ResampleFilterType;

    typedef itk::IPT<double, 2U> IPT;

    ResampleFilterType::Pointer resample = ResampleFilterType::New();

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

double normalizeDerivative(unsigned int gridPoints, unsigned int dim)
{
    double frac = pow(gridPoints / 5.0, (double)dim);
    return frac;
}

double maxDerivative(itk::Array<double> &array)
{
    double sum = 0.0;
    unsigned int count = array.GetSize();
    for (unsigned int i = 0; i < count; ++i)
    {
        double value = fabs(array[i]);
        if (value > sum)
            sum = value;
    }
    return sum;
}

double norm(itk::Array<double> &array, double p, double eps = 1e-7)
{
    double sum = 0.0;
    unsigned int count = array.GetSize();
    for (unsigned int i = 0; i < count; ++i)
    {
        double value = pow(fabs(array[i]), p);
        sum += value;
    }
    sum = pow(sum, 1.0 / p);
    if (sum < eps)
        sum = eps;
    return sum;
}

#include <fstream>
#include <sstream>
#include <string>
#include "itkRawImageIO.h"
#include "itkImageFileReader.h"

std::vector<std::string> read_strings(std::string path) {
    std::vector<std::string> result;
    std::ifstream infile(path.c_str());

    std::string line;
    while (std::getline(infile, line))
    {
        result.push_back(line);
    }

    return result;
}

template <typename TPixelType, unsigned int TImageDimension>
ImagePointer readRawIntegerFile(std::string path) {
    std::vector<std::string> hdr = read_strings(path);
    
    assert(hdr.size() >= 1 + TImageDimension*2);

    std::string data_path = hdr[0];
    int nPixels[TImageDimension];
    double szVoxels[TImageDimension];

    for(int i = 0; i < TImageDimension; ++i) {
        nPixels[i] = atoi(hdr[1+i].c_str());
        szVoxels[i] = atof(hdr[1+TImageDimension+i].c_str());
    }

    typedef itk::RawImageIO<TPixelType, TImageDimension> IOType;
    typedef typename IOType::Pointer IOPointer;

    IOPointer io = IOType::New();

    for(int i = 0; i < TImageDimension; ++i) {
        io->SetDimensions(i, nPixels[i]);
        io->SetSpacing(i, szVoxels[i]);
    }

    io->SetHeaderSize(io->GetImageSizeInPixels()*0);

    typedef itk::Image<TPixelType, TImageDimension> IntermediateImageType;

    typedef itk::ImageFileReader<IntermediateImageType> ReaderType;

    typedef itk::IPT<double, ImageDimension> IPT;

    typename ReaderType::Pointer reader = ReaderType::New();

    reader->SetFileName(data_path.c_str());
    reader->SetImageIO(io);

    return IPT::ConvertImageFromIntegerFormat<TPixelType>(reader->GetOutput());
}

#include "itkExtractImageFilter.h"

void saveSlice(ImagePointer image, int d, int ind, std::string path) {
    typedef itk::Image<double, ImageDimension-1> SliceImageType;

    ImageType::RegionType slice = image->GetLargestPossibleRegion();
    slice.SetIndex( d, ind );
    slice.SetSize( d, 0 );
    typedef itk::IPT<double, ImageDimension-1> IPT;

    using ExtractFilterType = itk::ExtractImageFilter< ImageType, SliceImageType>;
    typename ExtractFilterType::Pointer extract = ExtractFilterType::New();
    extract->SetDirectionCollapseToIdentity();
    extract->InPlaceOn();
    extract->SetInput( image );
    extract->SetExtractionRegion(slice);
    
    typename SliceImageType::Pointer sliceImage = extract->GetOutput();
    IPT::SaveImage(path.c_str(), sliceImage, false);
}

// Parameters ref_image.png flo_image.png learning_rate1 learning_rate2 symmetry_factor iterations
int main(int argc, char **argv)
{
    if (argc < 9)
    {
        std::cout << "registerBSpline3D ref_image.png flo_image.png learning_rate symmetry_factor iterations sampling_fraction control_point_count" << std::endl;
        return -1;
    }

    std::string file1 = argv[1];
    std::string file2 = argv[2];

    double learningRate1 = atof(argv[3]);
    double learningRate2 = atof(argv[4]);
    double lambdaFactor = atof(argv[5]);
    int iterations = atoi(argv[6]);
    double samplingFraction = atof(argv[7]);
    int controlPoints = atoi(argv[8]);

    typedef itk::AlphaSMDObjectToObjectMetricDeformv4<ImageType, ImageDimension, double, splineOrder> MetricType;
    typedef typename MetricType::Pointer MetricPointer;
    typedef itk::IPT<double, ImageDimension> IPT;

    typedef double CoordinateRepType;

    ImageType::Pointer fixedImage = readRawIntegerFile<unsigned short, ImageDimension>(file1);
    ImageType::Pointer movingImage = readRawIntegerFile<unsigned short, ImageDimension>(file2);

    for(int i = 0; i < ImageDimension; ++i) {
        ImageType::RegionType reg = fixedImage->GetLargestPossibleRegion();
        ImageType::SizeType sz = reg.GetSize();
        char pth[256];
        sprintf(pth, "./fixed_slice_%d.png", i+1);
        saveSlice(fixedImage, i, (int)(sz[i]/2), pth);
    }
    for(int i = 0; i < ImageDimension; ++i) {
        ImageType::RegionType reg = movingImage->GetLargestPossibleRegion();
        ImageType::SizeType sz = reg.GetSize();
        char pth[256];
        sprintf(pth, "./moving_slice_%d.png", i+1);
        saveSlice(movingImage, i, (int)(sz[i]/2), pth);
    }

    std::cout << fixedImage << std::endl;
    std::cout << movingImage << std::endl;

    //return 0;
    //ImageType::Pointer fixedImage = IPT::LoadImage(argv[1]);
    //ImageType::Pointer movingImage = IPT::LoadImage(argv[2]);

    // Create transforms

    unsigned int numberOfGridNodes = controlPoints;
    TransformPointer transformForward = CreateBSplineTransform(fixedImage, numberOfGridNodes);
    TransformPointer transformInverse = CreateBSplineTransform(movingImage, numberOfGridNodes);

    MetricPointer metric = MetricType::New();

    metric->SetRandomSeed(1337);

    metric->SetFixedImage(fixedImage);
    metric->SetMovingImage(movingImage);

    metric->SetForwardTransformPointer(transformForward);
    metric->SetInverseTransformPointer(transformInverse);

    metric->SetFixedSamplingPercentage(samplingFraction);
    metric->SetMovingSamplingPercentage(samplingFraction);

    metric->Update();

    typedef typename MetricType::MeasureType MeasureType;
    typedef typename MetricType::DerivativeType DerivativeType;

    MeasureType value;
    DerivativeType derivative(metric->GetNumberOfParameters());

    typedef itk::IPT<double, ImageDimension> IPT;

    IPT::SaveImageU8("fixed.png", fixedImage);
    IPT::SaveImageU8("moving.png", movingImage);

    // optimize

    metric->SetSymmetryLambda(lambdaFactor);
    double p = 20.0;

    for (int i = 0; i < iterations; ++i)
    {
        double alpha = (double)i / iterations;
        double learningRate = learningRate1 * (1.0 - alpha) + learningRate2 * alpha;
        metric->GetValueAndDerivative(value, derivative);

        double maxGrad = norm(derivative, p);
        double curLR = learningRate / maxGrad;

        metric->UpdateTransformParameters(derivative, curLR); //learningRate);

        learningRate = learningRate * 0.999;
        if (i % 50 == 0 || (i + 1) == iterations)
        {
            std::cout << "Iteration " << (i + 1) << "... Value: " << value << ", Derivative: " << maxGrad << ", lr: " << curLR << std::endl;
        }
    }

    metric->SetSymmetryLambda(1.0);
    metric->SetFixedSamplingPercentage(1.0);
    metric->SetMovingSamplingPercentage(1.0);
    metric->GetValueAndDerivative(value, derivative);

    std::cout << "Final Symmetry Loss: " << value << std::endl;

    std::cout << "Generating transfomed images." << std::endl;

    ImageType::Pointer movingTransformed = ApplyTransform(fixedImage, movingImage, metric->GetTransformForwardPointer());
    ImageType::Pointer fixedTransformed = ApplyTransform(movingImage, fixedImage, metric->GetTransformInversePointer());

    std::cout << "Generating diff images." << std::endl;

    ImageType::Pointer movingDiff = IPT::DifferenceImage(movingTransformed, fixedImage);
    ImageType::Pointer fixedDiff = IPT::DifferenceImage(fixedTransformed, movingImage);

    std::cout << "Saving transformed images." << std::endl;

    IPT::SaveImageU8("moving_transformed_to_fixed.png", movingTransformed);
    IPT::SaveImageU8("fixed_transformed_to_moving.png", fixedTransformed);

    std::cout << "Saving diff images." << std::endl;

    IPT::SaveImageU8("moving_diff.png", movingDiff);
    IPT::SaveImageU8("fixed_diff.png", fixedDiff);

    return 0;
}
