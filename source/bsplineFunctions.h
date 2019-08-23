
#ifndef BSPLINE_FUNCTIONS_H_
#define BSPLINE_FUNCTIONS_H_

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

#include "itkTimeProbesCollectorBase.h"
#include "itkMemoryProbesCollectorBase.h"

#include "metric/itkAlphaSMDMetricDeform.h"

// For BSpline transform resampling
#include "itkBSplineResampleImageFunction.h"
#include "itkIdentityTransform.h"
#include "itkBSplineDecompositionImageFilter.h"

#include "itkCheckerBoardImageFilter.h"
#include "itkTransformToDisplacementFieldFilter.h"
#include "itkFlatStructuringElement.h"
#include "itkBinaryDilateImageFilter.h"

#include "itkGradientMagnitudeImageFilter.h"
#include "itkDisplacementFieldJacobianDeterminantFilter.h"

#include "itkRegularStepGradientDescentOptimizerv4.h"
#include "itkImageRegistrationMethodv4.h"
#include "itkMeanSquaresImageToImageMetricv4.h"
#include "itkMattesMutualInformationImageToImageMetricv4.h"
#include "itkRegistrationParameterScalesFromPhysicalShift.h"

#include <fstream>
#include <sstream>
#include <string>
#include "itkRawImageIO.h"
#include "itkImageFileReader.h"

#include "itkExtractImageFilter.h"

#include "nlohmann/json.hpp"

struct BSplineRegParamInner {
    double learningRate;
    double lambdaFactor;
    long long iterations;
    long long controlPoints;
};

// Parameters for 
struct BSplineRegParam
{
    std::string optimizer;
    double samplingFraction;
    unsigned long long downsamplingFactor;
    double smoothingSigma;
    unsigned long long alphaLevels;
    bool gradientMagnitude;
    double normalization;
    double learningRate;
    double lambdaFactor;
    std::vector<BSplineRegParamInner> innerParams;
};

struct BSplineRegParamOuter
{
    std::vector<BSplineRegParam> paramSets;
};

using json = nlohmann::json;

json readJSON(std::string path) {
    std::ifstream i(path);
    json j;
    i >> j;
    return j;
}

template <typename C, typename T>
void readJSONKey(C& c, std::string key, T *out) {
    if(c.find(key) != c.end()) {
        *out = c[key];
    }
}

BSplineRegParamOuter readConfig(std::string path) {
    BSplineRegParamOuter param;
    json jc = readJSON(path);
    std::cout << jc << std::endl;

    //std::cout << jc["paramSets"].size() << std::endl;
    for(size_t i = 0; i < jc["paramSets"].size(); ++i) {
        auto m_i = jc["paramSets"][i];

        BSplineRegParam paramSet;
        
        paramSet.samplingFraction = 0.05;
        readJSONKey(m_i, "samplingFraction", &paramSet.samplingFraction);
        //if(m_i.find("samplingFraction") != m_i.end())
        //    paramSet.samplingFraction = m_i["samplingFraction"];
        //else
        //    paramSet.samplingFraction = 0.05;

        //std::cout << "Access samplingFraction" << std::endl;
        paramSet.optimizer = "adam";
        readJSONKey(m_i, "optimizer", &paramSet.optimizer);
        paramSet.downsamplingFactor = 1;
        readJSONKey(m_i, "downsamplingFactor", &paramSet.downsamplingFactor);
        paramSet.smoothingSigma = 0.0;
        readJSONKey(m_i, "smoothingSigma", &paramSet.smoothingSigma);
        paramSet.alphaLevels = 7;
        readJSONKey(m_i, "alphaLevels", &paramSet.alphaLevels);
        paramSet.gradientMagnitude = false;
        readJSONKey(m_i, "gradientMagnitude", &paramSet.gradientMagnitude);
        paramSet.normalization = 0.0;
        readJSONKey(m_i, "normalization", &paramSet.normalization);
        paramSet.learningRate = 0.1;
        readJSONKey(m_i, "learningRate", &paramSet.learningRate);
        paramSet.lambdaFactor = 0.01;
        readJSONKey(m_i, "lambdaFactor", &paramSet.lambdaFactor);

        //auto innerConfig = config[i]["inner"];
        //std::cout << "Access innerConfig" << i << " of size " << jc["paramSets"][i]["innerParams"].size() << std::endl;
        for(size_t j = 0; j < m_i["innerParams"].size(); ++j) {
            //std::cout << "Access innerParam" << j << std::endl;
            auto m_i_j = m_i["innerParams"][j];

            BSplineRegParamInner innerParam;// = jc["paramSets"][i]["inner"][j].get<BSplineRegParamInner>();
           
            innerParam.learningRate = paramSet.learningRate;
            readJSONKey(m_i_j, "learningRate", &innerParam.learningRate);
            innerParam.lambdaFactor = paramSet.lambdaFactor;
            readJSONKey(m_i_j, "lambdaFactor", &innerParam.lambdaFactor);
            innerParam.iterations = 500;
            readJSONKey(m_i_j, "iterations", &innerParam.iterations);
            innerParam.controlPoints = 15;
            readJSONKey(m_i_j, "controlPoints", &innerParam.controlPoints);

            paramSet.innerParams.push_back(innerParam);
        }

        param.paramSets.push_back(paramSet);
        //std::cout << "Destroying innerConfig" << std::endl;
    }
    //std::cout << "Done reading config, before returning" << std::endl;

    //param = jc.get<BSplineRegParamOuter>();

    return param;
}


template <unsigned int ImageDimension = 3U>
class BSplines {
public:

typedef double PixelType;
typedef double CoordinateRepType;

typedef typename itk::Image<PixelType, ImageDimension> ImageType;
typedef typename ImageType::Pointer ImagePointer;

typedef typename itk::Vector<PixelType, ImageDimension> VectorPixelType;
typedef typename itk::Image<VectorPixelType, ImageDimension> DisplacementFieldImageType;
typedef typename DisplacementFieldImageType::Pointer DisplacementFieldImagePointer;

typedef typename itk::IPT<double, ImageDimension> IPT;


template <typename TransformType>
DisplacementFieldImagePointer TransformToDisplacementField(typename TransformType::Pointer transform, ImagePointer reference_image) {
  typedef typename itk::TransformToDisplacementFieldFilter<DisplacementFieldImageType, CoordinateRepType> DisplacementFieldGeneratorType;

  typename DisplacementFieldGeneratorType::Pointer dfield_gen = DisplacementFieldGeneratorType::New();

  dfield_gen->UseReferenceImageOn();
  dfield_gen->SetReferenceImage(reference_image);
  dfield_gen->SetTransform(transform);
  try {
    dfield_gen->Update();
  } catch (itk::ExceptionObject & err) {
    std::cerr << "Error while generating deformation field: " << err << std::endl;
  }

  return dfield_gen->GetOutput();
}

ImagePointer JacobianDeterminantFilter(DisplacementFieldImagePointer dfield) {
  typedef typename itk::DisplacementFieldJacobianDeterminantFilter<DisplacementFieldImageType, PixelType, ImageType> FilterType;
  typename FilterType::Pointer filter = FilterType::New();

  filter->SetInput(dfield);
  filter->SetUseImageSpacingOn();

  filter->Update();

  return filter->GetOutput();
}

DisplacementFieldImagePointer LoadDisplacementField(std::string path) {
  typedef typename itk::ImageFileReader<DisplacementFieldImageType> FieldReaderType;
  typename FieldReaderType::Pointer reader = FieldReaderType::New();

  reader->SetFileName(path.c_str());

  reader->Update();

  return reader->GetOutput();
}

void SaveDisplacementField(DisplacementFieldImagePointer image, std::string path) {
  typedef typename itk::ImageFileWriter<DisplacementFieldImageType> FieldWriterType;
  typename FieldWriterType::Pointer writer = FieldWriterType::New();

  writer->SetInput(image);

  writer->SetFileName(path.c_str());

  try {
    writer->Update();
  } catch (itk::ExceptionObject & err) {
    std::cerr << "Error while writing displacement field: " << err << std::endl;
  }
}

void SaveJacobianDeterminantImage(ImagePointer image, std::string path) {
  typedef typename itk::ImageFileWriter<ImageType> FieldWriterType;
  typename FieldWriterType::Pointer writer = FieldWriterType::New();

  writer->SetInput(image);

  writer->SetFileName(path.c_str());

  try {
    writer->Update();
  } catch (itk::ExceptionObject & err) {
    std::cerr << "Error while writing jacobian determinant: " << err << std::endl;
  }
}

static void CreateEllipseImage(typename ImageType::Pointer image);
static void CreateCircleImage(typename ImageType::Pointer image);

constexpr static unsigned int splineOrder = 3;

typedef typename itk::BSplineTransform<double, ImageDimension, splineOrder> TransformType;
typedef typename TransformType::Pointer TransformPointer;

typedef itk::AlphaSMDObjectToObjectMetricDeformv4<ImageType, ImageDimension, double, splineOrder> MetricType;
typedef typename MetricType::Pointer MetricPointer;


ImagePointer GradientMagnitudeImage(ImagePointer image, double sigma) {
    typedef typename itk::IPT<double, ImageDimension> IPT;
    if(sigma > 0.0) {
        image = IPT::SmoothImage(image, sigma);
    }
    typedef typename itk::GradientMagnitudeImageFilter<ImageType, ImageType> GradientMagnitudeFilterType;
    typename GradientMagnitudeFilterType::Pointer gmFilter = GradientMagnitudeFilterType::New();

    gmFilter->SetInput(image);
    gmFilter->SetUseImageSpacingOn();
    gmFilter->Update();

    return gmFilter->GetOutput();
}

typename IPT::BinaryImagePointer DilateMask(typename IPT::BinaryImagePointer mask, int radiusValue) {
    using StructuringElementType = itk::FlatStructuringElement< ImageDimension >;
    typename StructuringElementType::RadiusType radius;
    radius.Fill( radiusValue );
    StructuringElementType structuringElement = typename StructuringElementType::Ball( radius );

    using BinaryDilateImageFilterType = itk::BinaryDilateImageFilter<typename IPT::BinaryImageType, typename IPT::BinaryImageType, StructuringElementType>;
    typename BinaryDilateImageFilterType::Pointer dilateFilter = BinaryDilateImageFilterType::New();
    dilateFilter->SetInput(mask);
    dilateFilter->SetKernel(structuringElement);

    dilateFilter->Update();
    return dilateFilter->GetOutput();
}

ImagePointer Chessboard(ImagePointer image1, ImagePointer image2, int cells)
{
    itk::FixedArray<unsigned int, ImageDimension> pattern;
    pattern.Fill(cells);

    typedef typename itk::IPT<double, ImageDimension> IPT;
    typedef typename itk::CheckerBoardImageFilter<ImageType> CheckerBoardFilterType;
    typename CheckerBoardFilterType::Pointer checkerBoardFilter = CheckerBoardFilterType::New();
    checkerBoardFilter->SetInput1(image1);
    checkerBoardFilter->SetInput2(image2);
    checkerBoardFilter->SetCheckerPattern(pattern);
    checkerBoardFilter->Update();
    return checkerBoardFilter->GetOutput();
}

ImagePointer BlackAndWhiteChessboard(ImagePointer refImage, int cells)
{
    typedef itk::IPT<double, ImageDimension> IPT;
    return Chessboard(IPT::ZeroImage(refImage->GetLargestPossibleRegion().GetSize()), IPT::ConstantImage(1.0, refImage->GetLargestPossibleRegion().GetSize()), cells);
}


std::vector<std::string> read_strings(std::string path)
{
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
ImagePointer readRawIntegerFile(std::string path)
{
    std::vector<std::string> hdr = read_strings(path);

    assert(hdr.size() >= 1 + TImageDimension * 2);

    std::string data_path = hdr[0];
    int nPixels[TImageDimension];
    double szVoxels[TImageDimension];

    for (int i = 0; i < TImageDimension; ++i)
    {
        nPixels[i] = atoi(hdr[1 + i].c_str());
        szVoxels[i] = atof(hdr[1 + TImageDimension + i].c_str());
    }

    typedef itk::RawImageIO<TPixelType, TImageDimension> IOType;
    typedef typename IOType::Pointer IOPointer;

    IOPointer io = IOType::New();

    for (int i = 0; i < TImageDimension; ++i)
    {
        io->SetDimensions(i, nPixels[i]);
        io->SetSpacing(i, szVoxels[i]);
    }

    io->SetHeaderSize(io->GetImageSizeInPixels() * 0);
    io->SetByteOrderToLittleEndian();

    typedef itk::Image<TPixelType, TImageDimension> IntermediateImageType;

    typedef itk::ImageFileReader<IntermediateImageType> ReaderType;

    typedef itk::IPT<double, ImageDimension> IPT;

    typename ReaderType::Pointer reader = ReaderType::New();

    reader->SetFileName(data_path.c_str());
    reader->SetImageIO(io);

    return itk::ConvertImageFromIntegerFormat<TPixelType>(reader->GetOutput());
}


void saveSlice(ImagePointer image, int d, int ind, std::string path)
{
    typedef itk::Image<double, ImageDimension - 1> SliceImageType;

    typename ImageType::RegionType slice = image->GetLargestPossibleRegion();
    slice.SetIndex(d, ind);
    slice.SetSize(d, 0);
    typedef itk::IPT<double, ImageDimension - 1> IPT;

    using ExtractFilterType = itk::ExtractImageFilter<ImageType, SliceImageType>;
    typename ExtractFilterType::Pointer extract = ExtractFilterType::New();
    extract->SetDirectionCollapseToIdentity();
    extract->InPlaceOn();
    extract->SetInput(image);
    extract->SetExtractionRegion(slice);

    typename SliceImageType::Pointer sliceImage = extract->GetOutput();
    IPT::SaveImage(path.c_str(), sliceImage, false);
}

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

void UpsampleBSplineTransform(ImagePointer image, TransformPointer newTransform, TransformPointer oldTransform, unsigned int numberOfGridNodes)
{
    typedef typename TransformType::ParametersType ParametersType;
    ParametersType parameters(newTransform->GetNumberOfParameters());
    parameters.Fill(0.0);

    unsigned int counter = 0;
    for (unsigned int k = 0; k < ImageDimension; k++)
    {
        using ParametersImageType = typename TransformType::ImageType;
        using ResamplerType = itk::ResampleImageFilter<ParametersImageType, ParametersImageType>;
        typename ResamplerType::Pointer upsampler = ResamplerType::New();
        using FunctionType = itk::BSplineResampleImageFunction<ParametersImageType, double>;
        typename FunctionType::Pointer function = FunctionType::New();
        using IdentityTransformType = itk::IdentityTransform<double, ImageDimension>;
        typename IdentityTransformType::Pointer identity = IdentityTransformType::New();
        upsampler->SetInput(oldTransform->GetCoefficientImages()[k]);
        upsampler->SetInterpolator(function);
        upsampler->SetTransform(identity);
        upsampler->SetSize(newTransform->GetCoefficientImages()[k]->GetLargestPossibleRegion().GetSize());
        upsampler->SetOutputSpacing(
            newTransform->GetCoefficientImages()[k]->GetSpacing());
        upsampler->SetOutputOrigin(
            newTransform->GetCoefficientImages()[k]->GetOrigin());
        upsampler->SetOutputDirection(image->GetDirection());
        using DecompositionType =
            itk::BSplineDecompositionImageFilter<ParametersImageType, ParametersImageType>;
        typename DecompositionType::Pointer decomposition = DecompositionType::New();
        decomposition->SetSplineOrder(splineOrder);
        decomposition->SetInput(upsampler->GetOutput());
        decomposition->Update();
        typename ParametersImageType::Pointer newCoefficients = decomposition->GetOutput();
        // copy the coefficients into the parameter array
        using Iterator = itk::ImageRegionIterator<ParametersImageType>;
        Iterator it(newCoefficients,
                    newTransform->GetCoefficientImages()[k]->GetLargestPossibleRegion());
        while (!it.IsAtEnd())
        {
            parameters[counter++] = it.Get();
            ++it;
        }
    }
    newTransform->SetParameters(parameters);
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

double normalizeDerivative(unsigned int gridPoints, unsigned int dim)
{
    // / 5.0
    double frac = pow(gridPoints, (double)dim);
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

void adam_optimizer(MetricType* metric, double learningRate, unsigned int iterations) {
    
    typedef typename MetricType::DerivativeType DerivativeType;

    unsigned int N = metric->GetNumberOfParameters();

    DerivativeType d(N);

    DerivativeType mt(N);
    DerivativeType vt(N);

    mt.Fill(0.0);
    vt.Fill(0.0);

    d.Fill(0.0);
    double beta1 = 0.9;
    double beta2 = 0.999;
    double eps = 1e-8;

    for(unsigned int i = 0; i < iterations; ++i) {
        double value;
        metric->GetValueAndDerivative(value, d);

        for(unsigned int j = 0; j < N; ++j) {
            double mt_j = beta1 * mt[j] + (1.0 - beta1) * d[j];
            double vt_j = beta2 * vt[j] + (1.0 - beta2) * (d[j]*d[j]);

            mt[j] = mt_j;
            vt[j] = vt_j;

            mt_j = mt_j / (1.0 - pow(beta1, i+1.0));
            vt_j = vt_j / (1.0 - pow(beta2, i+1.0));

            d[j] = mt_j / (sqrt(vt_j) + eps);
        }

        metric->UpdateTransformParameters(d, learningRate);
    }
}

void register_func(typename ImageType::Pointer fixedImage, typename ImageType::Pointer movingImage, TransformPointer& transformForward, TransformPointer& transformInverse, BSplineRegParam param, ImagePointer fixedMask, ImagePointer movingMask, bool verbose=false)
{
    typedef itk::IPT<double, ImageDimension> IPT;

    MetricPointer metric = MetricType::New();

    metric->SetRandomSeed(1337);

    metric->SetFixedImage(fixedImage);
    metric->SetMovingImage(movingImage);

    metric->SetAlphaLevels(param.alphaLevels);

    metric->SetForwardTransformPointer(transformForward);
    metric->SetInverseTransformPointer(transformInverse);

    metric->SetFixedSamplingPercentage(param.samplingFraction);
    metric->SetMovingSamplingPercentage(param.samplingFraction);

    if(fixedMask) {
        typename IPT::BinaryImagePointer maskBin = IPT::ThresholdImage(fixedMask, 0.01);

        metric->SetFixedMask(maskBin);
    }
    if(movingMask) {
        typename IPT::BinaryImagePointer maskBin = IPT::ThresholdImage(movingMask, 0.01);

        metric->SetMovingMask(maskBin);
    }

    itk::TimeProbesCollectorBase chronometer;
    itk::MemoryProbesCollectorBase memorymeter;

    chronometer.Start("Pre-processing");
    memorymeter.Start("Pre-processing");

    metric->Update();

    chronometer.Stop("Pre-processing");
    memorymeter.Stop("Pre-processing");

    typedef typename MetricType::MeasureType MeasureType;
    typedef typename MetricType::DerivativeType DerivativeType;

    MeasureType value = 0.0;
    DerivativeType derivative(metric->GetNumberOfParameters());

    typedef itk::IPT<double, ImageDimension> IPT;
    
    metric->SetFixedSamplingPercentage(param.samplingFraction);
    metric->SetMovingSamplingPercentage(param.samplingFraction);

    for (int q = 0; q < param.innerParams.size(); ++q) {
        double lr1 = param.innerParams[q].learningRate;
        unsigned int iterations = param.innerParams[q].iterations;
        unsigned int controlPoints = param.innerParams[q].controlPoints;

        metric->SetSymmetryLambda(param.innerParams[q].lambdaFactor);

        if(q > 0) {
            TransformPointer tforNew = CreateBSplineTransform(fixedImage, controlPoints);
            TransformPointer tinvNew = CreateBSplineTransform(movingImage, controlPoints);
            int curNumberOfGridNodes = controlPoints;
            UpsampleBSplineTransform(fixedImage, tforNew, transformForward, curNumberOfGridNodes);
            UpsampleBSplineTransform(movingImage, tinvNew, transformInverse, curNumberOfGridNodes);
            transformForward = tforNew;
            transformInverse = tinvNew;

            metric->SetForwardTransformPointer(transformForward);
            metric->SetInverseTransformPointer(transformInverse);
            metric->UpdateAfterTransformChange();
            derivative = DerivativeType(metric->GetNumberOfParameters());
        }

    chronometer.Start("Registration");
    memorymeter.Start("Registration");

    if(param.optimizer == "adam") {
        adam_optimizer(metric.GetPointer(), lr1, iterations);
    } else if(param.optimizer == "sgd") {
        typedef itk::RegularStepGradientDescentOptimizerv4<double> OptimizerType;
        typename OptimizerType::Pointer optimizer = OptimizerType::New();

        optimizer->SetNumberOfIterations(iterations);
        optimizer->SetLearningRate(lr1);
        optimizer->DoEstimateLearningRateOnceOff();
        optimizer->SetMinimumStepLength(0.0);
        optimizer->SetGradientMagnitudeTolerance(1e-8);
        optimizer->SetRelaxationFactor(0.95);
        optimizer->DoEstimateScalesOff();

        optimizer->SetMetric(metric.GetPointer());
        try {
            optimizer->StartOptimization();
        } catch(itk::ExceptionObject& err) {
            std::cout << "Optimization failed: " << err << std::endl;
        }
    }
   
    chronometer.Stop("Registration");
    memorymeter.Stop("Registration");

    typename MetricType::ParametersType parameters(metric->GetParameters());
    double absVal = 0.0;
    double val = 0.0;
    for(unsigned int j = 0; j < metric->GetNumberOfParameters(); ++j) {
        if(fabs(parameters[j]) > absVal) {
            absVal = fabs(parameters[j]);
            val = parameters[j];
        }
    }
    std::cout << "Largest parameters: " << absVal << std::endl;

    }

    //chronometer.Report(std::cout);
    //memorymeter.Report(std::cout);
}

/*

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
    */

template <typename MT>
void register_func_baseline(typename ImageType::Pointer fixedImage, typename ImageType::Pointer movingImage, TransformPointer& transformForward, TransformPointer& transformInverse, BSplineRegParam param, ImagePointer fixedMask, ImagePointer movingMask, typename MT::Pointer metric, bool verbose=false)
{
    typedef itk::IPT<double, ImageDimension> IPT;
/*
    MetricPointer metric = MetricType::New();

    metric->SetRandomSeed(1337);

    metric->SetFixedImage(fixedImage);
    metric->SetMovingImage(movingImage);

    metric->SetAlphaLevels(param.alphaLevels);

    metric->SetForwardTransformPointer(transformForward);
    metric->SetInverseTransformPointer(transformInverse);

    metric->SetFixedSamplingPercentage(param.samplingFraction);
    metric->SetMovingSamplingPercentage(param.samplingFraction);
    */

	typedef itk::RegularStepGradientDescentOptimizerv4<double>              OptimizerType;
	typedef itk::InterpolateImageFunction<ImageType, double> InterpolatorType;

	typename OptimizerType::Pointer      optimizer = OptimizerType::New();

	typename InterpolatorType::Pointer fixedInterpolator = IPT::MakeInterpolator(IPT::kImwarpInterpNearest);
	typename InterpolatorType::Pointer movingInterpolator = IPT::MakeInterpolator(IPT::kImwarpInterpCubic);

	metric->SetFixedInterpolator(fixedInterpolator);
	metric->SetMovingInterpolator(movingInterpolator);
/*
    if(fixedMask) {
        typename IPT::BinaryImagePointer maskBin = IPT::ThresholdImage(fixedMask, 0.01);

        metric->SetFixedMask(maskBin);
    }
    if(movingMask) {
        typename IPT::BinaryImagePointer maskBin = IPT::ThresholdImage(movingMask, 0.01);

        metric->SetMovingMask(maskBin);
    }
*/

    typedef typename MetricType::MeasureType MeasureType;
    typedef typename MetricType::DerivativeType DerivativeType;

    MeasureType value = 0.0;

    typedef itk::IPT<double, ImageDimension> IPT;
    
    //metric->SetFixedSamplingPercentage(param.samplingFraction);
    //metric->SetMovingSamplingPercentage(param.samplingFraction);

    metric->SetFixedImage(fixedImage);
    metric->SetMovingImage(movingImage);

    metric->Initialize();

    for (int q = 0; q < param.innerParams.size(); ++q) {
        double lr1 = param.innerParams[q].learningRate;
        unsigned int iterations = param.innerParams[q].iterations;
        unsigned int controlPoints = param.innerParams[q].controlPoints;

        if(q > 0) {
            TransformPointer tforNew = CreateBSplineTransform(fixedImage, controlPoints);
            TransformPointer tinvNew = CreateBSplineTransform(movingImage, controlPoints);
            int curNumberOfGridNodes = controlPoints;
            UpsampleBSplineTransform(fixedImage, tforNew, transformForward, curNumberOfGridNodes);
            UpsampleBSplineTransform(movingImage, tinvNew, transformInverse, curNumberOfGridNodes);
            transformForward = tforNew;
            transformInverse = tinvNew;
        }

        typedef itk::ImageRegistrationMethodv4<ImageType, ImageType, TransformType> RegistrationType;
        typedef typename RegistrationType::Pointer RegistrationPointer;

        RegistrationPointer registration = RegistrationType::New();

	    registration->SetInitialTransform(transformForward);
        registration->InPlaceOn();

	    registration->SetFixedImage(fixedImage);
	    registration->SetMovingImage(movingImage);

    	typename RegistrationType::ShrinkFactorsArrayType shrinkFactorsPerLevel;
	    shrinkFactorsPerLevel.SetSize(1);

	    typename RegistrationType::SmoothingSigmasArrayType smoothingSigmasPerLevel;
	    smoothingSigmasPerLevel.SetSize(1);

	    shrinkFactorsPerLevel[0] = 1;
	    smoothingSigmasPerLevel[0] = 0.0;

	    registration->SetNumberOfLevels(1);
	    registration->SetSmoothingSigmasPerLevel(smoothingSigmasPerLevel);
	    registration->SetShrinkFactorsPerLevel(shrinkFactorsPerLevel);

	    typename RegistrationType::MetricSamplingStrategyType  samplingStrategy =
		    RegistrationType::RANDOM;

	    registration->SetMetricSamplingStrategy(samplingStrategy);
	    registration->SetMetricSamplingPercentage(param.samplingFraction);
	    registration->MetricSamplingReinitializeSeed(1337+q);

    	registration->SetNumberOfThreads(1);

	    registration->SetMetric(metric);

    //if(param.optimizer == "adam") {
    //    adam_optimizer(metric.GetPointer(), lr1, iterations);
    //} else if(param.optimizer == "sgd") {
        typedef itk::RegularStepGradientDescentOptimizerv4<double> OptimizerType;
        typename OptimizerType::Pointer optimizer = OptimizerType::New();

        using ScalesEstimatorType = itk::RegistrationParameterScalesFromPhysicalShift<MT>;
        typename ScalesEstimatorType::Pointer scalesEstimator = ScalesEstimatorType::New();
        scalesEstimator->SetMetric( metric );
        scalesEstimator->SetTransformForward( true );
        //scalesEstimator->SetSmallParameterVariation( 0.001 );

        optimizer->SetNumberOfIterations(iterations);
        optimizer->SetLearningRate(lr1);
        optimizer->SetScalesEstimator(scalesEstimator);
        //optimizer->DoEstimateLearningRateOnceOn();
        //optimizer->DoEstimateLearningRateAtEachIterationOn();
        optimizer->SetMinimumStepLength(0.0);
        optimizer->SetGradientMagnitudeTolerance(1e-8);
        optimizer->SetRelaxationFactor(0.95);
        optimizer->DoEstimateScalesOff();
    	optimizer->SetNumberOfThreads(1);

        registration->SetOptimizer(optimizer);

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
    //}
   
    transformForward = registration->GetModifiableTransform();
   
    }
}

void bspline_register(
    typename ImageType::Pointer fixedImage,
    typename ImageType::Pointer movingImage,
    BSplineRegParamOuter param,
    ImagePointer fixedMask,
    ImagePointer movingMask,
    bool verbose,
    TransformPointer& transformForwardOut,
    TransformPointer& transformInverseOut) {

    TransformPointer transformForward;
    TransformPointer transformInverse;

    for(size_t i = 0; i < param.paramSets.size(); ++i) {
        auto paramSet = param.paramSets[i];

        ImagePointer fixedImagePrime = IPT::SmoothImage(fixedImage, paramSet.smoothingSigma);
        ImagePointer movingImagePrime = IPT::SmoothImage(movingImage, paramSet.smoothingSigma);
        ImagePointer fixedMaskPrime = fixedMask;
        ImagePointer movingMaskPrime = movingMask;
        if(paramSet.downsamplingFactor != 1) {
            fixedImagePrime = IPT::SubsampleImage(fixedImagePrime, paramSet.downsamplingFactor);
            movingImagePrime = IPT::SubsampleImage(movingImagePrime, paramSet.downsamplingFactor);
            fixedMaskPrime = IPT::SubsampleImage(fixedMaskPrime, paramSet.downsamplingFactor);
            movingMaskPrime = IPT::SubsampleImage(movingMaskPrime, paramSet.downsamplingFactor);
        }
        if(paramSet.gradientMagnitude) {
            fixedImagePrime = GradientMagnitudeImage(fixedImagePrime, 0.0);
            movingImagePrime = GradientMagnitudeImage(movingImagePrime, 0.0);
        }
        fixedImagePrime = IPT::NormalizeImage(fixedImagePrime, IPT::IntensityMinMax(fixedImagePrime, paramSet.normalization));
        movingImagePrime = IPT::NormalizeImage(movingImagePrime, IPT::IntensityMinMax(movingImagePrime, paramSet.normalization));

        if(i == 0) {
            transformForward = CreateBSplineTransform(fixedImagePrime, paramSet.innerParams[0].controlPoints);
            transformInverse = CreateBSplineTransform(movingImagePrime, paramSet.innerParams[0].controlPoints);
        } else {
            TransformPointer tforNew = CreateBSplineTransform(fixedImagePrime, paramSet.innerParams[0].controlPoints);
            TransformPointer tinvNew = CreateBSplineTransform(movingImagePrime, paramSet.innerParams[0].controlPoints);
            UpsampleBSplineTransform(fixedImagePrime, tforNew, transformForward, paramSet.innerParams[0].controlPoints);
            UpsampleBSplineTransform(movingImagePrime, tinvNew, transformInverse, paramSet.innerParams[0].controlPoints);
            transformForward = tforNew;
            transformInverse = tinvNew;            
        }

        //typename ImageType::Pointer fixedImage, typename ImageType::Pointer movingImage, TransformPointer& transformForward, TransformPointer& transformInverse, BSplineRegParam param, ImagePointer fixedMask, ImagePointer movingMask, bool verbose=false
        register_func(fixedImagePrime, movingImagePrime, transformForward, transformInverse, paramSet, fixedMaskPrime, movingMaskPrime, verbose);       
    }

    transformForwardOut = transformForward;
    transformInverseOut = transformInverse;
}


void bspline_register_baseline(
    typename ImageType::Pointer fixedImage,
    typename ImageType::Pointer movingImage,
    BSplineRegParamOuter param,
    ImagePointer fixedMask,
    ImagePointer movingMask,
    bool verbose,
    TransformPointer& transformForwardOut,
    TransformPointer& transformInverseOut,
    unsigned int metricID) {

    TransformPointer transformForward;
    TransformPointer transformInverse;

    for(size_t i = 0; i < param.paramSets.size(); ++i) {
        auto paramSet = param.paramSets[i];

        ImagePointer fixedImagePrime = IPT::SmoothImage(fixedImage, paramSet.smoothingSigma);
        ImagePointer movingImagePrime = IPT::SmoothImage(movingImage, paramSet.smoothingSigma);
        ImagePointer fixedMaskPrime = fixedMask;
        ImagePointer movingMaskPrime = movingMask;
        if(paramSet.downsamplingFactor != 1) {
            fixedImagePrime = IPT::SubsampleImage(fixedImagePrime, paramSet.downsamplingFactor);
            movingImagePrime = IPT::SubsampleImage(movingImagePrime, paramSet.downsamplingFactor);
            fixedMaskPrime = IPT::SubsampleImage(fixedMaskPrime, paramSet.downsamplingFactor);
            movingMaskPrime = IPT::SubsampleImage(movingMaskPrime, paramSet.downsamplingFactor);
        }
        if(paramSet.gradientMagnitude) {
            fixedImagePrime = GradientMagnitudeImage(fixedImagePrime, 0.0);
            movingImagePrime = GradientMagnitudeImage(movingImagePrime, 0.0);
        }
        fixedImagePrime = IPT::NormalizeImage(fixedImagePrime, IPT::IntensityMinMax(fixedImagePrime, paramSet.normalization));
        movingImagePrime = IPT::NormalizeImage(movingImagePrime, IPT::IntensityMinMax(movingImagePrime, paramSet.normalization));

        if(i == 0) {
            transformForward = CreateBSplineTransform(fixedImagePrime, paramSet.innerParams[0].controlPoints);
            transformInverse = CreateBSplineTransform(movingImagePrime, paramSet.innerParams[0].controlPoints);
        } else {
            TransformPointer tforNew = CreateBSplineTransform(fixedImagePrime, paramSet.innerParams[0].controlPoints);
            TransformPointer tinvNew = CreateBSplineTransform(movingImagePrime, paramSet.innerParams[0].controlPoints);
            UpsampleBSplineTransform(fixedImagePrime, tforNew, transformForward, paramSet.innerParams[0].controlPoints);
            UpsampleBSplineTransform(movingImagePrime, tinvNew, transformInverse, paramSet.innerParams[0].controlPoints);
            transformForward = tforNew;
            transformInverse = tinvNew;            
        }

        //typename ImageType::Pointer fixedImage, typename ImageType::Pointer movingImage, TransformPointer& transformForward, TransformPointer& transformInverse, BSplineRegParam param, ImagePointer fixedMask, ImagePointer movingMask, bool verbose=false
        if(metricID == 0) {
            typedef itk::MeanSquaresImageToImageMetricv4<ImageType, ImageType> BLMetricType;
            typedef typename BLMetricType ::Pointer BLMetricPointer;
            BLMetricPointer metric = BLMetricType::New();

            register_func_baseline<itk::MeanSquaresImageToImageMetricv4<ImageType, ImageType> >(fixedImagePrime, movingImagePrime, transformForward, transformInverse, paramSet, fixedMaskPrime, movingMaskPrime, metric, verbose);
        } else if(metricID == 1) {
            typedef itk::MattesMutualInformationImageToImageMetricv4<ImageType, ImageType> BLMetricType;
            typedef typename BLMetricType ::Pointer BLMetricPointer;
            BLMetricPointer metric = BLMetricType::New();

            register_func_baseline<itk::MattesMutualInformationImageToImageMetricv4<ImageType, ImageType> >(fixedImagePrime, movingImagePrime, transformForward, transformInverse, paramSet, fixedMaskPrime, movingMaskPrime, metric, verbose);
        }
        
    }

    transformForwardOut = transformForward;
    transformInverseOut = transformInverse;
}

void print_difference_image_stats(ImagePointer image1, ImagePointer image2, const char* name) {
    typedef itk::IPT<double, ImageDimension> IPT;
    typename ImageType::Pointer diff = IPT::DifferenceImage(image1, image2);

    typename IPT::ImageStatisticsData movingStats = IPT::ImageStatistics(diff);

    std::cout << name << " mean: " << movingStats.mean << ", std: " << movingStats.sigma << std::endl;
}

};

#endif
