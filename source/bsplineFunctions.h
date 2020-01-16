
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

#include "registration/alphaBSplineRegistration.h"
#include "samplers/pointSampler.h"
#include "metric/mcAlphaCutPointToSetDistance.h"

#include <fstream>
#include <sstream>
#include <string>
#include "itkRawImageIO.h"
#include "itkImageFileReader.h"

#include "itkExtractImageFilter.h"

#include "nlohmann/json.hpp"

template <typename TTransformType, unsigned int Dim>
struct BSplineRegistrationCallback : public Command
{
    public:

    using TransformType = TTransformType;
    using TransformPointer = typename TransformType::Pointer;

    TransformPointer m_TransformForward;
    TransformPointer m_TransformReverse;

    virtual void SetTransforms(TransformPointer transformForward, TransformPointer transformReverse)
    {
        m_TransformForward = transformForward;
        m_TransformReverse = transformReverse;
    }

    virtual void Invoke()
    {
        ;
    }
};

struct BSplineRegParamInner {
    double learningRate;
    double samplingFraction;
    double momentum;
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
    double momentum;
    double lambdaFactor;
    unsigned long long seed;
    bool enableCallbacks;
    bool verbose;
    std::string samplingMode;
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
        paramSet.optimizer = "sgdm";
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
        paramSet.momentum = 0.1;
        readJSONKey(m_i, "momentum", &paramSet.momentum);
        paramSet.lambdaFactor = 0.01;
        readJSONKey(m_i, "lambdaFactor", &paramSet.lambdaFactor);
        paramSet.samplingMode = "quasi";
        readJSONKey(m_i, "samplingMode", &paramSet.samplingMode);
        paramSet.seed = 1337;
        readJSONKey(m_i, "seed", &paramSet.seed);
        paramSet.enableCallbacks = false;
        readJSONKey(m_i, "enableCallbacks", &paramSet.enableCallbacks);
        paramSet.verbose = false;
        readJSONKey(m_i, "verbose", &paramSet.verbose);

        //auto innerConfig = config[i]["inner"];
        //std::cout << "Access innerConfig" << i << " of size " << jc["paramSets"][i]["innerParams"].size() << std::endl;
        for(size_t j = 0; j < m_i["innerParams"].size(); ++j) {
            //std::cout << "Access innerParam" << j << std::endl;
            auto m_i_j = m_i["innerParams"][j];

            BSplineRegParamInner innerParam;// = jc["paramSets"][i]["inner"][j].get<BSplineRegParamInner>();
           
            innerParam.learningRate = paramSet.learningRate;
            readJSONKey(m_i_j, "learningRate", &innerParam.learningRate);
            innerParam.samplingFraction = paramSet.samplingFraction;
            readJSONKey(m_i_j, "samplingFraction", &innerParam.samplingFraction);
            innerParam.momentum = paramSet.momentum;
            readJSONKey(m_i_j, "momentum", &innerParam.momentum);
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
    constexpr static unsigned int Dim = ImageDimension;
typedef double PixelType;
typedef double CoordinateRepType;

typedef typename itk::Image<PixelType, ImageDimension> ImageType;
typedef typename ImageType::Pointer ImagePointer;

typedef typename itk::Vector<PixelType, ImageDimension> VectorPixelType;
typedef typename itk::Image<VectorPixelType, ImageDimension> DisplacementFieldImageType;
typedef typename DisplacementFieldImageType::Pointer DisplacementFieldImagePointer;

typedef typename itk::IPT<double, ImageDimension> IPT;

constexpr static unsigned int splineOrder = 3;

typedef typename itk::BSplineTransform<double, ImageDimension, splineOrder> TransformType;
typedef typename TransformType::Pointer TransformPointer;

typedef itk::AlphaSMDObjectToObjectMetricDeformv4<ImageType, ImageDimension, double, splineOrder> MetricType;
typedef typename MetricType::Pointer MetricPointer;

using CallbackType = BSplineRegistrationCallback<TransformType, ImageDimension>;

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

void sgdm(MetricType* metric, double learningRate, double momentum, unsigned int iterations) {
    
    typedef typename MetricType::DerivativeType DerivativeType;

    unsigned int N = metric->GetNumberOfParameters();

    DerivativeType d(N);
    DerivativeType dcur(N);

    d.Fill(0.0);
    dcur.Fill(0.0);

    for(unsigned int i = 0; i < iterations; ++i) {
        double value;
        metric->GetValueAndDerivative(value, d);

        for(unsigned int j = 0; j < N; ++j) {
	  double d_j = momentum * dcur[j] + (1.0 - momentum) * d[j];
          if(fabs(d_j) < 1e-15) {
            dcur[j] = 0.0;
          } else {
            dcur[j] = d_j;
          }
        }

        metric->UpdateTransformParameters(dcur, learningRate);
    }
}

void register_func(typename ImageType::Pointer fixedImage, typename ImageType::Pointer movingImage, TransformPointer& transformForward, TransformPointer& transformInverse, BSplineRegParam param, ImagePointer fixedMask, ImagePointer movingMask, bool verbose=false)
{
    typedef itk::IPT<double, ImageDimension> IPT;

    MetricPointer metric = MetricType::New();

    metric->SetRandomSeed(param.seed);

    metric->SetFixedImage(fixedImage);
    metric->SetMovingImage(movingImage);

    metric->SetAlphaLevels(param.alphaLevels);

    metric->SetForwardTransformPointer(transformForward);
    metric->SetInverseTransformPointer(transformInverse);

    if(param.samplingMode == "quasi") {
        metric->SetUseQuasiRandomSampling(true);
    } else if(param.samplingMode == "uniform") {
        metric->SetUseQuasiRandomSampling(false);
    }

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
    
    for (int q = 0; q < param.innerParams.size(); ++q) {
        double lr1 = param.innerParams[q].learningRate;
        unsigned int iterations = param.innerParams[q].iterations;
        unsigned int controlPoints = param.innerParams[q].controlPoints;

        metric->SetSymmetryLambda(param.innerParams[q].lambdaFactor);

        metric->SetFixedSamplingPercentage(param.innerParams[q].samplingFraction);
        metric->SetMovingSamplingPercentage(param.innerParams[q].samplingFraction);

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
    } else if(param.optimizer == "sgdm") {
      sgdm(metric.GetPointer(), lr1, param.momentum, iterations);
    } else if(param.optimizer == "sgd") {
        typedef itk::RegularStepGradientDescentOptimizerv4<double> OptimizerType;
        typename OptimizerType::Pointer optimizer = OptimizerType::New();

        optimizer->SetNumberOfIterations(iterations);
        optimizer->SetLearningRate(lr1);
        optimizer->DoEstimateLearningRateOnceOff();
        optimizer->SetMinimumStepLength(0.0);
        optimizer->SetGradientMagnitudeTolerance(1e-8);
        optimizer->SetRelaxationFactor(0.999999);
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
    double absAcc = 0.0;
    double absAccSq = 0.0;
    unsigned int N = metric->GetNumberOfParameters();
    for(unsigned int j = 0; j < N; ++j) {
        double absparam = fabs(parameters[j]);
        if(absparam > absVal) {
            absVal = absparam;
            val = parameters[j];
        }
        absAcc += absparam;
        absAccSq += absparam*absparam;
    }
    absAcc /= N;
    absAccSq /= N;

    std::cout << "Max: " << absVal << ", Mean: " << (absAcc / N) << ", Sd: " << (absAccSq-absAcc*absAcc) << std::endl;

    }

    //chronometer.Report(std::cout);
    //memorymeter.Report(std::cout);
}

typename PointSamplerBase<ImageType, itk::Image<bool, Dim>, ImageType>::Pointer CreateHybridPointSampler(ImagePointer im, ImagePointer maskImage, double w1 = 0.5, bool binaryMode = false, double sigma = 0.0, unsigned int seed = 1000U)
{
    using PointSamplerType = PointSamplerBase<ImageType, itk::Image<bool, Dim>, ImageType>;
    using PointSamplerPointer = typename PointSamplerType::Pointer;
    PointSamplerPointer sampler1 = QuasiRandomPointSampler<ImageType, itk::Image<bool, Dim>, ImageType>::New().GetPointer();
    typename GradientWeightedPointSampler<ImageType, itk::Image<bool, Dim>, ImageType>::Pointer sampler2 =
        GradientWeightedPointSampler<ImageType, itk::Image<bool, Dim>, ImageType>::New().GetPointer();
    sampler2->SetSigma(sigma);
    sampler2->SetBinaryMode(binaryMode);
    sampler2->SetTolerance(1e-9);
    if(maskImage)
    {
        using MaskPointer = typename itk::Image<bool, Dim>::Pointer;
        MaskPointer maskBin = IPT::ThresholdImage(maskImage, 0.01);
        
        sampler1->SetMaskImage(maskBin);
        sampler2->SetMaskImage(maskBin);
    }

    typename HybridPointSampler<ImageType, itk::Image<bool, Dim>, ImageType>::Pointer sampler3 =
        HybridPointSampler<ImageType, itk::Image<bool, Dim>, ImageType>::New();

    sampler3->AddSampler(sampler1, w1);
    sampler3->AddSampler(sampler2.GetPointer(), 1.0-w1);
    sampler3->SetImage(im);
    sampler3->SetSeed(seed);
    sampler3->Initialize();

    return sampler3.GetPointer();
}

typename PointSamplerBase<ImageType, itk::Image<bool, Dim>, ImageType>::Pointer CreateUniformPointSampler(ImagePointer im, ImagePointer maskImage, unsigned int seed = 1000U)
{
    using PointSamplerType = PointSamplerBase<ImageType, itk::Image<bool, Dim>, ImageType>;
    using PointSamplerPointer = typename PointSamplerType::Pointer;

    PointSamplerPointer sampler1 = UniformPointSampler<ImageType, itk::Image<bool, Dim>, ImageType>::New().GetPointer();
    if(maskImage)
    {
        using MaskPointer = typename itk::Image<bool, Dim>::Pointer;
        MaskPointer maskBin = IPT::ThresholdImage(maskImage, 0.01);
        
        sampler1->SetMaskImage(maskBin);
    }

    sampler1->SetImage(im);
    sampler1->Initialize();
    sampler1->SetSeed(seed);

    return sampler1.GetPointer();
}

typename PointSamplerBase<ImageType, itk::Image<bool, Dim>, ImageType>::Pointer CreateQuasiRandomPointSampler(ImagePointer im, ImagePointer maskImage, unsigned int seed = 1000U)
{
    using PointSamplerType = PointSamplerBase<ImageType, itk::Image<bool, Dim>, ImageType>;
    using PointSamplerPointer = typename PointSamplerType::Pointer;

    PointSamplerPointer sampler1 = QuasiRandomPointSampler<ImageType, itk::Image<bool, Dim>, ImageType>::New().GetPointer();
    if(maskImage)
    {
        using MaskPointer = typename itk::Image<bool, Dim>::Pointer;
        MaskPointer maskBin = IPT::ThresholdImage(maskImage, 0.01);
        
        sampler1->SetMaskImage(maskBin);
    }

    sampler1->SetImage(im);
    sampler1->Initialize();
    sampler1->SetSeed(seed);

    return sampler1.GetPointer();
}

size_t ImagePixelCount(typename ImageType::Pointer image)
{
    auto region = image->GetLargestPossibleRegion();
    auto sz = region.GetSize();
    size_t cnt = 1;
    for(unsigned int i = 0; i < Dim; ++i)
    {
        cnt *= sz[i];
    }

    return cnt;
}

#include <chrono>

void mcalpha_register_func(typename ImageType::Pointer fixedImage, typename ImageType::Pointer movingImage, TransformPointer& transformForward, TransformPointer& transformInverse, BSplineRegParam param, ImagePointer fixedMask, ImagePointer movingMask, bool verbose=false, CallbackType* callback=nullptr)
{
    typedef itk::IPT<double, ImageDimension> IPT;

    using DistType = MCAlphaCutPointToSetDistance<ImageType, unsigned short>;
    using DistPointer = typename DistType::Pointer;

    DistPointer distStructRefImage = DistType::New();
    DistPointer distStructFloImage = DistType::New();

    distStructRefImage->SetSampleCount(param.alphaLevels);
    distStructRefImage->SetImage(fixedImage);
    distStructRefImage->SetMaxDistance(0);
    distStructRefImage->SetApproximationThreshold(20.0);
    distStructRefImage->SetApproximationFraction(0.2);

    distStructFloImage->SetSampleCount(param.alphaLevels);
    distStructFloImage->SetImage(movingImage);
    distStructFloImage->SetMaxDistance(0);
    distStructFloImage->SetApproximationThreshold(20.0);
    distStructFloImage->SetApproximationFraction(0.2);

    distStructRefImage->Initialize();
    distStructFloImage->Initialize();

    using RegistrationType = AlphaBSplineRegistration<ImageType, DistType, 3U>;
    using RegistrationPointer = typename RegistrationType::Pointer;

    RegistrationPointer reg = RegistrationType::New();

    using PointSamplerType = PointSamplerBase<ImageType, itk::Image<bool, Dim>, ImageType>;
    using PointSamplerPointer = typename PointSamplerType::Pointer;
    
    PointSamplerPointer sampler1; 
    PointSamplerPointer sampler2;
    constexpr double SIGMA = 0.5;
    
    if(param.samplingMode == "gw" || param.samplingMode == "gw50")
    {
        sampler1 = CreateHybridPointSampler(fixedImage, fixedMask, 0.5, false, SIGMA, param.seed);
        sampler2 = CreateHybridPointSampler(movingImage, movingMask, 0.5, false, SIGMA, param.seed);
    } else if(param.samplingMode == "gw25")
    {
        sampler1 = CreateHybridPointSampler(fixedImage, fixedMask, 0.25, false, SIGMA, param.seed);
        sampler2 = CreateHybridPointSampler(movingImage, movingMask, 0.25, false, SIGMA, param.seed);
    } else if(param.samplingMode == "gw75")
    {
        sampler1 = CreateHybridPointSampler(fixedImage, fixedMask, 0.75, false, SIGMA, param.seed);
        sampler2 = CreateHybridPointSampler(movingImage, movingMask, 0.75, false, SIGMA, param.seed);
    } else if(param.samplingMode == "quasi")
    {
        sampler1 = CreateQuasiRandomPointSampler(fixedImage, fixedMask, param.seed);
        sampler2 = CreateQuasiRandomPointSampler(movingImage, movingMask, param.seed);
    } else if(param.samplingMode == "uniform")
    {
        sampler1 = CreateUniformPointSampler(fixedImage, fixedMask, param.seed);
        sampler2 = CreateUniformPointSampler(movingImage, movingMask, param.seed);
    }

    reg->SetPointSamplerRefImage(sampler1);
    reg->SetPointSamplerFloImage(sampler2);

    reg->SetDistDataStructRefImage(distStructRefImage);
    reg->SetDistDataStructFloImage(distStructFloImage);

    constexpr unsigned int iterations = 1000U;
    constexpr double learningRate = 0.8;
    constexpr double momentum = 0.1;
    constexpr double symmetryLambda = 0.05;

    unsigned int sampleCountRefToFlo = 128;
    unsigned int sampleCountFloToRef = 128;

    reg->SetTransformRefToFlo(transformForward);
    reg->SetTransformFloToRef(transformInverse);

    itk::TimeProbesCollectorBase chronometer;
    itk::MemoryProbesCollectorBase memorymeter;

    for (int q = 0; q < param.innerParams.size(); ++q) {
        double lr1 = param.innerParams[q].learningRate;
        unsigned int iterations = param.innerParams[q].iterations;
        unsigned int controlPoints = param.innerParams[q].controlPoints;

        sampleCountRefToFlo = ImagePixelCount(fixedImage) * param.innerParams[q].samplingFraction;
        sampleCountFloToRef = ImagePixelCount(movingImage) * param.innerParams[q].samplingFraction;

        reg->SetSampleCountRefToFlo(sampleCountRefToFlo);
        reg->SetSampleCountFloToRef(sampleCountFloToRef);
        reg->SetIterations(iterations);
        reg->SetLearningRate(lr1);
        reg->SetSymmetryLambda(param.innerParams[q].lambdaFactor);

        if(q > 0) {
            TransformPointer tforNew = CreateBSplineTransform(fixedImage, controlPoints);
            TransformPointer tinvNew = CreateBSplineTransform(movingImage, controlPoints);
            int curNumberOfGridNodes = controlPoints;
            UpsampleBSplineTransform(fixedImage, tforNew, transformForward, curNumberOfGridNodes);
            UpsampleBSplineTransform(movingImage, tinvNew, transformInverse, curNumberOfGridNodes);
            transformForward = tforNew;
            transformInverse = tinvNew;
        }

        if (param.enableCallbacks) {
            callback->SetTransforms(transformForward, transformInverse);
            reg->AddCallback(callback);
        }

        reg->SetTransformRefToFlo(transformForward);
        reg->SetTransformFloToRef(transformInverse);
        reg->Initialize();

        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

        reg->Run();

        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        if(verbose) {
            std::cout << "Time elapsed: " << std::chrono::duration_cast<std::chrono::seconds>(end - begin).count() << "[s]" << std::endl;
        }

        transformForward = reg->GetTransformRefToFlo();
        transformInverse = reg->GetTransformFloToRef();
    }

    //chronometer.Report(std::cout);
    //memorymeter.Report(std::cout);
}


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
    //bool verbose,
    TransformPointer& transformForwardOut,
    TransformPointer& transformInverseOut,
    CallbackType* callback) {

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
        // Here we can switch between (1) the distance transform-based method
        //register_func(fixedImagePrime, movingImagePrime, transformForward, transformInverse, paramSet, fixedMaskPrime, movingMaskPrime, verbose);       
        // or (2) the monte carlo-based method
        mcalpha_register_func(fixedImagePrime, movingImagePrime, transformForward, transformInverse, paramSet, fixedMaskPrime, movingMaskPrime, paramSet.verbose, callback);       
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
