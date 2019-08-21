
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

constexpr    unsigned int    ImageDimension = 2;
typedef  double           PixelType;

typedef itk::Image< PixelType, ImageDimension >  ImageType;
typedef typename ImageType::Pointer ImagePointer;

static void CreateEllipseImage(ImageType::Pointer image);
static void CreateCircleImage(ImageType::Pointer image);

constexpr unsigned int splineOrder = 3;

typedef itk::BSplineTransform<double, 2, splineOrder> TransformType;
typedef typename TransformType::Pointer TransformPointer;

ImagePointer Chessboard(ImagePointer image1, ImagePointer image2, int cells) {
    itk::FixedArray<unsigned int, ImageDimension> pattern;
    pattern.Fill(cells);

    typedef itk::IPT<double, 2U> IPT;
    typedef itk::CheckerBoardImageFilter< ImageType > CheckerBoardFilterType;
    CheckerBoardFilterType::Pointer checkerBoardFilter = CheckerBoardFilterType::New();
    checkerBoardFilter->SetInput1(image1);
    checkerBoardFilter->SetInput2(image2);
    checkerBoardFilter->SetCheckerPattern(pattern);
    checkerBoardFilter->Update();
    return checkerBoardFilter->GetOutput();
}

ImagePointer BlackAndWhiteChessboard(ImagePointer refImage, int cells) {
    typedef itk::IPT<double, 2U> IPT;
    return Chessboard(IPT::ZeroImage(refImage->GetLargestPossibleRegion().GetSize()),  IPT::ConstantImage(1.0, refImage->GetLargestPossibleRegion().GetSize()), cells);
}

TransformPointer CreateBSplineTransform(ImagePointer image, unsigned int numberOfGridNodes) {
  TransformType::PhysicalDimensionsType   fixedPhysicalDimensions;
  TransformType::MeshSizeType             meshSize;

  TransformPointer transform = TransformType::New();

  for( unsigned int i=0; i < ImageDimension; i++ )
    {
    fixedPhysicalDimensions[i] = image->GetSpacing()[i] *
      static_cast<double>(
        image->GetLargestPossibleRegion().GetSize()[i] - 1 );
    }
  meshSize.Fill( numberOfGridNodes - splineOrder );
  transform->SetTransformDomainOrigin( image->GetOrigin() );
  transform->SetTransformDomainPhysicalDimensions( fixedPhysicalDimensions );
  transform->SetTransformDomainMeshSize( meshSize );
  transform->SetTransformDomainDirection( image->GetDirection() );

  return transform;
}

void UpsampleBSplineTransform(ImagePointer image, TransformPointer newTransform, TransformPointer oldTransform, unsigned int numberOfGridNodes) {
  typedef TransformType::ParametersType ParametersType;
  ParametersType parameters( newTransform->GetNumberOfParameters() );
  parameters.Fill( 0.0 );

  unsigned int counter = 0;
  for ( unsigned int k = 0; k < ImageDimension; k++)
    {
    using ParametersImageType = TransformType::ImageType;
    using ResamplerType = itk::ResampleImageFilter<ParametersImageType,ParametersImageType>;
    ResamplerType::Pointer upsampler = ResamplerType::New();
    using FunctionType = itk::BSplineResampleImageFunction<ParametersImageType,double>;
    FunctionType::Pointer function = FunctionType::New();
    using IdentityTransformType = itk::IdentityTransform<double,ImageDimension>;
    IdentityTransformType::Pointer identity = IdentityTransformType::New();
    upsampler->SetInput( oldTransform->GetCoefficientImages()[k] );
    upsampler->SetInterpolator( function );
    upsampler->SetTransform( identity );
    upsampler->SetSize( newTransform->GetCoefficientImages()[k]->
      GetLargestPossibleRegion().GetSize() );
    upsampler->SetOutputSpacing(
      newTransform->GetCoefficientImages()[k]->GetSpacing() );
    upsampler->SetOutputOrigin(
      newTransform->GetCoefficientImages()[k]->GetOrigin() );
    upsampler->SetOutputDirection( image->GetDirection() );
    using DecompositionType =
        itk::BSplineDecompositionImageFilter<ParametersImageType,ParametersImageType>;
    DecompositionType::Pointer decomposition = DecompositionType::New();
    decomposition->SetSplineOrder( splineOrder );
    decomposition->SetInput( upsampler->GetOutput() );
    decomposition->Update();
    ParametersImageType::Pointer newCoefficients = decomposition->GetOutput();
    // copy the coefficients into the parameter array
    using Iterator = itk::ImageRegionIterator<ParametersImageType>;
    Iterator it( newCoefficients,
      newTransform->GetCoefficientImages()[k]->GetLargestPossibleRegion() );
    while ( !it.IsAtEnd() )
      {
      parameters[ counter++ ] = it.Get();
      ++it;
      }
    }
    newTransform->SetParameters( parameters );
}

    /*static ImagePointer TransformImage(
        ImagePointer inputImage,
        typename itk::Transform<double, Dim, Dim>::Pointer transform,
        PointType originPoint,
        SizeType outputSize,
        ImwarpInterpolationMode interpMode,
        ValueType bgValue = 0)*/
        /*
TransformPointer ModifyBSplineTransform(ImagePointer image, unsigned int numberOfGridNodes, TransformPointer prevTransform) {
  typedef TransformType::CoefficientImageArray CoefficientImageArray;
  typedef itk::IPT<double, 2U> IPT;

  TransformType::PhysicalDimensionsType   fixedPhysicalDimensions;

  TransformPointer transform = CreateBSplineTransform(image, numberOfGridNodes);

  CoefficientImageArray coeffImages = prevTransform->GetCoefficientImages();

  CoefficientImageArray newCoeffImages;
  for(unsigned int i = 0; i < 2U; ++i) {
      itk::Point<double, 2U> origin;
      ImageType::SizeType size;
      origin = 
      newCoeffImages[i] = IPT::TransformImage(coeffImages[i], itk::IdentityTransform<double, 2U>::New(), )


  }

  fixedPhysicalDimensions = transform->GetTransformDomainPhysicalDimensions();
  
  for(unsigned int i = 0; i < numberOfGridNodes; ++i) {
    for(unsigned int j = 0; j < numberOfGridNodes; ++j) {
      
    }
  }
  
  return transform;
}*/

ImageType::Pointer ApplyTransform(ImagePointer refImage, ImagePointer floImage, TransformPointer transform) {
    typedef itk::ResampleImageFilter<
                            ImageType,
                            ImageType >    ResampleFilterType;

    typedef itk::IPT<double, 2U> IPT;

    ResampleFilterType::Pointer resample = ResampleFilterType::New();

    resample->SetTransform( transform );
    resample->SetInput( floImage );

    resample->SetSize(    refImage->GetLargestPossibleRegion().GetSize() );
    resample->SetOutputOrigin(  refImage->GetOrigin() );
    resample->SetOutputSpacing( refImage->GetSpacing() );
    resample->SetOutputDirection( refImage->GetDirection() );
    resample->SetDefaultPixelValue(0.5);

    resample->UpdateLargestPossibleRegion();
    
    return resample->GetOutput();
}

double normalizeDerivative(unsigned int gridPoints) {
    //double sum = 0.0;
    double frac = (gridPoints*gridPoints) / 25.0;
    return frac;
    //for(unsigned int i = 0; i < array.GetSize(); ++i) {
        //sum += fabs(array[i]);
    //    array[i] = array[i] * frac;
    //}
    /*
    if(sum < 0.000001) {
        sum = 0.000001;
    }
    for(unsigned int i = 0; i < array.GetSize(); ++i) {
        array[i] = array[i] / sum;
        //sum += fabs(array[i]);
    } */   
}

double maxDerivative(itk::Array<double> &array) {
    double sum = 0.0;
    unsigned int count = array.GetSize();
    for(unsigned int i = 0; i < count; ++i) {
      double value = fabs(array[i]);
        if(value > sum)
            sum = value;
    }
    return sum;
}

double norm(itk::Array<double> &array, double p, double eps=1e-7) {
    double sum = 0.0;
    unsigned int count = array.GetSize();
    for(unsigned int i = 0; i < count; ++i) {
      double value = pow(fabs(array[i]), p);
      sum += value;
    }
    sum = pow(sum, 1.0/p);
    if(sum < eps)
      sum = eps;
    return sum;
}

struct BSplineRegParam {
  double learningRate1;
  double learningRate2;
  double lambdaFactor;
  int iterations;
  double samplingFraction;
  int controlPoints;
};

void register_func(typename ImageType::Pointer fixedImage, typename ImageType::Pointer movingImage, TransformPointer transformForward, TransformPointer transformInverse, BSplineRegParam param) {
    typedef itk::AlphaSMDObjectToObjectMetricDeformv4<ImageType, ImageDimension, double, splineOrder> MetricType;
    typedef typename MetricType::Pointer MetricPointer;

    MetricPointer metric = MetricType::New();

    metric->SetRandomSeed(1337);

    metric->SetFixedImage(fixedImage);
    metric->SetMovingImage(movingImage);

    metric->SetAlphaLevels(7);

    metric->SetForwardTransformPointer(transformForward);
    metric->SetInverseTransformPointer(transformInverse);

    metric->SetFixedSamplingPercentage(param.samplingFraction);
    metric->SetMovingSamplingPercentage(param.samplingFraction);
    itk::TimeProbesCollectorBase chronometer;
    itk::MemoryProbesCollectorBase memorymeter;

    chronometer.Start("Pre-processing");
    memorymeter.Start("Pre-processing");

    metric->Update();

    chronometer.Stop("Pre-processing");
    memorymeter.Stop("Pre-processing");
    
    //chronometer.Report( std::cout );
    //memorymeter.Report( std::cout );

    typedef typename MetricType::MeasureType    MeasureType;
    typedef typename MetricType::DerivativeType DerivativeType;

    MeasureType value;
    DerivativeType derivative(metric->GetNumberOfParameters());
    DerivativeType derAcc(metric->GetNumberOfParameters());

    derAcc.fill(0.0);
    
    typedef itk::IPT<double, 2U> IPT;

    metric->SetSymmetryLambda(param.lambdaFactor);
    double p = 2.0;
    double maxGrad = 1.0;
    double momentum = 0.0;
    double maxMomentum = 0.5;
    double meanValue;

    chronometer.Start("Registration");
    memorymeter.Start("Registration");
    
    for(int i = 0; i < param.iterations; ++i) {
      double alpha = (double)i / param.iterations;
      double learningRate = param.learningRate1 * (1.0-alpha) + param.learningRate2 * alpha;
        metric->GetValueAndDerivative(value, derivative);

	if(i == 0)
	  meanValue = value;
	else
	  meanValue = 0.95 * meanValue + 0.05 * value;
	
        for(unsigned int j = 0; j < metric->GetNumberOfParameters(); ++j) {
	  derivative[j] = derAcc[j] * (momentum) + derivative[j] * (1.0-momentum);
	  derAcc[j] = derivative[j];
	}
        //double maxGrad = maxDerivative(derivative);
	  //if(maxGrad < 1e-7)
        //    maxGrad = 1e-7;
	maxGrad = (1.0-maxMomentum) * norm(derivative, p) + maxMomentum * maxGrad;
        double curLR = learningRate;// / maxGrad;
	
        metric->UpdateTransformParameters(derivative, curLR);//learningRate);

        if(i % 50 == 0 || (i+1) == param.iterations) {
	  std::cout << "Iteration " << (i+1) << "... Value: " << meanValue << ", Derivative: " << maxGrad << ", lr: " << curLR << std::endl;
            //std::cout << metric->GetParameters() << std::endl;
            //std::cout << "Derivative: " << derivative << std::endl;
        }
    }

    chronometer.Stop("Registration");
    memorymeter.Stop("Registration");

    chronometer.Report( std::cout );
    memorymeter.Report( std::cout );

    metric->SetSymmetryLambda(1.0);
    metric->SetFixedSamplingPercentage(1.0);
    metric->SetMovingSamplingPercentage(1.0);
    metric->GetValueAndDerivative(value, derivative);

    std::cout << "Final Symmetry Loss: " << value << std::endl;
    
}

// Parameters ref_image.png flo_image.png learning_rate1 learning_rate2 symmetry_factor iterations
int main(int argc, char **argv)
{
  if(argc < 9) {
    std::cout << "ACTestBSpline ref_image.png flo_image.png learning_rate symmetry_factor iterations sampling_fraction control_point_count" << std::endl;
    return -1;
  }
  BSplineRegParam param;
    param.learningRate1 = atof(argv[3]);
    param.learningRate2 = atof(argv[4]);
    param.lambdaFactor = atof(argv[5]);
    param.iterations = atoi(argv[6]);
    param.samplingFraction = atof(argv[7]);
    param.controlPoints = atoi(argv[8]);
    const char* in_extra = (argc >= 10) ? argv[9] : 0;
    
    typedef itk::IPT<double, 2U> IPT;

    typedef double CoordinateRepType;

    // Create the synthetic images
    //ImageType::Pointer  fixedImage  = ImageType::New();
    //CreateCircleImage(fixedImage);

    //ImageType::Pointer movingImage = ImageType::New();
    //CreateEllipseImage(movingImage);

    //ImageType::Pointer fixedImage = IPT::LoadImage("fixed_circle.png");
    //ImageType::Pointer movingImage = IPT::LoadImage("moving_circle.png");

    ImageType::Pointer fixedImage = IPT::LoadImage(argv[1]);
    ImageType::Pointer movingImage = IPT::LoadImage(argv[2]);

    // Create transforms

    unsigned int numberOfGridNodes = param.controlPoints;
    TransformPointer transformForward = CreateBSplineTransform(fixedImage, numberOfGridNodes);
    TransformPointer transformInverse = CreateBSplineTransform(movingImage, numberOfGridNodes);


    typedef itk::IPT<double, 2U> IPT;

    IPT::SaveImageU8("fixed.png", fixedImage);
    IPT::SaveImageU8("moving.png", movingImage);

    // optimize

    //double learningRate = 2.0;// * normalizeDerivative(numberOfGridNodes);
    //double lambdaFactor = 0.05;
    //int iterations = 5000;
    //double learningRate = atof(argv[3]);
    //double lambdaFactor = atof(argv[4]);
    //int iterations = atoi(argv[5]);

    //std::cout << derivative << std::endl;

    int levels = 4;
    double sigma[] = {3.0, 0.0, 0.0, 0.0};
    int subsamplingFactor[] = {2, 2, 1, 1};
    unsigned int levelNumberOfGridNodes[] = {numberOfGridNodes, (unsigned int)(numberOfGridNodes*2), numberOfGridNodes*3, numberOfGridNodes*4};

    double normalizationPercentage = 0.05;
    unsigned int curNumberOfGridNodes = numberOfGridNodes;
    ImageType::Pointer movingTransformed;
    ImageType::Pointer fixedTransformed;

    for(unsigned int i = 0; i < levels; ++i) {
      ImagePointer fixedImagePrime = IPT::SmoothImage(fixedImage, sigma[i]);
      ImagePointer movingImagePrime = IPT::SmoothImage(movingImage, sigma[i]);
      fixedImagePrime = IPT::SubsampleImage(fixedImagePrime, subsamplingFactor[i]);
      movingImagePrime = IPT::SubsampleImage(movingImagePrime, subsamplingFactor[i]);
      fixedImagePrime = IPT::NormalizeImage(fixedImagePrime, IPT::IntensityMinMax(fixedImagePrime, normalizationPercentage));
      movingImagePrime = IPT::NormalizeImage(movingImagePrime, IPT::IntensityMinMax(movingImagePrime, normalizationPercentage));
      if(levelNumberOfGridNodes[i] != curNumberOfGridNodes) {
          TransformPointer tforNew = CreateBSplineTransform(fixedImage, levelNumberOfGridNodes[i]);
          TransformPointer tinvNew = CreateBSplineTransform(movingImage, levelNumberOfGridNodes[i]);
          curNumberOfGridNodes = levelNumberOfGridNodes[i];
          UpsampleBSplineTransform(fixedImage, tforNew, transformForward, curNumberOfGridNodes);
          UpsampleBSplineTransform(movingImage, tinvNew, transformInverse, curNumberOfGridNodes);
          transformForward = tforNew;
          transformInverse = tinvNew;
      }
      register_func(fixedImagePrime, movingImagePrime, transformForward, transformInverse, param);
      movingTransformed = ApplyTransform(fixedImage, movingImage, transformForward);
      fixedTransformed = ApplyTransform(movingImage, fixedImage, transformInverse);
      
      char buf[512];
      sprintf(buf, "moving_transformed_to_fixed_%d.png", i+1);
      IPT::SaveImageU8(buf, movingTransformed);
      sprintf(buf, "fixed_transformed_to_moving_%d.png", i+1);
      IPT::SaveImageU8(buf, fixedTransformed);
      
    }

    //std::cout << "Generating transfomed images." << std::endl;
    
    //ImageType::Pointer movingTransformed = ApplyTransform(fixedImage, movingImage, transformForward);
    //ImageType::Pointer fixedTransformed = ApplyTransform(movingImage, fixedImage, transformInverse);

    ImageType::Pointer chessboard = BlackAndWhiteChessboard(movingImage, 24);
    ImageType::Pointer chessboardTransformed = ApplyTransform(fixedImage, chessboard, transformForward);

    std::cout << "Generating diff images." << std::endl;
    
    ImageType::Pointer movingDiff = IPT::DifferenceImage(movingTransformed, fixedImage);
    ImageType::Pointer fixedDiff = IPT::DifferenceImage(fixedTransformed, movingImage);

    std::cout << "Saving diff images." << std::endl;

    IPT::SaveImageU8("moving_diff.png", movingDiff);
    IPT::SaveImageU8("fixed_diff.png", fixedDiff);

    IPT::SaveImageU8("chessboardFixed.png", chessboardTransformed);

    if(in_extra) {
        ImageType::Pointer imExtra = IPT::LoadImage(in_extra);
        ImageType::Pointer imExtraTransformed = ApplyTransform(fixedImage, imExtra, transformForward);
        IPT::SaveImageU8("extra.png", imExtraTransformed);
    }

    return 0;
}

void CreateEllipseImage(ImageType::Pointer image)
{
  typedef itk::EllipseSpatialObject< ImageDimension >   EllipseType;

  typedef itk::SpatialObjectToImageFilter<
    EllipseType, ImageType >   SpatialObjectToImageFilterType;

  SpatialObjectToImageFilterType::Pointer imageFilter =
    SpatialObjectToImageFilterType::New();

  ImageType::SizeType size;
  size[ 0 ] =  100;
  size[ 1 ] =  100;

  imageFilter->SetSize( size );

  ImageType::SpacingType spacing;
  spacing.Fill(1);
  imageFilter->SetSpacing(spacing);

  EllipseType::Pointer ellipse    = EllipseType::New();
  EllipseType::ArrayType radiusArray;
  radiusArray[0] = 10;
  radiusArray[1] = 20;
  ellipse->SetRadius(radiusArray);

  typedef EllipseType::TransformType                 TransformType;
  TransformType::Pointer transform = TransformType::New();
  transform->SetIdentity();

  TransformType::OutputVectorType  translation;
  TransformType::CenterType        center;

  translation[ 0 ] =  65;
  translation[ 1 ] =  45;
  transform->Translate( translation, false );

  ellipse->SetObjectToParentTransform( transform );

  imageFilter->SetInput(ellipse);

  ellipse->SetDefaultInsideValue(1.0);
  ellipse->SetDefaultOutsideValue(0);
  imageFilter->SetUseObjectValue( true );
  imageFilter->SetOutsideValue( 0 );

  imageFilter->Update();

  image->Graft(imageFilter->GetOutput());

}

void CreateCircleImage(ImageType::Pointer image)
{
 typedef itk::EllipseSpatialObject< ImageDimension >   EllipseType;

  typedef itk::SpatialObjectToImageFilter<
    EllipseType, ImageType >   SpatialObjectToImageFilterType;

  SpatialObjectToImageFilterType::Pointer imageFilter =
    SpatialObjectToImageFilterType::New();

  ImageType::SizeType size;
  size[ 0 ] =  100;
  size[ 1 ] =  100;

  imageFilter->SetSize( size );

  ImageType::SpacingType spacing;
  spacing.Fill(1);
  imageFilter->SetSpacing(spacing);

  EllipseType::Pointer ellipse    = EllipseType::New();
  EllipseType::ArrayType radiusArray;
  radiusArray[0] = 10;
  radiusArray[1] = 10;
  ellipse->SetRadius(radiusArray);

  typedef EllipseType::TransformType                 TransformType;
  TransformType::Pointer transform = TransformType::New();
  transform->SetIdentity();

  TransformType::OutputVectorType  translation;
  TransformType::CenterType        center;

  translation[ 0 ] =  50;
  translation[ 1 ] =  50;
  transform->Translate( translation, false );

  ellipse->SetObjectToParentTransform( transform );

  imageFilter->SetInput(ellipse);

  ellipse->SetDefaultInsideValue(1.0);
  ellipse->SetDefaultOutsideValue(0.0);
  imageFilter->SetUseObjectValue( true );
  imageFilter->SetOutsideValue( 0 );

  imageFilter->Update();

  image->Graft(imageFilter->GetOutput());
}
