
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

const    unsigned int    ImageDimension = 2;
typedef  double           PixelType;

typedef itk::Image< PixelType, ImageDimension >  ImageType;
typedef typename ImageType::Pointer ImagePointer;

static void CreateEllipseImage(ImageType::Pointer image);
static void CreateCircleImage(ImageType::Pointer image);

constexpr unsigned int splineOrder = 3;

typedef itk::BSplineTransform<double, 2, splineOrder> TransformType;
typedef typename TransformType::Pointer TransformPointer;

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

double normalizeDerivative(unsigned int gridPoints, unsigned int dim) {
    //double sum = 0.0;
    double frac = pow(gridPoints/5.0, (double)dim);
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

// Parameters ref_image.png flo_image.png learning_rate1 learning_rate2 symmetry_factor iterations
int main(int argc, char **argv)
{
  if(argc < 9) {
    std::cout << "ACTestBSpline ref_image.png flo_image.png learning_rate symmetry_factor iterations sampling_fraction control_point_count" << std::endl;
    return -1;
  }
    double learningRate1 = atof(argv[3]);
    double learningRate2 = atof(argv[4]);
    double lambdaFactor = atof(argv[5]);
    int iterations = atoi(argv[6]);
    double samplingFraction = atof(argv[7]);
    int controlPoints = atoi(argv[8]);
    
    typedef itk::AlphaSMDObjectToObjectMetricDeformv4<ImageType, ImageDimension, double, splineOrder> MetricType;
    typedef typename MetricType::Pointer MetricPointer;
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

    typedef typename MetricType::MeasureType    MeasureType;
    typedef typename MetricType::DerivativeType DerivativeType;

    MeasureType value;
    DerivativeType derivative(metric->GetNumberOfParameters());

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

    metric->SetSymmetryLambda(lambdaFactor);
    double p = 20.0;
    
    for(int i = 0; i < iterations; ++i) {
      double alpha = (double)i / iterations;
      double learningRate = learningRate1 * (1.0-alpha) + learningRate2 * alpha;
        metric->GetValueAndDerivative(value, derivative);
        //double maxGrad = maxDerivative(derivative);
	  //if(maxGrad < 1e-7)
        //    maxGrad = 1e-7;
	double maxGrad = norm(derivative, p);
        double curLR = learningRate / maxGrad;
    
        metric->UpdateTransformParameters(derivative, curLR);//learningRate);

        learningRate = learningRate * 0.999;
        if(i % 50 == 0 || (i+1) == iterations) {
	  std::cout << "Iteration " << (i+1) << "... Value: " << value << ", Derivative: " << maxGrad << ", lr: " << curLR << std::endl;
            //std::cout << metric->GetParameters() << std::endl;
            //std::cout << "Derivative: " << derivative << std::endl;
        }
    }

    metric->SetSymmetryLambda(1.0);
    metric->SetFixedSamplingPercentage(1.0);
    metric->SetMovingSamplingPercentage(1.0);
    metric->GetValueAndDerivative(value, derivative);

    std::cout << "Final Symmetry Loss: " << value << std::endl;

    //std::cout << derivative << std::endl;

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
