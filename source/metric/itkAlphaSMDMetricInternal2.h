
#ifndef ALPHASMD_METRIC_INTERNAL2_H
#define ALPHASMD_METRIC_INTERNAL2_H

//
// ITK dependecies
//

#include <cstdlib>

// Core ITK facilities
#include "itkPoint.h"
#include "itkMath.h"
#include "itkImage.h"
#include "itkSmartPointer.h"
#include "itkNumericTraits.h"

// Filters
#include "itkShiftScaleImageFilter.h"
#include "itkBinaryThresholdImageFilter.h"
#include "itkDanielssonDistanceMapImageFilter.h"
#include <itkSignedMaurerDistanceMapImageFilter.h>
#include "itkGradientImageFilter.h"
#include <itkMinimumMaximumImageCalculator.h>
#include "itkChangeInformationImageFilter.h"
#include "itkImageDuplicator.h"

// Interpolators

#include "itkNearestNeighborInterpolateImageFunction.h"
#include "itkLinearInterpolateImageFunction.h"

namespace itk
{
namespace alphasmdinternal2
{
typedef float InternalValueType;
// Representation of one source point in an image, to be mapped (along with its height and weight)
// into the other image's space.

template <typename QType, unsigned int Dim>
struct SourcePoint {
	itk::Point<double, Dim> m_SourcePoint;
	double m_Weight;
	QType m_Height;
};

// Generate a size from a fixed array

template <typename ValueType, unsigned int Dim>
typename itk::Image<ValueType, Dim>::SizeType MakeSize(const itk::FixedArray<unsigned int, Dim> &sz)
{
	typename itk::Image<ValueType, Dim>::SizeType result;

	for (unsigned int i = 0; i < Dim; ++i)
	{
		result[i] = sz;
	}

	return result;
}

template <typename ValueType, unsigned int Dim>
typename itk::Image<ValueType, Dim>::SizeType GetSize(
	typename itk::Image<ValueType, Dim>::Pointer image) {

		typename itk::Image<ValueType, Dim>::RegionType r(image->GetLargestPossibleRegion());

		return r.GetSize();
}

// Function for creating a new image with value type 'ValueType' and spatial dimensionality 'Dim'

template <typename ValueType, unsigned int Dim>
typename itk::Image<ValueType, Dim>::Pointer MakeImage(
	typename itk::Image<ValueType, Dim>::RegionType region,
	const typename itk::Image<ValueType, Dim>::SpacingType& scaling,
	const typename itk::Image<ValueType, Dim>::PointType& origin)
{
	typedef itk::Image<ValueType, Dim> ImageType;
	typedef typename ImageType::Pointer ImagePointer;

	ImagePointer image = ImageType::New();

	image->SetRegions(region);
	image->SetSpacing(scaling);
	image->SetOrigin(origin);
	image->Allocate();

	return image;
}

template <typename ValueType, unsigned int Dim>
typename itk::Image<ValueType, Dim>::Pointer MakeImage(
	typename itk::Image<ValueType, Dim>::RegionType imageRegion,
	const typename itk::Image<ValueType, Dim>::SpacingType& scaling,
	const typename itk::Image<ValueType, Dim>::PointType& origin,
	ValueType initialValue)
{
	typedef itk::Image<ValueType, Dim> ImageType;
	typedef typename ImageType::Pointer ImagePointer;
/*
	ImagePointer image = ImageType::New();

	image->SetRegions(imageRegion);
	image->SetSpacing(scaling);
	image->SetOrigin(origin);
	image->Allocate();*/

	ImagePointer image = MakeImage<ValueType, Dim>(imageRegion, scaling, origin);

	image->FillBuffer(initialValue);
/*
	itk::ImageRegionIterator<ImageType> it(image, image->GetLargestPossibleRegion());
	it.GoToBegin();

	while (!it.IsAtEnd()) {
		it.Set(initialValue);

		++it;
	}*/

	return image;
}

template <typename ValueType, unsigned int Dim>
typename itk::Image<ValueType, Dim>::Pointer CloneImage(typename itk::Image<ValueType, Dim>::Pointer image)
{
    typedef itk::ImageDuplicator<itk::Image<ValueType, Dim> > DuplicatorType;
    typedef typename DuplicatorType::Pointer DuplicatorPointer;

    DuplicatorPointer dup = DuplicatorType::New();

    dup->SetInputImage(image);

    dup->Update();

    return dup->GetOutput();
}

template <typename ValueType, unsigned int Dim>
typename itk::Image<ValueType, Dim>::Pointer MakeCircularWeightImage(
	const typename itk::Image<ValueType, Dim>::RegionType& region,
	const typename itk::Image<ValueType, Dim>::SpacingType& scaling,
	ValueType zero, ValueType one) {

	typedef itk::Image<ValueType, Dim> ImageType;
	typedef typename ImageType::Pointer ImagePointer;

	ImagePointer image = ImageType::New();

	typename ImageType::IndexType start = region.GetIndex();
	typename ImageType::SizeType size = region.GetSize();

	itk::Point<double, Dim> center;
	ValueType radius[Dim];

	for(unsigned int i = 0; i < Dim; ++i) {
		center[i] = 0.5 * ((double)size[i] - (double)start[i]);
		radius[i] = 0.5 * ((double)size[i] - (double)start[i]);

		if(radius[i] > 0.0) {
			radius[i] = 1.0 / radius[i];
		} else {
			radius[i] = 0.0;
		}
	}

	image->SetRegions(region);
	image->SetSpacing(scaling);
	image->Allocate();

	itk::ImageRegionIterator<ImageType> writeIter(image, image->GetLargestPossibleRegion());
	writeIter.GoToBegin();

	itk::Point<double, Dim> point;
	
	while (!writeIter.IsAtEnd())
	{
		image->TransformIndexToPhysicalPoint(writeIter.GetIndex(), point);
		
		for(unsigned int i = 0; i < Dim; ++i) {
			point[i] *= radius[i];
		}

		if(point.EuclideanDistanceTo(center) <= 1.0)
			writeIter.Set(one);
		else
			writeIter.Set(zero);

		++writeIter;
	}

	return image;
}

template <typename ValueType, unsigned int Dim>
typename itk::Image<ValueType, Dim>::Pointer MakeConstantWeightImage(
	const typename itk::Image<ValueType, Dim>::RegionType& region,
	const typename itk::Image<ValueType, Dim>::SpacingType& scaling,
	ValueType value) {
	typedef itk::Image<ValueType, Dim> ImageType;
	typedef typename ImageType::Pointer ImagePointer;

	ImagePointer image = ImageType::New();

	image->SetRegions(region);
	image->SetSpacing(scaling);
	image->Allocate();

	itk::ImageRegionIterator<ImageType> writeIter(image, image->GetLargestPossibleRegion());
	writeIter.GoToBegin();

	while (!writeIter.IsAtEnd())
	{		
		writeIter.Set(value);

		++writeIter;
	}

	return image;
}

template <typename ValueType, unsigned int Dim>
void WriteToImage(
	typename itk::Image<ValueType, Dim>::Pointer dest,
	typename itk::Image<ValueType, Dim>::Pointer src)
{
	typedef itk::Image<ValueType, Dim> ValueImage;

	itk::ImageRegionIterator<ValueImage> writeIter(dest, dest->GetLargestPossibleRegion());
	itk::ImageRegionConstIterator<ValueImage> readIter(src, src->GetLargestPossibleRegion());

	readIter.GoToBegin();
	writeIter.GoToBegin();

	while (!writeIter.IsAtEnd())
	{
		assert(readIter.IsAtEnd() == false);

		writeIter.Set(readIter.Get());

		++readIter;
		++writeIter;
	}
}

template <typename ValueType, unsigned int Dim>
void AddToImage(
	typename itk::Image<ValueType, Dim>::Pointer dest,
	typename itk::Image<ValueType, Dim>::Pointer src)
{
	typedef itk::Image<ValueType, Dim> ValueImage;

	itk::ImageRegionIterator<ValueImage> writeIter(dest, dest->GetLargestPossibleRegion());
	itk::ImageRegionConstIterator<ValueImage> readIter(src, src->GetLargestPossibleRegion());

	readIter.GoToBegin();
	writeIter.GoToBegin();

	while (!writeIter.IsAtEnd())
	{
		assert(readIter.IsAtEnd() == false);

		ValueType tmp = writeIter.Get() + readIter.Get();
		writeIter.Set(tmp);

		++readIter;
		++writeIter;
	}
}

template <typename ValueType, unsigned int Dim>
void MultiplyImageWithConstant(
	typename itk::Image<ValueType, Dim>::Pointer image,
	ValueType c)
{
	typedef itk::Image<ValueType, Dim> ValueImage;

	itk::ImageRegionIterator<ValueImage> writeIter(image, image->GetRequestedRegion());

	writeIter.GoToBegin();

	while (!writeIter.IsAtEnd())
	{
		writeIter.Set(writeIter.Get() * c);

		++writeIter;
	}
}

template <typename ValueType, unsigned int Dim>
void SaturateImageWithConstant(
	typename itk::Image<ValueType, Dim>::Pointer image,
	ValueType c,
	bool saturateUpper = true)
{
	typedef itk::Image<ValueType, Dim> ValueImage;

	itk::ImageRegionIterator<ValueImage> writeIter(image, image->GetRequestedRegion());

	writeIter.GoToBegin();

	if(saturateUpper) {
		while (!writeIter.IsAtEnd())
		{
			if(writeIter.Get() > c)
				writeIter.Set(c);

			++writeIter;
		}
	} else {
		while (!writeIter.IsAtEnd())
		{
			if(writeIter.Get() < c)
				writeIter.Set(c);

			++writeIter;
		}
	}
}

template <typename ValueType, unsigned int Dim>
ValueType ImageMaximum(
	typename itk::Image<ValueType, Dim>::Pointer image) {

	typedef itk::MinimumMaximumImageCalculator<itk::Image<ValueType, Dim> > MinMaxCalculatorType;
	typedef typename MinMaxCalculatorType::Pointer MinMaxCalculatorPointer;

	MinMaxCalculatorPointer calc = MinMaxCalculatorType::New();

	calc->SetImage(image);

	calc->ComputeMaximum();

	return calc->GetMaximum();
}

template <typename ValueType, unsigned int Dim>
void ApplyMask(
	typename itk::Image<ValueType, Dim>::Pointer dest,
	typename itk::Image<bool, Dim>::Pointer mask,
	ValueType fillValue,
	bool maskValue = false)
{
	typedef itk::Image<ValueType, Dim> ValueImage;

	itk::ImageRegionIterator<ValueImage> writeIter(dest, dest->GetLargestPossibleRegion());
	itk::ImageRegionConstIterator<itk::Image<bool, Dim> > readIter(mask, mask->GetLargestPossibleRegion());

	readIter.GoToBegin();
	writeIter.GoToBegin();

	while (!writeIter.IsAtEnd())
	{
		assert(readIter.IsAtEnd() == false);

		if(readIter.Get() == maskValue) {
			writeIter.Set(fillValue);
		}

		++readIter;
		++writeIter;
	}
}

template <typename ValueType, unsigned int Dim>
typename itk::Image<bool, Dim>::Pointer ThresholdImage(
	typename itk::Image<ValueType, Dim>::Pointer image,
	ValueType threshold)
{
	typedef itk::BinaryThresholdImageFilter<itk::Image<ValueType, Dim>, itk::Image<bool, Dim> > ThresholderType;

	typename ThresholderType::Pointer thresholder = ThresholderType::New();

	thresholder->SetInput(image);
	thresholder->SetInsideValue(true);
	thresholder->SetOutsideValue(false);

	thresholder->SetLowerThreshold(threshold);

	thresholder->Update(); // Finalize thresholding

	return thresholder->GetOutput();
}

template <typename ValueType, unsigned int Dim>
typename itk::Image<ValueType, Dim>::Pointer ComplementImage(
	typename itk::Image<ValueType, Dim>::Pointer image,
	ValueType maxValue)
{

	typename itk::Image<ValueType, Dim>::Pointer dest = MakeImage<ValueType, Dim>(image->GetLargestPossibleRegion(), image->GetSpacing(), image->GetOrigin());

	itk::ImageRegionIterator<itk::Image<ValueType, Dim> > writeIter(dest, dest->GetLargestPossibleRegion());
	itk::ImageRegionConstIterator<itk::Image<ValueType, Dim> > readIter(image, image->GetLargestPossibleRegion());

	readIter.GoToBegin();
	writeIter.GoToBegin();

	while (!writeIter.IsAtEnd())
	{
		assert(readIter.IsAtEnd() == false);

		writeIter.Set(maxValue - readIter.Get());

		++readIter;
		++writeIter;
	}
	return dest;
}

template <unsigned int Dim>
typename itk::Image<bool, Dim>::Pointer ComplementBinaryImage(
	typename itk::Image<bool, Dim>::Pointer image)
{
	typename itk::Image<bool, Dim>::Pointer dest = MakeImage<bool, Dim>(image->GetLargestPossibleRegion(), image->GetSpacing(), image->GetOrigin());

	itk::ImageRegionIterator<itk::Image<bool, Dim> > writeIter(dest, dest->GetLargestPossibleRegion());
	itk::ImageRegionConstIterator<itk::Image<bool, Dim> > readIter(image, image->GetLargestPossibleRegion());

	readIter.GoToBegin();
	writeIter.GoToBegin();

	while (!writeIter.IsAtEnd())
	{
		assert(readIter.IsAtEnd() == false);

		writeIter.Set(!readIter.Get());

		++readIter;
		++writeIter;
	}
	return dest;
}

// Quantize image
// ValueType must be a floating point type

template <typename QType, typename ValueType, unsigned int Dim>
typename itk::Image<QType, Dim>::Pointer QuantizeImage(
	typename itk::Image<ValueType, Dim>::Pointer image,
	QType levels)
{
	typedef itk::Image<ValueType, Dim> ValueImage;
	typename itk::Image<QType, Dim>::Pointer dest = MakeImage<QType, Dim>(image->GetLargestPossibleRegion(), image->GetSpacing(), image->GetOrigin());

	itk::ImageRegionIterator<itk::Image<QType, Dim> > writeIter(dest, dest->GetLargestPossibleRegion());
	itk::ImageRegionConstIterator<ValueImage> readIter(image, image->GetLargestPossibleRegion());

	writeIter.GoToBegin();
	readIter.GoToBegin();

	while (!writeIter.IsAtEnd())
	{
		assert(readIter.IsAtEnd() == false);

		ValueType x = readIter.Get();
		int qtmp = 1 + itk::Math::floor((x - 0.5/levels) * levels);

		if(qtmp < 0)
			qtmp = 0;
		if(qtmp > levels)
			qtmp = levels;
		
		QType q = (QType)qtmp;

		assert(q >= 0);
		assert(q <= levels);

		writeIter.Set(q);

		++readIter;
		++writeIter;
	}

	return dest;
}

// Generates a source point set, with masked (and zero weighted) pixels removed.
// For kept points, store the height/value of the point and the weight.

template <typename QType, unsigned int Dim>
void ImageToMaskedSourcePointSet(
	typename itk::Image<QType, Dim>::Pointer image,
	typename itk::Image<bool, Dim>::Pointer mask,
	typename itk::Image<double, Dim>::Pointer weightImage,
	std::vector<SourcePoint<QType, Dim> >& spsOut) {

	typedef itk::ImageRegionConstIterator<itk::Image<QType, Dim> > IteratorType;
	typedef itk::ImageRegionConstIterator<itk::Image<double, Dim> > WeightIteratorType;
	typedef itk::ImageRegionConstIterator<itk::Image<bool, Dim> > MaskIteratorType;

	IteratorType it( image, image->GetLargestPossibleRegion() );
	WeightIteratorType itWeight( weightImage, weightImage->GetLargestPossibleRegion() );
	MaskIteratorType itMask( mask, mask->GetLargestPossibleRegion() );
  	it.GoToBegin();
  	itWeight.GoToBegin();
  	itMask.GoToBegin();

	itk::Point<double, Dim> point;
	SourcePoint<QType, Dim> sp;

	while( !it.IsAtEnd() )
    {
		assert(!itMask.IsAtEnd());
		assert(!itWeight.IsAtEnd());

		double wgt = itWeight.Get();

		if(itMask.Get() && wgt > 0.0) {
			// Read the point position
			image->TransformIndexToPhysicalPoint(it.GetIndex(), point);
			sp.m_SourcePoint = point; 

			// Read the height
			sp.m_Height = it.Get();

			// Initialize to unit weight
			sp.m_Weight = itWeight.Get();

			spsOut.push_back(sp);
		}

		++it;
		++itMask;
		++itWeight;
	}
}

template <typename ValueType, typename QType, unsigned int Dim>
class ImageStack
{
  public:
	typedef itk::Image<ValueType, Dim> ValueImage;
	typedef typename ValueImage::Pointer ValueImagePointer;

	void Clear() {
		m_ImageStack.clear();
	}

	void AddZeroLayer(typename ValueImage::RegionType region, typename ValueImage::SpacingType spacing, typename ValueImage::PointType origin, ValueType zero) {
		m_ImageStack.push_back(MakeImage<ValueType, Dim>(region, spacing, origin, zero));
	}

	void AddLayer(ValueImagePointer image) {
		m_ImageStack.push_back(image);
	}

	size_t GetLayerCount() const
	{
		return m_ImageStack.size();
	}

	typename ValueImage::SizeType GetLayerSize(QType level = 0) const
	{
		assert(level < m_ImageStack.size());

		return GetSize<ValueType, Dim>(m_ImageStack[level]);
	}

	ValueImage* GetLayerPtr(QType level) const {
		assert(level >= 0);
		assert(level < m_ImageStack.size());

		return m_ImageStack[level].GetPointer();		
	}

	ValueImagePointer GetLayer(QType level) const
	{
		assert(level >= 0);
		assert(level < m_ImageStack.size());

		return m_ImageStack[level];
	}

	void Reverse()
	{
		std::reverse(m_ImageStack.begin(), m_ImageStack.end());
	}

	void SumStack() {
		for(size_t ind = 1; ind < m_ImageStack.size(); ++ind) {
			AddToImage<ValueType, Dim>(m_ImageStack[ind], m_ImageStack[ind-1]);
		}
	}

	void AddStack(const ImageStack<ValueType, QType, Dim>& otherStack)
	{
		assert(m_ImageStack.size() == otherStack.m_ImageStack.size());

		for(size_t ind = 0; ind < m_ImageStack.size(); ++ind) {
			AddToImage<ValueType, Dim>(m_ImageStack[ind], otherStack.m_ImageStack[ind]);
		}
	}

  private:
	std::vector<ValueImagePointer> m_ImageStack;
};


/**
 * Representation of a computed point-to-set distance.
 */

template <typename QType, unsigned int Dim, typename ValueType = float>
struct PointToSetDistance
{
	// Source point
	SourcePoint<QType, Dim> m_SourcePoint;

	// Distance
	ValueType m_Distance; // The computed distance
	itk::CovariantVector<ValueType, Dim> m_SpatialGrad; // The computed spatial gradient
	bool m_IsValid;

	inline bool operator<(const PointToSetDistance<QType, Dim> &other) const
	{
		return m_Distance < other.m_Distance;
	}
};

template <typename QType, unsigned int Dim, typename ValueType = float, typename InputValueType = double>
class AlphaSMDInternal
{
  public:
	//  Types

	typedef itk::Vector<ValueType, Dim> Vector;
	typedef itk::CovariantVector<ValueType, Dim> CovariantVector;
	typedef itk::Point<ValueType, Dim> Point;

	//typedef double ValueType; // Maybe template?

	typedef itk::Image<ValueType, Dim> Image;
	typedef typename Image::Pointer ImagePointer;
	typedef itk::Image<InputValueType, Dim> InputImage;
	typedef typename InputImage::Pointer InputImagePointer;

	typedef typename itk::Image<ValueType, Dim>::SizeType SizeType;
	typedef typename itk::Image<ValueType, Dim>::IndexType IndexType;
	typedef typename itk::Image<ValueType, Dim>::SpacingType SpacingType;

	typedef itk::Image<Vector, Dim> VectorImage;
	typedef typename VectorImage::Pointer VectorImagePointer;

	typedef itk::Image<CovariantVector, Dim> CovariantVectorImage;
	typedef typename CovariantVectorImage::Pointer CovariantVectorImagePointer;

	typedef itk::Image<bool, Dim> BinaryImage;
	typedef typename BinaryImage::Pointer BinaryImagePointer;

	typedef itk::Image<unsigned char, Dim> ByteImage;
	typedef typename ByteImage::Pointer ByteImagePointer;

	typedef itk::Image<QType, Dim> QuantizedImage;
	typedef typename QuantizedImage::Pointer QuantizedImagePointer;

	typedef ImageStack<ValueType, QType, Dim> DistanceImageStack;
	typedef ImageStack<CovariantVector, QType, Dim> GradientImageStack;

	//Interpolator types

	typedef itk::InterpolateImageFunction<itk::Image<ValueType, Dim>, double> DistanceInterpolatorBaseType;
	typedef typename DistanceInterpolatorBaseType::Pointer DistanceInterpolatorBasePointer;
	typedef DistanceInterpolatorBaseType* DistanceInterpolatorBaseRawPointer;

	typedef itk::InterpolateImageFunction<itk::Image<CovariantVector, Dim>, double> GradientInterpolatorBaseType;
	typedef typename GradientInterpolatorBaseType::Pointer GradientInterpolatorBasePointer;
	typedef GradientInterpolatorBaseType* GradientInterpolatorBaseRawPointer; 

	typedef itk::NearestNeighborInterpolateImageFunction<itk::Image<ValueType, Dim>, double> DistanceNearestNeighborInterpolatorType;
	typedef typename DistanceNearestNeighborInterpolatorType::Pointer DistanceNearestNeighborInterpolatorPointer;

	typedef itk::NearestNeighborInterpolateImageFunction<itk::Image<CovariantVector, Dim>, double> GradientNearestNeighborInterpolatorType;
	typedef typename GradientNearestNeighborInterpolatorType::Pointer GradientNearestNeighborInterpolatorPointer;

	typedef itk::LinearInterpolateImageFunction<itk::Image<ValueType, Dim>, double> DistanceLinearInterpolatorType;
	typedef typename DistanceLinearInterpolatorType::Pointer DistanceLinearInterpolatorPointer;

	typedef itk::LinearInterpolateImageFunction<itk::Image<CovariantVector, Dim>, double> GradientLinearInterpolatorType;
	typedef typename GradientLinearInterpolatorType::Pointer GradientLinearInterpolatorPointer;

	typedef itk::NearestNeighborInterpolateImageFunction<itk::Image<bool, Dim>, double> MaskInterpolatorType;
	typedef typename MaskInterpolatorType::Pointer MaskInterpolatorPointer;
	typedef MaskInterpolatorType* MaskInterpolatorRawPointer;

	// Parameter setters

	QType GetAlphaLevels() const { return m_AlphaLevels; }

	void SetAlphaLevels(QType alphaLevels) { m_AlphaLevels = alphaLevels; }

	bool GetSquaredMeasure() const { return m_SquaredMeasure; }

	void SetSquaredMeasure(bool squaredMeasureFlag) { m_SquaredMeasure = squaredMeasureFlag; }

	ValueType GetMaxDistance() const { return m_MaxDistance; }

	void SetMaxDistance(ValueType maxDistance) {
		assert(maxDistance >= 0);

		m_MaxDistance = maxDistance;
	}

	bool GetLinearInterpolation() const { return m_LinearInterpolation; }

	void SetLinearInterpolation(bool linearInterpolationFlag) { m_LinearInterpolation = linearInterpolationFlag; }

	bool GetSymmetric() const { return m_Symmetric; }

	void SetSymmetric(bool symmetricFlag) { m_Symmetric = symmetricFlag; }

	// Input setters

	void SetRefImage(InputImagePointer refImage) {
		m_RefImage = refImage;
	}

	void SetFloImage(InputImagePointer floImage) {
		m_FloImage = floImage;
	}

	void SetRefMask(BinaryImagePointer refMask) {
		m_RefMask = refMask;		
	}

	void SetFloMask(BinaryImagePointer floMask) {
		m_FloMask = floMask;
	}

	void SetRefWeight(InputImagePointer refWeight) {
		m_RefWeight = refWeight;
	}

	void SetFloWeight(InputImagePointer floWeight) {
		m_FloWeight = floWeight;
	}

	// Generated products getters

	std::vector<SourcePoint<QType, Dim> > GetRefSourcePoints() const {
		return m_RefSourcePoints;
	}

	std::vector<SourcePoint<QType, Dim> > GetFloSourcePoints() const {
		return m_FloSourcePoints;
	}

	// Query functions for evaluation of the distance stacks

	inline
	ValueType EvalRef(SourcePoint<QType, Dim>& sp, Point transformedPoint, CovariantVector& gradOut, bool& isValid) {
		assert(sp.m_Height >= 0 && sp.m_Height <= m_AlphaLevels);
		assert(m_Symmetric); // This operation only allowed when the measure is in symmetric mode

		return EvalPoint(m_RefDistInterpolators[sp.m_Height], m_RefGradInterpolators[sp.m_Height], m_RefMaskInterpolator, transformedPoint, gradOut, isValid);
	}

	inline
	ValueType EvalFlo(SourcePoint<QType, Dim>& sp, Point transformedPoint, CovariantVector& gradOut, bool& isValid) {
		assert(sp.m_Height >= 0 && sp.m_Height <= m_AlphaLevels);

		return EvalPoint(m_FloDistInterpolators[sp.m_Height], m_FloGradInterpolators[sp.m_Height], m_FloMaskInterpolator, transformedPoint, gradOut, isValid);
	}

	inline
	void EvalRefPointToSetDistance(PointToSetDistance<QType, Dim>& ptsd, Point transformedPoint) {
		ptsd.m_Distance = EvalRef(ptsd.m_SourcePoint, transformedPoint, ptsd.m_SpatialGrad, ptsd.m_IsValid);
	}

	inline
	void EvalFloPointToSetDistance(PointToSetDistance<QType, Dim>& ptsd, Point transformedPoint) {
		ptsd.m_Distance = EvalFlo(ptsd.m_SourcePoint, transformedPoint, ptsd.m_SpatialGrad, ptsd.m_IsValid);
	}

	// Pre-processing, computing the stacks and point-sets - Call before evaluating points and retrieving source points!

	void Update() {
		{ // Preprocess reference image
			DistanceImageStack refCompDistStack;
			GradientImageStack refCompGradStack;
	
			QuantizedImagePointer refQuantImage;
	
			m_RefDistStack.Clear();
			m_RefGradStack.Clear();

			refQuantImage = QuantizeImage<QType, InputValueType, Dim>(m_RefImage, m_AlphaLevels);

			// Only create the dist/grad stacks for the reference image if the measure is set to symmetric mode

			if(m_Symmetric) {
				CreateStack(m_RefDistStack, m_RefGradStack, refQuantImage, m_RefMask);
				CreateStack(refCompDistStack, refCompGradStack, ComplementImage<QType, Dim>(refQuantImage, m_AlphaLevels), m_RefMask);

				// Flip the complement stacks

				refCompDistStack.Reverse();
				refCompGradStack.Reverse();

				// Add the reversed complement stacks to the main stacks
			
				m_RefDistStack.AddStack(refCompDistStack);
				m_RefGradStack.AddStack(refCompGradStack);
			} else {
				m_RefDistStack.Clear();
				m_RefGradStack.Clear();
			}

			// Compute the source point set
			
			m_RefSourcePoints.clear();

			ImageToMaskedSourcePointSet<QType, Dim>(refQuantImage, m_RefMask, m_RefWeight, m_RefSourcePoints);
		}
	
		{ // Proprocess floating image
			DistanceImageStack floCompDistStack;
			GradientImageStack floCompGradStack;

			QuantizedImagePointer floQuantImage;

			m_FloDistStack.Clear();
			m_FloGradStack.Clear();

			floQuantImage = QuantizeImage<QType, InputValueType, Dim>(m_FloImage, m_AlphaLevels);

			CreateStack(m_FloDistStack, m_FloGradStack, floQuantImage, m_FloMask);
			CreateStack(floCompDistStack, floCompGradStack, ComplementImage<QType, Dim>(floQuantImage, m_AlphaLevels), m_FloMask);

			// Flip the complement stacks

			floCompDistStack.Reverse();
			floCompGradStack.Reverse();

			// Add the reversed complement stacks to the main stacks

			m_FloDistStack.AddStack(floCompDistStack);
			m_FloGradStack.AddStack(floCompGradStack);

			// Compute the source point set

			m_FloSourcePoints.clear();

			// Only generate the floating image source point set if the measure is set to symmetric mode

			if(m_Symmetric) {
				ImageToMaskedSourcePointSet<QType, Dim>(floQuantImage, m_FloMask, m_FloWeight, m_FloSourcePoints);
			}
		}

		// Setup interpolators

		m_FloMaskInterpolatorPointer = MaskInterpolatorType::New();
		m_FloMaskInterpolator = m_FloMaskInterpolatorPointer.GetPointer();
		m_FloMaskInterpolator->SetInputImage(m_FloMask.GetPointer());

		if(m_Symmetric) {
			m_RefMaskInterpolatorPointer = MaskInterpolatorType::New();
			m_RefMaskInterpolator = m_RefMaskInterpolatorPointer.GetPointer();
			m_RefMaskInterpolator->SetInputImage(m_RefMask.GetPointer());
		}

		m_FloDistInterpolators.clear();
		m_FloGradInterpolators.clear();

		m_RefDistInterpolators.clear();
		m_RefGradInterpolators.clear();

		m_DistInterpolators.clear();
		m_GradInterpolators.clear();

		if(m_LinearInterpolation) {
			for(size_t i = 0; i < m_FloDistStack.GetLayerCount(); ++i) {
				QType q = static_cast<QType>(i);

				DistanceInterpolatorBasePointer distInterp = static_cast<DistanceInterpolatorBasePointer>(DistanceLinearInterpolatorType::New());
				GradientInterpolatorBasePointer gradInterp = static_cast<GradientInterpolatorBasePointer>(GradientLinearInterpolatorType::New());

				distInterp->SetInputImage(m_FloDistStack.GetLayerPtr(q));
				gradInterp->SetInputImage(m_FloGradStack.GetLayerPtr(q));

				m_DistInterpolators.push_back(distInterp);
				m_GradInterpolators.push_back(gradInterp);

				m_FloDistInterpolators.push_back(distInterp.GetPointer());
				m_FloGradInterpolators.push_back(gradInterp.GetPointer());
			}
			for(size_t i = 0; i < m_RefDistStack.GetLayerCount(); ++i) {
				QType q = static_cast<QType>(i);

				DistanceInterpolatorBasePointer distInterp = static_cast<DistanceInterpolatorBasePointer>(DistanceLinearInterpolatorType::New());
				GradientInterpolatorBasePointer gradInterp = static_cast<GradientInterpolatorBasePointer>(GradientLinearInterpolatorType::New());

				distInterp->SetInputImage(m_RefDistStack.GetLayerPtr(q));
				gradInterp->SetInputImage(m_RefGradStack.GetLayerPtr(q));

				m_DistInterpolators.push_back(distInterp);
				m_GradInterpolators.push_back(gradInterp);

				m_RefDistInterpolators.push_back(distInterp.GetPointer());
				m_RefGradInterpolators.push_back(gradInterp.GetPointer());
			}
		} else {
			for(size_t i = 0; i < m_FloDistStack.GetLayerCount(); ++i) {
				QType q = static_cast<QType>(i);

				DistanceInterpolatorBasePointer distInterp = static_cast<DistanceInterpolatorBasePointer>(DistanceNearestNeighborInterpolatorType::New());
				GradientInterpolatorBasePointer gradInterp = static_cast<GradientInterpolatorBasePointer>(GradientNearestNeighborInterpolatorType::New());

				distInterp->SetInputImage(m_FloDistStack.GetLayerPtr(q));
				gradInterp->SetInputImage(m_FloGradStack.GetLayerPtr(q));

				m_DistInterpolators.push_back(distInterp);
				m_GradInterpolators.push_back(gradInterp);

				m_FloDistInterpolators.push_back(distInterp.GetPointer());
				m_FloGradInterpolators.push_back(gradInterp.GetPointer());
			}
			for(size_t i = 0; i < m_RefDistStack.GetLayerCount(); ++i) {
				QType q = static_cast<QType>(i);

				DistanceInterpolatorBasePointer distInterp = static_cast<DistanceInterpolatorBasePointer>(DistanceNearestNeighborInterpolatorType::New());
				GradientInterpolatorBasePointer gradInterp = static_cast<GradientInterpolatorBasePointer>(GradientNearestNeighborInterpolatorType::New());

				distInterp->SetInputImage(m_RefDistStack.GetLayerPtr(q));
				gradInterp->SetInputImage(m_RefGradStack.GetLayerPtr(q));

				m_DistInterpolators.push_back(distInterp);
				m_GradInterpolators.push_back(gradInterp);

				m_RefDistInterpolators.push_back(distInterp.GetPointer());
				m_RefGradInterpolators.push_back(gradInterp.GetPointer());
			}			
		}
	}
  private:
	void CreateStack(DistanceImageStack& distStack, GradientImageStack& gradStack, QuantizedImagePointer image, BinaryImagePointer mask) {
		CovariantVector zeroGrad;
		zeroGrad.Fill(0);

		QuantizedImagePointer maskedImage = CloneImage<QType, Dim>(image);

		ApplyMask<QType, Dim>(maskedImage, mask, itk::NumericTraits<QType>::ZeroValue(), false);

		QType maxLevel = ImageMaximum<QType, Dim>(maskedImage);

        // Process zero'th layer, which trivially is the full set (and therefore zero-valued)
		
        ImagePointer dt0 = MakeImage<ValueType, Dim>(maskedImage->GetLargestPossibleRegion(), maskedImage->GetSpacing(), maskedImage->GetOrigin(), 0);
                        
        CovariantVectorImagePointer grad0 = MakeImage<CovariantVector, Dim>(dt0->GetLargestPossibleRegion(), dt0->GetSpacing(), maskedImage->GetOrigin(), zeroGrad);
        
        distStack.AddLayer(dt0);
        gradStack.AddLayer(grad0);

        // Process all populated layers, which are not trivially the full set

		const unsigned int startIt = 1U;
        const unsigned int endIt = static_cast<unsigned int>(maxLevel) + 1;

		for(unsigned int qIt = startIt; qIt < endIt; ++qIt) {
			QType q = static_cast<QType>(qIt);

			// Generate the q:th alpha-cut such that all object points height >= q

			BinaryImagePointer bip = ThresholdImage<QType, Dim>(maskedImage, q);
			
			// Compute the distance transform

			ImagePointer dt = DTransform(bip, m_SquaredMeasure);

			// Saturate the distance image to the maximum allowed distance (unless disabled, when m_MaxDistance == 0)

			if(m_MaxDistance > 0.0) {
				SaturateImageWithConstant<ValueType, Dim>(dt, m_MaxDistance);
			}

			// Multiply the layers with their thickness

			MultiplyImageWithConstant<ValueType, Dim>(dt, 1.0 / m_AlphaLevels);

			// Compute the gradient

			CovariantVectorImagePointer grad = GradientImage(dt);

			// Zero out the gradient on object points

			ApplyMask<CovariantVector, Dim>(grad, bip, zeroGrad, true);

			// Add the generated layers to the image stacks

			distStack.AddLayer(dt);
			gradStack.AddLayer(grad);
		}

		// Process empty levels by filling in the maximum distance value, and a zero gradient

		const unsigned int startIt2 = static_cast<unsigned int>(maxLevel) + 1;
        const unsigned int endIt2 = static_cast<unsigned int>(m_AlphaLevels) + 1;

		for(unsigned int qIt = startIt2; qIt < endIt2; ++qIt) {
			ImagePointer dt = MakeImage<ValueType, Dim>(maskedImage->GetLargestPossibleRegion(), maskedImage->GetSpacing(), maskedImage->GetOrigin(), m_MaxDistance / m_AlphaLevels);

			CovariantVectorImagePointer grad = MakeImage<CovariantVector, Dim>(dt->GetLargestPossibleRegion(), dt->GetSpacing(), maskedImage->GetOrigin(), zeroGrad);

			distStack.AddLayer(dt);
			gradStack.AddLayer(grad);
		}

		// Compute the cumulative distances (integrating from low to high alpha-cuts)

		distStack.SumStack();
		gradStack.SumStack();
	}

	// Compute the distance transform of 'binaryImage'.
	// IMPORTANT: Assumes that the set of object pixels is not empty.
	ImagePointer DTransform(
		BinaryImagePointer binaryImage,
		bool squaredMeasure = false)
	{
#define USE_MAURER

#ifdef USE_MAURER
		typedef itk::SignedMaurerDistanceMapImageFilter<
			BinaryImage, Image>
			DTType;
#else
		// TODO: Change to Maurer
		typedef itk::DanielssonDistanceMapImageFilter<
			BinaryImage, Image>
			DTType;
#endif
		typename DTType::Pointer dtFilter = DTType::New();

		dtFilter->SetUseImageSpacing(true);

		dtFilter->SetSquaredDistance(squaredMeasure);

#ifdef USE_MAURER
		dtFilter->InsideIsPositiveOff();
#else
		dtFilter->SetInputIsBinary(true);
#endif

		dtFilter->SetInput(binaryImage);

		dtFilter->Update();

		typename Image::Pointer output = dtFilter->GetOutput();
/*
		typedef itk::ChangeInformationImageFilter<Image> ChangeInformationFilterType;
		typedef typename ChangeInformationFilterType::Pointer ChangeInformationFilterPointer;

		ChangeInformationFilterPointer changeInformationFilter = ChangeInformationFilterType::New();

		changeInformationFilter->SetInput(output);
		changeInformationFilter->SetReferenceImage(refImage);
		changeInformationFilter->SetUseReferenceImage(true);

		changeInformationFilter->Update();

		output = changeInformationFilter->GetOutput();
*/
#ifdef USE_MAURER
		SaturateImageWithConstant<ValueType, Dim>(
			output, itk::NumericTraits<ValueType>::ZeroValue(), false);
#endif
		
		return output;
	}

	CovariantVectorImagePointer GradientImage(
		ImagePointer image)
	{

		typedef itk::GradientImageFilter<Image, ValueType, ValueType> GradientImageFilterType;
		typedef typename GradientImageFilterType::Pointer GradientImageFilterPointer;

		GradientImageFilterPointer gradFilter = GradientImageFilterType::New();

		gradFilter->SetInput(image);
		gradFilter->SetUseImageSpacing(true);
		gradFilter->SetUseImageDirection(true);

		gradFilter->Update();

		return gradFilter->GetOutput();
	}

	ValueType EvalPoint(
		DistanceInterpolatorBaseRawPointer distImage,
		GradientInterpolatorBaseRawPointer gradImage,
		MaskInterpolatorRawPointer maskImage,
		Point& transformedPoint,
		CovariantVector& gradOut,
		bool& isValid) {
	
		if(gradImage->IsInsideBuffer(transformedPoint)) {
			//assert(distImage->IsInsideBuffer(transformedPoint));
			//assert(maskImage->IsInsideBuffer(transformedPoint));
			/*if(!maskImage->IsInsideBuffer(transformedPoint))
			{
				std::cout << "Out side of mask image..." << std::endl;
				std::cout << (*maskImage) << std::endl;
				std::cout << (*gradImage) << std::endl;
				std::cout << *maskImage->GetInputImage() << std::endl;
				std::cout << *gradImage->GetInputImage() << std::endl;
				std::cout << (transformedPoint) << std::endl;
				std::exit(-1);
			}*/

			if(maskImage->Evaluate(transformedPoint)) {
				isValid = true;
				gradOut = gradImage->Evaluate(transformedPoint);
				return distImage->Evaluate(transformedPoint);
			} else {
				isValid = false;
				return m_MaxDistance;
			}
		} else {
			isValid = false;
			return m_MaxDistance;
		}
	}

	// --- Data ---

	// Generated data

	DistanceImageStack m_RefDistStack;
	DistanceImageStack m_FloDistStack;

	GradientImageStack m_RefGradStack;
	GradientImageStack m_FloGradStack;

	std::vector<DistanceInterpolatorBasePointer> m_DistInterpolators;
	std::vector<GradientInterpolatorBasePointer> m_GradInterpolators;

	std::vector<DistanceInterpolatorBaseRawPointer> m_RefDistInterpolators;
	std::vector<GradientInterpolatorBaseRawPointer> m_RefGradInterpolators;
	std::vector<DistanceInterpolatorBaseRawPointer> m_FloDistInterpolators;
	std::vector<GradientInterpolatorBaseRawPointer> m_FloGradInterpolators;

	std::vector<SourcePoint<QType, Dim> > m_RefSourcePoints;
	std::vector<SourcePoint<QType, Dim> > m_FloSourcePoints;

	MaskInterpolatorPointer m_FloMaskInterpolatorPointer;
	MaskInterpolatorPointer m_RefMaskInterpolatorPointer;

	MaskInterpolatorRawPointer m_FloMaskInterpolator;
	MaskInterpolatorRawPointer m_RefMaskInterpolator;

	// Settings

	QType m_AlphaLevels;
	ValueType m_MaxDistance;
	bool m_SquaredMeasure;
	bool m_LinearInterpolation;
	bool m_Symmetric;

	// Input

	InputImagePointer m_RefImage;
	InputImagePointer m_FloImage;

	BinaryImagePointer m_RefMask;
	BinaryImagePointer m_FloMask;

	InputImagePointer m_RefWeight;
	InputImagePointer m_FloWeight;
}; // class AlphaSMDInternal
} // namespace alphasmdinternal
} // namespace itk

#endif //ALPHASMD_METRIC_INTERNAL_H
