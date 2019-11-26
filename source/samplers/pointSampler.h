
//
// A set of point samplers which enables sampling from the pixels
// in an image, with masks and auxilliary weight-maps.
//
// Author: Johan Ofverstedt
//

#ifndef POINT_SAMPLER_H
#define POINT_SAMPLER_H

#include "itkImage.h"
#include "itkImageRegionConstIterator.h"
#include "itkMersenneTwisterRandomVariateGenerator.h"
#include "itkGradientMagnitudeImageFilter.h"
#include "itkDiscreteGaussianImageFilter.h"
#include "quasiRandomGenerator.h"

template <typename ImageType, typename WeightImageType=ImageType>
struct PointSample {
    typedef typename ImageType::PointType PointType;
    typedef typename ImageType::ValueType ValueType;
    typedef typename WeightImageType::ValueType WeightValueType;

    PointType m_Point;
    ValueType m_Value;
    WeightValueType m_Weight;
};

template <typename ImageType, typename MaskImageType, typename WeightImageType=ImageType>
class PointSamplerBase : public itk::Object {
public:
    using Self = PointSamplerBase;
    using Superclass = itk::Object;
    using Pointer = itk::SmartPointer<Self>;
    using ConstPointer = itk::SmartPointer<const Self>;
    
    typedef typename ImageType::Pointer ImagePointer;
    typedef typename MaskImageType::Pointer MaskImagePointer;
    typedef typename WeightImageType::Pointer WeightImagePointer;

    typedef typename ImageType::ValueType ValueType;
    typedef typename MaskImageType::ValueType MaskValueType;
    typedef typename WeightImageType::ValueType WeightValueType;

    typedef typename ImageType::SpacingType SpacingType;
    typedef typename ImageType::RegionType RegionType;
    typedef typename ImageType::IndexType IndexType;
    typedef typename ImageType::SizeType SizeType;
    typedef typename ImageType::IndexValueType IndexValueType;
    typedef typename ImageType::PointType PointType;

    typedef itk::Statistics::MersenneTwisterRandomVariateGenerator GeneratorType;
    typedef typename GeneratorType::Pointer GeneratorPointer;
    
    typedef PointSample<ImageType, WeightImageType> PointSampleType;
  
    // Leave out the new macro since this is an abstract base class

    itkTypeMacro(PointSamplerBase, itk::Object);

    virtual void SetImage(ImagePointer image) {
        m_Image = image;
        m_ImageRawPtr = image.GetPointer();
    }

    virtual void SetMaskImage(MaskImagePointer mask) {
        m_Mask = mask;
        m_MaskRawPtr = mask.GetPointer();
    }

    virtual void SetWeightImage(WeightImagePointer weights) {
        m_Weights = weights;
        m_WeightsRawPtr = weights.GetPointer();
    }

    virtual unsigned long long GetSeed() const {
        return m_Seed;
    }

    virtual void SetSeed(unsigned long long seed) {
        m_Seed = seed;
    }

    virtual void RestartFromSeed() {
        m_DitheringGenerator->SetSeed(m_Seed);
    }

    virtual bool IsDitheringOn() const { return m_Dithering; }

    virtual void SetDitheringOff() { m_Dithering = false; }

    virtual void SetDitheringOn() { m_Dithering = true; }

    virtual void Initialize() {
        m_Spacing = m_Image->GetSpacing();

        ComputeMaskBoundingBox();
    };

    // Abstract function for computing a random index given an origin and size
    // of a region of interest.
    virtual IndexType ComputeIndex(IndexType origin, SizeType size) = 0;

    virtual void Sample(PointSampleType& pointSampleOut, unsigned int attempts = 1)
    {
        ImageType* image = m_ImageRawPtr;
        MaskImageType* mask = m_MaskRawPtr;
        WeightImageType* weights = m_WeightsRawPtr;

        IndexType index = ComputeIndex(m_BBOrigin, m_BBSize);

        bool isMasked = PerformMaskTest(index);
        if(!isMasked) {
            if(attempts > 0) {
                Sample(pointSampleOut, attempts - 1U);
                return;
            }

            pointSampleOut.m_Weight = itk::NumericTraits<WeightValueType>::ZeroValue();
            return;
        }

        pointSampleOut.m_Value = m_ImageRawPtr->GetPixel(index);

        image->TransformIndexToPhysicalPoint(index, pointSampleOut.m_Point);

        if(weights)
        {
            pointSampleOut.m_Weight = weights->GetPixel(index);
        }
        else
        {
            pointSampleOut.m_Weight = itk::NumericTraits<WeightValueType>::OneValue();
        }

        DitherPoint(pointSampleOut.m_Point);
    }

    virtual void Sample(std::vector<PointSampleType>& pointSampleOut, unsigned int count, unsigned int attempts=1) {
        for(unsigned int i = 0; i < count; ++i) {
            PointSampleType pnt;
            Sample(pnt, attempts);
            pointSampleOut.push_back(pnt);
        }
    }
protected:
    PointSamplerBase() {
        m_Seed = 42U;
        m_ImageRawPtr = nullptr;
        m_MaskRawPtr = nullptr;
        m_WeightsRawPtr = nullptr;
        m_DitheringGenerator = GeneratorType::New();
        m_DitheringGenerator->SetSeed(m_Seed);
        m_Dithering = false;
    }
    virtual ~PointSamplerBase() {

    }

    void DitherPoint(PointType& point) {
        if(m_Dithering) {
            GeneratorType* gen = m_DitheringGenerator.GetPointer();

            for(unsigned int i = 0; i < ImageType::ImageDimension; ++i) {
                point[i] = point[i] + (gen->GetVariateWithClosedRange()-0.5)*m_Spacing[i];
            }
        }
    }

    bool PerformMaskTest(IndexType index) {
        if(m_MaskRawPtr) {
            // Assume here that the index is inside the buffer

            return m_MaskRawPtr->GetPixel(index);
        }
        return true;
    }

    // If there is a mask, compute the bounding box of the pixels
    // inside the mask
    void ComputeMaskBoundingBox() {
        if(m_MaskRawPtr) {
            IndexType minIndex;
            IndexType maxIndex;
            minIndex.Fill(itk::NumericTraits<IndexValueType>::max());
            maxIndex.Fill(itk::NumericTraits<IndexValueType>::min());

            typedef itk::ImageRegionConstIterator<MaskImageType> IteratorType;
            IteratorType it(m_MaskRawPtr, m_MaskRawPtr->GetLargestPossibleRegion());

            it.GoToBegin();
            while(!it.IsAtEnd()) {
                if(it.Value()) {
                    IndexType curIndex = it.GetIndex();
                    for(unsigned int i = 0; i < ImageType::ImageDimension; ++i) {
                        if(curIndex[i] < minIndex[i])
                            minIndex[i] = curIndex[i];
                        if(curIndex[i] > maxIndex[i])
                            maxIndex[i] = curIndex[i];
                    }
                }

                ++it;
            }

            m_BBOrigin = minIndex;
            for(unsigned int i = 0; i < ImageType::ImageDimension; ++i) {
                m_BBSize[i] = maxIndex[i]-minIndex[i];
            }
        } else {
            // No mask, just use the bounds of the whole image
            RegionType region = m_ImageRawPtr->GetLargestPossibleRegion();
            m_BBOrigin = region.GetIndex();
            m_BBSize = region.GetSize();
        }
    }

    // Attributes
    ImagePointer m_Image;
    MaskImagePointer m_Mask;
    WeightImagePointer m_Weights;

    ImageType* m_ImageRawPtr;
    MaskImageType* m_MaskRawPtr;
    WeightImageType* m_WeightsRawPtr;

    mutable GeneratorPointer m_DitheringGenerator;

    unsigned long long m_Seed;
    
    SpacingType m_Spacing;
    bool m_Dithering;
    IndexType m_BBOrigin;
    SizeType m_BBSize;
}; // End of class PointSamplerBase

// Uniform point sampler

template <typename ImageType, typename MaskImageType, typename WeightImageType=ImageType>
class UniformPointSampler : public PointSamplerBase<ImageType, MaskImageType, WeightImageType> {
public:
    using Self = UniformPointSampler;
    using Superclass = PointSamplerBase<ImageType, MaskImageType, WeightImageType>;
    using Pointer = itk::SmartPointer<Self>;
    using ConstPointer = itk::SmartPointer<const Self>;
    
    typedef typename ImageType::Pointer ImagePointer;
    typedef typename MaskImageType::Pointer MaskImagePointer;
    typedef typename WeightImageType::Pointer WeightImagePointer;

    typedef typename ImageType::ValueType ValueType;
    typedef typename MaskImageType::ValueType MaskValueType;
    typedef typename WeightImageType::ValueType WeightValueType;

    typedef typename ImageType::SpacingType SpacingType;
    typedef typename ImageType::RegionType RegionType;
    typedef typename ImageType::IndexType IndexType;
    typedef typename ImageType::SizeType SizeType;

    typedef itk::Statistics::MersenneTwisterRandomVariateGenerator GeneratorType;
    typedef typename GeneratorType::Pointer GeneratorPointer;
    
    typedef PointSample<ImageType, WeightImageType> PointSampleType;

    itkNewMacro(Self);
  
    itkTypeMacro(UniformPointSampler, PointSamplerBase);

    virtual void RestartFromSeed() {
        Superclass::RestartFromSeed();

        // Use a different seed from the dithering seed using an arbitrary factor and offset
        m_RNG->SetSeed(Superclass::GetSeed()*2U+13U);
    }

    virtual IndexType ComputeIndex(IndexType origin, SizeType size)
    {
        GeneratorType* gen = m_RNG.GetPointer();

        IndexType index;
        for(unsigned int i = 0; i < ImageType::ImageDimension; ++i) {
            if(size[i] > 0U)
            {
                unsigned int step = (gen->GetIntegerVariate() % size[i]);
                index[i] = origin[i] + step;
            }
            else
                index[i] = origin[i];
        }
        return index;
    }
protected:
    UniformPointSampler() {
        m_RNG = GeneratorType::New();
        m_RNG->SetSeed(Superclass::GetSeed()*2U+13U);
    }
    GeneratorPointer m_RNG;
}; // End of class UniformPointSampler

// Quasi random point sampler

template <typename ImageType, typename MaskImageType, typename WeightImageType=ImageType>
class QuasiRandomPointSampler : public PointSamplerBase<ImageType, MaskImageType, WeightImageType> {
public:
    using Self = QuasiRandomPointSampler;
    using Superclass = PointSamplerBase<ImageType, MaskImageType, WeightImageType>;
    using Pointer = itk::SmartPointer<Self>;
    using ConstPointer = itk::SmartPointer<const Self>;
    
    typedef typename ImageType::Pointer ImagePointer;
    typedef typename MaskImageType::Pointer MaskImagePointer;
    typedef typename WeightImageType::Pointer WeightImagePointer;

    typedef typename ImageType::ValueType ValueType;
    typedef typename MaskImageType::ValueType MaskValueType;
    typedef typename WeightImageType::ValueType WeightValueType;

    typedef typename ImageType::SpacingType SpacingType;
    typedef typename ImageType::RegionType RegionType;
    typedef typename ImageType::IndexType IndexType;
    typedef typename ImageType::SizeType SizeType;
    typedef typename ImageType::IndexValueType IndexValueType;

    typedef itk::Statistics::MersenneTwisterRandomVariateGenerator GeneratorType;
    typedef typename GeneratorType::Pointer GeneratorPointer;
    
    typedef QuasiRandomGenerator<ImageType::ImageDimension> QRGeneratorType;
    typedef typename QRGeneratorType::Pointer QRGeneratorPointer;

    typedef PointSample<ImageType, WeightImageType> PointSampleType;

    itkNewMacro(Self);
  
    itkTypeMacro(PointSamplerBase, itk::Object);

    virtual void RestartFromSeed() {
        Superclass::RestartFromSeed();

        // Use a different seed from the dithering seed using an arbitrary factor and offset
        m_QRGenerator->SetSeed(Superclass::GetSeed()*2U+13U);
        m_QRGenerator->Restart();
    }

    virtual IndexType ComputeIndex(IndexType origin, SizeType size)
    {
        QRGeneratorType* gen = m_QRGenerator.GetPointer();

        IndexType index;

        itk::FixedArray<double, ImageType::ImageDimension> v = gen->GetVariate();
        for(unsigned int i = 0; i < ImageType::ImageDimension; ++i) {
            index[i] = origin[i] + (IndexValueType)(v[i] * size[i]);
        }

        return index;
    }
protected:
    QuasiRandomPointSampler() {
        m_QRGenerator = QRGeneratorType::New();
        m_QRGenerator->SetSeed(2U*Superclass::GetSeed() + 13U);
    }

    mutable QRGeneratorPointer m_QRGenerator;
}; // End of class QuasiRandomPointSampler

// Gradient-importance weighted random point sampler
// Method discussed: A Scalable Asynchronous Distributed Algorithm for Topic Modeling, Hsiang-Fu Yu et. al, 2014.
template <typename ImageType, typename MaskImageType, typename WeightImageType=ImageType>
class GradientWeightedPointSampler : public PointSamplerBase<ImageType, MaskImageType, WeightImageType> {
public:
    using Self = GradientWeightedPointSampler;
    using Superclass = PointSamplerBase<ImageType, MaskImageType, WeightImageType>;
    using Pointer = itk::SmartPointer<Self>;
    using ConstPointer = itk::SmartPointer<const Self>;
    
    typedef typename ImageType::Pointer ImagePointer;
    typedef typename MaskImageType::Pointer MaskImagePointer;
    typedef typename WeightImageType::Pointer WeightImagePointer;

    typedef typename ImageType::ValueType ValueType;
    typedef typename MaskImageType::ValueType MaskValueType;
    typedef typename WeightImageType::ValueType WeightValueType;

    typedef typename ImageType::SpacingType SpacingType;
    typedef typename ImageType::RegionType RegionType;
    typedef typename ImageType::IndexType IndexType;
    typedef typename ImageType::SizeType SizeType;
    typedef typename ImageType::IndexValueType IndexValueType;

    typedef itk::Statistics::MersenneTwisterRandomVariateGenerator GeneratorType;
    typedef typename GeneratorType::Pointer GeneratorPointer;
    
    typedef PointSample<ImageType, WeightImageType> PointSampleType;

    itkNewMacro(Self);
  
    itkTypeMacro(GradientWeightedPointSampler, PointSamplerBase);

    virtual void RestartFromSeed() {
        Superclass::RestartFromSeed();

        // Use a different seed from the dithering seed using an arbitrary factor and offset
        m_RNG->SetSeed(Superclass::GetSeed()*2U+13U);
    }

    virtual double GetSigma() const
    {
        return m_Sigma;
    }

    virtual void SetSigma(double sigma) {
        m_Sigma = sigma;
    }

    virtual ValueType GetTolerance() const
    {
        return m_Tolerance;
    }

    virtual void SetTolerance(ValueType tol) {
        m_Tolerance = tol;
    }

    virtual void Initialize() {
        Superclass::Initialize();

        ImagePointer im = Superclass::m_Image;

        if(m_Sigma > 0.0) {
            typedef itk::DiscreteGaussianImageFilter<
                ImageType, ImageType>
                GaussianFilterType;

            // Create and setup a Gaussian filter
            typename GaussianFilterType::Pointer gaussianFilter = GaussianFilterType::New();
            gaussianFilter->SetInput(im);
	        gaussianFilter->SetUseImageSpacingOn();
            gaussianFilter->SetMaximumKernelWidth(128);
            gaussianFilter->SetVariance(m_Sigma * m_Sigma);
            gaussianFilter->Update();

            im = gaussianFilter->GetOutput();
        }

        // Compute the gradient magnitude image and generate a sparse list of cumulative probabilities
        //typedef itk::GradientMagnitudeRecursiveGaussianImageFilter<ImageType, ImageType> FilterType;
        typedef itk::GradientMagnitudeImageFilter<ImageType, ImageType> FilterType;

        typename FilterType::Pointer filter = FilterType::New();

        filter->SetInput(im);
        
        filter->Update();

        ImagePointer gradIm = filter->GetOutput();

        typedef itk::ImageRegionConstIterator<ImageType> IteratorType;
        typedef itk::ImageRegionConstIterator<MaskImageType> MaskIteratorType;

        m_Prob.clear();
        m_Indices.clear();

        const double tolerance = m_Tolerance;
        double totalValue = 0.0;

        if(Superclass::m_MaskRawPtr) {
        IteratorType it(gradIm, gradIm->GetLargestPossibleRegion());
        MaskIteratorType itMask(Superclass::m_MaskRawPtr, Superclass::m_MaskRawPtr->GetLargestPossibleRegion());

        it.GoToBegin();
        itMask.GoToBegin();
        
        while(!it.IsAtEnd() && !itMask.IsAtEnd())
        {
            if(itMask.Value())
            {
                double value = it.Value();

                if(value > tolerance)
                {
                    IndexType curIndex = it.GetIndex();
                    totalValue += value;

                    m_Prob.push_back(totalValue);
                    m_Indices.push_back(curIndex);
                }
            }
            ++it;
            ++itMask;
        }
        }
        else
        {
        IteratorType it(gradIm, gradIm->GetLargestPossibleRegion());

        it.GoToBegin();

        while(!it.IsAtEnd())
        {
            double value = it.Value();

            if(value > tolerance)
            {
                IndexType curIndex = it.GetIndex();
                totalValue += value;
                m_Prob.push_back(totalValue);
                m_Indices.push_back(curIndex);
            }
            ++it;
        }
        }

        if(totalValue < tolerance)
        {
            totalValue = tolerance;
        }
        for(size_t i = 0; i < m_Prob.size(); ++i)
        {
            m_Prob[i] /= totalValue;
        }
    }

    virtual IndexType ComputeIndex(IndexType origin, SizeType size)
    {
        IndexType index;
        GeneratorType* gen = m_RNG.GetPointer();

        if(m_Prob.size() > 0U)
        {
            ValueType p = static_cast<ValueType>(gen->GetVariateWithOpenRange());
            size_t sind = SearchCumProb(p);
            assert(sind < m_Prob.size());
            index = m_Indices[sind];
        }
        else
        {
            // In the case where the image is completely uniform (inside the mask)
            // we revert back to uniform random sampling (instead of failing with an error)

            for(unsigned int i = 0; i < ImageType::ImageDimension; ++i)
            {
                if(size[i] > 0U)
                {
                    unsigned int step = (gen->GetIntegerVariate() % size[i]);
                    index[i] = origin[i] + step;
                }
                else
                    index[i] = origin[i];
            }
        }

        return index;
    }
protected:
    GradientWeightedPointSampler() {
        m_RNG = GeneratorType::New();
        m_RNG->SetSeed(Superclass::GetSeed()*2U+13U);

        m_Sigma = 0.0;
        m_Tolerance = static_cast<ValueType>(1e-4);
    }

    size_t SearchCumProb(ValueType p) {
        ValueType* arr = m_Prob.data();

        // Use binary search to find the point with the minimal cumulative
        // probability greater than the random value 'p':
        // [0.2, 0.5, 0.8, 1.0]
        // SearchCumProb(p=0.6)
        // should give index 2 (corresponding to the 0.8 cumulative probability)
        if(m_Prob.size() > 0) {
            size_t s = 0;
            size_t e = m_Prob.size();

            while(s < e)
            {
                size_t m = s + (e-s) / 2U;
                ValueType pr = arr[m];

                if(p > pr)
                {
                    s = m + 1;
                }
                else
                {
                    e = m;
                }
            }

            return e;
        }
        return 0;
    }
    GeneratorPointer m_RNG;

    std::vector<ValueType> m_Prob;
    std::vector<IndexType> m_Indices;
    double m_Sigma;
    ValueType m_Tolerance;
}; // End of class UniformPointSampler

//
// Hybrid point sampler
// A collection of hybrid point samplers chosen at random
// for each point
//
template <typename ImageType, typename MaskImageType, typename WeightImageType=ImageType>
class HybridPointSampler : public PointSamplerBase<ImageType, MaskImageType, WeightImageType> {
public:
    using Self = HybridPointSampler;
    using Superclass = PointSamplerBase<ImageType, MaskImageType, WeightImageType>;
    using Pointer = itk::SmartPointer<Self>;
    using ConstPointer = itk::SmartPointer<const Self>;
    
    using SuperclassPointer = itk::SmartPointer<Superclass>;

    typedef typename ImageType::Pointer ImagePointer;
    typedef typename MaskImageType::Pointer MaskImagePointer;
    typedef typename WeightImageType::Pointer WeightImagePointer;

    typedef typename ImageType::ValueType ValueType;
    typedef typename MaskImageType::ValueType MaskValueType;
    typedef typename WeightImageType::ValueType WeightValueType;

    typedef typename ImageType::SpacingType SpacingType;
    typedef typename ImageType::RegionType RegionType;
    typedef typename ImageType::IndexType IndexType;
    typedef typename ImageType::SizeType SizeType;
    typedef typename ImageType::IndexValueType IndexValueType;

    typedef itk::Statistics::MersenneTwisterRandomVariateGenerator GeneratorType;
    typedef typename GeneratorType::Pointer GeneratorPointer;
    
    typedef PointSample<ImageType, WeightImageType> PointSampleType;

    itkNewMacro(Self);
  
    itkTypeMacro(HybridPointSampler, PointSamplerBase);

    virtual void Initialize() {
        for(size_t i = 0; i < m_Samplers.size(); ++i) {
            m_Samplers[i]->Initialize();
        }
    }

    virtual void AddSampler(SuperclassPointer sampler, double weight=1.0) {
        m_Samplers.push_back(sampler);
        m_TotalWeight += weight;
        m_SamplerWeights.push_back(m_TotalWeight);
    }

    virtual void SetSeed(unsigned int seed) {
        Superclass::SetSeed(seed);

        for(size_t i = 0; i < m_Samplers.size(); ++i) {
            m_Samplers[i]->SetSeed(seed + i * 17U);
        }
        m_RNG->SetSeed(seed * 2U + 19U);
    }
    virtual void RestartFromSeed() {
        for(size_t i = 0; i < m_Samplers.size(); ++i) {
            m_Samplers[i]->RestartFromSeed();
        }
        m_RNG->SetSeed(Superclass::GetSeed() * 2U + 19U);
    }

    virtual IndexType ComputeIndex(IndexType origin, SizeType size)
    {
        if(m_Samplers.size() == 0)
        {
            IndexType index;
            index.Fill(itk::NumericTraits<IndexValueType>::ZeroValue());
            return index;
        }

        double p = m_RNG->GetVariateWithOpenRange() * m_TotalWeight;
        size_t ind = 0;
        for(size_t i = 0; i < m_Samplers; ++i) {
            if(p < m_SamplerWeights[i]) {
                ind = i;
                break;
            }
        }
        
        return m_Samplers[ind]->ComputeIndex(origin, size);
    }

    virtual void Sample(PointSampleType& pointSampleOut, unsigned int attempts = 1)
    {
        if(m_Samplers.size() == 0)
        {
            pointSampleOut.m_Weight = 0.0;
            return;
        }

        double p = m_RNG->GetVariateWithOpenRange() * m_TotalWeight;
        size_t ind = 0;
        for(size_t i = 0; i < m_Samplers; ++i) {
            if(p < m_SamplerWeights[i]) {
                ind = i;
                break;
            }
        }
        
        m_Samplers[ind]->Sample(pointSampleOut, attempts); 
    }
protected:
    HybridPointSampler() {
        m_RNG = GeneratorType::New();
        m_RNG->SetSeed(Superclass::GetSeed() * 2U + 19U);
    }

    GeneratorPointer m_RNG;

    std::vector<SuperclassPointer> m_Samplers;
    std::vector<double> m_SamplerWeights;
    double m_TotalWeight;
}; // End of class HybridPointSampler

#endif
