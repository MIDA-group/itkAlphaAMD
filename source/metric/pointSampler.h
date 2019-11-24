
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
#include "itkGradientMagnitudeRecursiveGaussianImageFilter.h"
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
    using Pointer = SmartPointer<Self>;
    using ConstPointer = SmartPointer<const Self>;
    
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
  
    itkTypeMacro(PointSamplerBase, itk::Object);

    virtual void SetImage(ImagePointer image) {
        m_Image = image;
        m_ImageRawPtr = image.GetPointer();
    }

    virtual void SetMaskImage(MaskImagePointer mask) {
        m_Mask = mask;
        m_MaskRawPtr = mask.GetPointer();
    }

    virtual void SetWeightImage(WeightImageType weights) {
        m_Weights = weights;
        m_WeightsRawPtr = weights.GetPointer();
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

    virtual void Sample(PointSampleType& pointSampleOut, unsigned int attempts = 1) { assert(false); }

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
        m_DitheringGenerator = GeneratorType::New();
        m_DitheringGenerator->SetSeed(m_Seed);        
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
            maxIndex.Fill(itk::NumericTraits<IndexValueType>::min())

            typename itk::ImageRegionConstIterator<MaskImageType> IteratorType;
            IteratorType it(m_MaskRawPtr->GetLargestPossibleRegion(), m_MaskRawPtr);

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
    MaskImage* m_MaskRawPtr;
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
    using Pointer = SmartPointer<Self>;
    using ConstPointer = SmartPointer<const Self>;
    
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
        PointSamplerBase::RestartFromSeed();

        // Use a different seed from the dithering seed using an arbitrary factor and offset
        m_RNG->SetSeed(m_Seed*2U+13U);
    }

    virtual void Sample(PointSampleType& pointSampleOut, unsigned int attempts = 1)
    {
        IndexType index;

        GeneratorType* gen = m_RNG.GetPointer();
        ImageType* image = m_ImageRawPtr;
        MaskImageType* mask = m_MaskRawPtr;
        WeightImageType* weights = m_WeightsRawPtr;

        IndexType origin = m_BBOrigin;
        SizeType size = m_BBSize;

        for(unsigned int i = 0; i < ImageType::ImageDimension; ++i) {
            index[i] = origin[i] + gen->GetIntegerVariate(size[i]) - 1;
        }

        bool isMasked = PerformMaskTest(index);
        if(!isMasked) {
            if(attempts > 0) {
                Sample(pointSampleOut, attempts - 1U);
                return;
            }

            pointSampleOut.m_Weight = itk::NumericTraits<WeightValueType>::Zero();
            return;
        }

        image->TransformIndexToPhysicalPoint(index, pointSampleOut.m_Point);

        if(weights)
        {
            pointSampleOut.m_Weight = weights->GetPixel(index);
        }
        else
        {
            pointSampleOut.m_Weight = itk::NumericTraits<WeightValueType>::One();
        }

        DitherPoint(pointSampleOut.m_Point);
    }
protected:
    UniformPointSampler() {
        m_RNG = GeneratorType::New();
        m_RNG->SetSeed(m_Seed*2U+13U);
    }
    GeneratorPointer m_RNG;
}; // End of class UniformPointSampler

// Quasi random point sampler

template <typename ImageType, typename MaskImageType, typename WeightImageType=ImageType>
class QuasiRandomPointSampler : public PointSamplerBase<ImageType, MaskImageType, WeightImageType> {
public:
    using Self = QuasiRandomPointSampler;
    using Superclass = PointSamplerBase<ImageType, MaskImageType, WeightImageType>;
    using Pointer = SmartPointer<Self>;
    using ConstPointer = SmartPointer<const Self>;
    
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
    
    typedef QuasiRandomGenerator<ImageType::ImageDimension> QRGeneratorType;
    typedef QRGeneratorType::Pointer QRGeneratorPointer;

    typedef PointSample<ImageType, WeightImageType> PointSampleType;

    itkNewMacro(Self);
  
    itkTypeMacro(PointSamplerBase, itk::Object);

    virtual void RestartFromSeed() {
        PointSamplerBase::RestartFromSeed();

        // Use a different seed from the dithering seed using an arbitrary factor and offset
        m_QRGenerator->SetSeed(m_Seed*2U+13U);
        m_QRGenerator->Restart();
    }

    virtual void Sample(PointSampleType& pointSampleOut, unsigned int attempts = 1)
    {
        IndexType index;

        QRGeneratorType* gen = m_QRGenerator.GetPointer();
        ImageType* image = m_ImageRawPtr;
        MaskImageType* mask = m_MaskRawPtr;
        WeightImageType* weights = m_WeightsRawPtr;

        IndexType origin = m_BBOrigin;
        SizeType size = m_BBSize;

        itk::Vector<double, ImageType::ImageDimension> v = gen->GetVariate();
        for(unsigned int i = 0; i < ImageType::ImageDimension; ++i) {
            index[i] = origin[i] + (IndexValueType)(v[i] * size[i]);
        }

        bool isMasked = PerformMaskTest(index);
        if(!isMasked) {
            if(attempts > 0) {
                Sample(pointSampleOut, attempts - 1U);
                return;
            }

            pointSampleOut.m_Weight = itk::NumericTraits<WeightValueType>::Zero();
            return;
        }

        image->TransformIndexToPhysicalPoint(index, pointSampleOut.m_Point);

        if(weights)
        {
            pointSampleOut.m_Weight = weights->GetPixel(index);
        }
        else
        {
            pointSampleOut.m_Weight = itk::NumericTraits<WeightValueType>::One();
        }

        DitherPoint(pointSampleOut.m_Point);
    }
protected:
    QuasiRandomPointSampler() {
        m_QRGenerator = QRGeneratorType::New();
        m_QRGenerator->SetSeed(2U*m_Seed + 13U);
    }

    mutable QRGeneratorPointer m_QRGenerator;
}; // End of class QuasiRandomPointSampler

// Gradient-importance weighted random point sampler

template <typename ImageType, typename MaskImageType, typename WeightImageType=ImageType>
class GradientWeightedPointSampler : public PointSamplerBase<ImageType, MaskImageType, WeightImageType> {
public:
    using Self = GradientWeightedPointSampler;
    using Superclass = PointSamplerBase<ImageType, MaskImageType, WeightImageType>;
    using Pointer = SmartPointer<Self>;
    using ConstPointer = SmartPointer<const Self>;
    
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
  
    itkTypeMacro(GradientWeightedPointSampler, PointSamplerBase);

    virtual void RestartFromSeed() {
        PointSamplerBase::RestartFromSeed();

        // Use a different seed from the dithering seed using an arbitrary factor and offset
        m_RNG->SetSeed(m_Seed*2U+13U);
    }

    virtual void SetSigma(double sigma) {
        m_Sigma = sigma;
    }

    virtual void Initialize() {
        PointSamplerBase::Initialize();

        // Compute the gradient magnitude image and generate a sparse list of cumulative probabilities
        typedef itk::GradientMagnitudeRecursiveGaussianImageFilter<ImageType, ImageType> FilterType;

        typename FilterType::Pointer filter = FilterType::New();
        filter->SetSigma(m_Sigma);

        filter->SetInput(m_Image);
        
        filter->Update();

        ImagePointer gradIm = filter->GetOutput();

        typename itk::ImageRegionConstIterator<ImageType> IteratorType;
        typename itk::ImageRegionConstIterator<ImageType> MaskIteratorType;
        IteratorType it(m_ImageRawPtr->GetLargestPossibleRegion(), m_ImageRawPtr);
        MaskIteratorType itMask(m_MaskRawPtr->GetLargestPossibleRegion(), m_MaskRawPtr);

        m_Prob.clear();
        m_Indices.clear();

        it.GoToBegin();
        itMask.GoToBegin();

        double totalValue = 0.0;

        while(!it.IsAtEnd() && !itMask.IsAtEnd())
        {
            if(itMask.Value())
            {
                double value = it.Value();

                if(value > 0.0)
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

        if(totalValue < 1e-15)
        {
            totalValue = 1e-15;
        }
        for(size_t i = 0; i < m_Prob.size(); ++i)
        {
            m_Prob[i] /= totalValue;
        }
    }

    virtual void Sample(PointSampleType& pointSampleOut, unsigned int attempts = 1)
    {
        IndexType index;

        GeneratorType* gen = m_RNG.GetPointer();
        ImageType* image = m_ImageRawPtr;
        MaskImageType* mask = m_MaskRawPtr;
        WeightImageType* weights = m_WeightsRawPtr;

        if(m_Prob.size() == 0U)
        {
            size_t sind = SearchCumProb(gen->GetVariateWithOpenRange());
            assert(sind < m_Prob.size());
            index = m_Indices[sind];
        }
        else
        {
            // In the case where the image is completely uniform (inside the mask)
            // we revert back to uniform random sampling (instead of failing with an error)
            
            IndexType origin = m_BBOrigin;
            SizeType size = m_BBSize;

            itk::Vector<double, ImageType::ImageDimension> v = gen->GetVariate();
            for(unsigned int i = 0; i < ImageType::ImageDimension; ++i)
            {
                index[i] = origin[i] + (IndexValueType)(v[i] * size[i]);
            }
        }

        bool isMasked = PerformMaskTest(index);
        if(!isMasked) {
            if(attempts > 0)
            {
                Sample(pointSampleOut, attempts - 1U);
                return;
            }

            pointSampleOut.m_Weight = itk::NumericTraits<WeightValueType>::Zero();
            return;
        }

        image->TransformIndexToPhysicalPoint(index, pointSampleOut.m_Point);

        if(weights)
        {
            pointSampleOut.m_Weight = weights->GetPixel(index);
        }
        else
        {
            pointSampleOut.m_Weight = itk::NumericTraits<WeightValueType>::One();
        }

        DitherPoint(pointSampleOut.m_Point);
    }
protected:
    GradientWeightedPointSampler() {
        m_RNG = GeneratorType::New();
        m_RNG->SetSeed(m_Seed*2U+13U);
    }

    size_t SearchCumProb(double p) {
        double* arr = m_Prob.data();

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
                double pr = arr[m];

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

    std::vector<double> m_Prob;
    std::vector<IndexType> m_Indices;
}; // End of class UniformPointSampler


#endif
