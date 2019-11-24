
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
            minIndex.Fill(itk::NumericTraits<IndexValueType>::Max());
            maxIndex.Fill(itk::NumericTraits<IndexValueType>::Min())

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
    
    typedef PointSample<ImageType, WeightImageType> PointSampleType;

    itkNewMacro(Self);
  
    itkTypeMacro(PointSamplerBase, itk::Object);

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
    QuasiRandomPointSampler() {
        m_RNG = GeneratorType::New();
        m_RNG->SetSeed(m_Seed*2U+13U);
    }

    mutable GeneratorPointer m_RNG;
}; // End of class QuasiRandomPointSampler

// Gradient-importance weighted random point sampler

#endif
