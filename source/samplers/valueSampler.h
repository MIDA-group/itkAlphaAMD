
#ifndef VALUE_SAMPLER_H
#define VALUE_SAMPLER_H

#include "itkMersenneTwisterRandomVariateGenerator.h"
#include "quasiRandomGenerator.h"

template <typename ValueType, unsigned int Channels>
class ValueSamplerBase : public itk::Object {
public:
    using Self = ValueSamplerBase;
    using Superclass = itk::Object;
    using Pointer = itk::SmartPointer<Self>;
    using ConstPointer = itk::SmartPointer<const Self>;

    using VectorType = itk::FixedArray<ValueType, Channels>;
    
    itkNewMacro(Self);
  
    itkTypeMacro(ValueSamplerBase, itk::Object);

    virtual unsigned long long GetSeed() const
    {
        return m_Seed;
    }

    virtual void SetSeed(unsigned long long seed)
    {
        m_Seed = seed;
    }

    virtual void RestartFromSeed()
    {
        ;
    }

    virtual void Initialize()
    {
        ;
    }

    virtual void Sample(VectorType& out) { assert(false); }

    virtual void SampleN(std::vector<VectorType>& valuesOut, unsigned int count)
    {
        for(unsigned int i = 0; i < count; ++i) {
            VectorType v;
            Sample(v);
            valuesOut.push_back(v);
        }
    }

    template <typename OtherType>
    void SampleValue(itk::FixedArray<OtherType, Channels>& out, ValueType upper)
    {
        VectorType v;
        Sample(v);
        for(unsigned int i = 0; i < Channels; ++i)
        {
            out[i] = static_cast<OtherType>(v[i] * upper);
        }
    }

    template <typename OtherType>
    void SampleValues(std::vector<itk::FixedArray<OtherType, Channels> >&out, ValueType upper, unsigned int count)
    {
        for(unsigned int i = 0; i < count; ++i) {
            itk::FixedArray<OtherType, Channels> v;
            SampleValue(v, upper);
            out.push_back(v);
        }
    }
protected:
    ValueSamplerBase() {
        m_Seed = 42U;
    }
    virtual ~ValueSamplerBase() {

    }

    unsigned long long m_Seed;
    
}; // End of class ValueSamplerBase

template <typename ValueType, unsigned int Channels>
class UniformValueSampler : public ValueSamplerBase<ValueType, Channels> {
public:
    using Self = UniformValueSampler;
    using Superclass = ValueSamplerBase<ValueType, Channels>;
    using Pointer = itk::SmartPointer<Self>;
    using ConstPointer = itk::SmartPointer<const Self>;

    using VectorType = typename Superclass::VectorType;
    
    using GeneratorType = itk::Statistics::MersenneTwisterRandomVariateGenerator;
    using GeneratorPointer = typename GeneratorType::Pointer;

    itkNewMacro(Self);
  
    itkTypeMacro(UniformValueSampler, ValueSamplerBase);

    virtual void RestartFromSeed()
    {
        m_RNG->SetSeed(Superclass::GetSeed());
    }

    virtual void Initialize()
    {
        ;
    }

    virtual void Sample(VectorType& out) {
        for(unsigned int i = 0; i < Channels; ++i) {
            out[i] = m_RNG->GetVariateWithOpenRange();
        }
    }
protected:
    UniformValueSampler() {
        m_RNG = GeneratorType::New();
        m_RNG->SetSeed(Superclass::GetSeed());
    }
    virtual ~UniformValueSampler() {

    }

    GeneratorPointer m_RNG;
}; // End of class UniformValueSampler

template <typename ValueType, unsigned int Channels>
class QuasiRandomValueSampler : public ValueSamplerBase<ValueType, Channels> {
public:
    using Self = QuasiRandomValueSampler;
    using Superclass = ValueSamplerBase<ValueType, Channels>;
    using Pointer = itk::SmartPointer<Self>;
    using ConstPointer = itk::SmartPointer<const Self>;

    using VectorType = typename Superclass::VectorType;
    
    using GeneratorType = QuasiRandomGenerator<Channels>;
    using GeneratorPointer = typename GeneratorType::Pointer;

    itkNewMacro(Self);
  
    itkTypeMacro(QuasiRandomValueSampler, ValueSamplerBase);

    virtual void RestartFromSeed()
    {
        m_RNG->SetSeed(Superclass::GetSeed());
        m_RNG->Restart();
    }

    virtual void Initialize()
    {
        ;
    }

    virtual void Sample(VectorType& out) {
        out = m_RNG->GetVariate();
    }
protected:
    QuasiRandomValueSampler() {
        m_RNG = GeneratorType::New();
        m_RNG->SetSeed(Superclass::GetSeed());
    }
    virtual ~QuasiRandomValueSampler() {

    }

    GeneratorPointer m_RNG;
}; // End of class QuasiRandomValueSampler

#endif
