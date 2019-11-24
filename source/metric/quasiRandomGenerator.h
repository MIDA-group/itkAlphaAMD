
#ifndef QUASI_RANDOM_GENERATOR_H
#define QUASI_RANDOM_GENERATOR_H

#include "itkSize.h"
#include "itkPoint.h"
#include "itkImage.h"
#include "itkSmartPointer.h"
#include "itkImageRegionIterator.h"
#include "itkNumericTraits.h"
#include "itkMersenneTwisterRandomVariateGenerator.h"

template <unsigned int Dim>
itk::Vector<double, Dim> MakeQuasiRandomGeneratorAlpha();

template <>
itk::Vector<double, 1U> MakeQuasiRandomGeneratorAlpha<1U>() {
    itk::Vector<double, 1U> result;
    double phi1 = 1.61803398874989484820458683436563;
    result[0] = 1.0/phi1;
    
    return result;
}

template <>
itk::Vector<double, 2U> MakeQuasiRandomGeneratorAlpha<2U>() {
    itk::Vector<double, 2U> result;
    double phi2 = 1.32471795724474602596090885447809;
    result[0] = 1.0/phi2;
    result[1] = pow(1.0/phi2, 2.0);
    
    return result;
}

template <>
itk::Vector<double, 3U> MakeQuasiRandomGeneratorAlpha<3U>() {
    itk::Vector<double, 3U> result;
    double phi3 = 1.220744084605759475361686349108831;
    result[0] = 1.0/phi3;
    result[1] = pow(1.0/phi3, 2.0);
    result[2] = pow(1.0/phi3, 3.0);

    return result;
}

template <unsigned int Dim>
class QuasiRandomGenerator : public itk::Object {
public:
    using Self = QuasiRandomGenerator<Dim>;
    using Superclass = itk::Object;
    using Pointer = itk::SmartPointer<Self>;
    using ConstPointer = itk::SmartPointer<const Self>;

    using GeneratorType = itk::Statistics::MersenneTwisterRandomVariateGenerator;
    using GeneratorPointer = typename GeneratorType::Pointer;
    using ValueType = itk::Vector<double, Dim>;

    itkNewMacro(Self);
  
    itkTypeMacro(PointSamplerBase, itk::Object);

    void SetSeed(unsigned int seed) {
        m_Generator->SetSeed(seed);
    }

    void Restart() {
        m_State[0] = m_Generator->GetVariateWithOpenUpperRange();
        for(unsigned int i = 1; i < Dim; ++i) {
            m_State[i] = m_State[0];
        }
    }

    ValueType GetVariate() {
        for(unsigned j = 0; j < Dim; ++j) {
            m_State[j] = fmod((m_State[j] + m_Alpha[j]), 1.0);
        }
        return m_State;
    }
protected:
    QuasiRandomGenerator() {
        m_Alpha = MakeQuasiRandomGeneratorAlpha<Dim>();
        m_Generator = GeneratorType::New();
        m_Generator->SetSeed(42);
        Restart();
    }

    GeneratorPointer m_Generator;

    itk::Vector<double, Dim> m_Alpha;
    itk::Vector<double, Dim> m_State;
    itk::Size<Dim> m_Size;
};

#endif
