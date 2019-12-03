
// The main component of a registration framework
// based on distance measures between fuzzy sets.

#ifndef ALPHA_BSPLINE_REGISTRATION_H
#define ALPHA_BSPLINE_REGISTRATION_H

#include "itkImage.h"
#include "itkDomainThreader.h"
#include "itkThreadedIndexedContainerPartitioner.h"
#include "itkBSplineTransform.h"
#include "itkTimeProbesCollectorBase.h"

#include "../samplers/pointSampler.h"

template <typename TImageType, typename TDistType, unsigned int TSplineOrder>
struct AlphaBSplineRegistrationThreadState
{
    using ImageType = TImageType;
    using ImagePointer = typename ImageType::Pointer;

    constexpr static unsigned int ImageDimension = ImageType::ImageDimension;
    constexpr static unsigned int SplineOrder = TSplineOrder;

	using TransformType = itk::BSplineTransform<double, ImageDimension, SplineOrder>;
	using TransformPointer = typename TransformType::Pointer;
    using DerivativeType = typename TransformType::DerivativeType;

    using WeightsType = typename TransformType::WeightsType;
    using ParameterIndexArrayType = typename TransformType::ParameterIndexArrayType;

    using DistType = TDistType;
    using DistPointer = typename DistType::Pointer;

    using DistEvalContextType = typename DistType::EvalContextType;
    using DistEvalContextPointer = typename DistEvalContextType::Pointer;

    DerivativeType m_DerivativeRefToFlo;
    DerivativeType m_DerivativeFloToRef;

    DerivativeType m_WeightsRefToFlo;
    DerivativeType m_WeightsFloToRef;

    WeightsType m_ParamWeightsRefToFlo;
    WeightsType m_ParamWeightsFloToRef;

    ParameterIndexArrayType m_ParamIndicesRefToFlo;
    ParameterIndexArrayType m_ParamIndicesFloToRef;

    double m_DistanceRefToFlo;
    double m_DistanceFloToRef;
    double m_WeightRefToFlo;
    double m_WeightFloToRef;

    unsigned int m_ParamNumRefToFlo;
    unsigned int m_ParamNumFloToRef;

    unsigned int m_SupportSize;

    DistEvalContextPointer m_DistEvalContextRefImage;
    DistEvalContextPointer m_DistEvalContextFloImage;

    void Initialize(unsigned int paramNumRefToFlo, unsigned int paramNumFloToRef, unsigned int supportSize, DistEvalContextPointer distEvalContextRefImage, DistEvalContextPointer distEvalContextFloImage)
    {
        m_ParamNumRefToFlo = paramNumRefToFlo;
        m_ParamNumFloToRef = paramNumFloToRef;
        m_SupportSize = supportSize;

        m_DistEvalContextRefImage = distEvalContextRefImage;
        m_DistEvalContextFloImage = distEvalContextFloImage;

        m_DerivativeRefToFlo.SetSize(m_ParamNumRefToFlo);
        m_WeightsRefToFlo.SetSize(m_ParamNumRefToFlo);

        m_DerivativeFloToRef.SetSize(m_ParamNumFloToRef);
        m_WeightsFloToRef.SetSize(m_ParamNumRefToFlo);

        m_ParamWeightsRefToFlo.SetSize(supportSize);
        m_ParamWeightsFloToRef.SetSize(supportSize);
        m_ParamIndicesRefToFlo.SetSize(supportSize);
        m_ParamIndicesFloToRef.SetSize(supportSize);
    }

    void StartIteration()
    {
        m_WeightRefToFlo = 0.0;
        m_WeightFloToRef = 0.0;
        m_DistanceRefToFlo = 0.0;
        m_DistanceFloToRef = 0.0;
        m_WeightsRefToFlo.Fill(0);
        m_WeightsFloToRef.Fill(0);
        m_DerivativeRefToFlo.Fill(0);
        m_DerivativeFloToRef.Fill(0);
    }
};

template <typename ValueType, unsigned int Dim>
struct SymmetryLossTerm
{
	ValueType value;
	itk::Vector<ValueType, Dim> grad;
};

template <class TAssociate, class TImageType, class TTransformType, class TDistType>
class AlphaBSplineRegistrationVADThreader : public itk::DomainThreader<itk::ThreadedIndexedContainerPartitioner, TAssociate>
{
public:
  using Self = AlphaBSplineRegistrationVADThreader;
  using Superclass = itk::DomainThreader<itk::ThreadedIndexedContainerPartitioner, TAssociate>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  // The domain is an index range.
  using DomainType = typename Superclass::DomainType;

  using ImageType = TImageType;
  using TransformType = TTransformType;
  using TransformPointer = typename TransformType::Pointer;

  using DistType = TDistType;
  using DistPointer = typename DistType::Pointer;

  using ThreadStateType = AlphaBSplineRegistrationThreadState<ImageType, DistType, 3U>;

  using PointType = typename ImageType::PointType;

  using DerivativeType = typename TransformType::DerivativeType;
  using WeightsType = typename TransformType::WeightsType;
  using ParameterIndexArrayType = typename TransformType::ParameterIndexArrayType;

  using DistEvalContextType = typename DistType::EvalContextType;
  using DistEvalContextPointer = typename DistEvalContextType::Pointer;

  using PointSamplerType = typename TAssociate::PointSamplerType;
  using PointSamplerPointer = typename PointSamplerType::Pointer;
  using PointSampleType = typename TAssociate::PointSampleType;

  using SymmetryLossType = SymmetryLossTerm<double, TAssociate::ImageDimension>;

  constexpr static unsigned int ImageDimension = TAssociate::ImageDimension;

  // This creates the ::New() method for instantiating the class.
  itkNewMacro(Self);

protected:
  // We need a constructor for the itkNewMacro.
  AlphaBSplineRegistrationVADThreader() = default;

private:
  void
  BeforeThreadedExecution() override
  {
      std::vector<ThreadStateType>& threadStates = this->m_Associate->m_ThreadData;
    // Resize our per-thread data structures to the number of threads that we
    // are actually going to use.  At this point the number of threads that
    // will be used have already been calculated and are available.  The number
    // of threads used depends on the number of cores or processors available
    // on the current system.  It will also be truncated if, for example, the
    // number of cells in the CellContainer is smaller than the number of cores
    // available.
    const itk::ThreadIdType numberOfThreads = this->GetNumberOfThreadsUsed();
    //const itk::ThreadIdType numberOfThreads = this->GetNumberOfWorkUnitsUsed();

    assert(numberOfThreads <= threadStates.size());
    //std::cout << "Number of threads: " << numberOfThreads << std::endl;

  }

  void
  ThreadedExecution(const DomainType & subDomain, const itk::ThreadIdType threadId) override
  {
      //std::cout << "Thread: " << threadId << std::endl;
    ThreadStateType* state = &this->m_Associate->m_ThreadData[threadId];
    itk::IndexValueType refToFloSampleCount = static_cast<itk::IndexValueType>(this->m_Associate->m_RefToFloSampleCount);
    itk::IndexValueType floToRefSampleCount = static_cast<itk::IndexValueType>(this->m_Associate->m_FloToRefSampleCount);

    TransformType* transformRefToFlo = this->m_Associate->m_TransformRefToFloRawPtr;
    TransformType* transformFloToRef = this->m_Associate->m_TransformFloToRefRawPtr;

    state->StartIteration();

    PointSampleType pointSample;

    double lambda = this->m_Associate->m_SymmetryLambda;

    // Look only at the range of indices in the subDomain.
    for (itk::IndexValueType ii = subDomain[0]; ii <= subDomain[1]; ++ii)
    {
        //std::cout << ii << std::endl;
        if(ii < refToFloSampleCount) {
            // Reference to Floating sample
            this->m_Associate->m_PointSamplerRefImage->Sample(threadId, pointSample);
                
            ComputePointValueAndDerivative(
                pointSample,
                this->m_Associate->m_DistDataStructFloImage.GetPointer(),
                state->m_DistEvalContextFloImage.GetPointer(),
                transformRefToFlo,
                transformFloToRef,
                state->m_DistanceRefToFlo,
                state->m_WeightRefToFlo,
                state->m_DerivativeRefToFlo,
                state->m_DerivativeFloToRef,
                state->m_WeightsRefToFlo,
                state->m_WeightsFloToRef,
                state->m_ParamWeightsRefToFlo,
                state->m_ParamWeightsFloToRef,
                state->m_ParamIndicesRefToFlo,
                state->m_ParamIndicesFloToRef,
                state->m_SupportSize,
                lambda);
        }
        else
        {
            // Floating to Reference sample
            this->m_Associate->m_PointSamplerFloImage->Sample(threadId, pointSample);

            ComputePointValueAndDerivative(
                pointSample,
                this->m_Associate->m_DistDataStructRefImage.GetPointer(),
                state->m_DistEvalContextRefImage.GetPointer(),
                transformFloToRef,
                transformRefToFlo,
                state->m_DistanceFloToRef,
                state->m_WeightFloToRef,
                state->m_DerivativeFloToRef,
                state->m_DerivativeRefToFlo,
                state->m_WeightsFloToRef,
                state->m_WeightsRefToFlo,
                state->m_ParamWeightsFloToRef,
                state->m_ParamWeightsRefToFlo,
                state->m_ParamIndicesFloToRef,
                state->m_ParamIndicesRefToFlo,
                state->m_SupportSize,
                lambda);

        }
    }
  }

  void
  AfterThreadedExecution() override
  {

    //const itk::ThreadIdType numberOfThreads = this->GetNumberOfWorkUnitsUsed();
    const itk::ThreadIdType numberOfThreads = this->GetNumberOfThreadsUsed();

    this->m_Associate->m_ThreadsUsed = numberOfThreads;
  }

	inline static void ComputeSymmetryLoss(PointType originalPoint, PointType returnedPoint, SymmetryLossType &out)
	{
		itk::Vector<double, ImageDimension> vec = returnedPoint - originalPoint;
	
		out.value = 0.5 * vec.GetSquaredNorm();
		out.grad = vec;
	}

    inline static double ReciprocalValue(double value)
	{
		if (value < 1e-15)
		{
			return 0.0;
		}
		else
		{
			return 1.0 / value;
		}
	}

  void ComputePointValueAndDerivative(
      PointSampleType& pointSample,
      DistType* distStruct,
      DistEvalContextType* distEvalContext,
      TransformType* tfor,
      TransformType* trev,
      double& value,
      double &weight,
      DerivativeType& dfor,
      DerivativeType& drev,
      DerivativeType& wfor,
      DerivativeType& wrev,
      WeightsType& splineWeightsFor,
      WeightsType& splineWeightsRev,
      ParameterIndexArrayType& parameterIndicesFor,
      ParameterIndexArrayType& parameterIndicesRev,
      unsigned int supportSize,
      double lambda
      )
  {
    constexpr unsigned int Dim = TAssociate::ImageDimension;

	PointType transformedPoint;
	PointType returnedPoint;
	bool isInside;

    unsigned int paramPerDimFor = tfor->GetNumberOfParametersPerDimension();
    unsigned int paramPerDimRev = trev->GetNumberOfParametersPerDimension();

	tfor->TransformPoint(pointSample.m_Point, transformedPoint, splineWeightsFor, parameterIndicesFor, isInside);
	if (!isInside)
		return;
	trev->TransformPoint(transformedPoint, returnedPoint, splineWeightsRev, parameterIndicesRev, isInside);

	SymmetryLossType slt;

	ComputeSymmetryLoss(pointSample.m_Point, returnedPoint, slt);

    // Compute the point-to-set distance and gradient here

    double localValue = 0.0;
    itk::Vector<double, Dim> grad;
    bool flag = distStruct->ValueAndDerivative(
        distEvalContext,
        transformedPoint,
        pointSample.m_Value,
        localValue,
        grad);

    double w = pointSample.m_Weight;
    double valueW;

    if(!flag) {
        valueW = 0.0;
    } else {
        valueW = pointSample.m_Weight;
    }

    value += (1.0-lambda) * valueW * localValue + lambda * w * slt.value;
    weight += (1.0-lambda) * valueW + lambda * w;

	// Compute jacobian for metric and symmetry loss
	for (unsigned int dim = 0; dim < Dim; ++dim)
	{
		unsigned int offFor = dim * paramPerDimFor;
		unsigned int offRev = dim * paramPerDimRev;
		double gradVal = grad[dim];
		double sltGradVal = slt.grad[dim];

		for (unsigned int mu = 0; mu < supportSize; ++mu)
		{
			unsigned int parInd = offFor + parameterIndicesFor[mu];
			unsigned int parIndRev = offRev + parameterIndicesRev[mu];
            double sw = splineWeightsFor[mu];
			double swInv = splineWeightsRev[mu];

			dfor[parInd] -= ((1.0 - lambda) * sw * valueW * gradVal + lambda * sw * w * sltGradVal);
			wfor[parInd] += ((1.0 - lambda) * sw * w + lambda * sw * w);
            drev[parIndRev] -= (lambda * swInv * w * sltGradVal);
            wrev[parIndRev] += (lambda * swInv * w);
		}
	}
  }
};

template <class TAssociate, class TImageType, class TTransformType, class TDistType>
class AlphaBSplineRegistrationStepThreader : public itk::DomainThreader<itk::ThreadedIndexedContainerPartitioner, TAssociate>
{
public:
  using Self = AlphaBSplineRegistrationStepThreader;
  using Superclass = itk::DomainThreader<itk::ThreadedIndexedContainerPartitioner, TAssociate>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  // The domain is an index range.
  using DomainType = typename Superclass::DomainType;

  using ImageType = TImageType;

  using DistType = TDistType;
  using DistPointer = typename DistType::Pointer;

  using ThreadStateType = AlphaBSplineRegistrationThreadState<ImageType, DistType, 3U>;
  using TransformType = TTransformType;
  using TransformPointer = typename TransformType::Pointer;

  using DerivativeType = typename TransformType::DerivativeType;
  using WeightsType = typename TransformType::WeightsType;
  using ParameterIndexArrayType = typename TransformType::ParameterIndexArrayType;

  // This creates the ::New() method for instantiating the class.
  itkNewMacro(Self);

protected:
  // We need a constructor for the itkNewMacro.
  AlphaBSplineRegistrationStepThreader() = default;

private:
  void
  BeforeThreadedExecution() override
  {
      std::vector<ThreadStateType>& threadStates = this->m_Associate->m_ThreadData;
    // Resize our per-thread data structures to the number of threads that we
    // are actually going to use.  At this point the number of threads that
    // will be used have already been calculated and are available.  The number
    // of threads used depends on the number of cores or processors available
    // on the current system.  It will also be truncated if, for example, the
    // number of cells in the CellContainer is smaller than the number of cores
    // available.
    const itk::ThreadIdType numberOfThreads = this->GetNumberOfThreadsUsed();
    //const itk::ThreadIdType numberOfThreads = this->GetNumberOfWorkUnitsUsed();

    assert(numberOfThreads <= threadStates.size());
  }

  void
  ThreadedExecution(const DomainType & subDomain, const itk::ThreadIdType threadId) override
  {
    // The number of threads used for the previous step
    unsigned int threadsUsed = this->m_Associate->m_ThreadsUsed;
    unsigned int paramNumRefToFlo = this->m_Associate->m_TransformRefToFlo->GetNumberOfParameters();
    double momentum = this->m_Associate->m_Momentum;
    double learningRate = this->m_Associate->m_LearningRate;
    constexpr double eps = 0.01;

    DerivativeType& derRefToFlo = this->m_Associate->m_DerivativeRefToFlo;
    DerivativeType& derFloToRef = this->m_Associate->m_DerivativeFloToRef;
    if(threadId == 0)
    {
        double refToFloValue = 0.0;
        double floToRefValue = 0.0;
        double refToFloWeight = 0.0;
        double floToRefWeight = 0.0;
        for(unsigned int i = 0; i < threadsUsed; ++i) {
            refToFloValue += this->m_Associate->m_ThreadData[i].m_DistanceRefToFlo;
            floToRefValue += this->m_Associate->m_ThreadData[i].m_DistanceFloToRef;
            refToFloWeight += this->m_Associate->m_ThreadData[i].m_WeightRefToFlo;
            floToRefWeight += this->m_Associate->m_ThreadData[i].m_WeightFloToRef;
        }
        this->m_Associate->m_Value = 0.5 * (refToFloValue/(refToFloWeight + 0.001) + floToRefValue/(floToRefWeight + 0.001));
    }

    // Look only at the range of indices in the subDomain.
    for (itk::IndexValueType ii = subDomain[0]; ii <= subDomain[1]; ++ii)
    {
      
        if(ii < paramNumRefToFlo)
        {
            itk::IndexValueType localIndex = ii;

            double derVal = 0.0;
            double weightVal = 0.0;
            for(unsigned int i = 0; i < threadsUsed; ++i) {
                derVal += this->m_Associate->m_ThreadData[i].m_DerivativeRefToFlo[localIndex];
                weightVal += this->m_Associate->m_ThreadData[i].m_WeightsRefToFlo[localIndex];
            }
            
            double prevVal = this->m_Associate->m_DerivativeRefToFlo[localIndex];
            this->m_Associate->m_DerivativeRefToFlo[localIndex] = prevVal * momentum + (1.0 - momentum) * derVal / (weightVal + eps);
        }
        else
        {
            itk::IndexValueType localIndex = ii - paramNumRefToFlo;

            double derVal = 0.0;
            double weightVal = 0.0;
            for(unsigned int i = 0; i < threadsUsed; ++i) {
                derVal += this->m_Associate->m_ThreadData[i].m_DerivativeFloToRef[localIndex];
                weightVal += this->m_Associate->m_ThreadData[i].m_WeightsFloToRef[localIndex];
            }
            
            double prevVal = this->m_Associate->m_DerivativeFloToRef[localIndex];
            this->m_Associate->m_DerivativeFloToRef[localIndex] = prevVal * momentum + (1.0 - momentum) * derVal / (weightVal + eps);
        }
    }
  }

  void
  AfterThreadedExecution() override
  {
    const itk::ThreadIdType numberOfThreads = this->GetNumberOfThreadsUsed();
    //const itk::ThreadIdType numberOfThreads = this->GetNumberOfWorkUnitsUsed();

    this->m_Associate->m_ThreadsUsed = numberOfThreads;

  }
};

template <typename TImageType, typename TDistType, unsigned int TSplineOrder=3>
class AlphaBSplineRegistration : public itk::Object {
public:
    using Self = AlphaBSplineRegistration<TImageType, TDistType, TSplineOrder>;
    using Superclass = itk::Object;
    using Pointer = itk::SmartPointer<Self>;
    using ConstPointer = itk::SmartPointer<const Self>;
    
    constexpr static unsigned int ImageDimension = TImageType::ImageDimension;
    constexpr static unsigned int SplineOrder = TSplineOrder;

    using ImageType = TImageType;
    typedef typename ImageType::Pointer ImagePointer;
    using DistType = TDistType;
    
    typedef typename ImageType::ValueType ValueType;
    
    typedef typename ImageType::SpacingType SpacingType;
    typedef typename ImageType::RegionType RegionType;
    typedef typename ImageType::IndexType IndexType;
    typedef typename ImageType::SizeType SizeType;
    typedef typename ImageType::IndexValueType IndexValueType;
    typedef typename ImageType::PointType PointType;

    using DistPointer = typename DistType::Pointer;

    using DistEvalContextType = typename DistType::EvalContextType;
    using DistEvalContextPointer = typename DistEvalContextType::Pointer;

	using TransformType = itk::BSplineTransform<double, ImageDimension, SplineOrder>;
	using TransformPointer = typename TransformType::Pointer;
    using DerivativeType = typename TransformType::DerivativeType;

    using PointSamplerType = PointSamplerBase<ImageType, itk::Image<bool, ImageDimension> , ImageType>;
    using PointSamplerPointer = typename PointSamplerType::Pointer;
    using PointSampleType = PointSample<ImageType, ImageType>;

    using VADThreaderType = AlphaBSplineRegistrationVADThreader<Self, ImageType, TransformType, DistType>;
    using VADThreaderPointer = typename VADThreaderType::Pointer;
    using StepThreaderType = AlphaBSplineRegistrationStepThreader<Self, ImageType, TransformType, DistType>;
    using StepThreaderPointer = typename StepThreaderType::Pointer;

    using ThreadStateType = AlphaBSplineRegistrationThreadState<ImageType, DistType, 3U>;

    itkNewMacro(Self);

    itkTypeMacro(AlphaBSplineRegistration, itk::Object);

    virtual TransformPointer GetTransformRefToFlo() const
    {
        return m_TransformRefToFlo;
    }

    virtual TransformPointer GetTransformFloToRef() const
    {
        return m_TransformFloToRef;
    }

    virtual void SetTransformRefToFlo(TransformPointer transform)
    {
        m_TransformRefToFlo = transform;
        m_TransformRefToFloRawPtr = m_TransformRefToFlo.GetPointer();
    }

    virtual void SetTransformFloToRef(TransformPointer transform)
    {
        m_TransformFloToRef = transform;
        m_TransformFloToRefRawPtr = m_TransformFloToRef.GetPointer();
    }

    virtual void SetPointSamplerRefImage(PointSamplerPointer sampler)
    {
        m_PointSamplerRefImage = sampler;
    }
    
    virtual void SetPointSamplerFloImage(PointSamplerPointer sampler)
    {
        m_PointSamplerFloImage = sampler;
    }

    virtual void SetDistDataStructRefImage(DistPointer dist)
    {
        m_DistDataStructRefImage = dist;
    }

    virtual void SetDistDataStructFloImage(DistPointer dist)
    {
        m_DistDataStructFloImage = dist;
    }

    virtual void SetLearningRate(double learningRate)
    {
        m_LearningRate = learningRate;
    }

    virtual void SetMomentum(double momentum)
    {
        m_Momentum = momentum;
    }

    virtual void SetSymmetryLambda(double symmetryLambda)
    {
        m_SymmetryLambda = symmetryLambda;
    }

    virtual void SetIterations(unsigned int iterations)
    {
        m_Iterations = iterations;
    }

    virtual void SetSampleCountRefToFlo(unsigned int count)
    {
        m_RefToFloSampleCount = count;
    }

    virtual void SetSampleCountFloToRef(unsigned int count)
    {
        m_FloToRefSampleCount = count;
    }

    virtual void Initialize()
    {
        assert(m_TransformRefToFlo.GetPointer() != nullptr);
        assert(m_TransformFloToRef.GetPointer() != nullptr);

        m_DerivativeRefToFlo.SetSize(m_TransformRefToFlo->GetNumberOfParameters());
        m_DerivativeFloToRef.SetSize(m_TransformFloToRef->GetNumberOfParameters());

//    void Initialize(unsigned int paramNumRefToFlo, unsigned int paramNumFloToRef, unsigned int supportSize, DistEvalContextPointer distEvalContextRefImage, DistEvalContextPointer distEvalContextFloImage)

        m_ThreadData.reserve(128U);
        unsigned int supSize = (unsigned int)pow(4.0, ImageDimension);
        for(unsigned int i = 0; i < 32U; ++i)
        {
            //Initialize(unsigned int paramNumRefToFlo, unsigned int paramNumFloToRef, unsigned int supportSize, DistEvalContextPointer distEvalContextRefImage, DistEvalContextPointer distEvalContextFloImage)
            ThreadStateType ts;
            m_ThreadData.push_back(ts);
            DistEvalContextPointer evalContext1 = m_DistDataStructRefImage->MakeEvalContext();
            DistEvalContextPointer evalContext2 = m_DistDataStructFloImage->MakeEvalContext();
            m_ThreadData[i].Initialize(
                m_TransformRefToFlo->GetNumberOfParameters(),
                m_TransformFloToRef->GetNumberOfParameters(),
                supSize,
                evalContext1,
                evalContext2
            );
        }
    }

    virtual void Run()
    {
        unsigned int iterations = m_Iterations;
        for(unsigned int i = 0; i < iterations; ++i)
        {
            // Compute the distance value and derivatives for a sampled subset of the image
            //std::cout << "Stage 1" << std::endl;

            //chronometer.Start("Stage 1");            
            typename VADThreaderType::DomainType completeDomain1;
            completeDomain1[0] = 0;
            completeDomain1[1] = this->m_RefToFloSampleCount + this->m_FloToRefSampleCount - 1;
            this->m_VADThreader->Execute(this, completeDomain1);
            //chronometer.Stop("Stage 1");

            //std::cout << "Stage 2" << std::endl;
            // Aggregate, normalize, and apply a step counter to the gradient direction

            //chronometer.Start("Stage 2");            
            typename StepThreaderType::DomainType completeDomain2;
            completeDomain2[0] = 0;
            completeDomain2[1] = m_TransformRefToFlo->GetNumberOfParameters() + m_TransformFloToRef->GetNumberOfParameters() - 1;
            this->m_StepThreader->Execute(this, completeDomain2);
            //chronometer.Stop("Stage 2");

            m_TransformRefToFlo->UpdateTransformParameters(m_DerivativeRefToFlo, m_LearningRate);
            m_TransformFloToRef->UpdateTransformParameters(m_DerivativeFloToRef, m_LearningRate);

            //if(i % 50 == 0)
                //std::cout << "Value: " << m_Value << std::endl;
        }
        //chronometer.Report(std::cout);
    }   
protected:
    AlphaBSplineRegistration()
    {
        m_Value = 0.0;
        m_Momentum = 0.1;
        m_LearningRate = 1.0;
        m_SymmetryLambda = 0.05;
        m_DoMultiThreadValueAndDerivative = true;
        m_DoMultiThreadStep = true;

        m_RefToFloSampleCount = 4096;
        m_FloToRefSampleCount = 4096;

        m_Iterations = 300;

        m_VADThreader = VADThreaderType::New();
        m_StepThreader = StepThreaderType::New();
    }

    // Transformations
    TransformPointer m_TransformRefToFlo;
    TransformPointer m_TransformFloToRef;
    TransformType* m_TransformRefToFloRawPtr;
    TransformType* m_TransformFloToRefRawPtr;

    // Point samplers
    PointSamplerPointer m_PointSamplerRefImage;
    PointSamplerPointer m_PointSamplerFloImage;

    // Target distance data structures
    DistPointer m_DistDataStructRefImage;
    DistPointer m_DistDataStructFloImage;

    double m_Momentum;
    double m_LearningRate;
    double m_SymmetryLambda;
    // Current state
    double m_Value;

    // Thread-local data for value and derivatives computation
    std::vector<ThreadStateType > m_ThreadData;
    unsigned int m_ThreadsUsed;
    VADThreaderPointer m_VADThreader;
    StepThreaderPointer m_StepThreader;

    unsigned int m_Iterations;

    unsigned int m_RefToFloSampleCount;
    unsigned int m_FloToRefSampleCount;

    // Current/last derivative
    DerivativeType m_DerivativeRefToFlo;
    DerivativeType m_DerivativeFloToRef;

    // Enable/disable threading
    bool m_DoMultiThreadValueAndDerivative;
    bool m_DoMultiThreadStep;

    friend class AlphaBSplineRegistrationVADThreader<Self, ImageType, TransformType, DistType>;
    friend class AlphaBSplineRegistrationStepThreader<Self, ImageType, TransformType, DistType>;

   
    //itk::TimeProbesCollectorBase chronometer;

};

#endif
