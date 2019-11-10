
#ifndef ALPHA_SMD_METRIC_DEFORM_H
#define ALPHA_SMD_METRIC_DEFORM_H

#include <vector>
#include "itkOptimizerParameters.h"
#include "itkObjectToObjectMetricBase.h"
//#include "itkAffineTransform.h"
//#include "itkTranslationTransform.h"
//#include "itkCompositeTransform.h"
#include "itkBSplineTransform.h"

#include "itkMersenneTwisterRandomVariateGenerator.h"
#include "samplers.h"

#include "itkAlphaSMDMetricInternal2.h"

namespace itk
{
template <typename ArrayType, typename WeightsType, typename ParameterIndexArrayType>
struct InternalDerivatives
{
	ArrayType dMetric;
	ArrayType dSymmetry;
	ArrayType wMetric;
	ArrayType wSymmetry;

	WeightsType splineWeights;
	ParameterIndexArrayType parameterIndices;
	WeightsType splineWeightsInv;
	ParameterIndexArrayType parameterIndicesInv;
};

template <typename ValueType, unsigned int Dim>
struct SymmetryLossTerm
{
	ValueType value;
	itk::Vector<ValueType, Dim> grad;
};

/**
	* Metric which computes the Alpha-SMD Fuzzy Set Distance between a pair of Fixed and Moving images.
	* Since it does not satisfy the assumptions of the ImageToImageMetric-base classes, the measure
	* is implemented as an ObjectToObjectMetric instead.
	*/
template <typename ImageType, unsigned int Dim, typename TInternalComputationValueType = double, unsigned int SplineOrder = 3>
class AlphaSMDObjectToObjectMetricDeformv4 : public ObjectToObjectMetricBaseTemplate<TInternalComputationValueType>
{
  public:
	/** Standard class typedefs. */
	typedef AlphaSMDObjectToObjectMetricDeformv4 Self;
	typedef ObjectToObjectMetricBaseTemplate<TInternalComputationValueType> Superclass;
	typedef SmartPointer<Self> Pointer;
	typedef SmartPointer<const Self> ConstPointer;

	/** Method for creation through the object factory. */
	itkNewMacro(Self);

	/** Run-time type information (and related methods). */
	itkTypeMacro(AlphaSMDObjectToObjectMetricDeformv4, ObjectToObjectMetricBaseTemplate);

	typedef itk::Statistics::MersenneTwisterRandomVariateGenerator GeneratorType;

	typedef typename ImageType::Pointer ImagePointer;
	typedef typename Superclass::DerivativeType DerivativeType;
	typedef typename Superclass::DerivativeValueType DerivativeValueType;

	typedef typename Superclass::NumberOfParametersType NumberOfParametersType;
	typedef typename Superclass::ParametersType ParametersType;

	typedef typename ImageType::IndexType IndexType;
	typedef typename ImageType::PointType PointType;
	typedef typename std::vector<PointType> PointSetType;

	typedef typename Superclass::MeasureType MeasureType;

	typedef MeasureType ValueType;

	typedef double ParametersValueType;

	typedef itk::BSplineTransform<double, Dim, SplineOrder> TransformType;
	typedef typename TransformType::Pointer TransformPointer;
	typedef typename TransformType::JacobianType JacobianType;

	typedef unsigned char QType;
	typedef alphasmdinternal2::SourcePoint<QType, Dim> SourcePointType;

	typedef SymmetryLossTerm<double, Dim> SymmetryLossType;

	typedef typename TransformType::WeightsType WeightsType;
	typedef typename TransformType::ParameterIndexArrayType ParameterIndexArrayType;

	// Methods

	virtual void Initialize() override
	{
	}

  private:
	inline static void ComputeSymmetryLoss(PointType originalPoint, PointType returnedPoint, SymmetryLossType &out)
	{
		itk::Vector<double, Dim> vec = returnedPoint - originalPoint;
	
		out.value = 0.5 * vec.GetSquaredNorm();
		out.grad = vec;
	}

	inline static void NormalizeDerivative(DerivativeType &derivative, DerivativeType &derivativeWeights, unsigned int startIndex, unsigned int endIndex, double multiplier, DerivativeType &out, bool addToOut)
	{
		if(addToOut) {
			for (unsigned int i = startIndex; i < endIndex; ++i)
			{
				out[i] += ((multiplier * derivative[i]) / derivativeWeights[i]);
			}
		} else {
			for (unsigned int i = startIndex; i < endIndex; ++i)
			{
				out[i] = (multiplier * derivative[i]) / derivativeWeights[i];
			}
		}
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

	void EvaluateAsymmetricMetric(DerivativeType &derivative, bool forwardDirection, TransformType *transformForward, TransformType *transformInverse, std::vector<SourcePointType> &srcPoints, std::vector<unsigned int>& srcPointIndices, unsigned int count, double factor, itk::Vector<double, 2U> &values) const
	{
		double smoothingFactor = 0.1;
		double metricAcc = 0.0;
		double symmetryLossAcc = 0.0;

		double totalMetricWeight = 0.0;
		double totalSymmetryWeight = 0.0;

		unsigned paramPerDim = transformForward->GetNumberOfParametersPerDimension();
		unsigned int supportSize = transformForward->GetNumberOfAffectedWeights();

		DerivativeType& dMetric = m_InternalDerivatives.dMetric;
		DerivativeType& dSymmetry = m_InternalDerivatives.dSymmetry;
		DerivativeType& wMetric = m_InternalDerivatives.wMetric;
		DerivativeType& wSymmetry = m_InternalDerivatives.wSymmetry;

		ParameterIndexArrayType& parameterIndices = m_InternalDerivatives.parameterIndices;
		ParameterIndexArrayType& parameterIndicesInv = m_InternalDerivatives.parameterIndicesInv;

		WeightsType& splineWeights = m_InternalDerivatives.splineWeights;
		WeightsType& splineWeightsInv = m_InternalDerivatives.splineWeightsInv;

		dMetric.Fill(0.0);
		dSymmetry.Fill(0.0);
		wMetric.Fill(smoothingFactor);
		wSymmetry.Fill(smoothingFactor);

		unsigned int derivativeIndexOffset = forwardDirection ? 0U : transformForward->GetNumberOfParameters();
		unsigned int derivativeIndexOffsetInv = forwardDirection ? transformForward->GetNumberOfParameters() : 0U;

		for (unsigned int i = 0; i < count; ++i)
		{
			SourcePointType sp = srcPoints[srcPointIndices[i]];

			// Apply transformation

			PointType transformedPoint;
			PointType returnedPoint;
			bool isInside;

			transformForward->TransformPoint(sp.m_SourcePoint, transformedPoint, splineWeights, parameterIndices, isInside);
			if (!isInside)
				continue;
			transformInverse->TransformPoint(transformedPoint, returnedPoint, splineWeightsInv, parameterIndicesInv, isInside);
			//PointType transformedPoint = transformForward->TransformPoint(sp.m_SourcePoint);

			//PointType returnedPoint = transformInverse->TransformPoint(transformedPoint);

			SymmetryLossType slt;

			ComputeSymmetryLoss(sp.m_SourcePoint, returnedPoint, slt);

			alphasmdinternal2::PointToSetDistance<QType, Dim> ptsd;
			ptsd.m_SourcePoint = sp;

			if (forwardDirection)
			{
				internals.EvalFloPointToSetDistance(ptsd, transformedPoint);
			}
			else
			{
				internals.EvalRefPointToSetDistance(ptsd, transformedPoint);
			}

			if (ptsd.m_IsValid)
			{
				TInternalComputationValueType w = ptsd.m_SourcePoint.m_Weight;
				metricAcc += w * ptsd.m_Distance;
	    		symmetryLossAcc += w * slt.value;

				// Compute jacobian for metric and symmetry loss
				for (unsigned int dim = 0; dim < Dim; ++dim)
				{
					unsigned int off = derivativeIndexOffset + dim * paramPerDim;
					unsigned int offInv = derivativeIndexOffsetInv + dim * paramPerDim;
					TInternalComputationValueType gradVal = ptsd.m_SpatialGrad[dim];
					TInternalComputationValueType sltGradVal = slt.grad[dim];

					for (unsigned int mu = 0; mu < supportSize; ++mu)
					{
						unsigned int parInd = off + parameterIndices[mu];
						unsigned int parIndInv = offInv + parameterIndicesInv[mu];

						double sw = splineWeights[mu] * w;
						double swInv = splineWeightsInv[mu] * w;

						dMetric[parInd] -= sw * gradVal;//metricGrad[dim];
						wMetric[parInd] += sw;
						dSymmetry[parInd] -= sw * sltGradVal;
						wSymmetry[parInd] += sw;
						dSymmetry[parIndInv] -= swInv * sltGradVal;
						wSymmetry[parIndInv] += swInv;
					}
				}
				totalMetricWeight += w;
				totalSymmetryWeight += w;
			} else {
				TInternalComputationValueType w = sp.m_Weight;
	    		symmetryLossAcc += w * slt.value;

				// Compute jacobian for symmetry loss only
				for (unsigned int dim = 0; dim < Dim; ++dim)
				{
					unsigned int off = derivativeIndexOffset + dim * paramPerDim;
					unsigned int offInv = derivativeIndexOffsetInv + dim * paramPerDim;
					TInternalComputationValueType sltGradVal = slt.grad[dim];

					for (unsigned int mu = 0; mu < supportSize; ++mu)
					{
						unsigned int parInd = off + parameterIndices[mu];
						unsigned int parIndInv = offInv + parameterIndicesInv[mu];

						double sw = splineWeights[mu] * w;
						double swInv = splineWeightsInv[mu] * w;
						dSymmetry[parInd] -= sw * sltGradVal;
						wSymmetry[parInd] += sw;
						dSymmetry[parIndInv] -= swInv * sltGradVal;
						wSymmetry[parIndInv] += swInv;
					}
				}

				totalSymmetryWeight += w;
			}
		}

		NormalizeDerivative(
			dSymmetry,
			wSymmetry,
			0U,
			derivative.GetSize(),
			factor * m_SymmetryLambda,
			derivative,
			!forwardDirection);
		NormalizeDerivative(
			dMetric,
			wMetric,
			derivativeIndexOffset,
			derivativeIndexOffset + transformForward->GetNumberOfParameters(),
			factor * (1.0 - m_SymmetryLambda),
			derivative,
			true);

		totalMetricWeight = ReciprocalValue(totalMetricWeight);
		totalSymmetryWeight = ReciprocalValue(totalSymmetryWeight);

		if (forwardDirection)
		{
			values[0] = factor * metricAcc * totalMetricWeight;
			values[1] = factor * symmetryLossAcc * totalSymmetryWeight;
		}
		else
		{
			values[0] = values[0] + factor * metricAcc * totalMetricWeight;
			values[1] = values[1] + factor * symmetryLossAcc * totalSymmetryWeight;
		}
	}

  public:
	//
	// Main method of the measure. Computes the value and derivative (in terms of the transform-parameters).
	//
	virtual void GetValueAndDerivative(MeasureType &value, DerivativeType &derivative) const override
	{

		double distAcc = 0.0;

		itk::Vector<TInternalComputationValueType, 2U> valuePair;

		// Shuffle the (Fixed) source points

		//unsigned fixedCount = (unsigned int)RandomShuffle(m_FixedSourcePoints, m_FixedSamplingPercentage, m_Fixed_RNG.GetPointer());
		SampleFixedPoints(m_FixedSamplingPercentage);

		//assert(fixedCount <= m_FixedSourcePoints.size());

		double factor = (m_SymmetricMeasure ? 0.5 : 1.0);

		EvaluateAsymmetricMetric(derivative, true, m_TransformForwardRawPtr, m_TransformInverseRawPtr, m_FixedSourcePoints, m_FixedSourceIndices, m_FixedSourceIndices.size(), factor, valuePair);

		if (m_SymmetricMeasure)
		{
			// Shuffle the (Moving) source points

			//unsigned int movingCount = (unsigned int)RandomShuffle(m_MovingSourcePoints, m_MovingSamplingPercentage, m_Moving_RNG.GetPointer());
			SampleMovingPoints(m_MovingSamplingPercentage);

			//assert(movingCount <= m_MovingSourcePoints.size());

			EvaluateAsymmetricMetric(derivative, false, m_TransformInverseRawPtr, m_TransformForwardRawPtr, m_MovingSourcePoints, m_MovingSourceIndices, m_MovingSourceIndices.size(), factor, valuePair);
		}

		m_MetricValues = valuePair;
		value = (1.0 - m_SymmetryLambda) * valuePair[0] + m_SymmetryLambda * valuePair[1];
	}

	// Should be updated with a faster version
	virtual MeasureType GetValue() const override
	{
		MeasureType result;
		DerivativeType tmp;

		GetValueAndDerivative(result, tmp);

		return result;
	}

	virtual void GetDerivative(DerivativeType &derivative) const override
	{
		MeasureType tmp;

		GetValueAndDerivative(tmp, derivative);
	}

	virtual void UpdateTransformParameters(
		const DerivativeType &derivative,
		ParametersValueType factor = NumericTraits<ParametersValueType>::OneValue()) override
	{
		unsigned int inIndex = 0;
		unsigned int forwardN = m_TransformForwardRawPtr->GetNumberOfParameters();
		unsigned int inverseN = m_TransformInverseRawPtr->GetNumberOfParameters();
		for (unsigned int i = 0; i < forwardN; ++i, ++inIndex)
		{
			m_ForwardDerivativeTmp[i] = derivative[inIndex];
		}
		for (unsigned int i = 0; i < inverseN; ++i, ++inIndex)
		{
			m_InverseDerivativeTmp[i] = derivative[inIndex];
		}

		m_TransformForwardRawPtr->UpdateTransformParameters(m_ForwardDerivativeTmp, factor);
		m_TransformInverseRawPtr->UpdateTransformParameters(m_InverseDerivativeTmp, factor);
		/*
		const ParametersType &param1 = m_TransformForwardRawPtr->GetParameters();
		const ParametersType &param2 = m_TransformInverseRawPtr->GetParameters();

		ParametersType newParam1(m_TransformForwardRawPtr->GetNumberOfParameters());
		ParametersType newParam2(m_TransformInverseRawPtr->GetNumberOfParameters());

		NumberOfParametersType numberOfParameters = GetNumberOfParameters();

		if (factor == NumericTraits<ParametersValueType>::OneValue())
		{
			unsigned int inIndex = 0;
			for (unsigned int i = 0; i < m_TransformForwardRawPtr->GetNumberOfParameters(); ++i, ++inIndex)
			{
				newParam1[i] = param1[i] + derivative[inIndex];
				//m_TransformForwardRawPtr->SetParam(i, m_TransformForwardRawPtr->GetParam(i) + derivative[inIndex]);
			}
			for (unsigned int i = 0; i < m_TransformInverseRawPtr->GetNumberOfParameters(); ++i, ++inIndex)
			{
				newParam2[i] = param2[i] + derivative[inIndex];
			}
		}
		else
		{
			unsigned int inIndex = 0;
			for (unsigned int i = 0; i < m_TransformForwardRawPtr->GetNumberOfParameters(); ++i, ++inIndex)
			{
				newParam1[i] = param1[i] + derivative[inIndex] * factor;
			}
			for (unsigned int i = 0; i < m_TransformInverseRawPtr->GetNumberOfParameters(); ++i, ++inIndex)
			{
				newParam2[i] = param2[i] + derivative[inIndex] * factor;
			}
		}

		m_TransformForwardRawPtr->SetParameters(newParam1);
		m_TransformInverseRawPtr->SetParameters(newParam2);
	  */
	}

	virtual const ParametersType &GetParameters() const override
	{
		NumberOfParametersType num = GetNumberOfParameters();

		const ParametersType &param1 = m_TransformForwardRawPtr->GetParameters();
		const ParametersType &param2 = m_TransformInverseRawPtr->GetParameters();

		unsigned int outIndex = 0;
		for (unsigned int i = 0; i < m_TransformForwardRawPtr->GetNumberOfParameters(); ++i, ++outIndex)
		{
			m_Param[outIndex] = param1[i];
		}
		for (unsigned int i = 0; i < m_TransformInverseRawPtr->GetNumberOfParameters(); ++i, ++outIndex)
		{
			m_Param[outIndex] = param2[i];
		}

		return m_Param;
	}

	virtual void SetParameters(ParametersType &parameters) override
	{
		assert(parameters.Size() == GetNumberOfParameters());

		ParametersType newParam1(m_TransformForwardRawPtr->GetNumberOfParameters());
		ParametersType newParam2(m_TransformInverseRawPtr->GetNumberOfParameters());

		unsigned int outIndex = 0;
		for (unsigned int i = 0; i < m_TransformForwardRawPtr->GetNumberOfParameters(); ++i, ++outIndex)
		{
			newParam1[i] = parameters[outIndex];
		}
		for (unsigned int i = 0; i < m_TransformInverseRawPtr->GetNumberOfParameters(); ++i, ++outIndex)
		{
			newParam2[i] = parameters[outIndex];
		}

		m_TransformForwardRawPtr->SetParameters(newParam1);
		m_TransformInverseRawPtr->SetParameters(newParam2);
	}

	virtual NumberOfParametersType GetNumberOfParameters() const override { return m_TransformForwardRawPtr->GetNumberOfParameters() + m_TransformInverseRawPtr->GetNumberOfParameters(); }

	virtual NumberOfParametersType GetNumberOfLocalParameters() const override { return GetNumberOfParameters(); }

	virtual bool HasLocalSupport() const override { return false; }

	// Other

	virtual ImagePointer GetFixedImage() const
	{
		return m_FixedImage;
	}

	virtual void SetFixedImage(ImagePointer fixedImage)
	{
		m_FixedImage = fixedImage;
	}

	virtual ImagePointer GetMovingImage() const
	{
		return m_MovingImage;
	}

	virtual void SetMovingImage(ImagePointer movingImage)
	{
		m_MovingImage = movingImage;
	}

	virtual void SetFixedMask(typename itk::Image<bool, Dim>::Pointer fixedMask)
	{
		m_FixedMask = fixedMask;
	}

	virtual void SetMovingMask(typename itk::Image<bool, Dim>::Pointer movingMask)
	{
		m_MovingMask = movingMask;
	}

	virtual void SetFixedWeightImage(typename itk::Image<TInternalComputationValueType, Dim>::Pointer weightImage)
	{
		m_FixedWeightImage = weightImage;
	}

	virtual void SetMovingWeightImage(typename itk::Image<TInternalComputationValueType, Dim>::Pointer weightImage)
	{
		m_MovingWeightImage = weightImage;
	}

	virtual void SetFixedUnitWeightImage()
	{
		m_FixedWeightImage = itk::alphasmdinternal2::MakeConstantWeightImage<double, Dim>(m_FixedImage->GetLargestPossibleRegion(), m_FixedImage->GetSpacing(), 1.0);
	}

	virtual void SetMovingUnitWeightImage()
	{
		m_MovingWeightImage = itk::alphasmdinternal2::MakeConstantWeightImage<double, Dim>(m_MovingImage->GetLargestPossibleRegion(), m_MovingImage->GetSpacing(), 1.0);
	}

	virtual void SetFixedCircularWeightImage()
	{
		m_FixedWeightImage = itk::alphasmdinternal2::MakeCircularWeightImage<double, Dim>(m_FixedImage->GetLargestPossibleRegion(), m_FixedImage->GetSpacing(), 0.0, 1.0);
	}

	virtual void SetMovingCircularWeightImage()
	{
		m_MovingWeightImage = itk::alphasmdinternal2::MakeCircularWeightImage<double, Dim>(m_MovingImage->GetLargestPossibleRegion(), m_MovingImage->GetSpacing(), 0.0, 1.0);
	}

	virtual QType GetAlphaLevels() const
	{
		return m_AlphaLevels;
	}

	virtual void SetAlphaLevels(QType alphaLevels)
	{
		m_AlphaLevels = alphaLevels;
	}

	virtual bool GetSymmetricMeasure() const
	{
		return m_SymmetricMeasure;
	}

	virtual void SetSymmetricMeasure(bool symmetricMeasureFlag)
	{
		m_SymmetricMeasure = symmetricMeasureFlag;
	}

	virtual void SetFixedSamplingPercentage(double p)
	{
		m_FixedSamplingPercentage = p;
	}

	virtual void SetMovingSamplingPercentage(double p)
	{
		m_MovingSamplingPercentage = p;
	}

	virtual void SetSquaredMeasure(bool squaredMeasureFlag)
	{
		m_SquaredMeasure = squaredMeasureFlag;
	}

	virtual bool GetSquaredMeasure() const
	{
		return m_SquaredMeasure;
	}

	virtual void SetLinearInterpolation(bool linearInterpolationFlag)
	{
		m_LinearInterpolation = linearInterpolationFlag;
	}

	virtual bool GetLinearInterpolation() const
	{
		return m_LinearInterpolation;
	}

	virtual double GetMaxDistance() const
	{
		return m_MaxDistance;
	}

	virtual void SetMaxDistance(double d)
	{
		m_MaxDistance = d;
	}

	virtual bool GetUseQuasiRandomSampling() const {
		return m_DoQuasiRandomSampling;
	}

	virtual void SetUseQuasiRandomSampling(bool flag) {
		m_DoQuasiRandomSampling = flag;
	}

	virtual void SetRandomSeed(int seed)
	{
		m_Fixed_RNG->SetSeed(seed);
		m_Moving_RNG->SetSeed(seed);
		m_FixedQMCSampler.SetSeed(seed);
		m_MovingQMCSampler.SetSeed(seed);
		m_FixedQMCSampler.Restart();
		m_MovingQMCSampler.Restart();
	}
	// Do the pre-processing to set up the measure for evaluating the distance and derivative

	virtual void Update()
	{
		if (!m_FixedWeightImage)
		{
			SetFixedUnitWeightImage();
		}
		if (!m_MovingWeightImage)
		{
			SetMovingUnitWeightImage();
		}

		if (!m_FixedMask)
		{
			SetFixedMask(itk::alphasmdinternal2::MakeConstantWeightImage<bool, Dim>(m_FixedImage->GetLargestPossibleRegion(), m_FixedImage->GetSpacing(), true));
		}

		if (!m_MovingMask)
		{
			SetMovingMask(itk::alphasmdinternal2::MakeConstantWeightImage<bool, Dim>(m_MovingImage->GetLargestPossibleRegion(), m_MovingImage->GetSpacing(), true));
		}

		internals.SetRefImage(m_FixedImage);
		internals.SetFloImage(m_MovingImage);

		internals.SetRefMask(m_FixedMask);
		internals.SetFloMask(m_MovingMask);

		internals.SetRefWeight(m_FixedWeightImage);
		internals.SetFloWeight(m_MovingWeightImage);

		internals.SetAlphaLevels(m_AlphaLevels);
		internals.SetMaxDistance(m_MaxDistance);
		internals.SetSquaredMeasure(m_SquaredMeasure);
		internals.SetLinearInterpolation(m_LinearInterpolation);
		internals.SetSymmetric(m_SymmetricMeasure);

		internals.SetExcludeMaskedPoints(!m_DoQuasiRandomSampling);

		internals.Update();

		m_FixedSourcePoints = internals.GetRefSourcePoints();
		m_MovingSourcePoints = internals.GetFloSourcePoints();

		m_FixedQMCSampler.SetSize(m_FixedImage->GetLargestPossibleRegion().GetSize());
		m_MovingQMCSampler.SetSize(m_MovingImage->GetLargestPossibleRegion().GetSize());
		m_FixedRandomSampler.SetTotalIndices(m_FixedSourcePoints.size());
		m_MovingRandomSampler.SetTotalIndices(m_MovingSourcePoints.size());

		//m_Param = ParametersType(GetNumberOfParameters());
		UpdateAfterTransformChange();
	}

	// Must be called after setting the transformation
	virtual void UpdateAfterTransformChange()
	{
		auto n = GetNumberOfParameters();
		unsigned int supportSizeForward = GetTransformForwardPointer()->GetNumberOfAffectedWeights();
		unsigned int supportSizeInverse = GetTransformInversePointer()->GetNumberOfAffectedWeights();
		unsigned int supportSize = supportSizeForward < supportSizeInverse ? supportSizeInverse : supportSizeForward;

		m_Param = ParametersType(n);
		m_InternalDerivatives.dMetric = DerivativeType(n);
		m_InternalDerivatives.dSymmetry = DerivativeType(n);
		m_InternalDerivatives.wMetric = DerivativeType(n);
		m_InternalDerivatives.wSymmetry = DerivativeType(n);
		m_InternalDerivatives.splineWeights = WeightsType(supportSize);
		m_InternalDerivatives.splineWeightsInv = WeightsType(supportSize);
		m_InternalDerivatives.parameterIndices = ParameterIndexArrayType(supportSize);
		m_InternalDerivatives.parameterIndicesInv = ParameterIndexArrayType(supportSize);

		m_ForwardDerivativeTmp = DerivativeType(GetTransformForwardPointer()->GetNumberOfParameters());
		m_InverseDerivativeTmp = DerivativeType(GetTransformInversePointer()->GetNumberOfParameters());
	}

	virtual TransformPointer GetTransformForwardPointer() const
	{
		return m_TransformForward;
	}
	virtual TransformPointer GetTransformInversePointer() const
	{
		return m_TransformInverse;
	}

	virtual TransformType *GetTransformForwardRawPointer() const
	{
		return m_TransformForwardRawPtr;
	}
	virtual TransformType *GetTransformInverseRawPointer() const
	{
		return m_TransformInverseRawPtr;
	}

	virtual void SetForwardTransformPointer(TransformPointer transform)
	{
		m_TransformForward = transform;
		m_TransformForwardRawPtr = transform.GetPointer();
	}
	virtual void SetInverseTransformPointer(TransformPointer transform)
	{
		m_TransformInverse = transform;
		m_TransformInverseRawPtr = transform.GetPointer();
	}

	void SetSymmetryLambda(double lambda)
	{
		m_SymmetryLambda = lambda;
	}
	void SetSymmetryLambdaFromLearningRate(double learningRate, double fraction = 0.1)
	{
		double lambda = fraction / learningRate;
		if(lambda > 0.5)
			lambda = 0.5;
		m_SymmetryLambda = lambda;
	}

	bool GetDoRandomize() const
	{
		return m_DoRandomize;
	}

	void SetDoRandomize(bool flag)
	{
		m_DoRandomize = flag;
	}

	itk::Vector<TInternalComputationValueType, 2U> GetMetricValues() const
	{
		return m_MetricValues;
	}

  protected:
	AlphaSMDObjectToObjectMetricDeformv4() : m_AlphaLevels(7), m_SymmetricMeasure(true), m_FixedSamplingPercentage(1.0), m_MovingSamplingPercentage(1.0), m_SquaredMeasure(false), m_LinearInterpolation(true), m_MaxDistance(0.0)
	{
		m_Fixed_RNG = GeneratorType::New();
		m_Moving_RNG = GeneratorType::New();
		m_Fixed_RNG->SetSeed(42);
		m_Moving_RNG->SetSeed(42);
		m_SymmetryLambda = 0.0;
		m_DoRandomize = true;
		m_DoQuasiRandomSampling = true;
		m_FixedQMCSampler.SetSeed(42);
		m_MovingQMCSampler.SetSeed(42);
		m_FixedQMCSampler.Restart();
		m_MovingQMCSampler.Restart();
	}
	virtual ~AlphaSMDObjectToObjectMetricDeformv4() {}

	void SampleFixedPoints(double samplingPercentage) const {
		unsigned int count = (unsigned int)(samplingPercentage * m_FixedSourcePoints.size() + 0.5);

		if(m_DoQuasiRandomSampling) {
			//m_FixedQMCSampler.Restart();
			m_FixedQMCSampler.SampleIndices(count, m_FixedSourceIndices);
			//std::cout << "Point: " << m_FixedSourcePoints[m_FixedSourceIndices[count-1]].m_SourcePoint << std::endl;
		} else {
			m_FixedRandomSampler.SampleIndices(count, m_FixedSourceIndices);
		}
	}

	void SampleMovingPoints(double samplingPercentage) const {
		unsigned int count = (unsigned int)(samplingPercentage * m_MovingSourcePoints.size() + 0.5);

		if(m_DoQuasiRandomSampling) {
			//m_MovingQMCSampler.Restart();
			m_MovingQMCSampler.SampleIndices(count, m_MovingSourceIndices);
		} else {
			m_MovingRandomSampler.SampleIndices(count, m_MovingSourceIndices);
		}
	}

	int RandomShuffle(std::vector<SourcePointType> &ps, double samplingPercentage, GeneratorType *rng) const
	{

		int count = (int)(samplingPercentage * ps.size() + 0.5);

		if (count >= ps.size())
			return (int)ps.size();

		int sz = ps.size() - 1;

		if (m_DoRandomize == false)
			return count;

		for (int i = 0; i < count; ++i, --sz)
		{
			int index = i + rng->GetIntegerVariate(sz);

			assert(index >= i);
			assert(index < ps.size());

			if (index > i)
				std::swap(ps[index], ps[i]);
		}

		return count;
	}

	//
	// Metric Settings
	//

	QType m_AlphaLevels;

	// If true, both compute dist A to B and B to A, otherwise only from A to B
	bool m_SymmetricMeasure;

	// The fraction of the points in the fixed image (A) to sample
	double m_FixedSamplingPercentage;
	// The fraction of the points in the moving image (B) to sample
	double m_MovingSamplingPercentage;

	// Use the squared measure instead of the normal linear measure
	bool m_SquaredMeasure;

	// Use linear interpolation for the evaluation
	bool m_LinearInterpolation;

	// Maximal distance, which the distances are clipped to (0 means no max distance)
	double m_MaxDistance;

	double m_SymmetryLambda;

	mutable std::vector<alphasmdinternal2::PointToSetDistance<QType, Dim>> m_FixedPointToSetDistances;

	mutable std::vector<alphasmdinternal2::PointToSetDistance<QType, Dim>> m_MovingPointToSetDistances;

	mutable std::vector<SymmetryLossType> m_FixedSymmetryLosses;
	mutable std::vector<SymmetryLossType> m_MovingSymmetryLosses;

	mutable std::vector<SourcePointType> m_FixedSourcePoints;
	mutable std::vector<SourcePointType> m_MovingSourcePoints;

	mutable std::vector<unsigned int> m_FixedSourceIndices;
	mutable std::vector<unsigned int> m_MovingSourceIndices;

	// Images

	ImagePointer m_FixedImage;
	ImagePointer m_MovingImage;

	typename itk::Image<bool, Dim>::Pointer m_FixedMask;
	typename itk::Image<bool, Dim>::Pointer m_MovingMask;

	typename itk::Image<double, Dim>::Pointer m_FixedWeightImage;
	typename itk::Image<double, Dim>::Pointer m_MovingWeightImage;

	// Random number generator

	mutable QMCSampler<Dim> m_FixedQMCSampler;
	mutable QMCSampler<Dim> m_MovingQMCSampler;
	mutable RandomSampler m_FixedRandomSampler;
	mutable RandomSampler m_MovingRandomSampler;

	typename GeneratorType::Pointer m_Fixed_RNG;
	typename GeneratorType::Pointer m_Moving_RNG;

	bool m_DoRandomize;
	bool m_DoQuasiRandomSampling;

	// Transform

	mutable TransformPointer m_TransformForward;
	mutable TransformPointer m_TransformInverse;

	mutable TransformType *m_TransformForwardRawPtr;
	mutable TransformType *m_TransformInverseRawPtr;

	mutable alphasmdinternal2::AlphaSMDInternal<QType, Dim> internals;

	mutable InternalDerivatives<DerivativeType, WeightsType, ParameterIndexArrayType> m_InternalDerivatives;

	mutable DerivativeType m_ForwardDerivativeTmp;
	mutable DerivativeType m_InverseDerivativeTmp;

	mutable ParametersType m_Param;

	mutable itk::Vector<TInternalComputationValueType, 2U> m_MetricValues;
};
} // namespace itk

#endif
