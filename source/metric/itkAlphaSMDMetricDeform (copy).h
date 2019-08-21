
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

#include "itkAlphaSMDMetricInternal2.h"

namespace itk {
	template <typename ValueType, unsigned int Dim>
	struct SymmetryLossTerm {
		ValueType value;
		itk::CovariantVector<ValueType, Dim> grad;
	};

	/**
	* Metric which computes the Alpha-SMD Fuzzy Set Distance between a pair of Fixed and Moving images.
	* Since it does not satisfy the assumptions of the ImageToImageMetric-base classes, the measure
	* is implemented as an ObjectToObjectMetric instead.
	*/
	template <typename ImageType, unsigned int Dim, typename TInternalComputationValueType = double, unsigned int SplineOrder = 3>
	class AlphaSMDObjectToObjectMetricDeformv4 :
		public ObjectToObjectMetricBaseTemplate<TInternalComputationValueType>
	{
	public:
		/** Standard class typedefs. */
		typedef AlphaSMDObjectToObjectMetricDeformv4                            Self;
		typedef ObjectToObjectMetricBaseTemplate<TInternalComputationValueType> Superclass;
		typedef SmartPointer<Self>                                              Pointer;
		typedef SmartPointer<const Self>                                        ConstPointer;

		/** Method for creation through the object factory. */
		itkNewMacro(Self);

		/** Run-time type information (and related methods). */
		itkTypeMacro(AlphaSMDObjectToObjectMetricDeformv4, ObjectToObjectMetricBaseTemplate);

		typedef itk::Statistics::MersenneTwisterRandomVariateGenerator GeneratorType;

		typedef typename ImageType::Pointer                  ImagePointer;
		typedef typename Superclass::DerivativeType          DerivativeType;
		typedef typename Superclass::DerivativeValueType     DerivativeValueType;

		typedef typename Superclass::NumberOfParametersType  NumberOfParametersType;
		typedef typename Superclass::ParametersType          ParametersType;

		typedef typename ImageType::IndexType                IndexType;
		typedef typename ImageType::PointType                PointType;
		typedef typename std::vector<PointType>              PointSetType;

		typedef typename Superclass::MeasureType MeasureType;

		typedef MeasureType ValueType;

		typedef double ParametersValueType;

		typedef itk::BSplineTransform<double, Dim, SplineOrder>		 TransformType;
		typedef typename TransformType::Pointer              TransformPointer;
		typedef typename TransformType::JacobianType         JacobianType;

		typedef unsigned char 								  QType;
		typedef alphasmdinternal2::SourcePoint<QType, Dim>    SourcePointType;
		
		typedef SymmetryLossTerm<double, Dim> SymmetryLossType;
		
		typedef typename TransformType::WeightsType WeightsType;
		typedef typename TransformType::ParameterIndexArrayType ParameterIndexArrayType;

		// Methods

		virtual void Initialize() throw (ExceptionObject) override {

		}

private:
		inline
		void ComputeDerivativeBSpline(
			TransformType* transform,
			DerivativeType &derivative,
			unsigned int derivative_offset,
			PointType point,
			itk::CovariantVector<float, Dim> spatial_grad,
			WeightsType & splineWeights,
			ParameterIndexArrayType & parameterIndices,
			unsigned int paramPerDim,
			unsigned int supportSize) const {

			transform->ComputeJacobianFromBSplineWeightsWithRespectToPosition(point, splineWeights, parameterIndices);
			for (unsigned int dim=0; dim < Dim; ++dim) {
				unsigned int off = derivative_offset + dim * paramPerDim;
				for (unsigned int mu = 0; mu < supportSize; ++mu) {
					unsigned int par_ind = off + parameterIndices[mu];
					derivative[par_ind] += splineWeights[mu] * spatial_grad[dim];
				}
			}
		}
		inline
		void ComputeDerivativeBSplineAndWeights(
			TransformType* transform,
			DerivativeType &derivative,
			DerivativeType &derivative_weights,
			unsigned int derivative_offset,
			PointType point,
			itk::CovariantVector<float, Dim> spatial_grad,
			double pointWeight,
			WeightsType & splineWeights,
			ParameterIndexArrayType & parameterIndices,
			unsigned int paramPerDim,
			unsigned int supportSize) const {

			transform->ComputeJacobianFromBSplineWeightsWithRespectToPosition(point, splineWeights, parameterIndices);
			for (unsigned int dim=0; dim < Dim; ++dim) {
				unsigned int off = derivative_offset + dim * paramPerDim;
				for (unsigned int mu = 0; mu < supportSize; ++mu) {
					unsigned int par_ind = off + parameterIndices[mu];
					double sw = splineWeights[mu];
					derivative[par_ind] += sw * spatial_grad[dim];
					derivative_weights[par_ind] += sw * pointWeight;
				}
			}
		}
		inline
		void ComputeDerivative(TransformType* transform, DerivativeType &derivative, unsigned int derivative_offset, PointType point, itk::CovariantVector<float, Dim> spatial_grad, JacobianType& jac, JacobianType& jacPos) const {
			typedef JacobianType & JacobianReferenceType;

			// For dense transforms, this returns identity //
			transform->ComputeJacobianWithRespectToParametersCachedTemporaries(point, jac, jacPos);
			unsigned int param_count = transform->GetNumberOfParameters();

			for (unsigned int par = 0; par < param_count; par++) {
				unsigned int par_ind = derivative_offset + par;
				for ( SizeValueType dim = 0; dim < Dim; dim++) {
					derivative[par_ind] += spatial_grad[dim] * jac(dim, par);
				}
			}
		}

		inline
		void ComputeSymmetryLoss(PointType originalPoint, PointType returnedPoint, SymmetryLossType& out) const {
			itk::CovariantVector<double, Dim> deltaVector;
			for(unsigned int j = 0; j < Dim; ++j) {
				deltaVector[j] = returnedPoint[j] - originalPoint[j];
			}

			double vecNorm = deltaVector.GetNorm();
			double multiplier = 2.0;// * vecNorm;		

			for(unsigned int j = 0; j < Dim; ++j) {
				deltaVector[j] = multiplier * deltaVector[j];//(deltaVector[j] / (vecNorm + 1e-15));
			}

			out.value = vecNorm * vecNorm;
			out.grad = deltaVector;
		}

	double EvaluateAsymmetricMetric(MeasureType &value, DerivativeType &derivative, bool forwardDirection, TransformType* transformForward, TransformType* transformInverse, JacobianType& jac, JacobianType& jacPos, std::vector<SourcePointType>& srcPoints, unsigned int count) const {
	  double distAcc = 0.0;
	  double totalWeight = 0.0;
	  double nodes = transformForward->GetNumberOfParameters() / (double)Dim;

		//unsigned int supportSize = transform->GetNumberOfAffectedWeights();
		unsigned paramPerDim = transformForward->GetNumberOfParametersPerDimension();
		unsigned int supportSize = transformForward->GetNumberOfAffectedWeights();

		WeightsType splineWeights(supportSize);
		ParameterIndexArrayType parameterIndices(supportSize); 
		
		DerivativeType derivative_weights(derivative.GetSize());
	    derivative_weights.Fill(0.0);
      
	  unsigned int derivativeIndexOffset = forwardDirection ? 0U : transformForward->GetNumberOfParameters();

	  //double derivativeScaling = (transformForward->GetNumberOfParameters() / (double)Dim) / pow(SplineOrder+1.0, Dim);
	  double derivativeScaling = nodes;// / pow(SplineOrder+1.0, Dim);
			for(unsigned int i = 0; i < count; ++i) {
				SourcePointType sp = srcPoints[i];

				// Apply transformation

				PointType transformedPoint;
				bool isInside;

				transformForward->TransformPoint(sp.m_SourcePoint, transformedPoint, splineWeights, parameterIndices, isInside);
				if(!isInside)
					continue;
				//PointType transformedPoint = transformForward->TransformPoint(sp.m_SourcePoint);

				PointType returnedPoint = transformInverse->TransformPoint(transformedPoint);

				SymmetryLossType slt;

				ComputeSymmetryLoss(sp.m_SourcePoint, returnedPoint, slt);

				alphasmdinternal2::PointToSetDistance<QType, Dim> ptsd;
				ptsd.m_SourcePoint = sp;

				if(forwardDirection) {
				    internals.EvalFloPointToSetDistance(ptsd, transformedPoint);
				} else {
				    internals.EvalRefPointToSetDistance(ptsd, transformedPoint);
				}
				
				if(ptsd.m_IsValid)
				{
				TInternalComputationValueType w = ptsd.m_SourcePoint.m_Weight;
				
				distAcc += w * ((1.0 - m_SymmetryLambda) * ptsd.m_Distance + m_SymmetryLambda * slt.value);

				// Combine data-term and symmetry-term
				for(unsigned int j = 0; j < Dim; ++j) {
					double alpha_distance_grad = (1.0 - m_SymmetryLambda) * ptsd.m_SpatialGrad[j] * w;
					double symmetry_loss_grad = m_SymmetryLambda * slt.grad[j] * w;
					ptsd.m_SpatialGrad[j] = alpha_distance_grad + symmetry_loss_grad;
				}
				// Compute jacobian
				for (unsigned int dim=0; dim < Dim; ++dim) {
					unsigned int off = derivativeIndexOffset + dim * paramPerDim;
					for (unsigned int mu = 0; mu < supportSize; ++mu) {
						unsigned int par_ind = off + parameterIndices[mu];
						double sw = splineWeights[mu];
						derivative[par_ind] -= sw * ptsd.m_SpatialGrad[dim];
						derivative_weights[par_ind] += sw * w;
					}
				}

				/*
				ComputeDerivativeBSplineAndWeights(
					transformForward,
					derivative,
					derivative_weights,
					derivativeIndexOffset,
					ptsd.m_SourcePoint.m_SourcePoint,
					-ptsd.m_SpatialGrad,
					w,
					splineWeights,
					parameterIndices,
					paramPerDim,
					supportSize);*/
				/*
				ComputeDerivativeBSpline(
					transformForward,
					derivative,
					derivativeIndexOffset,
					ptsd.m_SourcePoint.m_SourcePoint,
					-ptsd.m_SpatialGrad,
					splineWeights,
					parameterIndices,
					paramPerDim,
					supportSize);
					*/
				/*
				ComputeDerivative(
					transformForward,
					derivative,
					derivativeIndexOffset,
					ptsd.m_SourcePoint.m_SourcePoint,
					-ptsd.m_SpatialGrad,
					jac,
					jacPos);*/
				  
				totalWeight += w;
				}

			}

		double smoothingFactor = 0.1;
		for(unsigned int i = 0; i < transformForward->GetNumberOfParameters(); ++i) {
			unsigned int par_ind = derivativeIndexOffset + i;
			double der = derivative[par_ind];
			double parWeight = derivative_weights[par_ind];
			double recWeight = 1.0 / (parWeight+smoothingFactor);
			derivative[par_ind] = der * recWeight;
		}

			if(totalWeight == 0.0)
				totalWeight = 0.0;
			else
				totalWeight = 1.0 / totalWeight;

	  value += distAcc * totalWeight;
	  
	  //return totalWeight * derivativeScaling;
	  return 1.0;
	  
     }
	
public:

		//
		// Main method of the measure. Computes the value and derivative (in terms of the transform-parameters).
	    //
		virtual void GetValueAndDerivative(MeasureType &value, DerivativeType &derivative) const override {

			double distAcc = 0.0;

			value = 0.0;
			derivative.Fill(0);

			double totalFixedWeight;
			double totalMovingWeight;
			
			// Shuffle the (Fixed) source points

			unsigned fixedCount = (unsigned int)RandomShuffle(m_FixedSourcePoints, m_FixedSamplingPercentage, m_Fixed_RNG.GetPointer());

			assert(fixedCount <= m_FixedSourcePoints.size());

			totalFixedWeight = EvaluateAsymmetricMetric(value, derivative, true, m_TransformForwardRawPtr, m_TransformInverseRawPtr, m_JacobianForward, m_JacobianForwardPos, m_FixedSourcePoints, fixedCount);
			
			if(m_SymmetricMeasure) {
				// Shuffle the (Moving) source points

				unsigned int movingCount = (unsigned int)RandomShuffle(m_MovingSourcePoints, m_MovingSamplingPercentage, m_Moving_RNG.GetPointer());

				assert(movingCount <= m_MovingSourcePoints.size());

				totalMovingWeight = EvaluateAsymmetricMetric(value, derivative, false, m_TransformInverseRawPtr, m_TransformForwardRawPtr, m_JacobianInverse, m_JacobianInversePos, m_MovingSourcePoints, movingCount);

				value *= 0.5;
				totalFixedWeight *= 0.5;
				totalMovingWeight *= 0.5;
			}
			
			for(unsigned int i = 0; i < GetNumberOfParameters()/2; ++i) {
			  derivative[i] = derivative[i] * totalFixedWeight;
			}
			for(unsigned int i = GetNumberOfParameters()/2; i < GetNumberOfParameters(); ++i) {
			  derivative[i] = derivative[i] * totalMovingWeight;
			}
		}

		// Should be updated with a faster version
		virtual MeasureType GetValue() const override {
			MeasureType result;
			DerivativeType tmp;

			GetValueAndDerivative(result, tmp);

			return result;
		}

		virtual void GetDerivative(DerivativeType& derivative) const override {
			MeasureType tmp;

			GetValueAndDerivative(tmp, derivative);
		}

		virtual void UpdateTransformParameters(
			const DerivativeType &derivative,
			ParametersValueType factor = NumericTraits< ParametersValueType >::OneValue()) override {

			const ParametersType& param1 = m_TransformForwardRawPtr->GetParameters();
			const ParametersType& param2 = m_TransformInverseRawPtr->GetParameters();

			ParametersType newParam1(m_TransformForwardRawPtr->GetNumberOfParameters());
			ParametersType newParam2(m_TransformInverseRawPtr->GetNumberOfParameters());

			NumberOfParametersType numberOfParameters = GetNumberOfParameters();

			if (factor == NumericTraits< ParametersValueType >::OneValue()) {
				unsigned int inIndex = 0;
				for (unsigned int i = 0; i < m_TransformForwardRawPtr->GetNumberOfParameters(); ++i, ++inIndex) {
					newParam1[i] = param1[i] + derivative[inIndex];
					//m_TransformForwardRawPtr->SetParam(i, m_TransformForwardRawPtr->GetParam(i) + derivative[inIndex]);
				}
				for (unsigned int i = 0; i < m_TransformInverseRawPtr->GetNumberOfParameters(); ++i, ++inIndex) {
					newParam2[i] = param2[i] + derivative[inIndex];
				}
			}
			else {
				unsigned int inIndex = 0;
				for (unsigned int i = 0; i < m_TransformForwardRawPtr->GetNumberOfParameters(); ++i, ++inIndex) {
					newParam1[i] = param1[i] + derivative[inIndex] * factor;
				}
				for (unsigned int i = 0; i < m_TransformInverseRawPtr->GetNumberOfParameters(); ++i, ++inIndex) {
					newParam2[i] = param2[i] + derivative[inIndex] * factor;
				}
			}

			m_TransformForwardRawPtr->SetParameters(newParam1);
			m_TransformInverseRawPtr->SetParameters(newParam2);
		}

		virtual const ParametersType& GetParameters() const override {
			NumberOfParametersType num = GetNumberOfParameters();

			const ParametersType& param1 = m_TransformForwardRawPtr->GetParameters();
			const ParametersType& param2 = m_TransformInverseRawPtr->GetParameters();

			unsigned int outIndex = 0;
			for (unsigned int i = 0; i < m_TransformForwardRawPtr->GetNumberOfParameters(); ++i, ++outIndex) {
				m_Param[outIndex] = param1[i];
			}
			for (unsigned int i = 0; i < m_TransformInverseRawPtr->GetNumberOfParameters(); ++i, ++outIndex) {
				m_Param[outIndex] = param2[i];
			}
						
			return m_Param;
		}

		virtual void SetParameters(ParametersType &parameters) override {
			assert(parameters.Size() == GetNumberOfParameters());

			ParametersType newParam1(m_TransformForwardRawPtr->GetNumberOfParameters());
			ParametersType newParam2(m_TransformInverseRawPtr->GetNumberOfParameters());

			unsigned int outIndex = 0;
			for (unsigned int i = 0; i < m_TransformForwardRawPtr->GetNumberOfParameters(); ++i, ++outIndex) {
				newParam1[i] = parameters[outIndex];
			}
			for (unsigned int i = 0; i < m_TransformInverseRawPtr->GetNumberOfParameters(); ++i, ++outIndex) {
				newParam2[i] = parameters[outIndex];
			}

			m_TransformForwardRawPtr->SetParameters(newParam1);
			m_TransformInverseRawPtr->SetParameters(newParam2);
		}

		virtual NumberOfParametersType GetNumberOfParameters() const override { return m_TransformForwardRawPtr->GetNumberOfParameters() + m_TransformInverseRawPtr->GetNumberOfParameters(); }

		virtual NumberOfParametersType GetNumberOfLocalParameters() const override { return GetNumberOfParameters(); }

		virtual bool HasLocalSupport() const override { return false; }

		// Other

		virtual ImagePointer GetFixedImage() const {
			return m_FixedImage;
		}

		virtual void SetFixedImage(ImagePointer fixedImage) {
			m_FixedImage = fixedImage;
		}

		virtual ImagePointer GetMovingImage() const {
			return m_MovingImage;
		}

		virtual void SetMovingImage(ImagePointer movingImage) {
			m_MovingImage = movingImage;
		}

		virtual void SetFixedMask(typename itk::Image<bool, Dim>::Pointer fixedMask) {
			m_FixedMask = fixedMask;
		}

		virtual void SetMovingMask(typename itk::Image<bool, Dim>::Pointer movingMask) {
			m_MovingMask = movingMask;
		}

		virtual void SetFixedWeightImage(typename itk::Image<TInternalComputationValueType, Dim>::Pointer weightImage) {
			m_FixedWeightImage = weightImage;
		}

		virtual void SetMovingWeightImage(typename itk::Image<TInternalComputationValueType, Dim>::Pointer weightImage) {
			m_MovingWeightImage = weightImage;
		}

		virtual void SetFixedUnitWeightImage() {
			m_FixedWeightImage = itk::alphasmdinternal2::MakeConstantWeightImage<double, Dim>(m_FixedImage->GetLargestPossibleRegion(), m_FixedImage->GetSpacing(), 1.0);
		}

		virtual void SetMovingUnitWeightImage() {
			m_MovingWeightImage = itk::alphasmdinternal2::MakeConstantWeightImage<double, Dim>(m_MovingImage->GetLargestPossibleRegion(), m_MovingImage->GetSpacing(), 1.0);
		}

		virtual void SetFixedCircularWeightImage() {
			m_FixedWeightImage = itk::alphasmdinternal2::MakeCircularWeightImage<double, Dim>(m_FixedImage->GetLargestPossibleRegion(), m_FixedImage->GetSpacing(), 0.0, 1.0);
		}

		virtual void SetMovingCircularWeightImage() {
			m_MovingWeightImage = itk::alphasmdinternal2::MakeCircularWeightImage<double, Dim>(m_MovingImage->GetLargestPossibleRegion(), m_MovingImage->GetSpacing(), 0.0, 1.0);
		}

		virtual QType GetAlphaLevels() const {
			return m_AlphaLevels;
		}

		virtual void SetAlphaLevels(QType alphaLevels) {
			m_AlphaLevels = alphaLevels;
		}

		virtual bool GetSymmetricMeasure() const {
			return m_SymmetricMeasure;
		}

		virtual void SetSymmetricMeasure(bool symmetricMeasureFlag) {
			m_SymmetricMeasure = symmetricMeasureFlag;
		}

		virtual void SetFixedSamplingPercentage(double p) {
			m_FixedSamplingPercentage = p;
		}

		virtual void SetMovingSamplingPercentage(double p) {
			m_MovingSamplingPercentage = p;
		}
        
		virtual void SetSquaredMeasure(bool squaredMeasureFlag) {
			m_SquaredMeasure = squaredMeasureFlag;
		}

		virtual bool GetSquaredMeasure() const {
			return m_SquaredMeasure;
		}

		virtual void SetLinearInterpolation(bool linearInterpolationFlag) {
			m_LinearInterpolation = linearInterpolationFlag;
		}

		virtual bool GetLinearInterpolation() const {
			return m_LinearInterpolation;
		}

		virtual double GetMaxDistance() const {
			return m_MaxDistance;
		}

		virtual void SetMaxDistance(double d) {
			m_MaxDistance = d;
		}

		virtual void SetRandomSeed(int seed) {
			m_Fixed_RNG->SetSeed(seed);
			m_Moving_RNG->SetSeed(seed);
		}
		// Do the pre-processing to set up the measure for evaluating the distance and derivative

		virtual void Update() {
			if(!m_FixedWeightImage) {
				SetFixedUnitWeightImage();
			}
			if(!m_MovingWeightImage) {
				SetMovingUnitWeightImage();
			}

			if(!m_FixedMask) {
				SetFixedMask(itk::alphasmdinternal2::MakeConstantWeightImage<bool, Dim>(m_FixedImage->GetLargestPossibleRegion(), m_FixedImage->GetSpacing(), true));
			}

			if(!m_MovingMask) {
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

			internals.Update();

			m_FixedSourcePoints = internals.GetRefSourcePoints();
			m_MovingSourcePoints = internals.GetFloSourcePoints();

			//m_Param = ParametersType(GetNumberOfParameters());
			UpdateAfterTransformChange();
		}

		// Must be called after setting the transformation
		virtual void UpdateAfterTransformChange() {
			m_Param = ParametersType(GetNumberOfParameters());
		}

		virtual TransformPointer GetTransformForwardPointer() const {
			return m_TransformForward;
		}
		virtual TransformPointer GetTransformInversePointer() const {
			return m_TransformInverse;
		}

		virtual TransformType* GetTransformForwardRawPointer() const {
			return m_TransformForwardRawPtr;
		}
		virtual TransformType* GetTransformInverseRawPointer() const {
			return m_TransformInverseRawPtr;
		}

		virtual void SetForwardTransformPointer(TransformPointer transform) {
			m_TransformForward = transform;
			m_TransformForwardRawPtr = transform.GetPointer();
			m_JacobianForward.SetSize(Dim, transform->GetNumberOfParameters());
			m_JacobianForwardPos.SetSize(Dim, transform->GetNumberOfParameters());
		}
		virtual void SetInverseTransformPointer(TransformPointer transform) {
			m_TransformInverse = transform;
			m_TransformInverseRawPtr = transform.GetPointer();
			m_JacobianInverse.SetSize(Dim, transform->GetNumberOfParameters());
			m_JacobianInversePos.SetSize(Dim, transform->GetNumberOfParameters());
		}

		void SetSymmetryLambda(double lambda) {
			m_SymmetryLambda = lambda;
		}
	protected:
		AlphaSMDObjectToObjectMetricDeformv4() : m_AlphaLevels(7), m_SymmetricMeasure(true), m_FixedSamplingPercentage(1.0), m_MovingSamplingPercentage(1.0), m_SquaredMeasure(false), m_LinearInterpolation(false), m_MaxDistance(0.0) {
			m_Fixed_RNG = GeneratorType::New();
			m_Moving_RNG = GeneratorType::New();
			m_Fixed_RNG->SetSeed(42);
			m_Moving_RNG->SetSeed(42);
			m_SymmetryLambda = 0.0;
		}
		virtual ~AlphaSMDObjectToObjectMetricDeformv4() {}

		int RandomShuffle(std::vector<SourcePointType>& ps, double samplingPercentage, GeneratorType* rng) const {
			int count = (int)(samplingPercentage * ps.size() + 0.5);

			if(count >= ps.size())
				return (int)ps.size();

			int sz = ps.size() - 1;

			for(int i = 0; i < count; ++i, --sz) {
				int index = i + rng->GetIntegerVariate(sz);
				
				assert(index >= i);
				assert(index < ps.size());

				if(index > i)
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

		mutable std::vector<alphasmdinternal2::PointToSetDistance<QType, Dim> > m_FixedPointToSetDistances;
		
		mutable std::vector<alphasmdinternal2::PointToSetDistance<QType, Dim> > m_MovingPointToSetDistances;
		
		mutable std::vector<SymmetryLossType> m_FixedSymmetryLosses;
		mutable std::vector<SymmetryLossType> m_MovingSymmetryLosses;

		mutable std::vector<SourcePointType> m_FixedSourcePoints;
		mutable std::vector<SourcePointType> m_MovingSourcePoints;
		
		// Images

		ImagePointer m_FixedImage;
		ImagePointer m_MovingImage;

		typename itk::Image<bool, Dim>::Pointer m_FixedMask;
		typename itk::Image<bool, Dim>::Pointer m_MovingMask;

		typename itk::Image<double, Dim>::Pointer m_FixedWeightImage;
		typename itk::Image<double, Dim>::Pointer m_MovingWeightImage;

		// Random number generator

		typename GeneratorType::Pointer m_Fixed_RNG;
		typename GeneratorType::Pointer m_Moving_RNG;

		// Transform

		mutable TransformPointer m_TransformForward;
		mutable TransformPointer m_TransformInverse;

		mutable TransformType* m_TransformForwardRawPtr;
		mutable TransformType* m_TransformInverseRawPtr;

		mutable alphasmdinternal2::AlphaSMDInternal<QType, Dim> internals;

		mutable JacobianType m_JacobianForward;
		mutable JacobianType m_JacobianForwardPos;
		mutable JacobianType m_JacobianInverse;
		mutable JacobianType m_JacobianInversePos;

		mutable ParametersType m_Param;
	};
}

#endif
