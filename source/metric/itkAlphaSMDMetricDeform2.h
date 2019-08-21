
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

		// Methods

		virtual void Initialize() throw (ExceptionObject) override {

		}

private:
		inline
		void ComputeDerivative(TransformType* transform, DerivativeType &derivative, unsigned int derivative_offset, PointType point, itk::CovariantVector<float, Dim> spatial_grad, JacobianType& jac, JacobianType& jacPos) const {
			typedef JacobianType & JacobianReferenceType;

			/** For dense transforms, this returns identity */
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
public:

		//
		// Main method of the measure. Computes the value and derivative (in terms of the transform-parameters).
	    //
		virtual void GetValueAndDerivative(MeasureType &value, DerivativeType &derivative) const override {

			double distAcc = 0.0;

			derivative.Fill(0);

			// Shuffle the (Fixed) source points

			unsigned fixedCount = (unsigned int)RandomShuffle(m_FixedSourcePoints, m_FixedSamplingPercentage, m_Fixed_RNG.GetPointer());

			assert(fixedCount <= m_FixedSourcePoints.size());

			m_FixedPointToSetDistances.clear();
			m_FixedSymmetryLosses.clear();

			for(unsigned int i = 0; i < fixedCount; ++i) {
				SourcePointType sp = m_FixedSourcePoints[i];

				// Apply transformation (with centering)

				PointType transformedPoint = m_TransformForwardRawPtr->TransformPoint(sp.m_SourcePoint);

				PointType returnedPoint = m_TransformInverseRawPtr->TransformPoint(transformedPoint);

				SymmetryLossType slt;

				ComputeSymmetryLoss(sp.m_SourcePoint, returnedPoint, slt);

				alphasmdinternal2::PointToSetDistance<QType, Dim> ptsd;
				ptsd.m_SourcePoint = sp;

				internals.EvalFloPointToSetDistance(ptsd, transformedPoint);
				
				if(ptsd.m_IsValid)
				{
					m_FixedPointToSetDistances.push_back(ptsd);
					m_FixedSymmetryLosses.push_back(slt);
				}
			}

			fixedCount = static_cast<unsigned int>(m_FixedPointToSetDistances.size());

			double totalFixedWeight = 0.0;
			for(unsigned int i = 0; i < fixedCount; ++i) {
				totalFixedWeight += m_FixedPointToSetDistances[i].m_SourcePoint.m_Weight;
			}
			if(totalFixedWeight == 0.0)
				totalFixedWeight = 0.0;
			else
				totalFixedWeight = 1.0 / totalFixedWeight;

			if(m_SymmetricMeasure)
				totalFixedWeight *= 0.5;
			
			// Aggregate the distances (average of minimal distances)

			for(unsigned int i = 0; i < fixedCount; ++i) {
				alphasmdinternal2::PointToSetDistance<QType, Dim> ptsd = m_FixedPointToSetDistances[i];
				SymmetryLossType slt = m_FixedSymmetryLosses[i];
				TInternalComputationValueType tmpWeight = ptsd.m_SourcePoint.m_Weight * totalFixedWeight;
				
				distAcc += tmpWeight * ((1.0 - m_SymmetryLambda) * ptsd.m_Distance + m_SymmetryLambda * slt.value);

				// Combine data-term and symmetry-term
				for(unsigned int j = 0; j < Dim; ++j) {
					double alpha_distance = (1.0 - m_SymmetryLambda) * ptsd.m_SpatialGrad[j] * tmpWeight;
					double symmetry_loss = m_SymmetryLambda * slt.grad[j] * tmpWeight;
					ptsd.m_SpatialGrad[j] = alpha_distance + symmetry_loss;
				}

				ComputeDerivative(
					m_TransformForwardRawPtr,
					derivative,
					0U,
					ptsd.m_SourcePoint.m_SourcePoint,
					-ptsd.m_SpatialGrad,
					m_JacobianForward,
					m_JacobianForwardPos);
			}

			if(m_SymmetricMeasure) {
				// Shuffle the (Moving) source points

				unsigned int movingCount = (unsigned int)RandomShuffle(m_MovingSourcePoints, m_MovingSamplingPercentage, m_Moving_RNG.GetPointer());

				assert(movingCount <= m_MovingSourcePoints.size());

				m_MovingPointToSetDistances.clear();
				m_MovingSymmetryLosses.clear();

				for(unsigned int i = 0; i < movingCount; ++i) {
					SourcePointType sp = m_MovingSourcePoints[i];

					// Apply transformation (with centering)

					PointType transformedPoint = m_TransformInverseRawPtr->TransformPoint(sp.m_SourcePoint);
				
					PointType returnedPoint = m_TransformForwardRawPtr->TransformPoint(transformedPoint);

					SymmetryLossType slt;

					ComputeSymmetryLoss(sp.m_SourcePoint, returnedPoint, slt);

					alphasmdinternal2::PointToSetDistance<QType, Dim> ptsd;
					ptsd.m_SourcePoint = sp;

					internals.EvalRefPointToSetDistance(ptsd, transformedPoint);
				
					if(ptsd.m_IsValid)
					{
						m_MovingPointToSetDistances.push_back(ptsd);
						m_MovingSymmetryLosses.push_back(slt);
					}
				}

				movingCount = static_cast<unsigned int>(m_MovingPointToSetDistances.size());
				
				double totalMovingWeight = 0.0;
				for(unsigned int i = 0; i < movingCount; ++i) {
					totalMovingWeight += m_MovingPointToSetDistances[i].m_SourcePoint.m_Weight;
				}
				if(totalMovingWeight == 0.0)
					totalMovingWeight = 0.0;
				else
					totalMovingWeight = 1.0 / totalMovingWeight;

				totalMovingWeight *= 0.5;
				
				// Aggregate the distances (average of minimal distances)

				for(unsigned int i = 0; i < movingCount; ++i) {
					alphasmdinternal2::PointToSetDistance<QType, Dim> ptsd = m_MovingPointToSetDistances[i];
					SymmetryLossType slt = m_MovingSymmetryLosses[i];
					TInternalComputationValueType tmpWeight = ptsd.m_SourcePoint.m_Weight * totalMovingWeight;
					
					//distAcc += tmpWeight * (1.0 - m_SymmetryLambda) * ptsd.m_Distance + m_SymmetryLambda * slt.value * tmpWeight;
					distAcc += tmpWeight * ((1.0 - m_SymmetryLambda) * ptsd.m_Distance + m_SymmetryLambda * slt.value);

					// Combine data-term and symmetry-term
					for(unsigned int j = 0; j < Dim; ++j) {
						double alpha_distance = (1.0 - m_SymmetryLambda) * ptsd.m_SpatialGrad[j] * tmpWeight;
						double symmetry_loss = m_SymmetryLambda * slt.grad[j] * tmpWeight;
						ptsd.m_SpatialGrad[j] = alpha_distance + symmetry_loss;
					}

					ComputeDerivative(
						m_TransformInverseRawPtr,
						derivative,
						m_TransformForwardRawPtr->GetNumberOfParameters(),
						ptsd.m_SourcePoint.m_SourcePoint,
						-ptsd.m_SpatialGrad,
						m_JacobianInverse,
						m_JacobianInversePos);
				}
			}

			value = distAcc;
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
		}
		virtual void SetInverseTransformPointer(TransformPointer transform) {
			m_TransformInverse = transform;
			m_TransformInverseRawPtr = transform.GetPointer();
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
