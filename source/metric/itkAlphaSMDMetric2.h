
#ifndef ALPHA_SMD_METRIC2_H
#define ALPHA_SMD_METRIC2_H

#include <vector>
#include "itkOptimizerParameters.h"
#include "itkObjectToObjectMetricBase.h"
#include "itkAffineTransform.h"
#include "itkTranslationTransform.h"
#include "itkCompositeTransform.h"

#include "itkMersenneTwisterRandomVariateGenerator.h"

#include "itkAlphaSMDMetricInternal2.h"
#include "../transforms/itkAlphaSMDAffineTransform.h"

namespace itk {
	/**
	* Metric which computes the Alpha-SMD Fuzzy Set Distance between a pair of Fixed and Moving images.
	* Since it does not satisfy the assumptions of the ImageToImageMetric-base classes, the measure
	* is implemented as an ObjectToObjectMetric instead.
	*/
	template <typename ImageType, unsigned int Dim, typename TInternalComputationValueType = double, typename SymmetricTransformType = itk::AlphaSMDAffineTransform<double, Dim> >
	class AlphaSMDObjectToObjectMetric2v4 :
		public ObjectToObjectMetricBaseTemplate<TInternalComputationValueType>
	{
	public:
		/** Standard class typedefs. */
		typedef AlphaSMDObjectToObjectMetric2v4                         Self;
		typedef ObjectToObjectMetricBaseTemplate<TInternalComputationValueType> Superclass;
		typedef SmartPointer<Self>                                              Pointer;
		typedef SmartPointer<const Self>                                        ConstPointer;

		/** Method for creation through the object factory. */
		itkNewMacro(Self);

		/** Run-time type information (and related methods). */
		itkTypeMacro(AlphaSMDObjectToObjectMetricv4, ObjectToObjectMetricBaseTemplate);

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

		typedef double ParametersValueType;

		typedef SymmetricTransformType 		     			 TransformType;
//		typedef typename SymmetricTransformType::Pointer     TransformPointer;

		typedef unsigned char 								 QType;
		typedef alphasmdinternal2::SourcePoint<QType, Dim>    SourcePointType;

		// Interface implementation

		virtual void Initialize() throw (ExceptionObject) override {

		}

		//
		// Main method of the measure. Computes the value and derivative (in terms of the transform-parameters).
	    //
		virtual void GetValueAndDerivative(MeasureType &value, DerivativeType &derivative) const override {

			double distAcc = 0.0;

			derivative.Fill(0);

			m_Transform.Begin();

			// Shuffle the (Fixed) source points

			unsigned int fixedCount = (unsigned int)RandomShuffle(m_FixedSourcePoints, m_FixedSamplingPercentage);

			assert(fixedCount <= m_FixedSourcePoints.size());

			m_FixedPointToSetDistances.clear();

			for(unsigned int i = 0; i < fixedCount; ++i) {
				SourcePointType sp = m_FixedSourcePoints[i];

				// Apply transformation (with centering)

				PointType transformedPoint = m_Transform.TransformForward(sp.m_SourcePoint - m_A_CenterPoint);
				for(unsigned int d = 0; d < Dim; ++d) {
					transformedPoint[d] = transformedPoint[d] + m_B_CenterPoint[d];
				}

				alphasmdinternal2::PointToSetDistance<QType, Dim> ptsd;
				ptsd.m_SourcePoint = sp;

				internals.EvalFloPointToSetDistance(ptsd, transformedPoint);
				
				if(ptsd.m_IsValid)
				{
					m_FixedPointToSetDistances.push_back(ptsd);
				}
			}

			fixedCount = static_cast<unsigned int>(m_FixedPointToSetDistances.size());

			unsigned int fixedOutlierRejectionCount = (unsigned int)(fixedCount * m_OutlierRejectionPercentage + 0.5);

			if(fixedOutlierRejectionCount > 1) {
				assert(fixedOutlierRejectionCount <= fixedCount);

				std::sort(m_FixedPointToSetDistances.begin(), m_FixedPointToSetDistances.end());

				for(size_t i = fixedCount - fixedOutlierRejectionCount; i < fixedCount; ++i) {
					m_FixedPointToSetDistances[i].m_Distance *= m_OutlierRejectionWeight;
					m_FixedPointToSetDistances[i].m_SpatialGrad *= m_OutlierRejectionWeight;
				}
			}

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
			
			// Aggregate the distances (sum of minimal distances)

			for(unsigned int i = 0; i < fixedCount; ++i) {
				alphasmdinternal2::PointToSetDistance<QType, Dim> ptsd = m_FixedPointToSetDistances[i];
				TInternalComputationValueType tmpWeight = ptsd.m_SourcePoint.m_Weight * totalFixedWeight;
				
				distAcc += ptsd.m_Distance * tmpWeight;

				m_Transform.TotalDiffForward(
					ptsd.m_SourcePoint.m_SourcePoint - m_A_CenterPoint,
					-ptsd.m_SpatialGrad,
					derivative,
					tmpWeight,
					true);
			}

			if(m_SymmetricMeasure) {
				// Shuffle the (Moving) source points

				unsigned int movingCount = (unsigned int)RandomShuffle(m_MovingSourcePoints, m_MovingSamplingPercentage);

				assert(movingCount <= m_MovingSourcePoints.size());

				m_MovingPointToSetDistances.clear();

				for(unsigned int i = 0; i < movingCount; ++i) {
					SourcePointType sp = m_MovingSourcePoints[i];

					// Apply transformation (with centering)

					PointType transformedPoint = m_Transform.TransformInverse(sp.m_SourcePoint - m_B_CenterPoint);
					for(unsigned int d = 0; d < Dim; ++d) {
						transformedPoint[d] = transformedPoint[d] + m_A_CenterPoint[d];
					}

					alphasmdinternal2::PointToSetDistance<QType, Dim> ptsd;
					ptsd.m_SourcePoint = sp;

					internals.EvalRefPointToSetDistance(ptsd, transformedPoint);
				
					if(ptsd.m_IsValid)
					{
						m_MovingPointToSetDistances.push_back(ptsd);
					}
				}

				movingCount = static_cast<unsigned int>(m_MovingPointToSetDistances.size());

				unsigned int movingOutlierRejectionCount = (unsigned int)(movingCount * m_OutlierRejectionPercentage + 0.5);

				if(movingOutlierRejectionCount > 1) {
					assert(movingOutlierRejectionCount <= movingCount);

					std::sort(m_MovingPointToSetDistances.begin(), m_MovingPointToSetDistances.end());

					for(size_t i = movingCount - movingOutlierRejectionCount; i < movingCount; ++i) {
						m_MovingPointToSetDistances[i].m_Distance *= m_OutlierRejectionWeight;
						m_MovingPointToSetDistances[i].m_SpatialGrad *= m_OutlierRejectionWeight;
					}
				}
					
				double totalMovingWeight = 0.0;
				for(unsigned int i = 0; i < movingCount; ++i) {
					totalMovingWeight += m_MovingPointToSetDistances[i].m_SourcePoint.m_Weight;
				}
				if(totalMovingWeight == 0.0)
					totalMovingWeight = 0.0;
				else
					totalMovingWeight = 1.0 / totalMovingWeight;

				totalMovingWeight *= 0.5;
				
				// Aggregate the distances (sum of minimal distances)

				for(unsigned int i = 0; i < movingCount; ++i) {
					alphasmdinternal2::PointToSetDistance<QType, Dim> ptsd = m_MovingPointToSetDistances[i];
					TInternalComputationValueType tmpWeight = ptsd.m_SourcePoint.m_Weight * totalMovingWeight;
					
					distAcc += ptsd.m_Distance * tmpWeight;

					m_Transform.TotalDiffInverse(
						ptsd.m_SourcePoint.m_SourcePoint - m_B_CenterPoint,
						-ptsd.m_SpatialGrad,
						derivative,
						tmpWeight,
						true);
				}
			}

			value = distAcc;
		}

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

			NumberOfParametersType numberOfParameters = m_Transform.GetParamCount();

			if (factor == NumericTraits< ParametersValueType >::OneValue()) {
				for (unsigned int i = 0; i < numberOfParameters; ++i) {
					m_Transform.SetParam(i, m_Transform.GetParam(i) + derivative[i]);
				}
			}
			else {
				for (unsigned int i = 0; i < numberOfParameters; ++i) {
					m_Transform.SetParam(i, m_Transform.GetParam(i) + factor * derivative[i]);
				}
			}
		}

		virtual const ParametersType& GetParameters() const override {
			for(unsigned int i = 0; i < m_Transform.GetParamCount(); ++i) {
				m_Param[i] = m_Transform.GetParam(i);
			}

			return m_Param;
		}

		virtual void SetParameters(ParametersType &parameters) override {
			assert(parameters.Size() == m_Transform.GetParamCount());

			for(unsigned int i = 0; i < m_Transform.GetParamCount(); ++i) {
				m_Transform.SetParam(i, parameters[i]);
			}
		}

		virtual NumberOfParametersType GetNumberOfParameters() const override { return m_Transform.GetParamCount(); }

		virtual NumberOfParametersType GetNumberOfLocalParameters() const override { return m_Transform.GetParamCount(); }

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

		virtual double GetOutlierRejectionPercentage() const {
			return m_OutlierRejectionPercentage;
		}

		virtual void SetOutlierRejectionPercentage(double p) {
			m_OutlierRejectionPercentage = p;
		}

		virtual void SetRandomSeed(int seed) {
			m_RNG->SetSeed(seed);
		}

		virtual PointType GetFixedCenterPoint() const {
			return m_A_CenterPoint;
		}

		virtual void SetFixedCenterPoint(PointType p) {
			m_A_CenterPoint = p;
		}

		virtual PointType GetMovingCenterPoint() const {
			return m_B_CenterPoint;
		}

		virtual void SetMovingCenterPoint(PointType p) {
			m_B_CenterPoint = p;
		}

		// Do the pre-processing to set up the measure for evaluating the distance and derivative

		virtual void Update() {
			if(!m_FixedWeightImage) {
				SetFixedUnitWeightImage();
			}
			if(!m_MovingWeightImage) {
				SetMovingUnitWeightImage();
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
		}

		template <typename T, typename U>
		itk::SmartPointer<U> cast_smart_pointer(itk::SmartPointer<T> ptr) const {
			itk::SmartPointer<U> result = itk::SmartPointer<U>((U*)ptr.GetPointer());
			return result;
		}

		virtual SymmetricTransformType* GetSymmetricTransformPointer() {
			return &m_Transform;
		}

		/**
		 * Generates an AffineTransform<double, 2U> object from the current parameters
		 * which is compatible with the rest of ITK.
		 */
		virtual typename itk::Transform<double, Dim, Dim>::Pointer MakeFinalTransform() const {
			typedef typename itk::CompositeTransform<double, Dim> TransformContainerType;
			typedef typename itk::TranslationTransform<double, Dim> TranslationTransformType;

			typename TranslationTransformType::Pointer firstTranslation = TranslationTransformType::New();

			typedef typename TranslationTransformType::ParametersType TranslationParametersType;

			TranslationParametersType transParam(Dim);
			for(unsigned int i = 0; i < Dim; ++i) {
				transParam[i] = -m_A_CenterPoint[i];
			}


			firstTranslation->SetParameters(transParam);

			//typedef typename itk::Transform<double, Dim, Dim>::ParametersType TransformParametersType;

			typename itk::Transform<double, Dim, Dim>::Pointer outTransform = static_cast<itk::Transform<double, Dim, Dim>*>(m_Transform.ConvertToITKTransform().GetPointer());

			TranslationParametersType transParam2(Dim);
			for(unsigned int i = 0; i < Dim; ++i) {
				transParam2[i] = m_B_CenterPoint[i];
			}

			typename TranslationTransformType::Pointer secondTranslation = TranslationTransformType::New();

			secondTranslation->SetParameters(transParam2);
			

			typename TransformContainerType::Pointer finalTransform = TransformContainerType::New();

			finalTransform->AddTransform(secondTranslation);
			finalTransform->AddTransform(outTransform);
			finalTransform->AddTransform(firstTranslation);

			return cast_smart_pointer<TransformContainerType, itk::Transform<double, Dim, Dim> >(finalTransform);
		}
	protected:
		AlphaSMDObjectToObjectMetric2v4() : m_AlphaLevels(31), m_SymmetricMeasure(true), m_FixedSamplingPercentage(1.0), m_MovingSamplingPercentage(1.0), m_SquaredMeasure(false), m_LinearInterpolation(false), m_MaxDistance(0.0), m_OutlierRejectionPercentage(0.0), m_Param(m_Transform.GetParamCount()) {
			m_Transform.SetIdentity();
			m_Param.Fill(0);
			m_RNG = GeneratorType::New();

			m_A_CenterPoint.Fill(0);
			m_B_CenterPoint.Fill(0);

			m_OutlierRejectionWeight = 0.1;
		}
		virtual ~AlphaSMDObjectToObjectMetric2v4() {}

		int RandomShuffle(std::vector<SourcePointType>& ps, double samplingPercentage) const {
			int count = (int)(samplingPercentage * ps.size() + 0.5);

			if(count >= ps.size())
				return (int)ps.size();

			int sz = ps.size() - 1;

			for(int i = 0; i < count; ++i, --sz) {
				int index = i + m_RNG->GetIntegerVariate(sz);
				
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

		// The fraction of the sampled points to be discarded (0 means no outlier rejection)
		double m_OutlierRejectionPercentage;
		double m_OutlierRejectionWeight;

		mutable std::vector<alphasmdinternal2::PointToSetDistance<QType, Dim> > m_FixedPointToSetDistances;
		
		mutable std::vector<alphasmdinternal2::PointToSetDistance<QType, Dim> > m_MovingPointToSetDistances;
		
		mutable ParametersType m_Param;

		mutable std::vector<SourcePointType> m_FixedSourcePoints;
		mutable std::vector<SourcePointType> m_MovingSourcePoints;
		
		mutable typename SymmetricTransformType::ParamArrayType tmpDerivative;

		// Images

		ImagePointer m_FixedImage;
		ImagePointer m_MovingImage;

		typename itk::Image<bool, Dim>::Pointer m_FixedMask;
		typename itk::Image<bool, Dim>::Pointer m_MovingMask;

		typename itk::Image<double, Dim>::Pointer m_FixedWeightImage;
		typename itk::Image<double, Dim>::Pointer m_MovingWeightImage;

		// Random number generator

		typename GeneratorType::Pointer m_RNG;

		// Center-of-rotation

		mutable PointType m_A_CenterPoint;
		mutable PointType m_B_CenterPoint;

		// Transform

		mutable SymmetricTransformType m_Transform;

		mutable alphasmdinternal2::AlphaSMDInternal<QType, Dim> internals;
	};
}

#endif
