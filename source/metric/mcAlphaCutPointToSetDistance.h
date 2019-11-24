
/**
 * Implementation of a Monte Carlo framework (insert reference here) for computing the fuzzy alpha-cut-based point-to-set distance
 * ("Linear time distances between fuzzy sets with applications to pattern matching and classification",
 * by J. Lindblad and N. Sladoje, IEEE Transactions on Image Processing, 2013).
 *
 * Author: Johan Ofverstedt
 */

#include <itkImage.h>
#include <itkNearestNeighborInterpolateImageFunction.h>
#include <cmath>
#include <algorithm>

#include "samplers.h"

// A namespace collecting auxilliary data-structures and functions
// used by the Monte Carlo distance framework.

namespace MCDSInternal {

template <unsigned int Dim>
struct ValuedCornerPoints;

template <>
struct ValuedCornerPoints<1U> {
  itk::Vector<double, 2U> m_Values;
};

template <>
struct ValuedCornerPoints<2U> {
  itk::Vector<double, 4U> m_Values;
};

template <>
struct ValuedCornerPoints<3U> {
  itk::Vector<double, 8U> m_Values;
};

template <typename T, unsigned int Dim>
struct CornerPoints;

template <typename T>
struct CornerPoints<T, 1U> {
  static constexpr unsigned int size = 2U;
  itk::FixedArray<itk::FixedArray<T, 1U>, size> m_Points;
};

template <typename T>
struct CornerPoints<T, 2U> {
  static constexpr unsigned int size = 4U;
  itk::FixedArray<itk::FixedArray<T, 2U>, size> m_Points;
};

template <typename T>
struct CornerPoints<T, 3U> {
  static constexpr unsigned int size = 8U;
  itk::FixedArray<itk::FixedArray<T, 3U>, size> m_Points;
};

template <typename T, unsigned int Dim>
void ComputeCornersRec(unsigned int cur, unsigned int& pos, itk::FixedArray<T, Dim>& index, CornerPoints<T, Dim>& out) {
  if(cur == 0) {
    out.m_Points[pos++] = index;
  } else {
    ComputeCornersRec(cur-1, pos, index, out);
    index[cur-1] = itk::NumericTraits<T>::OneValue();
    ComputeCornersRec(cur-1, pos, index, out);
    index[cur-1] = itk::NumericTraits<T>::ZeroValue();
  }

}

template <typename T, unsigned int Dim>
CornerPoints<T, Dim> ComputeCorners() {
  CornerPoints<T, Dim> res;
  itk::FixedArray<T, Dim> index;
  index.Fill(itk::NumericTraits<T>::ZeroValue());
  unsigned int pos = 0;
  ComputeCornersRec(Dim, pos, index, res);
  
  return res;
}

template <unsigned int ImageDimension>
inline double InterpolateDistances(itk::Vector<double, ImageDimension> frac, ValuedCornerPoints<ImageDimension>& distanceValues, itk::Vector<double, ImageDimension>& grad);

// Linear interpolation
template <>
inline double InterpolateDistances<1U>(itk::Vector<double, 1U> frac, ValuedCornerPoints<1U>& distanceValues, itk::Vector<double, 1U>& grad) {
  double xx = frac[0];
  double ixx = 1.0 - xx;
  grad[0] = distanceValues.m_Values[1] - distanceValues.m_Values[0];
  return ixx * distanceValues.m_Values[0] + xx * distanceValues.m_Values[1];
}

// Bilinear interpolation
template <>
inline double InterpolateDistances<2U>(itk::Vector<double, 2U> frac, ValuedCornerPoints<2U>& distanceValues, itk::Vector<double, 2U>& grad) {
  double xx = frac[0];
  double yy = frac[1];
  double ixx = 1.0 - xx;
  double iyy = 1.0 - yy;

  double v_00 = distanceValues.m_Values[0];
  double v_10 = distanceValues.m_Values[1];
  double v_01 = distanceValues.m_Values[2];
  double v_11 = distanceValues.m_Values[3];

  double step_10_00 = v_10 - v_00;
  double step_11_01 = v_11 - v_01;
  double step_01_00 = v_01 - v_00;
  double step_11_10 = v_11 - v_10;
  
  //grad[0] = iyy * step_10_00 + yy * step_11_01;
  //grad[1] = ixx * step_01_00 + xx * step_11_10;
  //return v_00 * ixx * iyy + v_10 * xx * iyy + v_01 * ixx * yy + v_11 * xx * yy;
  // Version that uses mid-point
  grad[0] = (step_10_00 + step_11_01) * 0.5;
  grad[1] = (step_01_00 + step_11_10) * 0.5;
  
  return v_00 * ixx * iyy + v_10 * xx * iyy + v_01 * ixx * yy + v_11 * xx * yy;
}

// Trilinear interpolation
template <>
inline double InterpolateDistances<3U>(itk::Vector<double, 3U> frac, ValuedCornerPoints<3U>& distanceValues, itk::Vector<double, 3U>& grad) {
  double xx = frac[0];
  double yy = frac[1];
  double zz = frac[2];
  double ixx = 1.0 - xx;
  double iyy = 1.0 - yy;
  double izz = 1.0 - zz;

  double v_000 = distanceValues.m_Values[0];
  double v_100 = distanceValues.m_Values[1];
  double v_010 = distanceValues.m_Values[2];
  double v_110 = distanceValues.m_Values[3];
  double v_001 = distanceValues.m_Values[4];
  double v_101 = distanceValues.m_Values[5];
  double v_011 = distanceValues.m_Values[6];
  double v_111 = distanceValues.m_Values[7];

  double v_00 = v_000 * ixx + v_100 * xx;
  double v_01 = v_001 * ixx + v_101 * xx;
  double v_10 = v_010 * ixx + v_110 * xx;
  double v_11 = v_011 * ixx + v_111 * xx;

  double v_0 = v_00 * iyy + v_10 * yy;
  double v_1 = v_01 * iyy + v_11 * yy;

  double v = v_0 * izz + v_1 * zz;

  /*
   v = v_0 * izz + v_1 * zz
   =>
   v = (v_00 * iyy + v_10 * yy) * izz +
         (v_01 * iyy + v_11 * yy) * zz =
   ((v_000 * ixx + v_100 * xx) * iyy + (v_010 * ixx + v_110 * xx) * yy) * izz +
     ((v_001 * ixx + v_101 * xx) * iyy + (v_011 * ixx + v_111 * xx) * yy) * zz =

   df/dx = ((-v_000 + v_100) * iyy + (-v_010 + v_110) * yy) * izz +
     ((-v_001 + v_101) * iyy + (-v_011 + v_111) * yy) * zz =     
  ((v_100 - v_000) * iyy + (v_110 - v_010) * yy) * izz + ((v_101 - v_001) * iyy + (v_111 - v_011) * yy) * zz

   df/dy = -(v_000 * ixx + v_100 * xx) * izz + (v_010 * ixx + v_110 * xx) * izz +
     -(v_001 * ixx + v_101 * xx) * zz + (v_011 * ixx + v_111 * xx) * zz =
   ((v_010 - v_000) * ixx + (v_110 - v_100) * xx) * izz + ((v_011 - v_001) * ixx + (v_111 - v_101) * xx) * zz

   df/dz = -((v_000 * ixx + v_100 * xx) * iyy + (v_010 * ixx + v_110 * xx) * yy) + 
     ((v_001 * ixx + v_101 * xx) * iyy + (v_011 * ixx + v_111 * xx) * yy) =
     ((v_001 - v_000) * ixx + (v_101 - v_100) * xx) * iyy + ((v_011 - v_010) * ixx + (v_111 - v_110) * xx) * yy

  */

  //grad[0] = ((v_100 - v_000) * iyy + (v_110 - v_010) * yy) * izz + ((v_101 - v_001) * iyy + (v_111 - v_011) * yy) * zz;
  //grad[1] = ((v_010 - v_000) * ixx + (v_110 - v_100) * xx) * izz + ((v_011 - v_001) * ixx + (v_111 - v_101) * xx) * zz;
  //grad[2] = ((v_001 - v_000) * ixx + (v_101 - v_100) * xx) * iyy + ((v_011 - v_010) * ixx + (v_111 - v_110) * xx) * yy;

  // Version that uses mid-point
  grad[0] = (((v_100 - v_000) + (v_110 - v_010)) + ((v_101 - v_001) + (v_111 - v_011))) * 0.25;
  grad[1] = (((v_010 - v_000) + (v_110 - v_100)) + ((v_011 - v_001) + (v_111 - v_101))) * 0.25;
  grad[2] = (((v_001 - v_000) + (v_101 - v_100)) + ((v_011 - v_010) + (v_111 - v_110))) * 0.25;
  return v;
}

template <unsigned int ImageDimension>
inline unsigned int PixelCount(itk::Size<ImageDimension> &sz)
{
  unsigned int acc = sz[0];
  for (unsigned int i = 1; i < ImageDimension; ++i)
  {
    acc *= sz[i];
  }
  return acc;
}

template <unsigned int ImageDimension>
inline unsigned int LargestDimension(itk::Size<ImageDimension> &sz)
{
  unsigned int res = 0;
  unsigned int s = sz[0];
  for (unsigned int i = 1U; i < ImageDimension; ++i)
  {
    if (sz[i] > s)
    {
      res = i;
      s = sz[i];
    }
  }
  return res;
}

template <typename IndexType, typename SizeType, unsigned int ImageDimension>
inline unsigned int SplitRectangle(IndexType index, SizeType size, IndexType& midIndexOut, SizeType& szOut1, SizeType& szOut2) {
  midIndexOut = index;

  unsigned int selIndex = MCDSInternal::LargestDimension<ImageDimension>(size);
  unsigned int maxSz = size[selIndex];
  unsigned int halfMaxSz = maxSz / 2;

  midIndexOut[selIndex] = index[selIndex] + halfMaxSz;
  szOut1 = size;
  szOut2 = size;

  szOut1[selIndex] = halfMaxSz;
  szOut2[selIndex] = maxSz - halfMaxSz;

  return selIndex;
}

template <typename IndexType, typename SizeType, unsigned int ImageDimension>
unsigned int MaxNodeIndex(IndexType index, SizeType size, unsigned int nodeIndex) {
      if(PixelCount(size) <= 1U) {
        return nodeIndex;
      }

      IndexType midIndex = index;
      unsigned int selIndex = LargestDimension<ImageDimension>(size);
      unsigned int maxSz = size[selIndex];

      midIndex[selIndex] = midIndex[selIndex] + maxSz / 2;
      SizeType sz1 = size;
      SizeType sz2 = size;

      sz1[selIndex] = sz1[selIndex] / 2;
      sz2[selIndex] = size[selIndex] - sz1[selIndex];

      unsigned int nodeIndex1 = nodeIndex * 2;
      unsigned int nodeIndex2 = nodeIndex * 2 + 1;

      return MaxNodeIndex<IndexType, SizeType, ImageDimension>(midIndex, sz2, nodeIndex2);
}

template <typename IndexType, typename SizeType, unsigned int ImageDimension, typename SpacingType>
inline double LowerBoundDistance(IndexType pnt, IndexType rectOrigin, SizeType rectSz, SpacingType sp)
{
  double d = 0;
  for (unsigned int i = 0; i < ImageDimension; ++i)
  {
    long long pnt_i = (long long)pnt[i];
    long long lowEdgePos_i = (long long)rectOrigin[i] - 1;
    long long highEdgePos_i = (long long)(rectOrigin[i] + rectSz[i] + 1);

    double d1_i = (double)(lowEdgePos_i - pnt_i);
    double d2_i = (double)(pnt_i - highEdgePos_i);
    double d_i = std::max(std::max(d1_i, d2_i), 0.0) * sp[i];
    d += d_i*d_i;
  } 
  return d;
}

template <unsigned int ImageDimension>
inline bool SizeIsEmpty(itk::Size<ImageDimension> &sz)
{
  for (unsigned int i = 0; i < ImageDimension; ++i)
  {
    if (sz[i] == 0U)
      return true;
  }
  return false;
}

template <typename T>
unsigned int PruneLevelsLinear(const T* values, unsigned int start, unsigned int end, T val) {
  for(; start < end; --end) {
    if(values[end-1] <= val) {
      break;
    }
  }
  return end;
}

template <typename T>
unsigned int PruneLevelsBinary(const std::vector<T>& values, unsigned int start, unsigned int end, T val) {
  if(start < end) {
    if(values[end-1] <= val)
      return end;
    else
      --end;
  } 
  while(start < end) {
    unsigned int mid = start + (end-start)/2;
    T midval = values[mid];
    if(midval > val) {
      end = mid;
    } else {
      start = mid + 1;
    }
  }
  return end;
}

};

//
// Evaluation context containing the auxilliary data-structures required
// to sample intensity values and compute the value and gradient in
// the Monte Carlo framework.
//
// In a multi-threaded scenario, each thread must command its own
// private eval context.
//
template <typename ImageType, typename SamplerType>
class MCDSEvalContext {
  static constexpr unsigned int ImageDimension = ImageType::ImageDimension;

  typedef typename ImageType::Pointer ImagePointer;
  typedef typename ImageType::RegionType RegionType;
  typedef typename ImageType::SizeType SizeType;
  typedef typename ImageType::IndexValueType IndexValueType;
  typedef typename ImageType::SizeValueType SizeValueType;
  typedef typename ImageType::IndexType IndexType;
  typedef typename itk::ContinuousIndex<double, ImageType::ImageDimension> ContinousIndexType;
  typedef typename ImageType::SpacingType SpacingType;
  typedef typename ImageType::ValueType ValueType;
  typedef itk::Vector<ValueType, 2U> NodeValueType;
  typedef typename ImageType::PointType PointType;



};

template <typename ImageType, typename SamplerType>
class MCAlphaCutPointToSetDistance
{
public:
  static constexpr unsigned int ImageDimension = ImageType::ImageDimension;

  typedef typename ImageType::Pointer ImagePointer;
  typedef typename ImageType::RegionType RegionType;
  typedef typename ImageType::SizeType SizeType;
  typedef typename ImageType::IndexValueType IndexValueType;
  typedef typename ImageType::SizeValueType SizeValueType;
  typedef typename ImageType::IndexType IndexType;
  typedef typename itk::ContinuousIndex<double, ImageType::ImageDimension> ContinousIndexType;
  typedef typename ImageType::SpacingType SpacingType;
  typedef typename ImageType::ValueType ValueType;
  typedef itk::Vector<ValueType, 2U> NodeValueType;
  typedef typename ImageType::PointType PointType;

  typedef itk::Image<bool, ImageDimension> MaskImageType;
  typedef typename MaskImageType::Pointer MaskImagePointer;
  typedef itk::NearestNeighborInterpolateImageFunction<MaskImageType, double> InterpolatorType;
  typedef typename InterpolatorType::Pointer InterpolatorPointer;

  typedef MCDSInternal::CornerPoints<IndexValueType, ImageType::ImageDimension> CornersType;
  
  typedef MCDSEvalContext<ImageType, SamplerType> EvalContextType;

  void SetImage(ImagePointer image)
  {
    m_Image = image;
    m_RawImagePtr = image.GetPointer();
  }

  void SetMaskImage(MaskImagePointer maskImage) {
    m_MaskImage = maskImage;
  }

  void SetOne(ValueType one)
  {
    m_One = one;
  }

  void SetMaxDistance(double dmax) {
    m_MaxDistance = dmax;
  }

  void SetSampleCount(unsigned int count)
  {
    m_SampleCount = count;
  }

  // Builds the kd-tree and initializes data-structures
  void Initialize()
  {
    // Compute height
    constexpr unsigned int dim = ImageType::ImageDimension;
    RegionType region = m_Image->GetLargestPossibleRegion();
    SizeType sz = region.GetSize();
    if(m_MaxDistance <= 0.0) {
      m_MaxDistance = 0.0;
      for(unsigned int i = 0; i < ImageDimension; ++i) {
        m_MaxDistance += sz[i]*sz[i];
      }
    }

    unsigned int nodeCount = MCDSInternal::MaxNodeIndex<IndexType, SizeType, ImageDimension>(region.GetIndex(), sz, 1);
    m_Array = std::move(std::unique_ptr<NodeValueType[]>(new NodeValueType[nodeCount]));

    m_Samples.reserve(m_SampleCount);
    m_InwardsValues.reserve(m_SampleCount);
    m_ComplementValues.reserve(m_SampleCount);

    if(MCDSInternal::PixelCount(sz) > 0)
      BuildTreeRec(1, region.GetIndex(), sz);

    m_Corners = MCDSInternal::ComputeCorners<IndexValueType, ImageType::ImageDimension>();

    m_DebugVisitCount = 0;

    // Initialize table
    m_Table = std::move(std::unique_ptr<double[]>(new double[m_SampleCount * CornersType::size]));

    m_RawImagePtr = m_Image.GetPointer();

    if(m_MaskImage) {
      m_MaskInterpolator = InterpolatorType::New();
      m_MaskInterpolator->SetInputImage(m_MaskImage);
    }
  }

  bool ValueAndDerivative(
    PointType point,
    ValueType h,
    double& valueOut,
    itk::Vector<double, ImageDimension>& gradOut) const {

    ImageType* image = m_RawImagePtr;

    // If we have a mask, check if we are inside the mask image bounds, and inside the mask
    if(m_MaskImage) {
      if(m_MaskInterpolator->IsInsideBuffer(point)) {
        if(!m_MaskInterpolator->Evaluate(point))
          return false;
      } else {
        return false;
      }
    }

    ContinousIndexType cIndex;
    IndexType pntIndex; // Generate the index
    itk::Vector<double, ImageDimension> frac;

    bool flag = image->TransformPhysicalPointToContinuousIndex(point, cIndex);
    for(unsigned int i = 0; i < ImageDimension; ++i) {
      pntIndex[i] = (long long)cIndex[i];
      frac[i] = cIndex[i] - (double)pntIndex[i];
    }
    
    // Sample
    m_Sampler.Sample(m_SampleCount, m_Samples);
    m_InwardsValues.clear();
    m_ComplementValues.clear();
    for(unsigned int i = 0; i < m_SampleCount; ++i) {
      if(m_Samples[i][0] <= h) {
        m_InwardsValues.push_back(m_Samples[i][0]);
      } else {
        m_ComplementValues.push_back(m_One - m_Samples[i][0]);
      }
    }
    std::sort(m_InwardsValues.begin(), m_InwardsValues.end());
    std::sort(m_ComplementValues.begin(), m_ComplementValues.end());

    RegionType region = image->GetLargestPossibleRegion();

    bool isFullyInside = true;
    for(unsigned int j = 0; j < ImageDimension; ++j) {
      if(pntIndex[j] < 0 || pntIndex[j] + 1 >= region.GetSize()[j]) {
        isFullyInside = false;
        break;
      }
    }

    unsigned int inwardsStart = 0;
    unsigned int complementStart = 0;

    ValueType minInVal = m_One;
    ValueType minCoVal = m_One;

    if(isFullyInside) {

      for(unsigned int i = 0; i < CornersType::size; ++i) {     
        IndexType cindex = pntIndex;
        for(unsigned int j = 0; j < ImageDimension; ++j) {
          cindex[j] = cindex[j] + m_Corners.m_Points[i][j];
        }

        ValueType valIn = image->GetPixel(cindex);
        ValueType valCo = m_One - valIn;
        if(valIn < minInVal)
          minInVal = valIn;
        if(valCo < minCoVal)
          minCoVal = valCo;
      }

      for(; inwardsStart < m_InwardsValues.size(); ++inwardsStart) {
        if(m_InwardsValues[inwardsStart] > minInVal)
          break;
      }
      for(; complementStart < m_ComplementValues.size(); ++complementStart) {
        if(m_ComplementValues[complementStart] > minCoVal)
          break;
      }
    }

    if(isFullyInside && (inwardsStart < m_InwardsValues.size() || complementStart < m_ComplementValues.size())) {
      unsigned int sampleCount = m_InwardsValues.size() + m_ComplementValues.size();   
    
      double dmax = m_MaxDistance;
      double dmaxSq = dmax * dmax;
      for(unsigned int i = 0; i < CornersType::size; ++i) {
        double* dists_i = m_Table.get() + (sampleCount * i);

        unsigned int j = 0;
        for(; j < inwardsStart; ++j)
          dists_i[j] = 0.0;
        for(; j < m_InwardsValues.size(); ++j)
          dists_i[j] = dmaxSq;
        for(; j < m_InwardsValues.size()+complementStart; ++j)
          dists_i[j] = 0.0;
        for(; j < sampleCount; ++j)
          dists_i[j] = dmaxSq;
      }

      Search(pntIndex, inwardsStart, complementStart);

      MCDSInternal::ValuedCornerPoints<ImageDimension> cornerValues;

      for(unsigned int i = 0; i < CornersType::size; ++i) {
        cornerValues.m_Values[i] = 0.0;
        double* dists_i = m_Table.get() + (m_SampleCount * i);

        for(unsigned int j = 0; j < m_SampleCount; ++j) {
          cornerValues.m_Values[i] += sqrt(dists_i[j]);
        }

        cornerValues.m_Values[i] = cornerValues.m_Values[i] / m_SampleCount;
      }

      valueOut = MCDSInternal::InterpolateDistances<ImageDimension>(frac, cornerValues, gradOut);

      // Apply spacing to gradient
      typedef typename ImageType::SpacingType SpacingType;
      SpacingType spacing = m_Image->GetSpacing();
    
      for(unsigned int i = 0; i < ImageDimension; ++i)
        gradOut[i] /= spacing[i];
    } else {
      valueOut = 0.0;
      for(unsigned int i = 0; i < ImageDimension; ++i)
        gradOut[i] = 0.0;
    }

    return true;
  }
  mutable size_t m_DebugVisitCount;
private:
  ImagePointer m_Image;
  ImageType* m_RawImagePtr;
  MaskImagePointer m_MaskImage;
  InterpolatorPointer m_MaskInterpolator;
  std::unique_ptr<NodeValueType[]> m_Array;
  unsigned int m_SampleCount;
  ValueType m_One;
  double m_MaxDistance;
  CornersType m_Corners;
  mutable std::unique_ptr<double[]> m_Table;
  mutable SamplerType m_Sampler;
  mutable std::vector<ValueType> m_InwardsValues;
  mutable std::vector<ValueType> m_ComplementValues;
  mutable std::vector<itk::Vector<double, 1U> > m_Samples;

  struct StackNode
  {
    IndexType m_Index;
    SizeType m_Size;
    unsigned int m_NodeIndex;
    unsigned int m_InStart;
    unsigned int m_InEnd;
    unsigned int m_CoStart;
    unsigned int m_CoEnd;
  };

  void BuildTreeRec(unsigned int nodeIndex, IndexType index, SizeType sz)
  {
    constexpr unsigned int dim = ImageType::ImageDimension;

    typedef itk::ImageRegionConstIterator<ImageType> IteratorType;
    NodeValueType* data = m_Array.get();

    unsigned int szCount = MCDSInternal::PixelCount<dim>(sz);

    if (szCount == 1U)
    {
      NodeValueType nv;
      if(m_MaskImage) {
        if(m_MaskImage->GetPixel(index)) {
          nv[0] = m_RawImagePtr->GetPixel(index);
          nv[1] = m_One - nv[0];
          data[nodeIndex - 1] = nv;
        } else {
          data[nodeIndex - 1].Fill(itk::NumericTraits<ValueType>::ZeroValue());
        }
      } else {
        nv[0] = m_RawImagePtr->GetPixel(index);
        nv[1] = m_One - nv[0];
        data[nodeIndex - 1] = nv;
      }
    }
    else
    {
      IndexType midIndex = index;
      unsigned int selIndex = MCDSInternal::LargestDimension<dim>(sz);
      unsigned int maxSz = sz[selIndex];

      midIndex[selIndex] = midIndex[selIndex] + maxSz / 2;
      SizeType sz1 = sz;
      SizeType sz2 = sz;

      sz1[selIndex] = sz1[selIndex] / 2;
      sz2[selIndex] = sz[selIndex] - sz1[selIndex];

      unsigned int nodeIndex1 = nodeIndex * 2;

      BuildTreeRec(nodeIndex1, index, sz1);
      BuildTreeRec(nodeIndex1+1, midIndex, sz2);

      NodeValueType n1 = *(data + (nodeIndex1 - 1));
      NodeValueType n2 = *(data + nodeIndex1);

      NodeValueType* dataCur = data + (nodeIndex-1);

      // Compute the maximum of the two nodes, for each channel
      for (unsigned int i = 0; i < 2U; ++i)
      {
        if (n2[i] > n1[i])
          (*dataCur)[i] = n2[i];
        else
          (*dataCur)[i] = n1[i];
      }
    }
  }

  void Search(
      IndexType index,
      unsigned int inwardsStart, unsigned int complementStart) const
  {
    ValueType* inwardsValues = m_InwardsValues.data();
    ValueType* complementValues = m_ComplementValues.data();
    unsigned int inwardsCount = m_InwardsValues.size();
    unsigned int complementCount = m_ComplementValues.size();

    typedef typename ImageType::SpacingType SpacingType;

    SpacingType spacing = m_Image->GetSpacing();

    double* distTable = m_Table.get();
    NodeValueType* data = m_Array.get(); // Node data

    unsigned int sampleCount = m_SampleCount;

    // Stack
    StackNode stackNodes[33];
    StackNode curStackNode;
    unsigned int stackIndex = 0;

    // Initialize the stack state
    curStackNode.m_Index = m_Image->GetLargestPossibleRegion().GetIndex();
    curStackNode.m_Size = m_Image->GetLargestPossibleRegion().GetSize();
    curStackNode.m_NodeIndex = 1;
    curStackNode.m_InStart = inwardsStart;
    curStackNode.m_InEnd = MCDSInternal::PruneLevelsLinear(inwardsValues, inwardsStart, inwardsCount, data[0][0]);
    curStackNode.m_CoStart = complementStart;
    curStackNode.m_CoEnd = MCDSInternal::PruneLevelsLinear(complementValues, complementStart, complementCount, data[0][1]);

    // All elements eliminated before entering the loop. Search is over. Return.
    if(curStackNode.m_InStart == curStackNode.m_InEnd && curStackNode.m_CoStart == curStackNode.m_CoEnd)
      return;
    
    itk::FixedArray<itk::Point<double, ImageDimension>, CornersType::size> corners;

    for (unsigned int i = 0; i < CornersType::size; ++i)
    {
      for (unsigned int j = 0; j < ImageDimension; ++j)
      {
        corners[i][j] = static_cast<double>(m_Corners.m_Points[i][j] + index[j]) * spacing[j];
      }
    }

    unsigned int visitCount = 0;
    while(true)
    {
      unsigned int npx = curStackNode.m_Size[0];
      for(unsigned int i = 1; i < ImageDimension; ++i) {
        npx *= curStackNode.m_Size[i];
      }

      unsigned int inStartLocal = curStackNode.m_InStart;
      unsigned int inEndLocal = curStackNode.m_InEnd;
      unsigned int coStartLocal = curStackNode.m_CoStart;
      unsigned int coEndLocal = curStackNode.m_CoEnd;

      // Is the node a leaf - compute distances
      if (npx == 1U)
      {
        ++visitCount;

        itk::Point<double, ImageDimension> leafPoint;
        for(unsigned int j = 0; j < ImageDimension; ++j)
          leafPoint[j] = static_cast<double>(curStackNode.m_Index[j]*spacing[j]);

        // Compare d with all the distances recorded for the alpha levels (which are still in play)
        for (unsigned int i = 0; i < CornersType::size; ++i)
        {
          const double d = corners[i].SquaredEuclideanDistanceTo(leafPoint);
          double* distTable_i = distTable + (sampleCount * i);
          double* coDistTable_i = distTable_i + inwardsCount;

          for (unsigned int j = coEndLocal; coStartLocal < j; --j)
          {
            unsigned int tabInd = j-1;//+inwardsCount
            double cur_j = coDistTable_i[tabInd];
            if (d < cur_j)
            {
              coDistTable_i[tabInd] = d;
            } else {
              break;
            }
          }          

          for (unsigned int j = inEndLocal; inStartLocal < j; --j)
          {
            unsigned int tabInd = j-1;
            double cur_j = distTable_i[tabInd];
            if (d < cur_j)
            {
              distTable_i[tabInd] = d;
            } else {
              break;
            }
          }
        }
      }
      else
      { // Continue traversing the tree
        // Compute lower bound on distance for all pixels in the node
        IndexType innerNodeInd = curStackNode.m_Index;
        SizeType innerNodeSz = curStackNode.m_Size;
        double lowerBoundDistance = MCDSInternal::LowerBoundDistance<IndexType, SizeType, ImageDimension>(index, innerNodeInd, innerNodeSz, spacing);

        // --- Approximation starts here ---
        constexpr double threshold = 20.0;
        constexpr double thresholdSq = threshold*threshold;
        constexpr double xponent = 1.1;
        constexpr double xponentSq = xponent*xponent;
        if(lowerBoundDistance > thresholdSq) {
          lowerBoundDistance = (thresholdSq-1.0) + ((lowerBoundDistance-thresholdSq)+1.0) * xponentSq;
        }
        // --- Approximation ends here ---

        // Eliminate inwards values based on distance bounds
        for (; inStartLocal < inEndLocal; ++inStartLocal)
        {
          double cur_j = distTable[inStartLocal];
          if (lowerBoundDistance <= cur_j)
            break;
        }

        // Eliminate complement values based on distance bounds
        for (; coStartLocal < coEndLocal; ++coStartLocal)
        {
          double cur_j = distTable[coStartLocal+inwardsCount];
          if (lowerBoundDistance <= cur_j)
            break;
        }

        // If all alpha levels are eliminated, backtrack
        // without doing the work to compute the node split
        // and tree value look-ups.
        if (inStartLocal == inEndLocal && coStartLocal == coEndLocal)
        {
          if(stackIndex == 0)
            break;
          curStackNode = stackNodes[--stackIndex];
          continue;
        }

        IndexType midIndex = innerNodeInd;
        unsigned int selIndex = MCDSInternal::LargestDimension<ImageDimension>(innerNodeSz);
        unsigned int maxSz = innerNodeSz[selIndex];
        unsigned int halfMaxSz = maxSz / 2;

        midIndex[selIndex] = midIndex[selIndex] + halfMaxSz;
        SizeType sz1 = innerNodeSz;
        SizeType sz2 = innerNodeSz;

        sz1[selIndex] = halfMaxSz;
        sz2[selIndex] = maxSz - halfMaxSz;

        unsigned int nodeIndex1 = curStackNode.m_NodeIndex * 2;

        unsigned int inEndLocal1 = MCDSInternal::PruneLevelsLinear(inwardsValues, inStartLocal, inEndLocal, data[nodeIndex1-1][0]);
        unsigned int coEndLocal1 = MCDSInternal::PruneLevelsLinear(complementValues, coStartLocal, coEndLocal, data[nodeIndex1-1][1]);
        unsigned int inEndLocal2 = MCDSInternal::PruneLevelsLinear(inwardsValues, inStartLocal, inEndLocal, data[nodeIndex1][0]);
        unsigned int coEndLocal2 = MCDSInternal::PruneLevelsLinear(complementValues, coStartLocal, coEndLocal, data[nodeIndex1][1]);

        if (index[selIndex] < midIndex[selIndex])
        {
          if(inStartLocal < inEndLocal1 || coStartLocal < coEndLocal1)
          {
            curStackNode.m_Index = innerNodeInd;
            curStackNode.m_Size = sz1;
            curStackNode.m_NodeIndex = nodeIndex1;
            curStackNode.m_InStart = inStartLocal;
            curStackNode.m_InEnd = inEndLocal1;
            curStackNode.m_CoStart = coStartLocal;
            curStackNode.m_CoEnd = coEndLocal1;
            if(inStartLocal < inEndLocal2 || coStartLocal < coEndLocal2)
            {
              stackNodes[stackIndex].m_Index = midIndex;
              stackNodes[stackIndex].m_Size = sz2;
              stackNodes[stackIndex].m_NodeIndex = nodeIndex1+1;
              stackNodes[stackIndex].m_InStart = inStartLocal;
              stackNodes[stackIndex].m_InEnd = inEndLocal2;
              stackNodes[stackIndex].m_CoStart = coStartLocal;
              stackNodes[stackIndex].m_CoEnd = coEndLocal2;
              ++stackIndex;
            }
            continue;
          }
          else if(inStartLocal < inEndLocal2 || coStartLocal < coEndLocal2)
          {
            curStackNode.m_Index = midIndex;
            curStackNode.m_Size = sz2;
            curStackNode.m_NodeIndex = nodeIndex1+1;
            curStackNode.m_InStart = inStartLocal;
            curStackNode.m_InEnd = inEndLocal2;
            curStackNode.m_CoStart = coStartLocal;
            curStackNode.m_CoEnd = coEndLocal2;
            continue;
          }      
        }
        else
        {
          if(inStartLocal < inEndLocal2 || coStartLocal < coEndLocal2)
          {
            curStackNode.m_Index = midIndex;
            curStackNode.m_Size = sz2;
            curStackNode.m_NodeIndex = nodeIndex1+1;
            curStackNode.m_InStart = inStartLocal;
            curStackNode.m_InEnd = inEndLocal2;
            curStackNode.m_CoStart = coStartLocal;
            curStackNode.m_CoEnd = coEndLocal2;
            if(inStartLocal < inEndLocal1 || coStartLocal < coEndLocal1)
            {
              stackNodes[stackIndex].m_Index = innerNodeInd;
              stackNodes[stackIndex].m_Size = sz1;
              stackNodes[stackIndex].m_NodeIndex = nodeIndex1;
              stackNodes[stackIndex].m_InStart = inStartLocal;
              stackNodes[stackIndex].m_InEnd = inEndLocal1;
              stackNodes[stackIndex].m_CoStart = coStartLocal;
              stackNodes[stackIndex].m_CoEnd = coEndLocal1;
              ++stackIndex;
            }
            continue;
          } else if(inStartLocal < inEndLocal1 || coStartLocal < coEndLocal1) 
          {
            curStackNode.m_Index = innerNodeInd;
            curStackNode.m_Size = sz1;
            curStackNode.m_NodeIndex = nodeIndex1;
            curStackNode.m_InStart = inStartLocal;
            curStackNode.m_InEnd = inEndLocal1;
            curStackNode.m_CoStart = coStartLocal;
            curStackNode.m_CoEnd = coEndLocal1;
            continue;
          }       
      } // End of else branch
      }

      // If we arrive here, we need to pop a stack node
      // unless the stack is empty, which would trigger a
      // termination of the loop.
      if(stackIndex == 0)
        break;
      curStackNode = stackNodes[--stackIndex];
    } // End main "recursion" loop

    m_DebugVisitCount += visitCount;
  } // End of Search function

}; // End of class
