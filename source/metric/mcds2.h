
#include <itkImage.h>
#include <itkArray.h>
//#include <itkImageRegionConstIterator.h>
#include <cmath>
#include <algorithm>

#include "samplers.h"

template <typename T, unsigned int Channels>
struct KDTreeNode {
  itk::Vector<T, Channels>  m_Max;
  unsigned int m_RightOffset;
};

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
/*
  void ComputeCorners(unsigned int cur, unsigned int dim, unsigned int &pos, IndexType index)
  {
    // Order for 2d: [(0, 0), (0, 1), (1, 0), (1, 1)]
    if (cur == 0)
    {
      m_Corners[pos++] = index;
    } else {
      ComputeCorners(cur - 1, dim, pos, index);
      index[cur-1] = 1;
      ComputeCorners(cur - 1, dim, pos, index);
    }
  }
*/
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
  
  grad[0] = iyy * step_10_00 + yy * step_11_01;
  grad[1] = ixx * step_01_00 + xx * step_11_10;
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

  grad[0] = ((v_100 - v_000) * iyy + (v_110 - v_010) * yy) * izz + ((v_101 - v_001) * iyy + (v_111 - v_011) * yy) * zz;
  grad[1] = ((v_010 - v_000) * ixx + (v_110 - v_100) * xx) * izz + ((v_011 - v_001) * ixx + (v_111 - v_101) * xx) * zz;
  grad[2] = ((v_001 - v_000) * ixx + (v_101 - v_100) * xx) * iyy + ((v_011 - v_010) * ixx + (v_111 - v_110) * xx) * yy;
  return v;
}

template <unsigned int ImageDimension>
inline unsigned int PixelCount(itk::Size<ImageDimension> &sz)
{
  unsigned int acc = 1;
  for (unsigned int i = 0; i < ImageDimension; ++i)
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

template <typename IndexType, typename SizeType, unsigned int ImageDimension, typename SpacingType>
inline double LowerBoundDistance(IndexType pnt, IndexType rectOrigin, SizeType rectSz, SpacingType &sp)
{
  double d = 0;
  for (unsigned int i = 0; i < ImageDimension; ++i)
  {
    long long pnt_i = (long long)pnt[i];
    long long lowEdgePos_i = (long long)rectOrigin[i] - 1;
    long long highEdgePos_i = (long long)(rectOrigin[i] + rectSz[i] + 1);

    // If outside:
    if (pnt_i < lowEdgePos_i)
    {
      double d_i = (double)(lowEdgePos_i - pnt_i) * sp[i];
      d += d_i * d_i;
    }
    else if (highEdgePos_i < pnt_i)
    {
      double d_i = (double)(pnt_i - highEdgePos_i) * sp[i];
      d += d_i * d_i;
    }
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
void FillVector(std::vector<T>& v, size_t count, const T& value) {
  v.clear();
  v.reserve(count);
  for(size_t i = 0; i < count; ++i) {
    v.push_back(value);
  }
}

template <typename ImageType, typename SamplerType>
class MCDS
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
  typedef KDTreeNode<ValueType, 2U> NodeValueType;
  typedef typename ImageType::PointType PointType;

  typedef CornerPoints<IndexValueType, ImageType::ImageDimension> CornersType;

  //MCDS(const MCDS&) = delete;
  //MCDS& operator=(const MCDS&) = delete;

  void SetImage(ImagePointer image)
  {
    m_Image = image;
    m_RawImagePtr = image.GetPointer();
  }

  void SetOne(ValueType one)
  {
    m_One = one;
  }

  void SetMaxDistance(double dmax) {
    m_MaxDistance = dmax;
  }

  void SetMaxSampleCount(unsigned int count)
  {
    m_MaxSampleCount = count;
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

    m_Height = 1;
    for (unsigned int i = 0; i < dim; ++i)
    {
      unsigned int logsz = (unsigned int)ceil(log2((double)sz[i]));
      m_Height += logsz;
    }

    unsigned int nodeCount = (unsigned int)(pow(2.0, (double)m_Height) + 0.5);
    NodeValueType nv;
    nv.m_Max = {itk::NumericTraits<ValueType>::ZeroValue()};
    nv.m_RightOffset = 0;
    FillVector<NodeValueType>(m_Array, nodeCount, nv);

    m_Samples.reserve(m_MaxSampleCount);
    m_InwardsValues.reserve(m_MaxSampleCount);
    m_ComplementValues.reserve(m_MaxSampleCount);

    BuildTreeRec(1, region.GetIndex(), sz, m_Height);

    m_Corners = ComputeCorners<IndexValueType, ImageType::ImageDimension>();

    // Initialize table
    m_Table.SetSize(m_MaxSampleCount, CornersType::size);

    m_RawImagePtr = m_Image.GetPointer();
  }
  /*
      IndexType index,
      itk::Array<ValueType> &inwardsValues,
      unsigned int inwardsCount,
      itk::Array<ValueType> &complementValues,
      unsigned int complementCount,
      itk::Array<double> &distOut
      */
  bool ValueAndDerivative(
    PointType point,
    ValueType h,
    unsigned int samples,
    double& valueOut,
    itk::Vector<double, ImageDimension>& gradOut) const {

/*
    itk::Array<ValueType>& inwardsValues,
    unsigned int inwardsCount,
    itk::Array<ValueType>& complementValues,
    unsigned int complementCount,*/

    ContinousIndexType cIndex;
    IndexType pntIndex; // Generate the index
    itk::Vector<double, ImageDimension> frac;

    bool flag = m_RawImagePtr->TransformPhysicalPointToContinuousIndex(point, cIndex);
    for(unsigned int i = 0; i < ImageDimension; ++i) {
      pntIndex[i] = (unsigned int)cIndex[i];
      frac[i] = cIndex[i] - (double)pntIndex[i];
    }
    
    // Sample
    m_Sampler.Sample(samples, m_Samples);
    m_InwardsValues.clear();
    m_ComplementValues.clear();
    for(unsigned int i = 0; i < samples; ++i) {
      if(m_Samples[i][0] <= h) {
        m_InwardsValues.push_back(m_Samples[i][0]);
      } else {
        m_ComplementValues.push_back(m_One - m_Samples[i][0]);
      }
    }
    std::sort(m_InwardsValues.begin(), m_InwardsValues.end());
    std::sort(m_ComplementValues.begin(), m_ComplementValues.end());

    //for(unsigned int i = 0; i < m_InwardsValues.size(); ++i) {
      //std::cout << m_InwardsValues[i] << ", ";
    //}
    //std::cout << std::endl;
    //for(unsigned int i = 0; i < m_ComplementValues.size(); ++i) {
      //std::cout << m_ComplementValues[i] << ", ";
    //}
    //std::cout << std::endl;

    double dmax = m_MaxDistance;
    for(unsigned int i = 0; i < CornersType::size; ++i) {
      for(unsigned int j = 0; j < samples; ++j) {
        m_Table.SetElement(j, i, dmax*dmax);
      }
    }

    Search(pntIndex);

    ValuedCornerPoints<ImageDimension> cornerValues;

    for(unsigned int i = 0; i < CornersType::size; ++i) {
      cornerValues.m_Values[i] = 0.0;

      for(unsigned int j = 0; j < samples; ++j) {
        cornerValues.m_Values[i] += sqrt(m_Table.GetElement(j, i));
      }

      cornerValues.m_Values[i] = cornerValues.m_Values[i] / samples;
    }

    valueOut = InterpolateDistances<ImageDimension>(frac, cornerValues, gradOut);

    return true;
  }
private:
  ImagePointer m_Image;
  ImageType* m_RawImagePtr;
  std::vector<NodeValueType> m_Array;
  unsigned int m_Height;
  unsigned int m_MaxSampleCount;
  ValueType m_One;
  double m_MaxDistance;
  CornersType m_Corners;
  mutable itk::Array2D<double> m_Table;
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

  unsigned int BuildTreeRec(unsigned int nodeIndex, IndexType index, SizeType sz, unsigned int depthCountDown)
  {
    constexpr unsigned int dim = ImageType::ImageDimension;

    typedef itk::ImageRegionConstIterator<ImageType> IteratorType;

    unsigned int szCount = PixelCount<dim>(sz);

    if (szCount == 0U)
    {
      return nodeIndex;
    }
    else if (szCount == 1U)
    {
      NodeValueType nv;
      nv.m_Max[0] = m_Image->GetPixel(index);
      nv.m_Max[1] = m_One - nv.m_Max[0];
      nv.m_RightOffset = 0;
      m_Array[nodeIndex - 1] = nv;
      return nodeIndex + 1;
    }
    else
    {
      IndexType midIndex = index;
      unsigned int selIndex = LargestDimension<dim>(sz);
      unsigned int maxSz = sz[selIndex];

      midIndex[selIndex] = midIndex[selIndex] + maxSz / 2;
      SizeType sz1 = sz;
      SizeType sz2 = sz;

      sz1[selIndex] = sz1[selIndex] / 2;
      sz2[selIndex] = sz[selIndex] - sz1[selIndex];

      //unsigned int nodeIndex1 = nodeIndex * 2;
      //unsigned int nodeIndex2 = nodeIndex * 2 + 1;

      unsigned int nodeIndex2 = BuildTreeRec(nodeIndex+1, index, sz1, depthCountDown - 1);

      unsigned int nodeIndex3 = BuildTreeRec(nodeIndex2, midIndex, sz2, depthCountDown - 1);

      NodeValueType n1 = m_Array[nodeIndex];
      NodeValueType n2 = m_Array[nodeIndex2 - 1];

      // Compute the maximum of the two nodes, for each channel
      for (unsigned int i = 0; i < 2U; ++i)
      {
        if (n2.m_Max[i] > n1.m_Max[i])
          n1.m_Max[i] = n2.m_Max[i];
      }
      n1.m_RightOffset = nodeIndex2;

      m_Array[nodeIndex - 1] = n1;

      return nodeIndex3;
    }
  }

  void Search(
      IndexType index) const
  {
    constexpr unsigned int dim = ImageType::ImageDimension;

    std::vector<ValueType>& inwardsValues = m_InwardsValues;
    std::vector<ValueType>& complementValues = m_ComplementValues;
    unsigned int inwardsCount = inwardsValues.size();
    unsigned int complementCount = complementValues.size();

    //typedef itk::ImageRegionConstIterator<ImageType> IteratorType;
    typedef typename ImageType::SpacingType SpacingType;

    ImageType *image = m_Image.GetPointer();
    SpacingType spacing = image->GetSpacing();
    ValueType one = m_One;

    RegionType region = image->GetLargestPossibleRegion();

    unsigned int rows = inwardsCount + complementCount;
    unsigned int cornerCount = CornersType::size;

    // Stack
    StackNode stackNodes[33];
    unsigned int stackIndex = 1;

    // Initialize the stack state
    stackNodes[0].m_Index = region.GetIndex();
    stackNodes[0].m_Size = region.GetSize();
    stackNodes[0].m_NodeIndex = 1;
    stackNodes[0].m_InStart = 0;
    stackNodes[0].m_InEnd = inwardsCount;
    stackNodes[0].m_CoStart = 0;
    stackNodes[0].m_CoEnd = complementCount;
    
    CornersType corners;

    for (unsigned int i = 0; i < cornerCount; ++i)
    {
      itk::FixedArray<long int, ImageDimension> crnr = m_Corners.m_Points[i];

      for (unsigned int j = 0; j < dim; ++j)
      {
        crnr[j] = crnr[j] + index[j];
      }
      corners.m_Points[i] = crnr;
    }

    while (stackIndex > 0)
    {
      --stackIndex;
      
      SizeType nodeSz = stackNodes[stackIndex].m_Size;
      unsigned int npx = PixelCount<dim>(nodeSz);
      //std::cout << "Pixels: " << npx << " ";

      // Can I remove this?
      if (npx == 0U)
      {
        continue;
      }

      unsigned int inStartLocal = stackNodes[stackIndex].m_InStart;
      unsigned int inEndLocal = stackNodes[stackIndex].m_InEnd;
      unsigned int coStartLocal = stackNodes[stackIndex].m_CoStart;
      unsigned int coEndLocal = stackNodes[stackIndex].m_CoEnd;

      unsigned int nodeIndex = stackNodes[stackIndex].m_NodeIndex;
      NodeValueType nv = m_Array[nodeIndex-1];

      //std::cout << "NV: " << nv << "\n";

      // Eliminate inwards values
      for (; inStartLocal < inEndLocal; --inEndLocal)
      {
        ValueType val = inwardsValues[inEndLocal - 1];
        if (val <= nv.m_Max[0]) {
          break;
        }
      }
      // Eliminate complement values
      for (; coStartLocal < coEndLocal; --coEndLocal)
      {
        ValueType val = complementValues[coEndLocal - 1];
        if (val <= nv.m_Max[1]) {
          break;
        }
      }

      // Is the node a leaf - compute distances
      if (npx == 1U)
      {
        IndexType leafInd = stackNodes[stackIndex].m_Index;

        for (unsigned int i = 0; i < cornerCount; ++i)
        {
          double d = 0.0;
          for (unsigned int j = 0; j < dim; ++j)
          {
            double inddiff_j = ((double)corners.m_Points[i][j] - (double)leafInd[j]) * spacing[j];
            d += inddiff_j * inddiff_j;
          }

          // Compare d with all the distances recorded for the alpha levels (which are still in play)

          for (unsigned int j = inEndLocal; inStartLocal < j; --j)
          {
            double cur_j = m_Table.GetElement(j-1, i);
            if (d < cur_j)
            {
              m_Table.SetElement(j-1, i, d);
            } else {
              break;
            }
          }
          for (unsigned int j = coEndLocal; coStartLocal < j; --j)
          {
            double cur_j = m_Table.GetElement(j+inwardsCount-1, i);
            if (d < cur_j)
            {
              m_Table.SetElement(j+inwardsCount-1, i, d);
            } else {
              break;
            }
          }
        }
      }
      else
      { // Continue traversing the tree
        // Compute lower bound on distance for all pixels in the node
        IndexType innerNodeInd = stackNodes[stackIndex].m_Index;
        SizeType innerNodeSz = stackNodes[stackIndex].m_Size;
        double lowerBoundDistance = LowerBoundDistance<IndexType, SizeType, dim>(index, innerNodeInd, innerNodeSz, spacing);

        //std::cout << "LB: " << lowerBoundDistance << "\n";
        // Eliminate inwards values based on distance bounds
        for (; inStartLocal < inEndLocal; ++inStartLocal)
        {
          double cur_j = m_Table.GetElement(inStartLocal, 0);
          if (lowerBoundDistance <= cur_j)
            break;
        }
        // Eliminate complement values based on distance bounds
        for (; coStartLocal < coEndLocal; ++coStartLocal)
        {
          double cur_j = m_Table.GetElement(coStartLocal+inwardsCount, 0);
          if (lowerBoundDistance <= cur_j)
            break;
        }

        //std::cout << "inStartLocal: " << inStartLocal << ", inEndLocal: " << inEndLocal << " ";
        //std::cout << "coStartLocal: " << coStartLocal << ", coEndLocal: " << coEndLocal << " ";
        // If all alpha levels are eliminated, backtrack...
        if (inStartLocal == inEndLocal && coStartLocal == coEndLocal)
        {
          continue;
        }

        IndexType midIndex = innerNodeInd;
        unsigned int selIndex = LargestDimension<dim>(innerNodeSz);
        unsigned int maxSz = innerNodeSz[selIndex];

        midIndex[selIndex] = midIndex[selIndex] + maxSz / 2;
        SizeType sz1 = innerNodeSz;
        SizeType sz2 = innerNodeSz;

        sz1[selIndex] = sz1[selIndex] / 2;
        sz2[selIndex] = innerNodeSz[selIndex] - sz1[selIndex];

        unsigned int nodeIndex1 = nodeIndex + 1;
        unsigned int nodeIndex2 = nv.m_RightOffset;

        if (index[selIndex] < midIndex[selIndex])
        {
          stackNodes[stackIndex].m_Index = midIndex;
          stackNodes[stackIndex].m_Size = sz2;
          stackNodes[stackIndex].m_NodeIndex = nodeIndex2;
          stackNodes[stackIndex].m_InStart = inStartLocal;
          stackNodes[stackIndex].m_InEnd = inEndLocal;
          stackNodes[stackIndex].m_CoStart = coStartLocal;
          stackNodes[stackIndex].m_CoEnd = coEndLocal;
          ++stackIndex;
          stackNodes[stackIndex].m_Index = innerNodeInd;
          stackNodes[stackIndex].m_Size = sz1;
          stackNodes[stackIndex].m_NodeIndex = nodeIndex1;
          stackNodes[stackIndex].m_InStart = inStartLocal;
          stackNodes[stackIndex].m_InEnd = inEndLocal;
          stackNodes[stackIndex].m_CoStart = coStartLocal;
          stackNodes[stackIndex].m_CoEnd = coEndLocal;
          ++stackIndex;
        }
        else
        {
          stackNodes[stackIndex].m_Index = innerNodeInd;
          stackNodes[stackIndex].m_Size = sz1;
          stackNodes[stackIndex].m_NodeIndex = nodeIndex1;
          stackNodes[stackIndex].m_InStart = inStartLocal;
          stackNodes[stackIndex].m_InEnd = inEndLocal;
          stackNodes[stackIndex].m_CoStart = coStartLocal;
          stackNodes[stackIndex].m_CoEnd = coEndLocal;
          ++stackIndex;
          stackNodes[stackIndex].m_Index = midIndex;
          stackNodes[stackIndex].m_Size = sz2;
          stackNodes[stackIndex].m_NodeIndex = nodeIndex2;
          stackNodes[stackIndex].m_InStart = inStartLocal;
          stackNodes[stackIndex].m_InEnd = inEndLocal;
          stackNodes[stackIndex].m_CoStart = coStartLocal;
          stackNodes[stackIndex].m_CoEnd = coEndLocal;
          ++stackIndex;
        }

      } // End of else branch

    } // End main "recursion" loop

  } // End of Search function

}; // End of class
