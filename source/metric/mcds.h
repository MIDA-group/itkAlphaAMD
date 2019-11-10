
#include <itkImage.h>
#include <itkArray.h>
#include <itkImageRegionConstIterator.h>
#include <cmath>

template <typename Dim>
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
  itk::FixedArray<itk::FixedArray<T, 1U>, 2U> m_Points;
};

template <typename T>
struct CornerPoints<T, 2U> {
  itk::FixedArray<itk::FixedArray<T, 2U>, 4U> m_Points;
};

template <typename T>
struct CornerPoints<T, 3U> {
  itk::FixedArray<itk::FixedArray<T, 3U>, 8U> m_Points;
};

template <typename T, unsigned int Dim>
void ComputeCornersRec(unsigned int cur, unsigned int& pos, itk::FixedArray<T, Dim>& index, CornerPoints<T, Dim>& out) {
  if(cur == 0) {
    out.m_Points[pos++] = index;
  } else {
    ComputeCornersRec(cur-1, pos, index, out);
    index[cur-1] = 1;
    ComputeCornersRec(cur-1, pos, index, out);
    index[cur-1] = 0;
  }

}

template <typename T, unsigned int Dim>
CornerPoints<T, Dim> ComputeCorners(unsigned int dim) {
  CornerPoints<T, Dim> res;
  itk::FixedArray<T, Dim> index;
  index.Fill(0);
  unsigned int pos = 0;
  ComputeCornersRec(Dim, pos, index, res);
  
  return res;
}

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

template <unsigned int ImageDimension>
inline double InterpolateDistances(itk::Vector<double, ImageDimension> frac, ValuedCornerPoints<ImageDimension>& distanceValues, itk::Vector<double, ImageDimension>& grad);

// Linear interpolation
template <>
inline double InterpolateDistances<1U>(itk::Vector<double, 1U> frac, ValuedCornerPoints<1U> distanceValues, itk::Vector<double, 1U>& grad) {
  double xx = frac[0];
  double ixx = 1.0 - xx;
  grad[0] = distanceValues.m_Values[1] - distanceValues.m_Values[0];
  return ixx * distanceValues.m_Values[0] + xx * distanceValues.m_Values[1];
}

// Bilinear interpolation
template <>
inline double InterpolateDistances<2U>(itk::Vector<double, 2U> frac, ValuedCornerPoints<2U> distanceValues, itk::Vector<double, 2U>& grad) {
  double xx = frac[0];
  double yy = frac[1];
  double ixx = 1.0 - xx;
  double iyy = 1.0 - yy;

  // 0
  // Order for 2d: [(0, 0), (0, 1), (1, 0), (1, 1)]
  // 
  double v_00 = distanceValues.m_Values[0];
  double v_01 = distanceValues.m_Values[1];
  double v_10 = distanceValues.m_Values[2];
  double v_11 = distanceValues.m_Values[3];

  double step_10_00 = v_10 - v_00;
  double step_11_01 = v_11 - v_01;
  double step_10_00 = v_10 - v_00;
  double step_11_01 = v_11 - v_01;
  
  grad[0] = iyy * step_10_00 + yy * step_11_01;
  grad[1] = ixx * step_01_00 + xx * step_11_10;
  return v_00 * ixx * iyy + v_10 * xx * iyy + v_01 * ixx * yy + v_11 * xx * yy;
}

// Trilinear interpolation
template <>
inline double InterpolateDistances<3U>(itk::Vector<double, 3U> frac, ValuedCornerPoints<3U> distanceValues, itk::Vector<double, 3U>& grad) {
  double xx = frac[0];
  double yy = frac[1];
  double zz = frac[2];
  double ixx = 1.0 - xx;
  double iyy = 1.0 - yy;
  double izz = 1.0 - zz;

  // 0
  // Order for 3d: [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]
  // 
  double v_000 = distanceValues.m_Values[0];
  double v_001 = distanceValues.m_Values[1];
  double v_010 = distanceValues.m_Values[2];
  double v_011 = distanceValues.m_Values[3];
  double v_100 = distanceValues.m_Values[4];
  double v_101 = distanceValues.m_Values[5];
  double v_110 = distanceValues.m_Values[6];
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
  unsigned int acc = 0;
  for (unsigned int i = 0; i < ImageDimension; ++i)
  {
    acc += sz[i];
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

template <typename IndexType, typename SizeType, unsigned int ImageDimension, SpacingType>
inline double LowerBoundDistance(IndexType pnt, IndexType rectOrigin, SizeType rectSz, SpacingType spacing &sp)
{
  double d = 0;
  for (unsigned int i = 0; i < ImageDimension; ++i)
  {
    // If outside:
    if (pnt[i] + 1 < rectOrigin[i])
    {
      double d_i = (double)(rectOrigin[i] - pnt[i] - 1) * sp[i];
      d += d_i * d_i;
    }
    if (rectOrigin[i] + rectSz[i] + 1 < pnt[i])
    {
      double d_i = (double)(pnt[i] - rectOrigin[i] - 1) * sp[i];
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

template <typename ImageType>
class MCDS
{
public:
  typedef typename ImageType::Pointer ImagePointer;
  typedef typename ImageType::RegionType RegionType;
  typedef typename ImageType::SizeType SizeType;
  typedef typename ImageType::IndexValueType IndexValueType;
  typedef typename ImageType::SizeValueType SizeValueType;
  typedef typename ImageType::IndexType IndexType;
  typedef typename ImageType::ValueType ValueType;
  typedef itk::Vector<ValueType, 2U> NodeValueType;

  void SetImage(ImagePointer image)
  {
    m_Image = image;
  }

  void SetOne(ValueType one)
  {
    m_One = one;
  }

  void SetMaxSampleCount(unsigned int count)
  {
    m_MaxSampleCount = count;
  }

  // Builds the kd-tree
  void Initialize()
  {
    m_Array.SetSize(0);

    // Compute height
    constexpr unsigned int dim = ImageType::ImageDimension;
    RegionType region = m_Image->GetLargestPossibleRegion();
    SizeType sz = region.GetSize();

    m_Height = 0;
    for (unsigned int i = 0; i < dim; ++i)
    {
      unsigned int logsz = (unsigned int)ceil(log2((double)sz[i]));
      m_Height += logsz;
    }

    unsigned int nodeCount = (unsigned int)(pow(2.0, (double)m_Height) + 0.5);
    m_Array.SetSize(nodeCount);
    m_Array.Fill(0);

    BuildTreeRec(1, region.GetIndex(), sz, m_Height);
    unsigned int pos = 0;
    IndexType cornerIndex;
    cornerIndex.Fill(0);

    ComputeCorners(0, ImageType::ImageDimension, pos, cornerIndex);

    // Initialize table
    m_Table.SetSize(m_MaxSampleCount, m_Corners.GetSize());
  }

private:
  ImagePointer m_Image;
  itk::Array<NodeValueType> m_Array;
  unsigned int m_Height;
  unsigned int m_MaxSampleCount;
  ValueType m_One;
  itk::Array<IndexType> m_Corners;
  itk::Array2D<double> m_Table;

  struct StackNode
  {
    IndexType m_Index;
    SizeType m_Size;
    unsigned int m_NodeIndex;
    unsigned int m_InwardsOffset;
    unsigned int m_ComplementOffset;
  };

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

  void BuildTreeRec(unsigned int nodeIndex, IndexType index, SizeType sz, unsigned int depthCountDown)
  {
    constexpr unsigned int dim = ImageType::ImageDimension;

    typedef itk::ImageRegionConstIterator<ImageType> IteratorType;

    unsigned int szCount = PixelCount<dim>(sz);

    if (szCount == 0U)
    {
      ;
    }
    else if (szCount == 1U)
    {
      NodeValueType nv;
      nv[0] = m_Image->GetPixel(index);
      nv[1] = m_One - nv[0];
      m_Array[nodeIndex - 1] = nv;
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

      unsigned int nodeIndex1 = nodeIndex * 2;
      unsigned int nodeIndex2 = nodeIndex * 2 + 1;

      BuildTreeRec(nodeIndex1, index, sz1, depthCountDown - 1);
      BuildTreeRec(nodeIndex2, midIndex, sz2, depthCountDown - 1);

      NodeValueType n1 = m_Array[nodeIndex1 - 1];
      NodeValueType n2 = m_Array[nodeIndex2 - 1];

      // Compute the maximum of the two nodes, for each channel
      for (unsigned int i = 0; i < 2U; ++i)
      {
        if (n2[i] > n1[i])
          n1[i] = n2[i];
      }

      m_Array[nodeIndex - 1] = n1;
    }
  }

  void Search(
      IndexType index,
      itk::Array<ValueType> &inwardsValues,
      unsigned int inwardsCount,
      itk::Array<ValueType> &complementValues,
      unsigned int complementCount,
      itk::Array<double> &distOut)
  {
    constexpr unsigned int dim = ImageType::ImageDimension;

    typedef itk::ImageRegionConstIterator<ImageType> IteratorType;
    typedef typename ImageType::SpacingType SpacingType;

    ImageType *image = m_Image.GetPointer();
    SpacingType spacing = image->GetSpacing();
    ValueType one = m_One;

    RegionType region = image->GetLargestPossibleRegion();

    unsigned int rows = inwardsCount + complementCount;
    unsigned int cornerCount = m_Corners.GetSize();

    // Stack
    StackNode stackNodes[92];
    unsigned int stackIndex = 1;

    // Initialize the stack state
    stackNodes[0].m_Index = region.GetIndex();
    stackNodes[0].m_Size = region.GetSize();
    stackNodes[0].m_NodeIndex = 1;
    stackNodes[0].m_InwardsOffset = inwardsCount;
    stackNodes[0].m_ComplementOffset = complementCount;

    while (stackIndex > 0)
    {
      --stackIndex;

      unsigned int npx = PixelCount<dim>(sizes[stackIndex]);
      // Can I remove this?
      if (npx == 0U)
      {
        continue;
      }

      int inwardsOffsetLocal = stackNodes[stackIndex].m_InwardsOffset[stackIndex];
      int complementOffsetLocal = stackNodes[stackIndex].m_ComplementOffset[stackIndex];
      unsigned int nodeIndex = stackNodes[stackIndex].m_NodeIndex;
      NodeValueType nv = m_Array[nodeIndex];

      // Eliminate inwards values
      for (; inwardsOffsetLocal > 0; --inwardsOffsetLocal)
      {
        if (inwardsValues[inwardsOffsetLocal - 1] <= nv[0])
          break;
      }
      // Eliminate complement values
      for (; complementOffsetLocal > 0; --complementOffsetLocal)
      {
        if (complementValues[complementOffsetLocal - 1] <= nv[1])
          break;
      }

      // Is the node a leaf - compute distances
      if (npx == 1U)
      {
        IndexType leafInd = stackNodes[stackIndex].m_Index;

        for (unsigned int i = 0; i < cornerCount; ++i)
        {
          IndexType ind = index;
          IndexType crnr = m_Corners[i];
          for (unsigned int j = 0; j < dim; ++j)
          {
            ind[j] = ind[j] + crnr[j];
          }
          double d = 0.0;
          for (unsigned int j = 0; j < dim; ++j)
          {
            double inddiff_j = ((double)ind[j] - (double)leafInd[j]) * spacing[j];
            d += inddiff_j * inddiff_j;
          }

          // Compare d with all the distances recorded for the alpha levels still in play

          for (unsigned int j = 0; j < inwardsOffsetLocal; ++j)
          {
            double cur_j = m_Table.GetElement(j, i);
            // Note - there may be an optimization possible here due to sorted alpha levels...
            if (d < cur_j)
            {
              m_Table.SetElement(j, i, d);
            }
          }
          for (unsigned int j = 0; j < complementOffsetLocal; ++j)
          {
            double cur_j = m_Table.GetElement(j + inwardsCount, i);
            // Note - there may be an optimization possible here due to sorted alpha levels...
            if (d < cur_j)
            {
              m_Table.SetElement(j, i, d);
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

        // Eliminate inwards values based on distance bounds
        for (; inwardsOffsetLocal > 0; --inwardsOffsetLocal)
        {
          double cur_j = m_Table.GetElement(inwardsOffsetLocal - 1, 0);
          if (lowerBoundDistance < cur_j)
            break;
        }
        // Eliminate complement values based on distance bounds
        for (; complementOffsetLocal > 0; --complementOffsetLocal)
        {
          double cur_j = m_Table.GetElement(complementOffsetLocal + inwardsCount - 1, 0);
          if (lowerBoundDistance < cur_j)
            break;
        }

        // If all alpha levels are eliminated, backtrack...
        if (inwardsOffsetLocal + complementOffsetLocal == 0U)
        {
          continue;
        }

        IndexType midIndex = index;
        unsigned int selIndex = LargestDimension<dim>(sz);
        unsigned int maxSz = sz[selIndex];

        midIndex[selIndex] = midIndex[selIndex] + maxSz / 2;
        SizeType sz1 = sz;
        SizeType sz2 = sz;

        sz1[selIndex] = sz1[selIndex] / 2;
        sz2[selIndex] = sz[selIndex] - sz1[selIndex];

        unsigned int nodeIndex1 = nodeIndex * 2;
        unsigned int nodeIndex2 = nodeIndex * 2 + 1;

        if (index[selIndex] < midIndex[selIndex])
        {
          stackNodes[stackIndex].m_Index = midIndex;
          stackNodes[stackIndex].m_Size = sz2;
          stackNodes[stackIndex].m_NodeIndex = nodeIndex2;
          stackNodes[stackIndex].m_InwardsOffset = inwardsOffsetLocal;
          stackNodes[stackIndex].m_ComplementOffset = complementOffsetLocal;
          ++stackIndex;
          stackNodes[stackIndex].m_Index = index;
          stackNodes[stackIndex].m_Size = sz1;
          stackNodes[stackIndex].m_NodeIndex = nodeIndex1;
          stackNodes[stackIndex].m_InwardsOffset = inwardsOffsetLocal;
          stackNodes[stackIndex].m_ComplementOffset = complementOffsetLocal;
          ++stackIndex;
        }
        else
        {
          stackNodes[stackIndex].m_Index = index;
          stackNodes[stackIndex].m_Size = sz1;
          stackNodes[stackIndex].m_NodeIndex = nodeIndex1;
          stackNodes[stackIndex].m_InwardsOffset = inwardsOffsetLocal;
          stackNodes[stackIndex].m_ComplementOffset = complementOffsetLocal;
          ++stackIndex;
          stackNodes[stackIndex].m_Index = midIndex;
          stackNodes[stackIndex].m_Size = sz2;
          stackNodes[stackIndex].m_NodeIndex = nodeIndex2;
          stackNodes[stackIndex].m_InwardsOffset = inwardsOffsetLocal;
          stackNodes[stackIndex].m_ComplementOffset = complementOffsetLocal;
          ++stackIndex;
        }

      } // End of else branch

    } // End main "recursion" loop

  } // End of Search function

}; // End of class
