
#include <itkImage.h>
#include <itkArray.h>
#include <itkImageRegionConstIterator.h>
#include <cmath>

template <unsigned int ImageDimension>
inline double InterpolateDistances(itk::Vector<double, ImageDimension> frac, itk::Array<double> distances, itk::Vector<double, ImageDimension>& grad);

template <>
inline double InterpolateDistances<2U>(itk::Vector<double, 2U> frac, itk::Array<double> distances, itk::Vector<double, 2U>& grad) {
  // 0
  // Order for 2d: [(0, 0), (0, 1), (1, 0), (1, 1)]
  // 
  double step01 = distances[2] - distances[0];
  double step02 = distances[3] - distances[1];
  
  grad[0] = (1.0-frac[1]) * step11 + frac[1] * step12;
  grad[1] = (1.0-frac[0]) * step01 + frac[0] * step02;

  
  w0 = (1.0 - frac[0]) * (1.0 - frac[1]);
  w1 = (1.0 - frac[0]) * frac[1];
  w2 = frac[0] * (1.0 - frac[1]);
  w3 = frac[0] * frac[1];

  double d = (frac[0]*(1.0 - frac[1]) * step01 + frac[0] * frac[1] * step02);

  double step11 = distances[]
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
    if (cur == dim)
    {
      m_Corners[pos++] = index;
    }
    ComputeCorners(cur + 1, dim, pos, index);
    index[cur] = 1;
    ComputeCorners(cur + 1, dim, pos, index);
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
