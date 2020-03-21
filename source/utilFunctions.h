
// TODO: Make all this stuff actually work, or delete it!
// /Johan

#ifndef UTIL_FUNCTIONS_H
#define UTIL_FUNCTIONS_H

#include "./common/itkImageProcessingTools.h"

#include "itkCheckerBoardImageFilter.h"
#include "itkFlatStructuringElement.h"
#include "itkBinaryDilateImageFilter.h"

#include <fstream>
#include <sstream>
#include <string>

template <typename ImageType>
typename ImageType::Pointer Chessboard(typename ImageType::Pointer image1, typename ImageType::Pointer image2, int cells)
{
    itk::FixedArray<unsigned int, ImageType::ImageDimension> pattern;
    pattern.Fill(cells);

    typedef typename itk::IPT<double, ImageType::ImageDimension> IPT;
    typedef typename itk::CheckerBoardImageFilter<ImageType> CheckerBoardFilterType;
    typename CheckerBoardFilterType::Pointer checkerBoardFilter = CheckerBoardFilterType::New();
    checkerBoardFilter->SetInput1(image1);
    checkerBoardFilter->SetInput2(image2);
    checkerBoardFilter->SetCheckerPattern(pattern);
    checkerBoardFilter->Update();
    return checkerBoardFilter->GetOutput();
}

template <typename ImageType>
typename itk:: BlackAndWhiteChessboard(typename ImageType::Pointer refImage, int cells)
{
    typedef itk::IPT<typename ImageType::ValueType, ImageType::ImageDimension> IPT;
    return Chessboard<ImageType>(IPT::ZeroImage(refImage->GetLargestPossibleRegion().GetSize()), IPT::ConstantImage(1.0, refImage->GetLargestPossibleRegion().GetSize()), cells);
}


typename IPT::BinaryImagePointer DilateMask(typename IPT::BinaryImagePointer mask, int radiusValue) {
    using StructuringElementType = itk::FlatStructuringElement< ImageDimension >;
    typename StructuringElementType::RadiusType radius;
    radius.Fill( radiusValue );
    StructuringElementType structuringElement = typename StructuringElementType::Ball( radius );

    using BinaryDilateImageFilterType = itk::BinaryDilateImageFilter<typename IPT::BinaryImageType, typename IPT::BinaryImageType, StructuringElementType>;
    typename BinaryDilateImageFilterType::Pointer dilateFilter = BinaryDilateImageFilterType::New();
    dilateFilter->SetInput(mask);
    dilateFilter->SetKernel(structuringElement);

    dilateFilter->Update();
    return dilateFilter->GetOutput();
}


std::vector<std::string> read_strings(std::string path)
{
    std::vector<std::string> result;
    std::ifstream infile(path.c_str());

    std::string line;
    while (std::getline(infile, line))
    {
        result.push_back(line);
    }

    return result;
}

template <typename TPixelType, unsigned int TImageDimension>
ImagePointer readRawIntegerFile(std::string path)
{
    std::vector<std::string> hdr = read_strings(path);

    assert(hdr.size() >= 1 + TImageDimension * 2);

    std::string data_path = hdr[0];
    int nPixels[TImageDimension];
    double szVoxels[TImageDimension];

    for (int i = 0; i < TImageDimension; ++i)
    {
        nPixels[i] = atoi(hdr[1 + i].c_str());
        szVoxels[i] = atof(hdr[1 + TImageDimension + i].c_str());
    }

    typedef itk::RawImageIO<TPixelType, TImageDimension> IOType;
    typedef typename IOType::Pointer IOPointer;

    IOPointer io = IOType::New();

    for (int i = 0; i < TImageDimension; ++i)
    {
        io->SetDimensions(i, nPixels[i]);
        io->SetSpacing(i, szVoxels[i]);
    }

    io->SetHeaderSize(io->GetImageSizeInPixels() * 0);
    io->SetByteOrderToLittleEndian();

    typedef itk::Image<TPixelType, TImageDimension> IntermediateImageType;

    typedef itk::ImageFileReader<IntermediateImageType> ReaderType;

    typedef itk::IPT<double, ImageDimension> IPT;

    typename ReaderType::Pointer reader = ReaderType::New();

    reader->SetFileName(data_path.c_str());
    reader->SetImageIO(io);

    return itk::ConvertImageFromIntegerFormat<TPixelType>(reader->GetOutput());
}



void saveSlice(ImagePointer image, int d, int ind, std::string path)
{
    typedef itk::Image<double, ImageDimension - 1> SliceImageType;

    typename ImageType::RegionType slice = image->GetLargestPossibleRegion();
    slice.SetIndex(d, ind);
    slice.SetSize(d, 0);
    typedef itk::IPT<double, ImageDimension - 1> IPT;

    using ExtractFilterType = itk::ExtractImageFilter<ImageType, SliceImageType>;
    typename ExtractFilterType::Pointer extract = ExtractFilterType::New();
    extract->SetDirectionCollapseToIdentity();
    extract->InPlaceOn();
    extract->SetInput(image);
    extract->SetExtractionRegion(slice);

    typename SliceImageType::Pointer sliceImage = extract->GetOutput();
    IPT::SaveImage(path.c_str(), sliceImage, false);
}

void print_difference_image_stats(ImagePointer image1, ImagePointer image2, const char* name) {
    typedef itk::IPT<double, ImageDimension> IPT;
    typename ImageType::Pointer diff = IPT::DifferenceImage(image1, image2);

    typename IPT::ImageStatisticsData movingStats = IPT::ImageStatistics(diff);

    std::cout << name << " mean: " << movingStats.mean << ", std: " << movingStats.sigma << std::endl;
}

typename ImageType::Pointer ApplyTransform(ImagePointer refImage, ImagePointer floImage, TransformPointer transform)
{
    typedef itk::ResampleImageFilter<
        ImageType,
        ImageType>
        ResampleFilterType;

    typedef itk::IPT<double, ImageDimension> IPT;

    typename ResampleFilterType::Pointer resample = ResampleFilterType::New();

    resample->SetTransform(transform);
    resample->SetInput(floImage);

    resample->SetSize(refImage->GetLargestPossibleRegion().GetSize());
    resample->SetOutputOrigin(refImage->GetOrigin());
    resample->SetOutputSpacing(refImage->GetSpacing());
    resample->SetOutputDirection(refImage->GetDirection());
    resample->SetDefaultPixelValue(0.5);

    resample->UpdateLargestPossibleRegion();

    return resample->GetOutput();
}

ImagePointer JacobianDeterminantFilter(DisplacementFieldImagePointer dfield) {
  typedef typename itk::DisplacementFieldJacobianDeterminantFilter<DisplacementFieldImageType, PixelType, ImageType> FilterType;
  typename FilterType::Pointer filter = FilterType::New();

  filter->SetInput(dfield);
  filter->SetUseImageSpacingOn();

  filter->Update();

  return filter->GetOutput();
}

DisplacementFieldImagePointer LoadDisplacementField(std::string path) {
  typedef typename itk::ImageFileReader<DisplacementFieldImageType> FieldReaderType;
  typename FieldReaderType::Pointer reader = FieldReaderType::New();

  reader->SetFileName(path.c_str());

  reader->Update();

  return reader->GetOutput();
}

void SaveDisplacementField(DisplacementFieldImagePointer image, std::string path) {
  typedef typename itk::ImageFileWriter<DisplacementFieldImageType> FieldWriterType;
  typename FieldWriterType::Pointer writer = FieldWriterType::New();

  writer->SetInput(image);

  writer->SetFileName(path.c_str());

  try {
    writer->Update();
  } catch (itk::ExceptionObject & err) {
    std::cerr << "Error while writing displacement field: " << err << std::endl;
  }
}

#endif
