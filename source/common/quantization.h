
#ifndef QUANTIZATION_H
#define QUANTIZATION_H

#include <cmath>

template <typename InType, typename OutType>
OutType QuantizeValue(InType input)
{
    return static_cast<OutType>(input);
}

template <>
auto QuantizeValue<unsigned char, unsigned short>(unsigned char input) -> unsigned short
{
    return static_cast<unsigned short>(input) * ((unsigned short)256U);
}

template <>
auto QuantizeValue<unsigned short, unsigned char>(unsigned short input) -> unsigned char
{
    return static_cast<unsigned char>(input / ((unsigned short)256U));
}

template <>
auto QuantizeValue<float, unsigned char>(float input) -> unsigned char
{
    float scaledInput = floor((double)input * 255.0 + 0.5);
    if(scaledInput <= 0.0f)
        return 0U;
    if(scaledInput >= 255.0f)
        return 255U;
    return static_cast<unsigned char>(scaledInput);
}

template <>
auto QuantizeValue<double, unsigned char>(double input) -> unsigned char
{
    double scaledInput = floor(input * 255.0 + 0.5);
    if(scaledInput <= 0.0)
        return (unsigned char)0U;
    if(scaledInput >= 255.0)
        return (unsigned char)255U;
    return static_cast<unsigned char>(scaledInput);
}

template <>
auto QuantizeValue<float, unsigned short>(float input) -> unsigned short
{
    float scaledInput = floor((double)input * 65535.0 + 0.5);
    if(scaledInput <= 0.0f)
        return (unsigned short)0U;
    if(scaledInput >= 65535.0f)
        return (unsigned short)65535U;
    return static_cast<unsigned short>(scaledInput);
}

template <>
auto QuantizeValue<double, unsigned short>(double input) -> unsigned short
{
    double scaledInput = floor(input * 65535.0 + 0.5);
    if(scaledInput <= 0.0)
        return (unsigned short)0U;
    if(scaledInput >= 65535.0)
        return (unsigned short)65535U;
    return static_cast<unsigned short>(scaledInput);
}

// We do not want to support signed types yet
/*
template <>
auto QuantizeValue<float, short>(float input) -> short
{
    float scaledInput = floor(input * 65535.0f + 0.5) - 32768.0f;
    if(scaledInput <= -32768.0f)
        return -32768;
    if(scaledInput >= 32767.0f)
        return 32767;
    return static_cast<short>(scaledInput);
}

template <>
auto QuantizeValue<double, short>(double input) -> short
{
    double scaledInput = floor(input * 65535.0 + 0.5) - 32768.0;
    if(scaledInput <= -32768.0)
        return -32768;
    if(scaledInput >= 32767.0)
        return 32767;
    return static_cast<short>(scaledInput);
}
*/

template <typename T>
inline T QuantizedValueMax()
{
    return static_cast<T>(1); // Only holds for floating point values
}

template <>
inline unsigned char QuantizedValueMax<unsigned char>()
{
    return ((unsigned char)255U);
}

template <>
inline unsigned short QuantizedValueMax<unsigned short>()
{
    return ((unsigned short)65535U);
}

template <typename T>
inline T QuantizedValueMin()
{
    return static_cast<T>(0);
}

template <>
inline unsigned char QuantizedValueMin<unsigned char>()
{
    return ((unsigned char)0U);
}

template <>
inline unsigned short QuantizedValueMin<unsigned short>()
{
    return ((unsigned short)0U);
}

#endif

