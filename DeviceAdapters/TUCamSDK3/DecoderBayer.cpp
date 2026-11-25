#include "DecoderBayer.h"
#include "./sdk3/inc/PFNC.h"
#include <string.h>

#if defined(_WIN32) || defined(__CYGWIN__)
#define CH0B  -4
#define CH0G  -3
#define CH0R  -2
#define CH1B  -1
#define CH1G  0
#define CH1R  1
#define CH2B  2
#define CH2G  3
#define CH2R  4
#elif defined(__linux)|| defined(__APPLE__)
#define CH0B  -2
#define CH0G  -3
#define CH0R  -4
#define CH1B  1
#define CH1G  0
#define CH1R  -1
#define CH2B  4
#define CH2G  3
#define CH2R  2
#endif

DecoderBayer::DecoderBayer()
{
}

DecoderBayer::~DecoderBayer()
{
}

bool DecoderBayer::execute(unsigned char* dst, unsigned char* src, const int& width, const int& height, const int& elemBytes, const int& format)
{
    if (nullptr == dst || nullptr == src)
    {
        return false;
    }

    BayerType bayerType = getBayerType(format);
    if (BayerType::UNKNOWN == bayerType)
    {
        return false;
    }

    if (2 == elemBytes)
    {
        auto rgb = (unsigned short*)(dst);
        auto bayer = (unsigned short*)(src);

        return decode(rgb, bayer, width, height, bayerType);
    }

    auto rgb = dst;
    auto bayer = src;

    return decode(rgb, bayer, width, height, bayerType);
}

BayerType DecoderBayer::getBayerType(const int& format)
{
    switch (format)
    {
    case PFNC_BayerRG8:
    case PFNC_BayerRG10:
    case PFNC_BayerRG12:
    case PFNC_BayerRG14:
    case PFNC_BayerRG16:
    case PFNC_BayerRG10p:
    case PFNC_BayerRG12p:
    case GVSP_BayerRG10Packed:
    case GVSP_BayerRG12Packed:
        return BayerType::RGGB;
    case PFNC_BayerGR8:
    case PFNC_BayerGR10:
    case PFNC_BayerGR12:
    case PFNC_BayerGR14:
    case PFNC_BayerGR16:
    case PFNC_BayerGR10p:
    case PFNC_BayerGR12p:
    case GVSP_BayerGR10Packed:
    case GVSP_BayerGR12Packed:
        return BayerType::GRBG;
    case PFNC_BayerGB8:
    case PFNC_BayerGB10:
    case PFNC_BayerGB12:
    case PFNC_BayerGB14:
    case PFNC_BayerGB16:
    case PFNC_BayerGB10p:
    case PFNC_BayerGB12p:
    case GVSP_BayerGB10Packed:
    case GVSP_BayerGB12Packed:
        return BayerType::GBRG;
    case PFNC_BayerBG8:
    case PFNC_BayerBG10:
    case PFNC_BayerBG12:
    case PFNC_BayerBG14:
    case PFNC_BayerBG16:
    case PFNC_BayerBG10p:
    case PFNC_BayerBG12p:
    case GVSP_BayerBG10Packed:
    case GVSP_BayerBG12Packed:
        return BayerType::BGGR;
    default:
        break;
    }

    return BayerType::UNKNOWN;
}

inline int DecoderBayer::getBlueIndex(const BayerType& bayer)
{
    return (BayerType::RGGB == bayer || BayerType::GRBG == bayer) ? -1 : 1;
}

inline bool DecoderBayer::isGreenFirst(const BayerType& bayer)
{
    return (BayerType::GRBG == bayer || BayerType::GBRG == bayer) ? true : false;
}

template<typename T>
bool DecoderBayer::decode(T* rgb, T* bayer, const int& cols, const int& rows, const BayerType& bayerType)
{
    if (nullptr == rgb || nullptr == bayer)
    {
        return false;
    }

    int subWidth = cols - 2;
    int subHeight = rows - 2;

    int srcStep = cols;
    int dstStep = cols * 3;

    int bytePerElement = sizeof(T);

    int srcNextRow1Col0 = srcStep;
    int srcNextRow1Col1 = srcStep + 1;
    int srcNextRow1Col2 = srcStep + 2;
    int srcNextRow1Col3 = srcStep + 3;
    int srcNextRow2Col0 = (srcStep << 1);
    int srcNextRow2Col1 = srcNextRow2Col0 + 1;
    int srcNextRow2Col2 = srcNextRow2Col0 + 2;
    int srcNextRow2Col3 = srcNextRow2Col0 + 3;

    bool isGFirst = isGreenFirst(bayerType);
    //  bool isGFirstFixed = isGFirst;

    int blueIndex = getBlueIndex(bayerType);

    auto src = bayer;
    auto dst = rgb + dstStep + 3 + 1;

    int redIndex1 = blueIndex * static_cast<int>(CH1R);
    int redIndex2 = blueIndex > 0 ? CH2R : CH2B;
    int blueIndex1 = blueIndex * static_cast<int>(CH1B);
    int blueIndex2 = 6 - redIndex2;

    for (; subHeight-- > 0; src += srcStep, dst += dstStep)
    {
        auto srcLine = src;
        auto dstLine = dst;
        auto srcLineEnd = srcLine + subWidth;
        auto srcLineEndSub = srcLineEnd - 2;

        if (subWidth <= 0)
        {
            dst[CH0B] = dst[CH0G] = dst[CH0R] = dst[dstStep - 1] = dst[dstStep] = dst[dstStep + 1] = 0;
        }

        if (isGFirst)
        {
            dstLine[blueIndex] = (srcLine[1] + srcLine[srcNextRow2Col1] + 1) >> 1;
            dstLine[0] = srcLine[srcNextRow1Col1];
            dstLine[-blueIndex] = (srcLine[srcNextRow1Col0] + srcLine[srcNextRow1Col2] + 1) >> 1;

            srcLine++;
            dstLine += 3;
        }

        for (; srcLine <= srcLineEndSub; srcLine += 2, dstLine += 6)
        {
            dstLine[redIndex1] = (srcLine[0] + srcLine[2] + srcLine[srcNextRow2Col0] + srcLine[srcNextRow2Col2] + 2) >> 2;
            dstLine[CH1G] = (srcLine[1] + srcLine[srcNextRow1Col0] + srcLine[srcNextRow1Col2] + srcLine[srcNextRow2Col1] + 2) >> 2;
            dstLine[blueIndex1] = srcLine[srcNextRow1Col1];

            dstLine[redIndex2] = (srcLine[2] + srcLine[srcNextRow2Col2] + 1) >> 1;
            dstLine[CH2G] = ((srcLine[srcNextRow1Col2] << 2) + srcLine[1] + srcLine[3] + srcLine[srcNextRow2Col1] + srcLine[srcNextRow2Col3]) >> 3;
            //          dstLine[CH2G] = srcLine[srcNextRow1Col2];
            dstLine[blueIndex2] = (srcLine[srcNextRow1Col1] + srcLine[srcNextRow1Col3] + 1) >> 1;
        }

        if (srcLine < srcLineEnd)
        {
            dstLine[blueIndex] = (srcLine[0] + srcLine[2] + srcLine[srcNextRow2Col0] + srcLine[srcNextRow2Col2] + 2) >> 2;
            dstLine[0] = (srcLine[1] + srcLine[srcNextRow1Col0] + srcLine[srcNextRow1Col2] + srcLine[srcNextRow2Col1] + 2) >> 2;
            dstLine[-blueIndex] = srcLine[srcNextRow1Col1];

            srcLine++;
            dstLine += 3;
        }

        // 修复右侧1列
        dstLine[CH1B] = dstLine[CH0B];
        dstLine[CH1G] = dstLine[CH0G];
        dstLine[CH1R] = dstLine[CH0R];

        // 修复左侧1列
        dstLine[-dstStep + 2] = dstLine[-dstStep + 5];
        dstLine[-dstStep + 3] = dstLine[-dstStep + 6];
        dstLine[-dstStep + 4] = dstLine[-dstStep + 7];

        blueIndex = -blueIndex;
        isGFirst = !isGFirst;

        redIndex1 = -redIndex1;
        blueIndex1 = -blueIndex1;
        redIndex2 = 6 - redIndex2;
        blueIndex2 = 6 - blueIndex2;
    }

    auto fixSize = static_cast<size_t>(dstStep * bytePerElement);

    // 修复第1行
    memcpy(rgb, rgb + dstStep, fixSize);

    // 修复最后1行
    auto lastLine = rgb + (rows - 1) * dstStep;
    memcpy(lastLine, lastLine - dstStep, fixSize);

    return true;
}

