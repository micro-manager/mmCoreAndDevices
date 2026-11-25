#include "DecoderCallbackRaw.h"
#include "DecoderBayer.h"
#include "./sdk3/inc/PFNC.h"

#include <cstdint>
#include <new>
#define TUIMG_SAVE_RAWHEADER 0x400

DecoderCallbackRaw::DecoderCallbackRaw()
    : m_bufUnpacked(nullptr)
    , m_bufWidthStep(nullptr)
    , m_decodedRaw(false)
{
}

DecoderCallbackRaw::~DecoderCallbackRaw()
{
    destroyWidthStepBuffer();
    destroyBayerUnpackedBuffer();
}

bool DecoderCallbackRaw::allocWidthStepBuffer(const int& size)
{
    m_bufWidthStep = new(std::nothrow) unsigned char[size];

    return nullptr == m_bufWidthStep ? false : true;
}

bool DecoderCallbackRaw::allocBayerUnpackedBuffer(const int& size)
{
    m_bufUnpacked = new(std::nothrow) unsigned char[size + TUIMG_SAVE_RAWHEADER];

    return nullptr == m_bufUnpacked ? false : true;
}

void DecoderCallbackRaw::destroyWidthStepBuffer()
{
    TU_DELBUF(m_bufWidthStep);
}

void DecoderCallbackRaw::destroyBayerUnpackedBuffer()
{
    TU_DELBUF(m_bufUnpacked);
}

bool DecoderCallbackRaw::doDecoding(PTuCamRawImgHeader dst, PTuCamRawImgHeader src, bool isVertical)
{
    // Determine whether the data requires Bayer decoding
    if (isBayerPixelFormat(src->uiPixelFormat))
    {
        return decodeBayer(dst, src, isVertical);
    }

    return decodeGray(dst, src, isVertical);
}

int DecoderCallbackRaw::getUnpackedMovedBit(const int& pixelFormat)
{
    int moveBit = 0;

    switch (pixelFormat)
    {
    case PFNC_Mono10p:
    case GVSP_Mono10Packed:
    case GVSP_BayerGR10Packed:
    case GVSP_BayerRG10Packed:
    case GVSP_BayerGB10Packed:
    case GVSP_BayerBG10Packed:
        moveBit = 2;
        break;
    case PFNC_Mono12:
    case PFNC_Mono12p:
    case GVSP_Mono12Packed:
    case GVSP_BayerGR12Packed:
    case GVSP_BayerRG12Packed:
    case GVSP_BayerGB12Packed:
    case GVSP_BayerBG12Packed:
        moveBit = 4;
        break;
    default:
        break;
    }

    return moveBit;
}

int DecoderCallbackRaw::getPFNCTypeAfterVertical(const int& format)
{
    switch (format)
    {
    case PFNC_BayerRG8: return PFNC_BayerGB8;
    case PFNC_BayerGR8: return PFNC_BayerBG8;
    case PFNC_BayerGB8: return PFNC_BayerRG8;
    case PFNC_BayerBG8: return PFNC_BayerGR8;

    case PFNC_BayerRG10: return PFNC_BayerGB10;
    case PFNC_BayerGR10: return PFNC_BayerBG10;
    case PFNC_BayerGB10: return PFNC_BayerRG10;
    case PFNC_BayerBG10: return PFNC_BayerGR10;

    case PFNC_BayerRG10p: return PFNC_BayerGB10p;
    case PFNC_BayerGR10p: return PFNC_BayerBG10p;
    case PFNC_BayerGB10p: return PFNC_BayerRG10p;
    case PFNC_BayerBG10p: return PFNC_BayerGR10p;

    case PFNC_BayerRG12: return PFNC_BayerGB12;
    case PFNC_BayerGR12: return PFNC_BayerBG12;
    case PFNC_BayerGB12: return PFNC_BayerRG12;
    case PFNC_BayerBG12: return PFNC_BayerGR12;

    case PFNC_BayerRG12p: return PFNC_BayerGB12p;
    case PFNC_BayerGR12p: return PFNC_BayerBG12p;
    case PFNC_BayerGB12p: return PFNC_BayerRG12p;
    case PFNC_BayerBG12p: return PFNC_BayerGR12p;

    case PFNC_BayerRG14: return PFNC_BayerGB14;
    case PFNC_BayerGR14: return PFNC_BayerBG14;
    case PFNC_BayerGB14: return PFNC_BayerRG14;
    case PFNC_BayerBG14: return PFNC_BayerGR14;

    case PFNC_BayerRG16: return PFNC_BayerGB16;
    case PFNC_BayerGR16: return PFNC_BayerBG16;
    case PFNC_BayerGB16: return PFNC_BayerRG16;
    case PFNC_BayerBG16: return PFNC_BayerGR16;

    case GVSP_BayerRG10Packed: return GVSP_BayerGB10Packed;
    case GVSP_BayerGR10Packed: return GVSP_BayerBG10Packed;
    case GVSP_BayerGB10Packed: return GVSP_BayerRG10Packed;
    case GVSP_BayerBG10Packed: return GVSP_BayerGR10Packed;

    case GVSP_BayerRG12Packed: return GVSP_BayerGB12Packed;
    case GVSP_BayerGR12Packed: return GVSP_BayerBG12Packed;
    case GVSP_BayerGB12Packed: return GVSP_BayerRG12Packed;
    case GVSP_BayerBG12Packed: return GVSP_BayerGR12Packed;

    default:
        break;
    }

    return format;
}

bool DecoderCallbackRaw::isBayerPixelFormat(const int& pixelFormat)
{
    bool isBayer = false;

    switch (pixelFormat)
    {
    case PFNC_BayerGR8:
    case PFNC_BayerRG8:
    case PFNC_BayerGB8:
    case PFNC_BayerBG8:
    case PFNC_BayerGR10:
    case PFNC_BayerRG10:
    case PFNC_BayerGB10:
    case PFNC_BayerBG10:
    case PFNC_BayerGR12:
    case PFNC_BayerRG12:
    case PFNC_BayerGB12:
    case PFNC_BayerBG12:
    case PFNC_BayerGR14:
    case PFNC_BayerRG14:
    case PFNC_BayerGB14:
    case PFNC_BayerBG14:
    case PFNC_BayerGR16:
    case PFNC_BayerRG16:
    case PFNC_BayerGB16:
    case PFNC_BayerBG16:
    case PFNC_BayerGR10p:
    case PFNC_BayerRG10p:
    case PFNC_BayerGB10p:
    case PFNC_BayerBG10p:
    case PFNC_BayerGR12p:
    case PFNC_BayerRG12p:
    case PFNC_BayerGB12p:
    case PFNC_BayerBG12p:
    case GVSP_BayerGR10Packed:
    case GVSP_BayerRG10Packed:
    case GVSP_BayerGB10Packed:
    case GVSP_BayerBG10Packed:
    case GVSP_BayerGR12Packed:
    case GVSP_BayerRG12Packed:
    case GVSP_BayerGB12Packed:
    case GVSP_BayerBG12Packed:
        isBayer = true;
        break;
    default:
        break;
    }

    return (isBayer && m_decodedRaw);
}

bool DecoderCallbackRaw::unpackData(unsigned char* dst, unsigned char* src, const int& length, const int& movedBit)
{
    if (nullptr == dst || nullptr == src || (2 != movedBit && 4 != movedBit))
    {
        return false;
    }

    uint16_t bitLow = 0;
    uint16_t bitHigh = 0;

    int packedSize = length;
    int quarterPixels = (packedSize >> 2);

    auto packed = src;
    auto unpacked = (uint16_t*)dst;

    for (int i = 0; i < quarterPixels; ++i)
    {
        int index2 = i * 2;
        int index3 = i * 3;
        int index31 = index3 + 1;

        bitLow = (packed[(index3 + 2)] << movedBit) + (packed[index31] >> 4);
        bitHigh = (packed[index3] << movedBit) + (packed[index31] & 0x0F);

        unpacked[index2] = bitHigh;
        unpacked[(index2 + 1)] = bitLow;
    }

    return true;
}

bool DecoderCallbackRaw::flipVertically(PTuCamRawImgHeader src)
{
    if (nullptr == src || nullptr == src->pImgData || nullptr == m_bufWidthStep)
    {
        return false;
    }

    int i = 0;
    int height = src->usHeight;
    int heightHalf = height >> 1;
    int widthStep = src->usWidth * src->ucElemBytes * src->ucChannels;

    auto swap = m_bufWidthStep;
    auto head = src->pImgData;
    auto tail = src->pImgData + (height - 1) * widthStep;

    for (i = 0; i < heightHalf; ++i)
    {
        memcpy(swap, tail, widthStep);
        memcpy(tail, head, widthStep);
        memcpy(head, swap, widthStep);

        head += widthStep;
        tail -= widthStep;
    }

    return true;
}

bool DecoderCallbackRaw::flipVertically(unsigned char* dst, unsigned char* src, const int& widthStep, const int& height)
{
    if (nullptr == dst || nullptr == src)
    {
        return false;
    }

    unsigned char* head = nullptr;
    unsigned char* tail = nullptr;

    head = dst;
    tail = src + (height - 1) * widthStep;

    for (int i = 0; i < height; ++i)
    {
        memcpy(head, tail, widthStep);

        head += widthStep;
        tail -= widthStep;
    }

    return true;
}

bool DecoderCallbackRaw::decodeGray(PTuCamRawImgHeader dst, PTuCamRawImgHeader src, bool isVertical)
{
    auto movedBit = getUnpackedMovedBit(src->uiPixelFormat);
    auto unpacked = dst->pImgData + TUIMG_SAVE_RAWHEADER;

    // Determine whether data decompression is necessary
    if (0 != movedBit)
    {
        auto unpackedSize = src->usWidth * src->usHeight * src->ucElemBytes;
        unpackData(unpacked, src->pImgData, unpackedSize, movedBit);

        // Determine whether the data requires vertical mirroring
        if (isVertical)
        {
            flipVertically(dst);
        }

        return true;
    }

    if (isVertical)
    {
        int widthStep = src->usWidth * src->ucElemBytes * src->ucChannels;
        flipVertically(dst->pImgData, unpacked, widthStep, src->usHeight);
    }
    else
    {
        memcpy(unpacked, src->pImgData, src->uiImgSize);
    }

    return true;
}

bool DecoderCallbackRaw::decodeBayer(PTuCamRawImgHeader dst, PTuCamRawImgHeader src, bool isVertical)
{
    auto movedBit = getUnpackedMovedBit(src->uiPixelFormat);
    auto unpacked = src->pImgData;

    // 1.Determine whether data decompression is necessary
    if (0 != movedBit)
    {
        auto unpackedSize = src->usWidth * src->usHeight * src->ucElemBytes;
        unpackData(m_bufUnpacked + TUIMG_SAVE_RAWHEADER, src->pImgData, unpackedSize, movedBit);

        unpacked = m_bufUnpacked + TUIMG_SAVE_RAWHEADER;
    }

    // 2.Determine whether the data requires vertical mirroring
    int format = src->uiPixelFormat;
    if (isVertical)
    {
        TuCamRawImgHeader rawHead;
        memcpy(&rawHead, src, sizeof(TuCamRawImgHeader));
        rawHead.pImgData = unpacked;
        flipVertically(&rawHead);
        format = getPFNCTypeAfterVertical(format);
    }

    // 3.Decoding bayer
    DecoderBayer decoder;
    decoder.execute(dst->pImgData + TUIMG_SAVE_RAWHEADER, unpacked, src->usWidth, src->usHeight, src->ucElemBytes, format);

    return true;

}
