#ifndef _TUCAM_TYPES_H_
#define _TUCAM_TYPES_H_

#include <cstdint>
#include <cstring>

#ifndef TU_DELBUF
#define TU_DELBUF(pBuf)  { if (NULL != (pBuf)) { delete[] (pBuf); (pBuf) = NULL; } }
#endif

#ifndef TU_DELOBJ
#define TU_DELOBJ(pObj)  { if (NULL != (pObj)) { delete (pObj); (pObj) = NULL; } }
#endif

/** Lightweight frame descriptor used by Snap/sequence copy path (maps from TUC_IMAGE_INFO). */
struct TuCamImageFrame
{
    uint32_t width;
    uint32_t height;
    uint32_t widthStep;
    uint32_t imgSize;
    uint8_t  channels;
    uint8_t  elemBytes;
    uint8_t* pBuffer;
    bool     owned;

    TuCamImageFrame()
        : width(0), height(0), widthStep(0), imgSize(0)
        , channels(1), elemBytes(1), pBuffer(NULL), owned(false)
    {}
};

/** RAW header for callback decoder (fields aligned with legacy TUCAM_RAWIMG_HEADER). */
struct TuCamRawImgHeader
{
    uint16_t usWidth;
    uint16_t usHeight;
    uint16_t usXOffset;
    uint16_t usYOffset;
    uint16_t usXPadding;
    uint16_t usYPadding;
    uint16_t usOffset;
    uint8_t  ucDepth;
    uint8_t  ucChannels;
    uint8_t  ucElemBytes;
    uint32_t uiIndex;
    uint32_t uiImgSize;
    uint32_t uiPixelFormat;
    double   dblExposure;
    uint8_t* pImgData;
    double   dblTimeStamp;
    double   dblTimeLast;
};

typedef TuCamRawImgHeader* PTuCamRawImgHeader;

#endif // _TUCAM_TYPES_H_
