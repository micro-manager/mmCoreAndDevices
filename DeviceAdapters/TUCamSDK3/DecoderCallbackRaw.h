#ifndef _DECODER_CALLBACK_RAW_H_
#define _DECODER_CALLBACK_RAW_H_

#include "TuCamTypes.h"

class DecoderCallbackRaw
{
public:
    explicit DecoderCallbackRaw();
    virtual ~DecoderCallbackRaw();

    bool allocWidthStepBuffer(const int& size);
    bool allocBayerUnpackedBuffer(const int& size);

    void destroyWidthStepBuffer();
    void destroyBayerUnpackedBuffer();

    bool doDecoding(PTuCamRawImgHeader dst, PTuCamRawImgHeader src, bool isVertical = false);

    inline void setDecodedRaw(bool decodedRaw = false) { m_decodedRaw = decodedRaw; }

protected:

    int getUnpackedShift(const int& pixelFormat);
    int getUnpackedMovedBit(const int& pixelFormat);
    int getPFNCTypeAfterVertical(const int& format);
    bool isBayerPixelFormat(const int& pixelFormat);

    bool unpackData(unsigned char* dst, unsigned char* src, const int& length, const int& movedBit);

    bool flipVertically(PTuCamRawImgHeader src);
    bool flipVertically(unsigned char* dst, unsigned char* src, const int& widthStep, const int& height);

    bool decodeGray(PTuCamRawImgHeader dst, PTuCamRawImgHeader src, bool isVertical = false);
    bool decodeBayer(PTuCamRawImgHeader dst, PTuCamRawImgHeader src, bool isVertical = false);

    bool m_decodedRaw;

    unsigned char* m_bufUnpacked;
    unsigned char* m_bufWidthStep;

};

#endif  // _DECODER_CALLBACK_RAW_H_
