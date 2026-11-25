#ifndef _DECODER_BAYER_H_
#define _DECODER_BAYER_H_

// Payload Type
enum class BayerType : int
{
    UNKNOWN = 0,    // unknown format
    RGGB,           // Red and blue pixels are arranged alternately, while green pixels are interlaced
    GRBG,           // Red and blue pixels are arranged alternately, with the first pixel in the first row being green
    GBRG,           // Red and blue pixels are arranged alternately, with the second pixel in the first row being blue
    BGGR            // Red and blue pixels are arranged alternately, with the second pixel in the first row being green
};

class DecoderBayer
{
public:
    explicit DecoderBayer();
    virtual ~DecoderBayer();

    bool execute(unsigned char* dst, unsigned char* src, const int& width, const int& height, const int& elemBytes, const int& format);

protected:

    BayerType getBayerType(const int& format);

    inline int getBlueIndex(const BayerType& bayer);
    inline bool isGreenFirst(const BayerType& bayer);

    template<typename T>
    bool decode(T* rgb, T* bayer, const int& cols, const int& rows, const BayerType& bayerType);
};

#endif  // _DECODER_BAYER_H_

