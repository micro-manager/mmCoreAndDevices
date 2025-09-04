///////////////////////////////////////////////////////////////////////////////
// MODULE:        PvDebayer.h
// SYSTEM:        ImageBase subsystem
// AUTHOR:        Jennifer West, jennifer_west@umanitoba.ca,
//                Nenad Amodaj, nenad@amodaj.com
//
// DESCRIPTION:   Debayer algorithms, adapted from:
//                http://www.umanitoba.ca/faculties/science/astronomy/jwest/plugins.html
//
// COPYRIGHT:     Jennifer West (University of Manitoba),
//                Exploratorium http://www.exploratorium.edu
//
// LICENSE:       This file is free for use, modification and distribution and
//                is distributed under terms specified in the BSD license
//                This file is distributed in the hope that it will be useful,
//                but WITHOUT ANY WARRANTY; without even the implied warranty
//                of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
//
//                IN NO EVENT SHALL THE COPYRIGHT OWNER OR
//                CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
//                INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES.
//
///////////////////////////////////////////////////////////////////////////////

#if !defined(_DEBAYER_)
#define _DEBAYER_

// MMDevice
#include "ImgBuffer.h"

/**
* RGB scales resulting from White Balance algorithm plugin
*/
typedef struct
{
    double r;
    double g;
    double b;
}
RGBscales;

#define CFA_RGGB 0
#define CFA_BGGR 1
#define CFA_GRBG 2
#define CFA_GBRG 3

#define ALG_REPLICATION 0
#define ALG_BILINEAR 1
#define ALG_SMOOTH_HUE 2
#define ALG_ADAPTIVE_SMOOTH_HUE 3

/**
* Utility class to build color image from the Bayer gray-scale image
* Based on the Debayer_Image plugin for ImageJ, by Jennifer West, University of Manitoba
*/
class PvDebayer
{
public:
    PvDebayer();
    ~PvDebayer();

    int Process(ImgBuffer& out, const ImgBuffer& in, int bitDepth);
    int Process(ImgBuffer& out, const unsigned char* in, int width, int height, int bitDepth);
    int Process(ImgBuffer& out, const unsigned short* in, int width, int height, int bitDepth);

    const std::vector<std::string>& GetOrders() const { return orders_; }
    const std::vector<std::string>& GetAlgorithms() const { return algorithms_; }
    const RGBscales& GetRGBScales() const { return rgbScales_; }

    void SetOrderIndex(int idx) { orderIndex_ = idx; }
    void SetAlgorithmIndex(int idx) { algoIndex_ = idx; }
    void SetRGBScales(const RGBscales& scales) { rgbScales_ = scales; }

private:
    template <typename T>
    int ProcessT(ImgBuffer& out, const T* input, int width, int height, int bitDepth);

    // Decodes input data and stores to scratch buffer vectors
    template<typename T>
    int DecodeT(const T* input, int width, int height, int rowOrder, int algorithm);

    template<typename T>
    void DecodeT_Replicate(const T* input, int width, int height, int rowOrder);
    template <typename T>
    void DecodeT_Bilinear(const T* input, int width, int height, int rowOrder);
    template <typename T>
    void DecodeT_Smooth(const T* input, int width, int height, int rowOrder);
    template <typename T>
    void DecodeT_AdaptiveSmooth(const T* input, int width, int height, int rowOrder);

    // Applies RGB scales to scratch buffers and stores to output
    int WhiteBalance(int* output, int width, int height, int bitDepth, int rowOrder);

    template <typename T>
    unsigned short GetPixel(const T* v, int x, int y, int width, int height);

    void SetPixel(std::vector<unsigned short>& v, unsigned short val, int x, int y, int width, int height);

private:
    std::vector<unsigned short> r_; // red scratch buffer
    std::vector<unsigned short> g_; // green scratch buffer
    std::vector<unsigned short> b_; // blue scratch buffer

    std::vector<std::string> orders_;
    std::vector<std::string> algorithms_;

    int orderIndex_;
    int algoIndex_;
    RGBscales rgbScales_;
};

#endif // !defined(_DEBAYER_)
