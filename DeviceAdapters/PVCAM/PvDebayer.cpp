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

#include "PvDebayer.h"

#include "MMDeviceConstants.h"

// System
#include <algorithm>
#include <cassert>
#include <cmath>

PvDebayer::PvDebayer()
{
    orders_.push_back("R-G-R-G");
    orders_.push_back("B-G-B-G");
    orders_.push_back("G-R-G-R");
    orders_.push_back("G-B-G-B");

    algorithms_.push_back("Replication");
    algorithms_.push_back("Bilinear");
    algorithms_.push_back("Smooth-Hue");
    algorithms_.push_back("Adaptive-Smooth-Hue");

    // Default settings
    orderIndex_ = CFA_RGGB;
    algoIndex_ = ALG_REPLICATION;
    rgbScales_.r = 1.0;
    rgbScales_.g = 1.0;
    rgbScales_.b = 1.0;
}

PvDebayer::~PvDebayer()
{
}

int PvDebayer::Process(ImgBuffer& out, const ImgBuffer& in, int bitDepth)
{
    switch (in.Depth())
    {
    case 1: {
        const unsigned char* input = in.GetPixels();
        return ProcessT(out, input, in.Width(), in.Height(), bitDepth);
    }
    case 2: {
        const unsigned short* input = reinterpret_cast<const unsigned short*>(in.GetPixels());
        return ProcessT(out, input, in.Width(), in.Height(), bitDepth);
    }
    default:
        return DEVICE_UNSUPPORTED_DATA_FORMAT;
    }
}

int PvDebayer::Process(ImgBuffer& out, const unsigned char* in, int width, int height, int bitDepth)
{
    return ProcessT(out, in, width, height, bitDepth);
}

int PvDebayer::Process(ImgBuffer& out, const unsigned short* in, int width, int height, int bitDepth)
{
    return ProcessT(out, in, width, height, bitDepth);
}

template <typename T>
int PvDebayer::ProcessT(ImgBuffer& out, const T* input, int width, int height, int bitDepth)
{
    if ((size_t)bitDepth > sizeof(T) * 8)
    {
        assert(false);
        return DEVICE_INVALID_INPUT_PARAM;
    }

    if (width < 2 || height < 2)
        return DEVICE_NOT_SUPPORTED;

    assert(sizeof(int) == 4);
    out.Resize(width, height, 4);

    int* output = reinterpret_cast<int*>(out.GetPixelsRW());

    const size_t numPixels = (size_t)width * height;
    // Resize scratch buffers, don't shrink, like ImgBuffer::Resize()
    if (r_.size() < numPixels)
    {
        r_.resize(numPixels);
        g_.resize(numPixels);
        b_.resize(numPixels);
    }

    // TODO: There is a bug in the conversion routines that not all R, G and B
    //       pixels are set, especially on the image border (see TODOs).
    //       When Binning is switched from 1x1 to 2x2 or other the image cannot
    //       be debayered (obviously, but it is incorrectly), however when
    //       switching back to 1x1 some color channels contain some incorrect
    //       values from the 2x2 binned image.
    //       The same happens after increasing ROI size.
    //       Initializing the scratch buffers to zeroes solves the issue until
    //       the conversion routines are fixes.
    std::fill(r_.begin(), r_.begin() + numPixels, (unsigned short)0);
    std::fill(g_.begin(), g_.begin() + numPixels, (unsigned short)0);
    std::fill(b_.begin(), b_.begin() + numPixels, (unsigned short)0);

    // TODO: The algorithm itself is very inefficient, we should consider using
    //       OpenCV or at least revisit the algorithms, at least the SetPixel(),
    //       GetPixel() and all the loops using vectors needs to be improved.
    //       (remove conditions from Get,SetPixel, avoid using std::vector)

    int err = DecodeT(input, width, height, orderIndex_, algoIndex_);
    if (err != DEVICE_OK)
        return err;

    return WhiteBalance(output, width, height, bitDepth, orderIndex_);
}

template<typename T>
int PvDebayer::DecodeT(const T* input, int width, int height, int rowOrder, int algorithm)
{
    switch (algorithm)
    {
    case ALG_REPLICATION:
        DecodeT_Replicate(input, width, height, rowOrder);
        return DEVICE_OK;
    case ALG_BILINEAR:
        DecodeT_Bilinear(input, width, height, rowOrder);
        return DEVICE_OK;
    case ALG_SMOOTH_HUE:
        DecodeT_Smooth(input, width, height, rowOrder);
        return DEVICE_OK;
    case ALG_ADAPTIVE_SMOOTH_HUE:
        DecodeT_AdaptiveSmooth(input, width, height, rowOrder);
        return DEVICE_OK;
    default:
        return DEVICE_NOT_SUPPORTED;
    }
}

// Replication algorithm
template <typename T>
void PvDebayer::DecodeT_Replicate(const T* input, int width, int height, int rowOrder)
{
    int x, y, i00, i01, i02, i10, i11, i12, i20, i21, i22;

    #define _SET_RGB_(i, ir, ig, ib) do { \
        r_[i] = input[ir]; \
        g_[i] = input[ig]; \
        b_[i] = input[ib]; \
    } while (0)

    if (rowOrder == CFA_RGGB || rowOrder == CFA_BGGR) // Green slash
    {
        // Code is for BGGR, RGGB needs to swap R & B channels, done in WhiteBalance()
        // BGGR y\x:    0      1      2
        //        0:  i00-B  i10-G  i20-B
        //        1:  i01-G  i11-R  i21-G
        //        2:  i02-B  i12-G  i22-B

        // All internal pixels without left & top edges,
        // and without right / bottom edges when width /height is even.
        for (y = 1; y+1 < height; y += 2)
        {
            for (x = 1; x+1 < width; x += 2)
            {
                i11 = x + y * width;
                i10 = i11 - width;
                i12 = i11 + width;
                i01 = i11 - 1;
                i21 = i11 + 1;
                i00 = i10 - 1;
                i20 = i10 + 1;
                i02 = i12 - 1;
                i22 = i12 + 1;
                _SET_RGB_(i11, i11, i01, i00);
                _SET_RGB_(i21, i11, i21, i20);
                _SET_RGB_(i12, i11, i12, i02);
                _SET_RGB_(i22, i11, i12, i22);
            }
        }
        // Left edge
        for (y = 1, x = 0; y+1 < height; y += 2)
        {
            i01 = x + y * width;
            i00 = i01 - width;
            i02 = i01 + width;
            i11 = i01 + 1;
            i12 = i02 + 1;
            _SET_RGB_(i01, i11, i01, i00);
            _SET_RGB_(i02, i11, i12, i02);
        }
        // Top edge
        for (y = 0, x = 1; x+1 < width; x += 2)
        {
            i10 = x + y * width;
            i11 = i10 + width;
            i00 = i10 - 1;
            i20 = i10 + 1;
            _SET_RGB_(i10, i11, i10, i00);
            _SET_RGB_(i20, i11, i10, i20);
        }
        // Right edge with top-right corner
        if ((width % 2) == 0)
        {
            x = width-1; // width is even thus x is odd
            for (y = 1; y+1 < height; y += 2)
            {
                i11 = x + y * width;
                i10 = i11 - width;
                i12 = i11 + width;
                i01 = i11 - 1;
                i00 = i10 - 1;
                i02 = i12 - 1;
                _SET_RGB_(i11, i11, i01, i00);
                _SET_RGB_(i12, i11, i12, i02);
            }
            // Top-right corner
            y = 0;
            i10 = x + y * width;
            i11 = i10 + width;
            i00 = i10 - 1;
            _SET_RGB_(i10, i11, i10, i00);
        }
        // Bottom edge with bottom-left corner
        if ((height % 2) == 0)
        {
            y = height-1; // height is even thus y is odd
            for (x = 1; x+1 < width; x += 2)
            {
                i11 = x + y * width;
                i10 = i11 - width;
                i01 = i11 - 1;
                i21 = i11 + 1;
                i00 = i10 - 1;
                i20 = i10 + 1;
                _SET_RGB_(i11, i11, i01, i00);
                _SET_RGB_(i21, i11, i21, i20);
            }
            // Bottom-left corner
            x = 0;
            i01 = x + y * width;
            i00 = i01 - width;
            i11 = i01 + 1;
            _SET_RGB_(i01, i11, i01, i00);
        }
        // Bottom-right corner
        if ((width % 2) == 0 && (height % 2) == 0)
        {
            x = width-1; // width is even thus x is odd
            y = height-1; // height is even thus y is odd
            i11 = x + y * width;
            i10 = i11 - width;
            i01 = i11 - 1;
            i00 = i10 - 1;
            _SET_RGB_(i11, i11, i01, i00);
        }
        // Top-left corner
        _SET_RGB_(0, 1+width, 1, 0);
    }
    else if (rowOrder == CFA_GRBG || rowOrder == CFA_GBRG) // Green backslash
    {
        // Code is for GRBG, GBRG needs to swap R & B channels, done in WhiteBalance()
        // GRBG y\x:    0      1      2
        //        0:  i00-G  i10-R  i20-G
        //        1:  i01-B  i11-G  i21-B
        //        2:  i02-G  i12-R  i22-G

        // All internal pixels without left & top edges,
        // and without right / bottom edges when width /height is even.
        for (y = 1; y+1 < height; y += 2)
        {
            for (x = 1; x+1 < width; x += 2)
            {
                i11 = x + y * width;
                i10 = i11 - width;
                i12 = i11 + width;
                i01 = i11 - 1;
                i21 = i11 + 1;
                i00 = i10 - 1;
                //i20 = i10 + 1;
                i02 = i12 - 1;
                i22 = i12 + 1;
                _SET_RGB_(i11, i10, i11, i01);
                _SET_RGB_(i21, i10, i11, i21);
                _SET_RGB_(i12, i12, i02, i01);
                _SET_RGB_(i22, i12, i22, i21);
            }
        }
        // Left edge
        for (y = 1, x = 0; y+1 < height; y += 2)
        {
            i01 = x + y * width;
            i00 = i01 - width;
            i02 = i01 + width;
            i10 = i00 + 1;
            i11 = i01 + 1;
            i12 = i02 + 1;
            _SET_RGB_(i01, i10, i11, i01);
            _SET_RGB_(i02, i12, i02, i01);
        }
        // Top edge
        for (y = 0, x = 1; x+1 < width; x += 2)
        {
            i10 = x + y * width;
            i11 = i10 + width;
            i00 = i10 - 1;
            i20 = i10 + 1;
            i01 = i11 - 1;
            i21 = i11 + 1;
            _SET_RGB_(i10, i10, i00, i01);
            _SET_RGB_(i20, i10, i20, i21);
        }
        // Right edge with top-right corner
        if ((width % 2) == 0)
        {
            x = width-1; // width is even thus x is odd
            for (y = 1; y+1 < height; y += 2)
            {
                i11 = x + y * width;
                i10 = i11 - width;
                i12 = i11 + width;
                i01 = i11 - 1;
                i00 = i10 - 1;
                i02 = i12 - 1;
                _SET_RGB_(i11, i10, i11, i01);
                _SET_RGB_(i12, i12, i02, i01);
            }
            // Top-right corner
            y = 0;
            i10 = x + y * width;
            i11 = i10 + width;
            i01 = i11 - 1;
            i00 = i10 - 1;
            _SET_RGB_(i10, i10, i00, i01);
        }
        // Bottom edge with bottom-left corner
        if ((height % 2) == 0)
        {
            y = height-1; // height is even thus y is odd
            for (x = 1; x+1 < width; x += 2)
            {
                i11 = x + y * width;
                i10 = i11 - width;
                i01 = i11 - 1;
                i21 = i11 + 1;
                i00 = i10 - 1;
                i20 = i10 + 1;
                _SET_RGB_(i11, i10, i11, i01);
                _SET_RGB_(i21, i10, i11, i21);
            }
            // Bottom-left corner
            x = 0;
            i01 = x + y * width;
            i00 = i01 - width;
            i10 = i00 + 1;
            i11 = i01 + 1;
            _SET_RGB_(i01, i10, i11, i01);
        }
        // Bottom-right corner
        if ((width % 2) == 0 && (height % 2) == 0)
        {
            x = width-1; // width is even thus x is odd
            y = height-1; // height is even thus y is odd
            i11 = x + y * width;
            i10 = i11 - width;
            i01 = i11 - 1;
            _SET_RGB_(i11, i10, i11, i01);
        }
        // Top-left corner
        _SET_RGB_(0, 1, 0, width);
    }

    #undef _SET_RGB_
}

// Bilinear algorithm
template <typename T>
void PvDebayer::DecodeT_Bilinear(const T* input, int width, int height, int rowOrder)
{
    unsigned int R1, R2, R3, R4;
    unsigned int G1, G2, G3, G4;
    unsigned int B1, B2, B3, B4;

    if (rowOrder == CFA_RGGB || rowOrder == CFA_BGGR) // Green slash
    {
        // Code is for BGGR, RGGB needs to swap R & B channels, done in WhiteBalance()
        // BGGR y\x:   0   1   2   3
        //       -1:   G   R   G   R
        //        0:   B   G   B   G
        //        1:   G   R   G   R
        //        2:   B   G   B   G
        //        3:   G   R   G   R

        // Top-right greens
        for (int y = 0; y < height; y += 2)
        {
            for (int x = -1; x < width; x += 2) // Starts at -1, not from +1!
            {
                // BGGR y\x:   0   1   2   3
                //       -1:   G   R   G4  R
                //        0:   B  *G1 #B  *G2
                //        1:   G   R   G3  R
                //        2:   B  *G  #B  *G
                //        3:   G   R   G   R
                //             ^
                //             \--- Green is missing on first column even rows if x starts at +1
                G1 = GetPixel(input, x  , y  , width, height);
                G2 = GetPixel(input, x+2, y  , width, height, (T)G1);
                if (G1 == 0) G1 = G2; // Fix on left edge
                G3 = GetPixel(input, x+1, y+1, width, height, (T)G1);
                G4 = GetPixel(input, x+1, y-1, width, height, (T)G3);

                SetPixel(g_, (unsigned short)( G1                      ), x  , y, width, height); // *
                SetPixel(g_, (unsigned short)((G1 + G2 + G3 + G4) / 4.0), x+1, y, width, height); // #
            }
        }
        // Bottom-left greens
        for (int y = 1; y < height; y += 2)
        {
            for (int x = 0; x < width; x += 2)
            {
                // BGGR y\x:   0   1   2   3
                //       -1:   G   R   G   R
                //        0:   B   G4  B   G
                //        1:  *G1 #R  *G2 #R
                //        2:   B   G3  B   G
                //        3:  *G  #R  *G  #R
                G1 = GetPixel(input, x  , y  , width, height);
                G2 = GetPixel(input, x+2, y  , width, height, (T)G1);
                G3 = GetPixel(input, x+1, y+1, width, height, (T)G1);
                G4 = GetPixel(input, x+1, y-1, width, height, (T)G3);

                SetPixel(g_, (unsigned short)( G1                      ), x  , y, width, height); // *
                SetPixel(g_, (unsigned short)((G1 + G2 + G3 + G4) / 4.0), x+1, y, width, height); // #
            }
        }

        // Blue channel for BGGR, red for RGGB (needs swap)
        for (int y = 0; y < height; y += 2)
        {
            for (int x = 0; x < width; x += 2)
            {
                // BGGR y\x:   0   1   2   3
                //       -1:   G   R   G   R
                //        0:  *B1 #G  *B2 #G
                //        1:  +G  ^R  +G  ^R
                //        2:  *B3 #G  *B4 #G
                //        3:  +G  ^R  +G  ^R
                B1 = GetPixel(input, x  , y  , width, height);
                B2 = GetPixel(input, x+2, y  , width, height, (T)B1);
                B3 = GetPixel(input, x  , y+2, width, height, (T)B1);
                B4 = GetPixel(input, x+2, y+2, width, height, (T)B3);

                SetPixel(b_, (unsigned short)( B1                      ), x  , y  , width, height); // *
                SetPixel(b_, (unsigned short)((B1 + B2          ) / 2.0), x+1, y  , width, height); // #
                SetPixel(b_, (unsigned short)((B1 + B3          ) / 2.0), x  , y+1, width, height); // +
                SetPixel(b_, (unsigned short)((B1 + B2 + B3 + B4) / 4.0), x+1, y+1, width, height); // ^
            }
        }

        // Red channel for BGGR, blue for RGGB (needs swap)
        for (int y = -1; y < height; y += 2) // Starts at -1, not from +1!
        {
            for (int x = -1; x < width; x += 2) // Starts at -1, not from +1!
            {
                // BGGR y\x:   0   1   2   3
                //       -1:   G   R   G   R
                //        0:   B   G   B   G   <--- Red is missing on first row if y starts at +1
                //        1:   G  *R1 #G  *R2
                //        2:   B  +G  ^B  +G
                //        3:   G  *R3 #G  *R4
                //             ^
                //             \--- Red is missing on first column if x starts at +1
                R1 = GetPixel(input, x  , y  , width, height);
                R2 = GetPixel(input, x+2, y  , width, height, (T)R1);
                if (R1 == 0) R1 = R2; // Fix on left edge
                R3 = GetPixel(input, x  , y+2, width, height, (T)R1);
                if (R1 == 0) R1 = R2 = R3; // Fix on top edge
                R4 = GetPixel(input, x+2, y+2, width, height, (T)R3);
                if (R3 == 0) R1 = R2 = R3 = R4; // Fix on top-left corner

                SetPixel(r_, (unsigned short)( R1                      ), x  , y  , width, height); // *
                SetPixel(r_, (unsigned short)((R1 + R2          ) / 2.0), x+1, y  , width, height); // #
                SetPixel(r_, (unsigned short)((R1 + R3          ) / 2.0), x  , y+1, width, height); // +
                SetPixel(r_, (unsigned short)((R1 + R2 + R3 + R4) / 4.0), x+1, y+1, width, height); // ^
            }
        }
    }
    else if (rowOrder == CFA_GRBG || rowOrder == CFA_GBRG) // Green backslash
    {
        // Code is for GRBG, GBRG needs to swap R & B channels, done in WhiteBalance()
        // GRBG y\x:   0   1   2   3
        //       -1:   B   G   B   G
        //        0:   G   R   G   R
        //        1:   B   G   B   G
        //        2:   G   R   G   R
        //        3:   B   G   B   G

        // Top-left greens
        for (int y = 0; y < height; y += 2)
        {
            for (int x = 0; x < width; x += 2)
            {
                // GRBG y\x:   0   1   2   3
                //       -1:   B   G4  B   G
                //        0:  *G1 #R  *G2 #R
                //        1:   B   G3  B   G
                //        2:  *G  #R  *G  #R
                //        3:   B   G   B   G
                G1 = GetPixel(input, x  , y  , width, height);
                G2 = GetPixel(input, x+2, y  , width, height, (T)G1);
                G3 = GetPixel(input, x+1, y+1, width, height, (T)G1);
                G4 = GetPixel(input, x+1, y-1, width, height, (T)G3);

                SetPixel(g_, (unsigned short)( G1                      ), x  , y, width, height); // *
                SetPixel(g_, (unsigned short)((G1 + G2 + G3 + G4) / 4.0), x+1, y, width, height); // #
            }
        }
        // Bottom-right greens
        for (int y = 1; y < height; y += 2)
        {
            for (int x = -1; x < width; x += 2) // Starts at -1, not from +1!
            {
                // GRBG y\x:   0   1   2   3
                //       -1:   B   G   B   G
                //        0:   G   R   G4  R
                //        1:   B  *G1 #B  *G2
                //        2:   G   R   G3  R
                //        3:   B  *G  #B  *G
                //             ^
                //             \--- Green is missing on first column odd rows if x starts at +1
                G1 = GetPixel(input, x  , y  , width, height);
                G2 = GetPixel(input, x+2, y  , width, height, (T)G1);
                if (G1 == 0) G1 = G2; // Fix on left edge
                G3 = GetPixel(input, x+1, y+1, width, height, (T)G1);
                G4 = GetPixel(input, x+1, y-1, width, height, (T)G3);

                SetPixel(g_, (unsigned short)( G1                      ), x  , y, width, height); // *
                SetPixel(g_, (unsigned short)((G1 + G2 + G3 + G4) / 4.0), x+1, y, width, height); // #
            }
        }

        // Blue channel for GRBG, red for GBRG (needs swap)
        for (int y = -1; y < height; y += 2) // Starts at -1, not from +1!
        {
            for (int x = 0; x < width; x += 2)
            {
                // GRBG y\x:   0   1   2   3
                //       -1:   B   G   B   G
                //        0:   G   R   G   R   <--- Blue is missing on first row if y starts at +1
                //        1:  *B1 #G  *B2 #G
                //        2:  +G  ^R  +G  ^R
                //        3:  *B3 #G  *B4 #G
                B1 = GetPixel(input, x  , y  , width, height);
                B2 = GetPixel(input, x+2, y  , width, height, (T)B1);
                B3 = GetPixel(input, x  , y+2, width, height, (T)B1);
                B4 = GetPixel(input, x+2, y+2, width, height, (T)B3);
                if (B2 == 0) B1 = B2 = B3; // Fix on top edge

                SetPixel(b_, (unsigned short)( B1                      ), x  , y  , width, height); // *
                SetPixel(b_, (unsigned short)((B1 + B2          ) / 2.0), x+1, y  , width, height); // #
                SetPixel(b_, (unsigned short)((B1 +      B3     ) / 2.0), x  , y+1, width, height); // +
                SetPixel(b_, (unsigned short)((B1 + B2 + B3 + B4) / 4.0), x+1, y+1, width, height); // ^
            }
        }

        // Red channel for GRBG, blue for GBRG (needs swap)
        for (int y = 0; y < height; y += 2)
        {
            for (int x = -1; x < width; x += 2) // Starts at -1, not from +1!
            {
                // GRBG y\x:   0   1   2   3
                //       -1:   B   G   B   G
                //        0:   G  *R1 #G  *R2
                //        1:   B  +G  ^B  +G
                //        2:   G  *R3 #G  *R4
                //        3:   B  +G  ^B  +G
                //             ^
                //             \--- Red is missing on first column if x starts at +1
                R1 = GetPixel(input, x  , y  , width, height);
                R2 = GetPixel(input, x+2, y  , width, height, (T)R1);
                if (R1 == 0) R1 = R2; // Fix on left edge
                R3 = GetPixel(input, x  , y+2, width, height, (T)R1);
                R4 = GetPixel(input, x+2, y+2, width, height, (T)R3);

                SetPixel(r_, (unsigned short)( R1                      ), x  , y  , width, height); // *
                SetPixel(r_, (unsigned short)((R1 + R2          ) / 2.0), x+1, y  , width, height); // #
                SetPixel(r_, (unsigned short)((R1 +      R3     ) / 2.0), x  , y+1, width, height); // +
                SetPixel(r_, (unsigned short)((R1 + R2 + R3 + R4) / 4.0), x+1, y+1, width, height); // ^
            }
        }
    }
}

// Smooth Hue algorithm
template <typename T>
void PvDebayer::DecodeT_Smooth(const T* input, int width, int height, int rowOrder)
{
    double G1, G2, G3, G4, G5, G6, G9;
    double B1, B2, B3, B4;
    double R1, R2, R3, R4;

    if (rowOrder == CFA_RGGB || rowOrder == CFA_BGGR) // Green slash
    {
        // Code is for BGGR, RGGB needs to swap R & B channels, done in WhiteBalance()
        // BGGR y\x:  -1   0   1   2   3
        //       -1:   R   G   R   G   R
        //        0:   G   B   G   B   G
        //        1:   R   G   R   G   R
        //        2:   G   B   G   B   G
        //        3:   R   G   R   G   R

        // Solve for green pixels first, it's needed for red and blue channels

        // Top-right greens
        for (int y = 0; y < height; y += 2)
        {
            for (int x = 1; x < width; x += 2)
            {
                // BGGR y\x:  -1   0   1   2   3
                //       -1:   R   G6  R   G4  R
                //        0:   G  ^B  *G1 +B  *G2
                //        1:   R   G5  R   G3  R
                //        2:   G  ^B  *G  #B  *G
                //        3:   R   G   R   G   R
                G1 = GetPixel(input, x  , y  , width, height);
                G2 = GetPixel(input, x+2, y  , width, height);
                G3 = GetPixel(input, x+1, y+1, width, height);
                G4 = GetPixel(input, x+1, y-1, width, height);

                SetPixel(g_, (unsigned short)G1, x, y, width, height); // *

                if (y == 0)
                    SetPixel(g_, (unsigned short)((G1 + G2 + G3     ) / 3.0), x+1, y, width, height); // +
                else
                    SetPixel(g_, (unsigned short)((G1 + G2 + G3 + G4) / 4.0), x+1, y, width, height); // #

                // TODO: No reason for this case, maybe only if (x == width-1)
                if (x == 1)
                {
                    // TODO: Should take G6 instead of G4.
                    //       We should exclude G4 and/or G5 from calculation if zero.
                    G5 = GetPixel(input, x-1, y+1, width, height);
                    SetPixel(g_, (unsigned short)((G1 + G4 + G5) / 3.0), x-1, y, width, height); // ^
                }
            }
        }
        // Bottom-left greens
        for (int y = 1; y < height; y += 2)
        {
            for (int x = 0; x < width; x += 2)
            {
                // BGGR y\x:  -1   0   1   2   3
                //       -1:   R   G   R   G   R
                //        0:   G   B   G4  B   G
                //        1:   R  *G1 +R  *G2 #R
                //        2:   G   B   G3  B   G
                //        3:   R  *G  +R  *G  #R
                G1 = GetPixel(input, x  , y  , width, height);
                G2 = GetPixel(input, x+2, y  , width, height);
                G3 = GetPixel(input, x+1, y+1, width, height);
                G4 = GetPixel(input, x+1, y-1, width, height);

                SetPixel(g_, (unsigned short)G1, x, y, width, height); // *

                // TODO: G1 and G4 are always valid.
                //       We should exclude G2 and/or G3 from calculation if zero.
                if (x == 0)
                    SetPixel(g_, (unsigned short)((G1 + G2 + G3     ) / 3.0), x+1, y, width, height); // +
                else
                    SetPixel(g_, (unsigned short)((G1 + G2 + G3 + G4) / 4.0), x+1, y, width, height); // #
            }
        }
        // BGGR y\x:  -1   0   1   2   3
        //       -1:   R   G   R   G   R
        //        0:   G  *B   G2  B   G
        //        1:   R   G1  R   G   R
        //        2:   G   B   G   B   G
        //        3:   R   G   R   G   R
        G1 = GetPixel(input, 0, 1, width, height);
        G2 = GetPixel(input, 1, 0, width, height);
        SetPixel(g_, (unsigned short)((G1 + G2) / 2.0), 0, 0, width, height); // *

        // Blue channel for BGGR, red for RGGB (needs swap)
        for (int y = 0; y < height; y += 2)
        {
            for (int x = 0; x < width; x += 2)
            {
                // BGGR y\x:  -1   0   1   2   3
                //       -1:   R   G   R   G   R
                //        0:   G  *B1 #G5 *B2 #G
                //        1:   R  +G6 ^R9 +G  ^R
                //        2:   G  *B3 #G  *B4 #G
                //        3:   R  +G  ^R  +G  ^R
                B1 = GetPixel(input, x  , y  , width, height);
                B2 = GetPixel(input, x+2, y  , width, height);
                B3 = GetPixel(input, x  , y+2, width, height);
                B4 = GetPixel(input, x+2, y+2, width, height);
                G1 = GetPixel(g_.data(), x  , y  , width, height);
                G2 = GetPixel(g_.data(), x+2, y  , width, height);
                G3 = GetPixel(g_.data(), x  , y+2, width, height);
                G4 = GetPixel(g_.data(), x+2, y+2, width, height);
                G5 = GetPixel(g_.data(), x+1, y  , width, height);
                G6 = GetPixel(g_.data(), x  , y+1, width, height);
                G9 = GetPixel(g_.data(), x+1, y+1, width, height);
                // TODO: Set to B1-B4 instead of 1
                if (G1 == 0) G1 = 1;
                if (G2 == 0) G2 = 1;
                if (G3 == 0) G3 = 1;
                if (G4 == 0) G4 = 1;

                SetPixel(b_, (unsigned short)(B1                                    ), x  , y  , width, height); // *
                SetPixel(b_, (unsigned short)(G5/2 * (B1/G1 + B2/G2                )), x+1, y  , width, height); // #
                SetPixel(b_, (unsigned short)(G6/2 * (B1/G1 +         B3/G3        )), x  , y+1, width, height); // +
                SetPixel(b_, (unsigned short)(G9/4 * (B1/G1 + B2/G2 + B3/G3 + B4/G4)), x+1, y+1, width, height); // ^
            }
        }

        // Red channel for BGGR, blue for RGGB (needs swap)
        for (int y = 1; y < height; y += 2)
        {
            for (int x = 1; x < width; x += 2)
            {
                // BGGR y\x:  -1   0   1   2   3
                //       -1:   R   G   R   G   R
                //        0:   G   B   G   B   G   <--- TODO: Red is missing on first row
                //        1:   R   G  *R1 #G5 *R2
                //        2:   G   B  +G6 ^B9 +G
                //        3:   R   G  *R3 #G  *R4
                //                 ^
                //                 \--- TODO: Red is missing on first column
                R1 = GetPixel(input, x  , y  , width, height);
                R2 = GetPixel(input, x+2, y  , width, height);
                R3 = GetPixel(input, x  , y+2, width, height);
                R4 = GetPixel(input, x+2, y+2, width, height);
                G1 = GetPixel(g_.data(), x  , y  , width, height);
                G2 = GetPixel(g_.data(), x+2, y  , width, height);
                G3 = GetPixel(g_.data(), x  , y+2, width, height);
                G4 = GetPixel(g_.data(), x+2, y+2, width, height);
                G5 = GetPixel(g_.data(), x+1, y  , width, height);
                G6 = GetPixel(g_.data(), x  , y+1, width, height);
                G9 = GetPixel(g_.data(), x+1, y+1, width, height);
                // TODO: Set to R1-R4 instead of 1
                if (G1 == 0) G1 = 1;
                if (G2 == 0) G2 = 1;
                if (G3 == 0) G3 = 1;
                if (G4 == 0) G4 = 1;

                SetPixel(r_, (unsigned short)(R1                                    ), x  , y  , width, height); // *
                SetPixel(r_, (unsigned short)(G5/2 * (R1/G1 + R2/G2                )), x+1, y  , width, height); // #
                SetPixel(r_, (unsigned short)(G6/2 * (R1/G1 +         R3/G3        )), x  , y+1, width, height); // +
                SetPixel(r_, (unsigned short)(G9/4 * (R1/G1 + R2/G2 + R3/G3 + R4/G4)), x+1, y+1, width, height); // ^
            }
        }
    }
    else if (rowOrder == CFA_GRBG || rowOrder == CFA_GBRG) // Green backslash
    {
        // Code is for GRBG, GBRG needs to swap R & B channels, done in WhiteBalance()
        // GRBG y\x:  -1   0   1   2   3
        //       -1:   G   B   G   B   G
        //        0:   R   G   R   G   R
        //        1:   G   B   G   B   G
        //        2:   R   G   R   G   R
        //        3:   G   B   G   B   G

        // Solve for green pixels first, it's needed for red and blue channels

        // Top-left greens
        for (int y = 0; y < height; y += 2)
        {
            for (int x = 0; x < width; x += 2)
            {
                // GRBG y\x:  -1   0   1   2   3
                //       -1:   G6  B   G4  B   G
                //        0:   R  *G1 +R  *G2 +R
                //        1:   G5  B   G3  B   G
                //        2:   R  *G  #R  *G  #R
                //        3:   G   B   G   B   G
                G1 = GetPixel(input, x  , y  , width, height);
                G2 = GetPixel(input, x+2, y  , width, height);
                G3 = GetPixel(input, x+1, y+1, width, height);
                G4 = GetPixel(input, x+1, y-1, width, height);

                SetPixel(g_, (unsigned short)G1, x, y, width, height); // *

                if (y == 0)
                    SetPixel(g_, (unsigned short)((G1 + G2 + G3     ) / 3.0), x+1, y, width, height); // +
                else
                    SetPixel(g_, (unsigned short)((G1 + G2 + G3 + G4) / 4.0), x+1, y, width, height); // #

                // TODO: x cannot be 1 because it's always even
                if (x == 1)
                {
                    // TODO: Should take G6 instead of G4.
                    //       We should exclude G4 and/or G5 from calculation if zero.
                    G5 = GetPixel(input, x-1, y+1, width, height);
                    SetPixel(g_, (unsigned short)((G1 + G4 + G5) / 3.0), x-1, y, width, height); // ^
                }
            }
        }
        // Bottom-right greens
        for (int y = 1; y < height; y += 2)
        {
            for (int x = 1; x < width; x += 2)
            {
                // GRBG y\x:  -1   0   1   2   3
                //       -1:   G   B   G   B   G
                //        0:   R   G   R   G4  R
                //        1:   G   B  *G1 #B  *G2
                //        2:   R   G   R   G3  R
                //        3:   G   B  *G  #B  *G
                //                 ^
                //                 \--- TODO: Green is missing on first column odd rows
                G1 = GetPixel(input, x  , y  , width, height);
                G2 = GetPixel(input, x+2, y  , width, height);
                G3 = GetPixel(input, x+1, y+1, width, height);
                G4 = GetPixel(input, x+1, y-1, width, height);

                SetPixel(g_, (unsigned short)G1, x, y, width, height); // *

                // TODO: x cannot be 0 because it's always odd
                if (x == 0)
                    SetPixel(g_, (unsigned short)((G1 + G2 + G3     ) / 3.0), x+1, y, width, height); // +
                else
                    SetPixel(g_, (unsigned short)((G1 + G2 + G3 + G4) / 4.0), x+1, y, width, height); // #
            }
        }
        // GRBG y\x:  -1   0   1   2   3
        //       -1:   G   B   G   B   G
        //        0:   R  *G  !R2  G   R
        //        1:   G  !B1  G   B   G
        //        2:   R   G   R   G   R
        //        3:   G   B   G   B   G
        G1 = GetPixel(input, 0, 1, width, height); // TODO: This is not green but blue pixel!
        G2 = GetPixel(input, 1, 0, width, height); // TODO: This is not green but red pixel!
        // TODO: This pixel is already set!
        SetPixel(g_, (unsigned short)((G1 + G2) / 2.0), 0, 0, width, height); // *

        // Blue channel for GRBG, red for GBRG (needs swap)
        for (int y = 1; y < height; y += 2)
        {
            for (int x = 0; x < width; x += 2)
            {
                // GRBG y\x:  -1   0   1   2   3
                //       -1:   G   B   G   B   G
                //        0:   R   G   R   G   R   <--- TODO: Blue is missing on first row
                //        1:   G  *B1 #G5 *B2 #G
                //        2:   R  +G6 ^R9 +G  ^R
                //        3:   G  *B3 #G  *B4 #G
                B1 = GetPixel(input, x  , y  , width, height);
                B2 = GetPixel(input, x+2, y  , width, height);
                B3 = GetPixel(input, x  , y+2, width, height);
                B4 = GetPixel(input, x+2, y+2, width, height);
                G1 = GetPixel(g_.data(), x  , y  , width, height);
                G2 = GetPixel(g_.data(), x+2, y  , width, height);
                G3 = GetPixel(g_.data(), x  , y+2, width, height);
                G4 = GetPixel(g_.data(), x+2, y+2, width, height);
                G5 = GetPixel(g_.data(), x+1, y  , width, height);
                G6 = GetPixel(g_.data(), x  , y+1, width, height);
                G9 = GetPixel(g_.data(), x+1, y+1, width, height);
                // TODO: Set to B1-B4 instead of 1
                if (G1 == 0) G1 = 1;
                if (G2 == 0) G2 = 1;
                if (G3 == 0) G3 = 1;
                if (G4 == 0) G4 = 1;

                SetPixel(b_, (unsigned short)(B1                                    ), x  , y  , width, height); // *
                SetPixel(b_, (unsigned short)(G5/2 * (B1/G1 + B2/G2                )), x+1, y  , width, height); // #
                SetPixel(b_, (unsigned short)(G6/2 * (B1/G1 +         B3/G3        )), x  , y+1, width, height); // +
                SetPixel(b_, (unsigned short)(G9/4 * (B1/G1 + B2/G2 + B3/G3 + B4/G4)), x+1, y+1, width, height); // ^
            }
        }

        // Red channel for GRBG, blue for GBRG (needs swap)
        for (int y = 0; y < height; y += 2)
        {
            for (int x = 1; x < width; x += 2)
            {
                // GRBG y\x:  -1   0   1   2   3
                //       -1:   G   B   G   B   G
                //        0:   R   G  *R1 #G5 *R2
                //        1:   G   B  +G6 ^B9 +G
                //        2:   R   G  *R3 #G  *R4
                //        3:   G   B  +G  ^B  +G
                //                 ^
                //                 \--- TODO: Red is missing on first column
                R1 = GetPixel(input, x  , y  , width, height);
                R2 = GetPixel(input, x+2, y  , width, height);
                R3 = GetPixel(input, x  , y+2, width, height);
                R4 = GetPixel(input, x+2, y+2, width, height);
                G1 = GetPixel(g_.data(), x  , y  , width, height);
                G2 = GetPixel(g_.data(), x+2, y  , width, height);
                G3 = GetPixel(g_.data(), x  , y+2, width, height);
                G4 = GetPixel(g_.data(), x+2, y+2, width, height);
                G5 = GetPixel(g_.data(), x+1, y  , width, height);
                G6 = GetPixel(g_.data(), x  , y+1, width, height);
                G9 = GetPixel(g_.data(), x+1, y+1, width, height);
                // TODO: Set to R1-R4 instead of 1
                if (G1 == 0) G1 = 1;
                if (G2 == 0) G2 = 1;
                if (G3 == 0) G3 = 1;
                if (G4 == 0) G4 = 1;

                SetPixel(r_, (unsigned short)(R1                                    ), x  , y  , width, height); // *
                SetPixel(r_, (unsigned short)(G5/2 * (R1/G1 + R2/G2                )), x+1, y  , width, height); // #
                SetPixel(r_, (unsigned short)(G6/2 * (R1/G1 +         R3/G3        )), x  , y+1, width, height); // +
                SetPixel(r_, (unsigned short)(G9/4 * (R1/G1 + R2/G2 + R3/G3 + R4/G4)), x+1, y+1, width, height); // ^
            }
        }
    }
}

// Adaptive Smooth Hue algorithm (edge detecting)
template <typename T>
void PvDebayer::DecodeT_AdaptiveSmooth(const T* input, int width, int height, int rowOrder)
{
    double G1, G2, G3, G4, G5, G6, G9;
    double B1, B2, B3, B4, B5;
    double R1, R2, R3, R4, R5;
    double N, S, E, W;

    if (rowOrder == CFA_RGGB || rowOrder == CFA_BGGR) // Green slash
    {
        // Code is for BGGR, RGGB needs to swap R & B channels, done in WhiteBalance()
        // BGGR y\x:  -1   0   1   2   3   4
        //       -2:   G   B   G   B   G   B
        //       -1:   R   G   R   G   R   G
        //        0:   G   B   G   B   G   B
        //        1:   R   G   R   G   R   G
        //        2:   G   B   G   B   G   B
        //        3:   R   G   R   G   R   G

        // Solve for green pixels first, it's needed for red and blue channels

        // Top-right greens
        for (int y = 0; y < height; y += 2)
        {
            for (int x = 1; x < width; x += 2)
            {
                // BGGR y\x:  -1   0   1   2   3   4
                //       -2:   G   B   G   B4   G   B
                //       -1:   R   G6  R   G4  R   G
                //        0:   G  ^B1 *G1 +B  *G2 +B2
                //        1:   R   G5  R   G3  R   G
                //        2:   G  ^B  *G  #B3 *G  #B
                //        3:   R   G   R   G   R   G
                G1 = GetPixel(input, x  , y  , width, height);
                G2 = GetPixel(input, x+2, y  , width, height);
                G3 = GetPixel(input, x+1, y+1, width, height);
                G4 = GetPixel(input, x+1, y-1, width, height);
                B1 = GetPixel(input, x-1, y  , width, height);
                B2 = GetPixel(input, x+3, y  , width, height);
                B3 = GetPixel(input, x+1, y+2, width, height);
                B4 = GetPixel(input, x+1, y-2, width, height);
                B5 = GetPixel(input, x+1, y+1, width, height); // TODO: Same as G3!

                N = fabs(B4 - B5) * 2 + fabs(G4 - G3);
                S = fabs(B5 - B3) * 2 + fabs(G4 - G3);
                E = fabs(B5 - B2) * 2 + fabs(G1 - G2);
                W = fabs(B1 - B5) * 2 + fabs(G1 - G2);

                SetPixel(g_, (unsigned short)G1, x, y, width, height); // *

                if      (N < S && N < E && N < W)
                    SetPixel(g_, (unsigned short)((G4*3 + B5 + G3 - B4) / 4.0), x+1, y, width, height);
                else if (S < N && S < E && S < W)
                    SetPixel(g_, (unsigned short)((G3*3 + B5 + G4 - B3) / 4.0), x+1, y, width, height);
                else if (W < N && W < E && W < S)
                    SetPixel(g_, (unsigned short)((G1*3 + B5 + G2 - B1) / 4.0), x+1, y, width, height);
                else if (E < N && E < S && E < W)
                    SetPixel(g_, (unsigned short)((G2*3 + B5 + G1 - B2) / 4.0), x+1, y, width, height);
                // TODO: This overwrites edge-detected values we've just set!
                //       So it is basically degraded to Smooth Hue algorithm.
                if (y == 0)
                    SetPixel(g_, (unsigned short)((G1 + G2 + G3     ) / 3.0), x+1, y, width, height); // +
                else
                    SetPixel(g_, (unsigned short)((G1 + G2 + G3 + G4) / 4.0), x+1, y, width, height); // #

                if (x == 1)
                {
                    // TODO: Should take G6 instead of G4
                    G5 = GetPixel(input, x-1, y+1, width, height);
                    SetPixel(g_, (unsigned short)((G1 + G4 + G5) / 3.0), x-1, y, width, height); // ^
                }
            }
        }

        // Bottom-left greens
        for (int y = 1; y < height; y += 2)
        {
            for (int x = 0; x < width; x += 2)
            {
                // BGGR y\x:  -1   0   1   2   3   4
                //       -2:   G   B   G   B   G   B
                //       -1:   R   G   R4  G   R   G
                //        0:   G   B   G4  B   G   B
                //        1:   R1 *G1 +R  *G2 #R2 *G
                //        2:   G   B   G3  B   G   B
                //        3:   R  *G  +R3 *G  #R  *G
                G1 = GetPixel(input, x  , y  , width, height);
                G2 = GetPixel(input, x+2, y  , width, height);
                G3 = GetPixel(input, x+1, y+1, width, height);
                G4 = GetPixel(input, x+1, y-1, width, height);
                R1 = GetPixel(input, x-1, y  , width, height);
                R2 = GetPixel(input, x+3, y  , width, height);
                R3 = GetPixel(input, x+1, y+2, width, height);
                R4 = GetPixel(input, x+1, y-2, width, height);
                R5 = GetPixel(input, x+1, y+1, width, height); // TODO: Same as G3!

                N = fabs(R4 - R5) * 2 + fabs(G4 - G3);
                S = fabs(R5 - R3) * 2 + fabs(G4 - G3);
                E = fabs(R5 - R2) * 2 + fabs(G1 - G2);
                W = fabs(R1 - R5) * 2 + fabs(G1 - G2);

                SetPixel(g_, (unsigned short)G1, x, y, width, height); // *

                if      (N < S && N < E && N < W)
                    SetPixel(g_, (unsigned short)((G4*3 + R5 + G3 - R4) / 4.0), x+1, y, width, height);
                else if (S < N && S < E && S < W)
                    SetPixel(g_, (unsigned short)((G3*3 + R5 + G4 - R3) / 4.0), x+1, y, width, height);
                else if (W < N && W < E && W < S)
                    SetPixel(g_, (unsigned short)((G1*3 + R5 + G2 - R1) / 4.0), x+1, y, width, height);
                else if (E < N && E < S && E < W)
                    SetPixel(g_, (unsigned short)((G2*3 + R5 + G1 - R2) / 4.0), x+1, y, width, height);
                // TODO: This overwrites edge-detected values we've just set!
                //       So it is basically degraded to Smooth Hue algorithm.
                if (x == 0)
                    SetPixel(g_, (unsigned short)((G1 + G2 + G3     ) / 3.0), x+1, y, width, height); // +
                else
                    SetPixel(g_, (unsigned short)((G1 + G2 + G3 + G4) / 4.0), x+1, y, width, height); // #
            }
        }
        // BGGR y\x:  -1   0   1   2   3   4
        //       -2:   G   B   G   B   G   B
        //       -1:   R   G   R   G   R   G
        //        0:   G  *B   G2  B   G   B
        //        1:   R   G1  R   G   R   G
        //        2:   G   B   G   B   G   B
        //        3:   R   G   R   G   R   G
        G1 = GetPixel(input, 0, 1, width, height);
        G2 = GetPixel(input, 1, 0, width, height);
        SetPixel(g_, (unsigned short)((G1 + G2) / 2.0), 0, 0, width, height);

        // Blue channel for BGGR, red for RGGB (needs swap)
        for (int y = 0; y < height; y += 2)
        {
            for (int x = 0; x < width; x += 2)
            {
                // BGGR y\x:  -1   0   1   2   3   4
                //       -2:   G   B   G   B   G   B
                //       -1:   R   G   R   G   R   G
                //        0:   G  *B1 #G5 *B2 #G  *B
                //        1:   R  +G6 ^R9 +G  ^R  +G
                //        2:   G  *B3 #G  *B4 #G  *B
                //        3:   R  +G  ^R  +G  ^R  +G
                B1 = GetPixel(input, x  , y  , width, height);
                B2 = GetPixel(input, x+2, y  , width, height);
                B3 = GetPixel(input, x  , y+2, width, height);
                B4 = GetPixel(input, x+2, y+2, width, height);
                G1 = GetPixel(g_.data(), x  , y  , width, height);
                G2 = GetPixel(g_.data(), x+2, y  , width, height);
                G3 = GetPixel(g_.data(), x  , y+2, width, height);
                G4 = GetPixel(g_.data(), x+2, y+2, width, height);
                G5 = GetPixel(g_.data(), x+1, y  , width, height);
                G6 = GetPixel(g_.data(), x  , y+1, width, height);
                G9 = GetPixel(g_.data(), x+1, y+1, width, height);
                // TODO: Set to B1-B4 instead of 1
                if (G1 == 0) G1 = 1;
                if (G2 == 0) G2 = 1;
                if (G3 == 0) G3 = 1;
                if (G4 == 0) G4 = 1;

                SetPixel(b_, (unsigned short)(B1                                    ), x  , y  , width, height); // *
                SetPixel(b_, (unsigned short)(G5/2 * (B1/G1 + B2/G2                )), x+1, y  , width, height); // #
                SetPixel(b_, (unsigned short)(G6/2 * (B1/G1 +         B3/G3        )), x  , y+1, width, height); // +
                SetPixel(b_, (unsigned short)(G9/4 * (B1/G1 + B2/G2 + B3/G3 + B4/G4)), x+1, y+1, width, height); // ^
            }
        }

        // Red channel for BGGR, blue for RGGB (needs swap)
        for (int y = 1; y < height; y += 2)
        {
            for (int x = 1; x < width; x += 2)
            {
                // BGGR y\x:  -1   0   1   2   3   4
                //       -2:   G   B   G   B   G   B
                //       -1:   R   G   R   G   R   G
                //        0:   G   B   G   B   G   B   <--- TODO: Red is missing on first row
                //        1:   R   G  *R1 #G5 *R2 #G
                //        2:   G   B  +G6 ^B9 +G  ^B
                //        3:   R   G  *R3 #G  *R4 #G
                //                 ^
                //                 \--- TODO: Red is missing on first column
                R1 = GetPixel(input, x  , y  , width, height);
                R2 = GetPixel(input, x+2, y  , width, height);
                R3 = GetPixel(input, x  , y+2, width, height);
                R4 = GetPixel(input, x+2, y+2, width, height);
                G1 = GetPixel(g_.data(), x  , y  , width, height);
                G2 = GetPixel(g_.data(), x+2, y  , width, height);
                G3 = GetPixel(g_.data(), x  , y+2, width, height);
                G4 = GetPixel(g_.data(), x+2, y+2, width, height);
                G5 = GetPixel(g_.data(), x+1, y  , width, height);
                G6 = GetPixel(g_.data(), x  , y+1, width, height);
                G9 = GetPixel(g_.data(), x+1, y+1, width, height);
                // TODO: Set to B1-B4 instead of 1
                if (G1 == 0) G1 = 1;
                if (G2 == 0) G2 = 1;
                if (G3 == 0) G3 = 1;
                if (G4 == 0) G4 = 1;

                SetPixel(r_, (unsigned short)(R1                                    ), x  , y  , width, height); // *
                SetPixel(r_, (unsigned short)(G5/2 * (R1/G1 + R2/G2                )), x+1, y  , width, height); // #
                SetPixel(r_, (unsigned short)(G6/2 * (R1/G1 +         R3/G3        )), x  , y+1, width, height); // +
                SetPixel(r_, (unsigned short)(G9/4 * (R1/G1 + R2/G2 + R3/G3 + R4/G4)), x+1, y+1, width, height); // ^
            }
        }
    }
    else if (rowOrder == CFA_GRBG || rowOrder == CFA_GBRG) // Green backslash
    {
        // Code is for GRBG, GBRG needs to swap R & B channels, done in WhiteBalance()
        // GRBG y\x:  -1   0   1   2   3   4
        //       -2:   R   G   R   G   R   G
        //       -1:   G   B   G   B   G   B
        //        0:   R   G   R   G   R   G
        //        1:   G   B   G   B   G   B
        //        2:   R   G   R   G   R   G
        //        3:   G   B   G   B   G   B

        // Solve for green pixels first, it's needed for red and blue channels

        // Top-left greens
        for (int y = 0; y < height; y += 2)
        {
            for (int x = 0; x < width; x += 2)
            {
                // GRBG y\x:  -1   0   1   2   3   4
                //       -2:   R   G   R4  G   R   G
                //       -1:   G   B   G4  B   G   B
                //        0:   R1 *G1 +R  *G2 +R2 *G
                //        1:   G   B   G3  B   G   B
                //        2:   R  *G  #R3 *G  #R  *G
                //        3:   G   B   G   B   G   B
                G1 = GetPixel(input, x  , y  , width, height);
                G2 = GetPixel(input, x+2, y  , width, height);
                G3 = GetPixel(input, x+1, y+1, width, height);
                G4 = GetPixel(input, x+1, y-1, width, height);
                R1 = GetPixel(input, x-1, y  , width, height);
                R2 = GetPixel(input, x+3, y  , width, height);
                R3 = GetPixel(input, x+1, y+2, width, height);
                R4 = GetPixel(input, x+1, y-2, width, height);
                R5 = GetPixel(input, x+1, y+1, width, height); // TODO: Same as G3!

                N = fabs(R4 - R5) * 2 + fabs(G4 - G3);
                S = fabs(R5 - R3) * 2 + fabs(G4 - G3);
                E = fabs(R5 - R2) * 2 + fabs(G1 - G2);
                W = fabs(R1 - R5) * 2 + fabs(G1 - G2);

                SetPixel(g_, (unsigned short)G1, x, y, width, height); // *

                if      (N < S && N < E && N < W)
                    SetPixel(g_, (unsigned short)((G4*3 + R5 + G3 - R4) / 4.0), x+1, y, width, height);
                else if (S < N && S < E && S < W)
                    SetPixel(g_, (unsigned short)((G3*3 + R5 + G4 - R3) / 4.0), x+1, y, width, height);
                else if (W < N && W < E && W < S)
                    SetPixel(g_, (unsigned short)((G1*3 + R5 + G2 - R1) / 4.0), x+1, y, width, height);
                else if (E < N && E < S && E < W)
                    SetPixel(g_, (unsigned short)((G2*3 + R5 + G1 - R2) / 4.0), x+1, y, width, height);
                // TODO: This overwrites edge-detected values we've just set!
                //       So it is basically degraded to Smooth Hue algorithm.
                if (y == 0)
                    SetPixel(g_, (unsigned short)((G1 + G2 + G3     ) / 3.0), x+1, y, width, height); // +
                else
                    SetPixel(g_, (unsigned short)((G1 + G2 + G3 + G4) / 4.0), x+1, y, width, height); // #

                // TODO: x cannot be 1 because it's always even
                if (x == 1)
                {
                    G5 = GetPixel(input, x-1, y+1, width, height);
                    SetPixel(g_, (unsigned short)((G1 + G4 + G5) / 3.0), x-1, y, width, height); // ^
                }
            }
        }
        // Bottom-right greens
        for (int y = 1; y < height; y += 2)
        {
            for (int x = 1; x < width; x += 2)
            {
                // GRBG y\x:  -1   0   1   2   3   4
                //       -2:   R   G   R   G   R   G
                //       -1:   G   B   G   B4  G   B
                //        0:   R   G   R   G4  R   G
                //        1:   G   B1 *G1 #B  *G2 #B2
                //        2:   R   G   R   G3  R   G
                //        3:   G   B  *G  #B3 *G  #B
                //                 ^
                //                 \--- TODO: Green is missing on first column odd rows
                G1 = GetPixel(input, x  , y  , width, height);
                G2 = GetPixel(input, x+2, y  , width, height);
                G3 = GetPixel(input, x+1, y+1, width, height);
                G4 = GetPixel(input, x+1, y-1, width, height);
                B1 = GetPixel(input, x-1, y  , width, height);
                B2 = GetPixel(input, x+3, y  , width, height);
                B3 = GetPixel(input, x+1, y+2, width, height);
                B4 = GetPixel(input, x+1, y-2, width, height);
                B5 = GetPixel(input, x+1, y+1, width, height); // TODO: Same as G3!

                N = fabs(B4 - B5) * 2 + fabs(G4 - G3);
                S = fabs(B5 - B3) * 2 + fabs(G4 - G3);
                E = fabs(B5 - B2) * 2 + fabs(G1 - G2);
                W = fabs(B1 - B5) * 2 + fabs(G1 - G2);

                SetPixel(g_, (unsigned short)G1, x, y, width, height);

                if      (N < S && N < E && N < W)
                    SetPixel(g_, (unsigned short)((G4*3 + B5 + G3 - B4) / 4.0), x+1, y, width, height);
                else if (S < N && S < E && S < W)
                    SetPixel(g_, (unsigned short)((G3*3 + B5 + G4 - B3) / 4.0), x+1, y, width, height);
                else if (W < N && W < E && W < S)
                    SetPixel(g_, (unsigned short)((G1*3 + B5 + G2 - B1) / 4.0), x+1, y, width, height);
                else if (E < N && E < S && E < W)
                    SetPixel(g_, (unsigned short)((G2*3 + B5 + G1 - B2) / 4.0), x+1, y, width, height);
                // TODO: This overwrites edge-detected values we've just set!
                //       So it is basically degraded to Smooth Hue algorithm.
                // TODO: x cannot be 0 because it's always odd
                if (x == 0)
                    SetPixel(g_, (unsigned short)((G1 + G2 + G3     ) / 3.0), x+1, y, width, height); // +
                else
                    SetPixel(g_, (unsigned short)((G1 + G2 + G3 + G4) / 4.0), x+1, y, width, height); // #
            }
        }
        // GRBG y\x:  -1   0   1   2   3   4
        //       -2:   R   G   R   G   R   G
        //       -1:   G   B   G   B   G   B
        //        0:   R  *G  !R2  G   R   G
        //        1:   G  !B1  G   B   G   B
        //        2:   R   G   R   G   R   G
        //        3:   G   B   G   B   G   B
        G1 = GetPixel(input, 0, 1, width, height); // TODO: This is not green but blue pixel!
        G2 = GetPixel(input, 1, 0, width, height); // TODO: This is not green but red pixel!
        SetPixel(g_, (unsigned short)((G1 + G2) / 2.0), 0, 0, width, height); // *

        // Blue channel for GRBG, red for GBRG (needs swap)
        for (int y = 1; y < height; y += 2)
        {
            for (int x = 0; x < width; x += 2)
            {
                // GRBG y\x:  -1   0   1   2   3   4
                //       -2:   R   G   R   G   R   G
                //       -1:   G   B   G   B   G   B
                //        0:   R   G   R   G   R   G   <--- TODO: Blue is missing on first row
                //        1:   G  *B1 #G5 *B2 #G  *B
                //        2:   R  +G6 ^R9 +G  ^R  +G
                //        3:   G  *B3 #G  *B4 #G  *B
                B1 = GetPixel(input, x  , y  , width, height);
                B2 = GetPixel(input, x+2, y  , width, height);
                B3 = GetPixel(input, x  , y+2, width, height);
                B4 = GetPixel(input, x+2, y+2, width, height);
                G1 = GetPixel(g_.data(), x  , y  , width, height);
                G2 = GetPixel(g_.data(), x+2, y  , width, height);
                G3 = GetPixel(g_.data(), x  , y+2, width, height);
                G4 = GetPixel(g_.data(), x+2, y+2, width, height);
                G5 = GetPixel(g_.data(), x+1, y  , width, height);
                G6 = GetPixel(g_.data(), x  , y+1, width, height);
                G9 = GetPixel(g_.data(), x+1, y+1, width, height);
                // TODO: Set to B1-B4 instead of 1
                if (G1 == 0) G1 = 1;
                if (G2 == 0) G2 = 1;
                if (G3 == 0) G3 = 1;
                if (G4 == 0) G4 = 1;

                SetPixel(b_, (unsigned short)(B1                                    ), x  , y  , width, height); // *
                SetPixel(b_, (unsigned short)(G5/2 * (B1/G1 + B2/G2                )), x+1, y  , width, height); // #
                SetPixel(b_, (unsigned short)(G6/2 * (B1/G1 +         B3/G3        )), x  , y+1, width, height); // +
                SetPixel(b_, (unsigned short)(G9/4 * (B1/G1 + B2/G2 + B3/G3 + B4/G4)), x+1, y+1, width, height); // ^
            }
        }

        // Red channel for GRBG, blue for GBRG (needs swap)
        for (int y = 0; y < height; y += 2)
        {
            for (int x = 1; x < width; x += 2)
            {
                // GRBG y\x:  -1   0   1   2   3   4
                //       -2:   R   G   R   G   R   G
                //       -1:   G   B   G   B   G   B
                //        0:   R   G  *R1 #G5 *R2 #G
                //        1:   G   B  +G6 ^B9 +G  ^B
                //        2:   R   G  *R3 #G  *R4 #G
                //        3:   G   B  +G  ^B  +G  ^B
                //                 ^
                //                 \--- TODO: Red is missing on first column
                R1 = GetPixel(input, x  , y  , width, height);
                R2 = GetPixel(input, x+2, y  , width, height);
                R3 = GetPixel(input, x  , y+2, width, height);
                R4 = GetPixel(input, x+2, y+2, width, height);
                G1 = GetPixel(g_.data(), x  , y  , width, height);
                G2 = GetPixel(g_.data(), x+2, y  , width, height);
                G3 = GetPixel(g_.data(), x  , y+2, width, height);
                G4 = GetPixel(g_.data(), x+2, y+2, width, height);
                G5 = GetPixel(g_.data(), x+1, y  , width, height);
                G6 = GetPixel(g_.data(), x  , y+1, width, height);
                G9 = GetPixel(g_.data(), x+1, y+1, width, height);
                // TODO: Set to R1-R4 instead of 1
                if (G1 == 0) G1 = 1;
                if (G2 == 0) G2 = 1;
                if (G3 == 0) G3 = 1;
                if (G4 == 0) G4 = 1;

                SetPixel(r_, (unsigned short)(R1                                    ), x  , y  , width, height); // *
                SetPixel(r_, (unsigned short)(G5/2 * (R1/G1 + R2/G2                )), x+1, y  , width, height); // #
                SetPixel(r_, (unsigned short)(G6/2 * (R1/G1 +         R3/G3        )), x  , y+1, width, height); // +
                SetPixel(r_, (unsigned short)(G9/4 * (R1/G1 + R2/G2 + R3/G3 + R4/G4)), x+1, y+1, width, height); // ^
            }
        }
    }
}

int PvDebayer::WhiteBalance(int* output, int width, int height, int bitDepth, int rowOrder)
{
    const bool swapRB = (rowOrder == CFA_RGGB || rowOrder == CFA_GBRG);

    const std::vector<unsigned short>& b = (swapRB) ? r_ : b_;
    const std::vector<unsigned short>& g =                 g_;
    const std::vector<unsigned short>& r = (swapRB) ? b_ : r_;

    const int bitShift = bitDepth - 8;
    const size_t numPixels = (size_t)width * height;

    unsigned short bVal;
    unsigned short gVal;
    unsigned short rVal;
    unsigned char* bytePix;

    for (size_t i = 0; i < numPixels; i++)
    {
        // TODO: The 'unsigned char' cast after bit-shift shouldn't be needed.
        //       Unfortunately, the Smooth Hue algorithms here generate values
        //       greater than fit given bit depth on the edges due to
        //       "fixing" division by zero by division by 1.
        bVal = (unsigned short)(rgbScales_.b * ((unsigned char)(b[i] >> bitShift)));
        gVal = (unsigned short)(rgbScales_.g * ((unsigned char)(g[i] >> bitShift)));
        rVal = (unsigned short)(rgbScales_.r * ((unsigned char)(r[i] >> bitShift)));

        bytePix = (unsigned char*)(&output[i]);

        bytePix[0] = (unsigned char)((bVal > 255) ? 255 : bVal);
        bytePix[1] = (unsigned char)((gVal > 255) ? 255 : gVal);
        bytePix[2] = (unsigned char)((rVal > 255) ? 255 : rVal);
        bytePix[3] = (unsigned char)0;
    }

    return DEVICE_OK;
}

template <typename T>
unsigned short PvDebayer::GetPixel(const T* v, int x, int y, int width, int height, T def)
{
    if (x < width && x >= 0 && y < height && y >= 0)
        return v[y * width + x];
    return def;
}

template <typename T>
void PvDebayer::SetPixel(std::vector<T>& v, T val, int x, int y, int width, int height)
{
    if (x < width && x >= 0 && y < height && y >= 0)
        v[y * width + x] = val;
}
