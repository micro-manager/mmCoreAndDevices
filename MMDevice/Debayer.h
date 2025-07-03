///////////////////////////////////////////////////////////////////////////////
// MODULE:			Debayer.h
// SYSTEM:        ImageBase subsystem
// AUTHOR:			Jennifer West, jennifer_west@umanitoba.ca,
//                Nenad Amodaj, nenad@amodaj.com
//
// DESCRIPTION:	Debayer algorithms, adapted from:
//                http://www.umanitoba.ca/faculties/science/astronomy/jwest/plugins.html
//                
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

#pragma once

#include "ImgBuffer.h"

#include <string>
#include <vector>

/**
 * Utility class to build color image from the Bayer grayscale image
 * Based on the Debayer_Image plugin for ImageJ, by Jennifer West, University of Manitoba
 */
class Debayer
{
public:
   Debayer();
   ~Debayer();

   int Process(ImgBuffer& out, const ImgBuffer& in, int bitDepth);
   int Process(ImgBuffer& out, const unsigned char* in, int width, int height, int bitDepth);
   int Process(ImgBuffer& out, const unsigned short* in, int width, int height, int bitDepth);

   const std::vector<std::string> GetOrders() const {return orders;}
   const std::vector<std::string> GetAlgorithms() const {return algorithms;}

   void SetOrderIndex(int idx) {orderIndex = idx;}
   void SetAlgorithmIndex(int idx) {algoIndex = idx;}

private:
   template <typename T>
   int ProcessT(ImgBuffer& out, const T* in, int width, int height, int bitDepth);
   template<typename T>
   void ReplicateDecode(const T* input, int* out, int width, int height, int bitDepth, int rowOrder);
   template <typename T>
   void SmoothDecode(const T* input, int* output, int width, int height, int bitDepth, int rowOrder);
   template<typename T>
   int Convert(const T* input, int* output, int width, int height, int bitDepth, int rowOrder, int algorithm);
   unsigned short GetPixel(const unsigned short* v, int x, int y, int width, int height);
   void SetPixel(std::vector<unsigned short>& v, unsigned short val, int x, int y, int width, int height);
   unsigned short GetPixel(const unsigned char* v, int x, int y, int width, int height);

   std::vector<unsigned short> r; // red scratch buffer
   std::vector<unsigned short> g; // green scratch buffer
   std::vector<unsigned short> b; // blue scratch buffer

   std::vector<std::string> orders;
   std::vector<std::string> algorithms;

   int orderIndex;
   int algoIndex;
};
