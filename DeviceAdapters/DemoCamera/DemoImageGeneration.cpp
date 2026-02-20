///////////////////////////////////////////////////////////////////////////////
// FILE:          DemoImageGeneration.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   Image generation utility functions for DemoCamera.
//                Extracted from DemoCamera.cpp for better code organization.
//
// AUTHOR:        Nenad Amodaj, nenad@amodaj.com, 06/08/2005
//
// COPYRIGHT:     University of California, San Francisco, 2006
// LICENSE:       This file is distributed under the BSD license.
//                License text is included with the source distribution.
//
//                This file is distributed in the hope that it will be useful,
//                but WITHOUT ANY WARRANTY; without even the implied warranty
//                of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
//
//                IN NO EVENT SHALL THE COPYRIGHT OWNER OR
//                CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
//                INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES.

#include "DemoCamera.h"
#include "WriteCompactTiffRGB.h"
#include <cmath>
#include <random>
#include <algorithm>
#include <sstream>

// External pixel type constants (defined in DemoCamera.cpp)
extern const char* g_PixelType_8bit;
extern const char* g_PixelType_16bit;
extern const char* g_PixelType_32bitRGB;
extern const char* g_PixelType_64bitRGB;
extern const char* g_PixelType_32bit;

// External mode constants (defined in DemoCamera.cpp)
extern const char* g_Sine_Wave;
extern const char* g_Norm_Noise;
extern const char* g_Color_Test;
extern const char* g_Beads;

/**
* Generate a synthetic test image based on the current mode
*/
void CDemoCamera::GenerateSyntheticImage(ImgBuffer& img, double exp)
{

   MMThreadGuard g(imgPixelsLock_);

   if (mode_ == MODE_BEADS)
   {
      GenerateBeadsImage(img, exp);
      return;
   }
   else if (mode_ == MODE_NOISE)
   {
      double max = 1 << GetBitDepth();
      int offset = 10;
      if (max > 256)
      {
         offset = 100;
      }
	   double readNoiseDN = readNoise_ / pcf_;
      AddBackgroundAndNoise(img, offset, readNoiseDN);
      AddSignal (img, photonFlux_, exp, pcf_);
      if (imgManpl_ != 0)
      {
         imgManpl_->ChangePixels(img);
      }
      return;
   }
   else if (mode_ == MODE_COLOR_TEST)
   {
      if (GenerateColorTestPattern(img))
         return;
   }

	//std::string pixelType;
	char buf[MM::MaxStrLength];
   GetProperty(MM::g_Keyword_PixelType, buf);
   std::string pixelType(buf);

	if (img.Height() == 0 || img.Width() == 0 || img.Depth() == 0)
      return;

   double lSinePeriod = 3.14159265358979 * stripeWidth_;
   unsigned imgWidth = img.Width();
   unsigned int* rawBuf = (unsigned int*) img.GetPixelsRW();
   double maxDrawnVal = 0;
   long lPeriod = (long) imgWidth / 2;
   double dLinePhase = 0.0;
   const double dAmp = exp;
   double cLinePhaseInc = 2.0 * lSinePeriod / 4.0 / img.Height();
   if (shouldRotateImages_) {
      // Adjust the angle of the sin wave pattern based on how many images
      // we've taken, to increase the period (i.e. time between repeat images).
      cLinePhaseInc *= (((int) dPhase_ / 6) % 24) - 12;
   }

   static bool debugRGB = false;
#ifdef TIFFDEMO
	debugRGB = true;
#endif
   static  unsigned char* pDebug  = NULL;
   static unsigned long dbgBufferSize = 0;
   static long iseq = 1;



	// for integer images: bitDepth_ is 8, 10, 12, 16 i.e. it is depth per component
   long maxValue = (1L << bitDepth_)-1;

	long pixelsToDrop = 0;
	if( dropPixels_)
		pixelsToDrop = (long)(0.5 + fractionOfPixelsToDropOrSaturate_*img.Height()*imgWidth);
	long pixelsToSaturate = 0;
	if( saturatePixels_)
		pixelsToSaturate = (long)(0.5 + fractionOfPixelsToDropOrSaturate_*img.Height()*imgWidth);

   unsigned j, k;
   if (pixelType.compare(g_PixelType_8bit) == 0)
   {
      double pedestal = 127 * exp / 100.0 * GetBinning() * GetBinning();
      unsigned char* pBuf = const_cast<unsigned char*>(img.GetPixels());
      for (j=0; j<img.Height(); j++)
      {
         for (k=0; k<imgWidth; k++)
         {
            long lIndex = imgWidth*j + k;
            unsigned char val = (unsigned char) (g_IntensityFactor_ * std::min(255.0, (pedestal + dAmp * sin(dPhase_ + dLinePhase + (2.0 * lSinePeriod * k) / lPeriod))));
            if (val > maxDrawnVal) {
                maxDrawnVal = val;
            }
            *(pBuf + lIndex) = val;
         }
         dLinePhase += cLinePhaseInc;
      }
	   for(int snoise = 0; snoise < pixelsToSaturate; ++snoise)
		{
			j = (unsigned)( (double)(img.Height()-1)*(double)rand()/(double)RAND_MAX);
			k = (unsigned)( (double)(imgWidth-1)*(double)rand()/(double)RAND_MAX);
			*(pBuf + imgWidth*j + k) = (unsigned char)maxValue;
		}
		int pnoise;
		for(pnoise = 0; pnoise < pixelsToDrop; ++pnoise)
		{
			j = (unsigned)( (double)(img.Height()-1)*(double)rand()/(double)RAND_MAX);
			k = (unsigned)( (double)(imgWidth-1)*(double)rand()/(double)RAND_MAX);
			*(pBuf + imgWidth*j + k) = 0;
		}

   }
   else if (pixelType.compare(g_PixelType_16bit) == 0)
   {
      double pedestal = maxValue/2 * exp / 100.0 * GetBinning() * GetBinning();
      double dAmp16 = dAmp * maxValue/255.0; // scale to behave like 8-bit
      unsigned short* pBuf = (unsigned short*) const_cast<unsigned char*>(img.GetPixels());
      for (j=0; j<img.Height(); j++)
      {
         for (k=0; k<imgWidth; k++)
         {
            long lIndex = imgWidth*j + k;
            unsigned short val = (unsigned short) (g_IntensityFactor_ * std::min((double)maxValue, pedestal + dAmp16 * sin(dPhase_ + dLinePhase + (2.0 * lSinePeriod * k) / lPeriod)));
            if (val > maxDrawnVal) {
                maxDrawnVal = val;
            }
            *(pBuf + lIndex) = val;
         }
         dLinePhase += cLinePhaseInc;
      }
	   for(int snoise = 0; snoise < pixelsToSaturate; ++snoise)
		{
			j = (unsigned)(0.5 + (double)img.Height()*(double)rand()/(double)RAND_MAX);
			k = (unsigned)(0.5 + (double)imgWidth*(double)rand()/(double)RAND_MAX);
			*(pBuf + imgWidth*j + k) = (unsigned short)maxValue;
		}
		int pnoise;
		for(pnoise = 0; pnoise < pixelsToDrop; ++pnoise)
		{
			j = (unsigned)(0.5 + (double)img.Height()*(double)rand()/(double)RAND_MAX);
			k = (unsigned)(0.5 + (double)imgWidth*(double)rand()/(double)RAND_MAX);
			*(pBuf + imgWidth*j + k) = 0;
		}

	}
   else if (pixelType.compare(g_PixelType_32bit) == 0)
   {
      double pedestal = 127 * exp / 100.0 * GetBinning() * GetBinning();
      float* pBuf = (float*) const_cast<unsigned char*>(img.GetPixels());
      float saturatedValue = 255.;
      memset(pBuf, 0, img.Height()*imgWidth*4);
      // static unsigned int j2;
      for (j=0; j<img.Height(); j++)
      {
         for (k=0; k<imgWidth; k++)
         {
            long lIndex = imgWidth*j + k;
            double value =  (g_IntensityFactor_ * std::min(255.0, (pedestal + dAmp * sin(dPhase_ + dLinePhase + (2.0 * lSinePeriod * k) / lPeriod))));
            if (value > maxDrawnVal) {
                maxDrawnVal = value;
            }
            *(pBuf + lIndex) = (float) value;
            if( 0 == lIndex)
            {
               std::ostringstream os;
               os << " first pixel is " << (float)value;
               LogMessage(os.str().c_str(), true);

            }
         }
         dLinePhase += cLinePhaseInc;
      }

	   for(int snoise = 0; snoise < pixelsToSaturate; ++snoise)
		{
			j = (unsigned)(0.5 + (double)img.Height()*(double)rand()/(double)RAND_MAX);
			k = (unsigned)(0.5 + (double)imgWidth*(double)rand()/(double)RAND_MAX);
			*(pBuf + imgWidth*j + k) = saturatedValue;
		}
		int pnoise;
		for(pnoise = 0; pnoise < pixelsToDrop; ++pnoise)
		{
			j = (unsigned)(0.5 + (double)img.Height()*(double)rand()/(double)RAND_MAX);
			k = (unsigned)(0.5 + (double)imgWidth*(double)rand()/(double)RAND_MAX);
			*(pBuf + imgWidth*j + k) = 0;
      }

	}
	else if (pixelType.compare(g_PixelType_32bitRGB) == 0)
	{
      double pedestal = 127 * exp / 100.0;
      unsigned int * pBuf = (unsigned int*) rawBuf;

      unsigned char* pTmpBuffer = NULL;

      if(debugRGB)
      {
         const unsigned long bfsize = img.Height() * imgWidth * 3;
         if(  bfsize != dbgBufferSize)
         {
            if (NULL != pDebug)
            {
               free(pDebug);
               pDebug = NULL;
            }
            pDebug = (unsigned char*)malloc( bfsize);
            if( NULL != pDebug)
            {
               dbgBufferSize = bfsize;
            }
         }
      }

		// only perform the debug operations if pTmpbuffer is not 0
      pTmpBuffer = pDebug;
      unsigned char* pTmp2 = pTmpBuffer;
      if( NULL!= pTmpBuffer)
			memset( pTmpBuffer, 0, img.Height() * imgWidth * 3);

      for (j=0; j<img.Height(); j++)
      {
         unsigned char theBytes[4];
         for (k=0; k<imgWidth; k++)
         {
            long lIndex = imgWidth*j + k;
            double factor = (2.0 * lSinePeriod * k) / lPeriod;
            unsigned char value0 =   (unsigned char) std::min(255.0, (pedestal + dAmp * sin(dPhase_ + dLinePhase + factor)));
            theBytes[0] = value0;
            if( NULL != pTmpBuffer)
               pTmp2[1] = value0;
            unsigned char value1 =   (unsigned char) std::min(255.0, (pedestal + dAmp * sin(dPhase_ + dLinePhase*2 + factor)));
            theBytes[1] = value1;
            if( NULL != pTmpBuffer)
               pTmp2[2] = value1;
            unsigned char value2 = (unsigned char) std::min(255.0, (pedestal + dAmp * sin(dPhase_ + dLinePhase*4 + factor)));
            theBytes[2] = value2;

            if( NULL != pTmpBuffer){
               pTmp2[3] = value2;
               pTmp2+=3;
            }
            theBytes[3] = 0;
            unsigned long tvalue = *(unsigned long*)(&theBytes[0]);
            if (tvalue > maxDrawnVal) {
                maxDrawnVal = tvalue;
            }
            *(pBuf + lIndex) =  tvalue ;  //value0+(value1<<8)+(value2<<16);
         }
         dLinePhase += cLinePhaseInc;
      }


      // ImageJ's AWT images are loaded with a Direct Color processor which expects big endian ARGB,
      // which on little endian architectures corresponds to BGRA (see: https://en.wikipedia.org/wiki/RGBA_color_model),
      // that's why we swapped the Blue and Red components in the generator above.
      if(NULL != pTmpBuffer)
      {
         // write the compact debug image...
         char ctmp[12];
         snprintf(ctmp,12,"%ld",iseq++);
         writeCompactTiffRGB(imgWidth, img.Height(), pTmpBuffer, ("democamera" + std::string(ctmp)).c_str());
      }

	}

	// generate an RGB image with bitDepth_ bits in each color
	else if (pixelType.compare(g_PixelType_64bitRGB) == 0)
	{
      double pedestal = maxValue/2 * exp / 100.0 * GetBinning() * GetBinning();
      double dAmp16 = dAmp * maxValue/255.0; // scale to behave like 8-bit

		double maxPixelValue = (1<<(bitDepth_))-1;
      unsigned long long * pBuf = (unsigned long long*) rawBuf;
      for (j=0; j<img.Height(); j++)
      {
         for (k=0; k<imgWidth; k++)
         {
            long lIndex = imgWidth*j + k;
            unsigned long long value0 = (unsigned short) std::min(maxPixelValue, (pedestal + dAmp16 * sin(dPhase_ + dLinePhase + (2.0 * lSinePeriod * k) / lPeriod)));
            unsigned long long value1 = (unsigned short) std::min(maxPixelValue, (pedestal + dAmp16 * sin(dPhase_ + dLinePhase*2 + (2.0 * lSinePeriod * k) / lPeriod)));
            unsigned long long value2 = (unsigned short) std::min(maxPixelValue, (pedestal + dAmp16 * sin(dPhase_ + dLinePhase*4 + (2.0 * lSinePeriod * k) / lPeriod)));
            unsigned long long tval = value0+(value1<<16)+(value2<<32);
            if (tval > maxDrawnVal) {
                maxDrawnVal = static_cast<double>(tval);
            }
            *(pBuf + lIndex) = tval;
			}
         dLinePhase += cLinePhaseInc;
      }
	}

    if (shouldDisplayImageNumber_) {
        // Draw a seven-segment display in the upper-left corner of the image,
        // indicating the image number.
        int divisor = 1;
        int numDigits = 0;
        while (imageCounter_ / divisor > 0) {
            divisor *= 10;
            numDigits += 1;
        }
        int remainder = imageCounter_;
        for (int i = 0; i < numDigits; ++i) {
            // Black out the background for this digit.
            // TODO: for now, hardcoded sizes, which will cause buffer
            // overflows if the image size is too small -- but that seems
            // unlikely.
            int xBase = (numDigits - i - 1) * 20 + 2;
            int yBase = 2;
            for (int x = xBase; x < xBase + 20; ++x) {
                for (int y = yBase; y < yBase + 20; ++y) {
                    long lIndex = imgWidth*y + x;

                    if (pixelType.compare(g_PixelType_8bit) == 0) {
                        *((unsigned char*) rawBuf + lIndex) = 0;
                    }
                    else if (pixelType.compare(g_PixelType_16bit) == 0) {
                        *((unsigned short*) rawBuf + lIndex) = 0;
                    }
                    else if (pixelType.compare(g_PixelType_32bit) == 0 ||
                             pixelType.compare(g_PixelType_32bitRGB) == 0) {
                        *((unsigned int*) rawBuf + lIndex) = 0;
                    }
                }
            }
            // Draw each segment, if appropriate.
            int digit = remainder % 10;
            for (int segment = 0; segment < 7; ++segment) {
                if (!((1 << segment) & SEVEN_SEGMENT_RULES[digit])) {
                    // This segment is not drawn.
                    continue;
                }
                // Determine if the segment is horizontal or vertical.
                int xStep = SEVEN_SEGMENT_HORIZONTALITY[segment];
                int yStep = (xStep + 1) % 2;
                // Calculate starting point for drawing the segment.
                int xStart = xBase + SEVEN_SEGMENT_X_OFFSET[segment] * 16;
                int yStart = yBase + SEVEN_SEGMENT_Y_OFFSET[segment] * 8 + 1;
                // Draw one pixel at a time of the segment.
                for (int pixNum = 0; pixNum < 8 * (xStep + 1); ++pixNum) {
                    long lIndex = imgWidth * (yStart + pixNum * yStep) + (xStart + pixNum * xStep);
                    if (pixelType.compare(g_PixelType_8bit) == 0) {
                        *((unsigned char*) rawBuf + lIndex) = static_cast<unsigned char>(maxDrawnVal);
                    }
                    else if (pixelType.compare(g_PixelType_16bit) == 0) {
                        *((unsigned short*) rawBuf + lIndex) = static_cast<unsigned short>(maxDrawnVal);
                    }
                    else if (pixelType.compare(g_PixelType_32bit) == 0 ||
                             pixelType.compare(g_PixelType_32bitRGB) == 0) {
                        *((unsigned int*) rawBuf + lIndex) = static_cast<unsigned int>(maxDrawnVal);
                    }
                }
            }
            remainder /= 10;
        }
    }
   if (multiROIXs_.size() > 0)
   {
      // Blank out all pixels that are not in an ROI.
      // TODO: it would be more efficient to only populate pixel values that
      // *are* in an ROI, but that would require substantial refactoring of
      // this function.
      for (unsigned int i = 0; i < imgWidth; ++i)
      {
         for (unsigned h = 0; h < img.Height(); ++h)
         {
            bool shouldKeep = false;
            for (unsigned int mr = 0; mr < multiROIXs_.size(); ++mr)
            {
               unsigned xOffset = multiROIXs_[mr] - roiX_;
               unsigned yOffset = multiROIYs_[mr] - roiY_;
               unsigned width = multiROIWidths_[mr];
               unsigned height = multiROIHeights_[mr];
               if (i >= xOffset && i < xOffset + width &&
                        h >= yOffset && h < yOffset + height)
               {
                  // Pixel is inside an ROI.
                  shouldKeep = true;
                  break;
               }
            }
            if (!shouldKeep)
            {
               // Blank the pixel.
               long lIndex = imgWidth * h + i;
               if (pixelType.compare(g_PixelType_8bit) == 0)
               {
                  *((unsigned char*) rawBuf + lIndex) = static_cast<unsigned char>(multiROIFillValue_);
               }
               else if (pixelType.compare(g_PixelType_16bit) == 0)
               {
                  *((unsigned short*) rawBuf + lIndex) = static_cast<unsigned short>(multiROIFillValue_);
               }
               else if (pixelType.compare(g_PixelType_32bit) == 0 ||
                        pixelType.compare(g_PixelType_32bitRGB) == 0)
               {
                  *((unsigned int*) rawBuf + lIndex) = static_cast<unsigned int>(multiROIFillValue_);
               }
            }
         }
      }
   }
   dPhase_ += lSinePeriod / 4.;
}


bool CDemoCamera::GenerateColorTestPattern(ImgBuffer& img)
{
   unsigned width = img.Width(), height = img.Height();
   switch (img.Depth())
   {
      case 1:
      {
         const unsigned char maxVal = 255;
         unsigned char* rawBytes = img.GetPixelsRW();
         for (unsigned y = 0; y < height; ++y)
         {
            for (unsigned x = 0; x < width; ++x)
            {
               if (y == 0)
               {
                  rawBytes[x] = (unsigned char) (maxVal * (x + 1) / (width - 1));
               }
               else {
                  rawBytes[x + y * width] = rawBytes[x];
               }
            }
         }
         return true;
      }
      case 2:
      {
         const unsigned short maxVal = 65535;
         unsigned short* rawShorts =
            reinterpret_cast<unsigned short*>(img.GetPixelsRW());
         for (unsigned y = 0; y < height; ++y)
         {
            for (unsigned x = 0; x < width; ++x)
            {
               if (y == 0)
               {
                  rawShorts[x] = (unsigned short) (maxVal * (x + 1) / (width - 1));
               }
               else {
                  rawShorts[x + y * width] = rawShorts[x];
               }
            }
         }
         return true;
      }
      case 4:
      {
         const unsigned long maxVal = 255;
         unsigned* rawPixels = reinterpret_cast<unsigned*>(img.GetPixelsRW());
         for (unsigned section = 0; section < 8; ++section)
         {
            unsigned ystart = section * (height / 8);
            unsigned ystop = section == 7 ? height : ystart + (height / 8);
            for (unsigned y = ystart; y < ystop; ++y)
            {
               for (unsigned x = 0; x < width; ++x)
               {
                  rawPixels[x + y * width] = 0;
                  for (unsigned component = 0; component < 4; ++component)
                  {
                     unsigned sample = 0;
                     if (component == section ||
                           (section >= 4 && section - 4 != component))
                     {
                        sample = maxVal * (x + 1) / (width - 1);
                     }
                     sample &= 0xff; // Just in case
                     rawPixels[x + y * width] |= sample << (8 * component);
                  }
               }
            }
         }
         return true;
      }
   }
   return false;
}


/**
* Generate an image with offset plus noise
*/
void CDemoCamera::AddBackgroundAndNoise(ImgBuffer& img, double mean, double stdDev)
{
	char buf[MM::MaxStrLength];
   GetProperty(MM::g_Keyword_PixelType, buf);
	std::string pixelType(buf);

   int maxValue = 1 << GetBitDepth();
   long nrPixels = img.Width() * img.Height();
   if (pixelType.compare(g_PixelType_8bit) == 0)
   {
      unsigned char* pBuf = (unsigned char*) const_cast<unsigned char*>(img.GetPixels());
      for (long i = 0; i < nrPixels; i++)
      {
         double value = GaussDistributedValue(mean, stdDev);
         if (value < 0)
         {
            value = 0;
         }
         else if (value > maxValue)
         {
            value = maxValue;
         }
         *(pBuf + i) = (unsigned char) value;
      }
   }
   else if (pixelType.compare(g_PixelType_16bit) == 0)
   {
      unsigned short* pBuf = (unsigned short*) const_cast<unsigned char*>(img.GetPixels());
      for (long i = 0; i < nrPixels; i++)
      {
         double value = GaussDistributedValue(mean, stdDev);
         if (value < 0)
         {
            value = 0;
         }
         else if (value > maxValue)
         {
            value = maxValue;
         }
         *(pBuf + i) = (unsigned short) value;
      }
   }
}


/**
* Adds signal to an image
* Assume a homogenuous illumination
* Calculates the signal for each pixel individually as:
* photon flux * exposure time / conversion factor
* Assumes QE of 100%
*/
void CDemoCamera::AddSignal(ImgBuffer& img, double photonFlux, double exp, double cf)
{
	char buf[MM::MaxStrLength];
   GetProperty(MM::g_Keyword_PixelType, buf);
	std::string pixelType(buf);

   int maxValue = (1 << GetBitDepth()) -1;
   long nrPixels = img.Width() * img.Height();
   double photons = photonFlux * exp;
   double shotNoise = sqrt(photons);
   double digitalValue = photons / cf;
   double shotNoiseDigital = shotNoise / cf;
   if (pixelType.compare(g_PixelType_8bit) == 0)
   {
      unsigned char* pBuf = (unsigned char*) const_cast<unsigned char*>(img.GetPixels());
      for (long i = 0; i < nrPixels; i++)
      {
         double value = *(pBuf + i) + GaussDistributedValue(digitalValue, shotNoiseDigital);
         if (value < 0)
         {
            value = 0;
         }
         else if (value > maxValue)
         {
            value = maxValue;
         }
         *(pBuf + i) =  (unsigned char) value;
      }
   }
   else if (pixelType.compare(g_PixelType_16bit) == 0)
   {
      unsigned short* pBuf = (unsigned short*) const_cast<unsigned char*>(img.GetPixels());
      for (long i = 0; i < nrPixels; i++)
      {
         double value = *(pBuf + i) + GaussDistributedValue(digitalValue, shotNoiseDigital);
         if (value < 0)
         {
            value = 0;
         }
         else if (value > maxValue)
         {
            value = maxValue;
         }
         *(pBuf + i) = (unsigned short) value;
      }
   }
}


/**
 * Uses Marsaglia polar method to generate Gaussian distributed value.
 * Then distributes this around mean with the desired std
 */
double CDemoCamera::GaussDistributedValue(double mean, double std)
{
   double s = 2;
   double u = 1; // inconsequential, but avoid potential use of uninitialized value
   double v;
   double halfRandMax = (double) RAND_MAX / 2.0;
   while (s >= 1 || s <= 0)
   {
      // get random values between -1 and 1
      u = (double) rand() / halfRandMax - 1.0;
      v = (double) rand() / halfRandMax - 1.0;
      s = u * u + v * v;
   }
   double tmp = sqrt( -2 * log(s) / s);
   double x = u * tmp;

   return mean + std * x;
}

///////////////////////////////////////////////////////////////////////////////
// Bead mode implementation
///////////////////////////////////////////////////////////////////////////////

// Hash function for tile coordinates - generates deterministic "random" seed
unsigned int CDemoCamera::HashTileCoords(int tileX, int tileY)
{
   // Simple hash combining tile coordinates
   // Using prime numbers for better distribution
   unsigned int hash = 0;
   hash = tileX * 73856093;
   hash ^= tileY * 19349663;
   return hash;
}

// Generate beads for a specific tile using procedural generation
void CDemoCamera::GenerateBeadsForTile(int tileX, int tileY, std::vector<Bead>& beads)
{
   const double tileSize = 512.0; // microns, matches typical image size
   
   // Use tile hash as seed for deterministic random generation
   unsigned int seed = HashTileCoords(tileX, tileY);
   std::srand(seed);
   
   // Calculate tile boundaries in world coordinates
   double tileWorldX = tileX * tileSize;
   double tileWorldY = tileY * tileSize;
   
   // Generate beads for this tile
   // beadDensity_ is per image (512x512), so use same density per tile
   int beadsPerTile = beadDensity_;
   
   for (int i = 0; i < beadsPerTile; ++i)
   {
      Bead bead;
      // Generate position within this tile
      bead.worldX = tileWorldX + (double)std::rand() / RAND_MAX * tileSize;
      bead.worldY = tileWorldY + (double)std::rand() / RAND_MAX * tileSize;
      bead.intensityFactor = 0.8 + 0.4 * (double)std::rand() / RAND_MAX;  // 80-120%
      bead.sizeFactor = 0.8 + 0.4 * (double)std::rand() / RAND_MAX;       // 80-120%
      beads.push_back(bead);
   }
}

void CDemoCamera::GenerateBeadPositions()
{
   beads_.clear();
   
   const double tileSize = 512.0; // microns
   
   // Get current stage position
   double stageX, stageY;
   GetCurrentXYPosition(stageX, stageY);
   
   // Calculate which tiles are visible
   // Image size in world coords (1px = 1um, typical 512x512)
   double imageWidth = 512.0;
   double imageHeight = 512.0;
   
   // Add margin to ensure beads near edges with blur are included
   double margin = 100.0; // microns
   
   // Calculate view bounds
   double viewLeft = stageX - imageWidth / 2.0 - margin;
   double viewRight = stageX + imageWidth / 2.0 + margin;
   double viewBottom = stageY - imageHeight / 2.0 - margin;
   double viewTop = stageY + imageHeight / 2.0 + margin;
   
   // Calculate tile range
   int tileXMin = (int)floor(viewLeft / tileSize);
   int tileXMax = (int)floor(viewRight / tileSize);
   int tileYMin = (int)floor(viewBottom / tileSize);
   int tileYMax = (int)floor(viewTop / tileSize);
   
   // Generate beads for all visible tiles
   for (int tileY = tileYMin; tileY <= tileYMax; ++tileY)
   {
      for (int tileX = tileXMin; tileX <= tileXMax; ++tileX)
      {
         GenerateBeadsForTile(tileX, tileY, beads_);
      }
   }
   
   beadsGenerated_ = true;
}

double CDemoCamera::GetCurrentZPosition()
{
   // Try "Z" label first (Micro-Manager default for demo)
   MM::Device* pStage = GetDevice("Z");
   if (!pStage)
      pStage = GetDevice(g_StageDeviceName);
      
   if (!pStage)
      return 0.0;
   
   // Get the position property
   char pos[MM::MaxStrLength];
   int ret = pStage->GetProperty(MM::g_Keyword_Position, pos);
   if (ret != DEVICE_OK)
      return 0.0;
      
   return atof(pos);
}

void CDemoCamera::GetCurrentXYPosition(double& x, double& y)
{
   x = 0.0;
   y = 0.0;
   
   // Try "XY" label first (Micro-Manager default for demo)
   MM::Device* pXYStage = GetDevice("XY");
   if (!pXYStage)
      pXYStage = GetDevice(g_XYStageDeviceName);
      
   if (!pXYStage)
      return;
   
   // Get XY stage interface
   MM::XYStage* pStage = dynamic_cast<MM::XYStage*>(pXYStage);
   if (!pStage)
      return;
   
   // Get position
   pStage->GetPositionUm(x, y);
}

void CDemoCamera::RenderBeadToImage(ImgBuffer& img, const Bead& bead, double blurRadius, double stageX, double stageY)
{
   unsigned width = img.Width();
   unsigned height = img.Height();
   unsigned depth = img.Depth();
   
   // Convert world coordinates to screen coordinates
   // 1px = 1um
   double pixelSizeUm = 1.0;
   double screenX = (bead.worldX - stageX) / pixelSizeUm + width / 2.0;
   double screenY = (bead.worldY - stageY) / pixelSizeUm + height / 2.0;
   
   // Total sigma is the sum of base size and defocus blur (root sum of squares for convolutions)
   double effectiveBaseSize = beadSize_ * bead.sizeFactor;
   double totalSigma = sqrt(effectiveBaseSize * effectiveBaseSize + blurRadius * blurRadius);
   if (totalSigma < 0.5) totalSigma = 0.5;
   
   double twoSigmaSq = 2.0 * totalSigma * totalSigma;
   
   // Render within 4-sigma radius
   int renderRadius = (int)(4.0 * totalSigma) + 1;
   
   int xMin = std::max(0, (int)(screenX - renderRadius));
   int xMax = std::min((int)width - 1, (int)(screenX + renderRadius));
   int yMin = std::max(0, (int)(screenY - renderRadius));
   int yMax = std::min((int)height - 1, (int)(screenY + renderRadius));
   
   // Use fixed maxValue of 255 regardless of bit depth. Users can scale brightness with beadBrightness param.
   double maxValue = 255.0;
   
   double amplitude = maxValue * bead.intensityFactor * beadBrightness_ * g_IntensityFactor_;
   
   for (int y = yMin; y <= yMax; ++y)
   {
      for (int x = xMin; x <= xMax; ++x)
      {
         double dx = x - screenX;
         double dy = y - screenY;
         double distSq = dx * dx + dy * dy;
         
         double value = amplitude * exp(-distSq / twoSigmaSq);
         
         unsigned char* pBuf = const_cast<unsigned char*>(img.GetPixels());
         unsigned long pixelIndex = (y * width + x);
         
         if (depth == 1)  // 8-bit
         {
            unsigned long idx = pixelIndex * depth;
            pBuf[idx] = (unsigned char)std::min(255.0, (double)pBuf[idx] + value);
         }
         else if (depth == 2)  // 16-bit
         {
            unsigned short* pShort = (unsigned short*)pBuf;
            pShort[pixelIndex] = (unsigned short)std::min(65535.0, (double)pShort[pixelIndex] + value);
         }
         else if (depth == 4 && nComponents_ == 1)  // 32-bit float
         {
            float* pFloat = (float*)pBuf;
            pFloat[pixelIndex] = std::min(1.0f, pFloat[pixelIndex] + (float)value);
         }
         else if (depth == 4 && nComponents_ == 4)  // 32-bit RGB - render to green channel
         {
            unsigned long idx = pixelIndex * 4;
            pBuf[idx + 1] = (unsigned char)std::min(255.0, (double)pBuf[idx + 1] + value);
         }
      }
   }
}

void CDemoCamera::GenerateBeadsImage(ImgBuffer& img, double exposure)
{
   if (img.Height() == 0 || img.Width() == 0 || img.Depth() == 0)
      return;
   
   // Clear image to dark background
   unsigned char* pBuf = const_cast<unsigned char*>(img.GetPixels());
   memset(pBuf, 0, img.Height() * img.Width() * img.Depth() * ((nComponents_ > 1) ? nComponents_ : 1));
   
   // Get current XY and Z positions
   double stageX, stageY;
   GetCurrentXYPosition(stageX, stageY);
   double zPos = GetCurrentZPosition();
   
   // Regenerate beads for current view (always regenerate since stage may have moved)
   GenerateBeadPositions();
   
   // Calculate blur (no cap)
   double blurRadius = beadBlurRate_ * std::abs(zPos);
   
   // 1px = 1um
   double pixelSizeUm = 1.0;
   
   // Render each bead
   for (const auto& bead : beads_)
   {
      // Convert world coordinates to screen coordinates for visibility check
      double screenX = (bead.worldX - stageX) / pixelSizeUm + img.Width() / 2.0;
      double screenY = (bead.worldY - stageY) / pixelSizeUm + img.Height() / 2.0;
      
      // Total sigma is sum of base size and blur
      double effectiveBaseSize = beadSize_ * bead.sizeFactor;
      double totalSigma = sqrt(effectiveBaseSize * effectiveBaseSize + blurRadius * blurRadius);
      
      // Only render beads that might be visible (with some margin for blur)
      double margin = totalSigma * 4.0;
      if (screenX >= -margin && screenX < img.Width() + margin &&
         screenY >= -margin && screenY < img.Height() + margin)
      {
         RenderBeadToImage(img, bead, blurRadius, stageX, stageY);
      }
   }
}
