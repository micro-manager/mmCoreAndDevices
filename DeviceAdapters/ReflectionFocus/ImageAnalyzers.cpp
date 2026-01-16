
#include "ReflectionFocus.h"
#include <algorithm>


int DetectTwoBrightSpots::AnalyzeImage(ImgBuffer img, double& score1, double& x1, double& y1, double& score2, double& x2, double& y2)
{
   // Find up to two spots in the image, return their (x,y) coordinates and scores

   const unsigned int width = img.Width();
   const unsigned int height = img.Height();
   const unsigned int pixDepth = img.Depth();
   const unsigned char* pixels = img.GetPixels();

   if (pixels == nullptr || width == 0 || height == 0)
   {
      score1 = score2 = 0.0;
      x1 = y1 = x2 = y2 = 0.0;
      return DEVICE_OK;
   }

   // Calculate threshold using Otsu's method or use a simple percentile-based threshold
   // For bright spots on dark background, we'll use a high percentile threshold
   std::vector<unsigned char> intensities;
   intensities.reserve(width * height);

   if (pixDepth == 1)
   {
      // 8-bit images: copy directly
      for (unsigned int i = 0; i < width * height; ++i)
      {
         intensities.push_back(pixels[i]);
      }
   }
   else if (pixDepth == 2)
   {
      // 16-bit images: find max value and scale to 8-bit range
      const unsigned short* pixels16 = (const unsigned short*)pixels;
      unsigned short maxValue = 0;

      // First pass: find maximum value
      for (unsigned int i = 0; i < width * height; ++i)
      {
         if (pixels16[i] > maxValue)
            maxValue = pixels16[i];
      }

      // Second pass: scale to 8-bit range
      if (maxValue > 0)
      {
         double scale = 255.0 / maxValue;
         for (unsigned int i = 0; i < width * height; ++i)
         {
            unsigned char scaledValue = static_cast<unsigned char>(pixels16[i] * scale);
            intensities.push_back(scaledValue);
         }
      }
      else
      {
         // All pixels are zero - no spots to find
         score1 = score2 = 0.0;
         x1 = y1 = x2 = y2 = 0.0;
         return DEVICE_OK;
      }
   }

   // Find threshold as 95th percentile to isolate bright spots
   std::vector<unsigned char> sorted = intensities;
   std::sort(sorted.begin(), sorted.end());
   unsigned char threshold = sorted[static_cast<size_t>(sorted.size() * 0.99)];

   // Label connected components using flood fill
   std::vector<int> labels(width * height, -1);
   int currentLabel = 0;

   struct Spot {
      double sumX = 0.0;
      double sumY = 0.0;
      double sumIntensity = 0.0;
      double sumX2 = 0.0; // For calculating variance/spread
      double sumY2 = 0.0;
      int pixelCount = 0;
      unsigned char maxIntensity = 0;
   };

   std::vector<Spot> spots;

   // Flood fill to find connected components
   for (unsigned int y = 0; y < height; ++y)
   {
      for (unsigned int x = 0; x < width; ++x)
      {
         unsigned int idx = y * width + x;
         unsigned char intensity = intensities[idx];

         if (intensity >= threshold && labels[idx] == -1)
         {
            // Start new component
            spots.push_back(Spot());
            std::vector<std::pair<unsigned int, unsigned int>> stack;
            stack.push_back(std::make_pair(x, y));
            labels[idx] = currentLabel;

            while (!stack.empty())
            {
               std::pair<unsigned int, unsigned int> pos = stack.back();
               stack.pop_back();
               unsigned int px = pos.first;
               unsigned int py = pos.second;
               unsigned int pidx = py * width + px;
               unsigned char pIntensity = intensities[pidx];

               // Add to spot statistics
               spots[currentLabel].sumX += px * pIntensity;
               spots[currentLabel].sumY += py * pIntensity;
               spots[currentLabel].sumIntensity += pIntensity;
               spots[currentLabel].sumX2 += px * px * pIntensity;
               spots[currentLabel].sumY2 += py * py * pIntensity;
               spots[currentLabel].pixelCount++;
               if (pIntensity > spots[currentLabel].maxIntensity)
                  spots[currentLabel].maxIntensity = pIntensity;

               // Check 8-connected neighbors
               for (int dy = -1; dy <= 1; ++dy)
               {
                  for (int dx = -1; dx <= 1; ++dx)
                  {
                     if (dx == 0 && dy == 0) continue;

                     int nx = px + dx;
                     int ny = py + dy;

                     if (nx >= 0 && nx < (int)width && ny >= 0 && ny < (int)height)
                     {
                        unsigned int nidx = ny * width + nx;
                        if (intensities[nidx] >= threshold && labels[nidx] == -1)
                        {
                           labels[nidx] = currentLabel;
                           stack.push_back(std::make_pair(nx, ny));
                        }
                     }
                  }
               }
            }
            currentLabel++;
         }
      }
   }

   // Calculate centroid, variance, and score for each spot
   struct SpotResult {
      double x = 0.0;
      double y = 0.0;
      double score = 0.0;
      double totalIntensity = 0.0;
   };

   std::vector<SpotResult> results;
   for (size_t i = 0; i < spots.size(); ++i)
   {
      if (spots[i].sumIntensity > 0)
      {
         SpotResult result;
         result.x = spots[i].sumX / spots[i].sumIntensity;
         result.y = spots[i].sumY / spots[i].sumIntensity;
         result.totalIntensity = spots[i].sumIntensity;

         // Calculate variance (spread) of the spot
         double varX = (spots[i].sumX2 / spots[i].sumIntensity) - (result.x * result.x);
         double varY = (spots[i].sumY2 / spots[i].sumIntensity) - (result.y * result.y);
         double spread = sqrt(varX + varY);

         // Score: smaller spread = better focus = higher score
         // Use inverse of spread, normalized by total intensity
         if (spread > 0.0)
            result.score = spots[i].sumIntensity / (spread * spread);
         else
            result.score = spots[i].sumIntensity * 1000.0; // Very small spot

         results.push_back(result);
      }
   }

   // Sort spots by intensity
   std::sort(results.begin(), results.end(),
      [](const SpotResult& a, const SpotResult& b) {
         return a.totalIntensity > b.totalIntensity;
      });

   // Return top 2 spots, highest score first
   if (results.size() >= 2)
   {
      if (results[0].score >= results[1].score)
      {
         x1 = results[0].x;
         y1 = results[0].y;
         score1 = results[0].score;
         x2 = results[1].x;
         y2 = results[1].y;
         score2 = results[1].score;
      }
      else
      {
         x1 = results[1].x;
         y1 = results[1].y;
         score1 = results[1].score;
         x2 = results[0].x;
         y2 = results[0].y;
         score2 = results[0].score;
      }
   }
   else if (results.size() == 1)
   {
      x1 = results[0].x;
      y1 = results[0].y;
      score1 = results[0].score;
      x2 = y2 = score2 = 0.0;
   }
   else
   {
      x1 = y1 = score1 = 0.0;
   }

   return DEVICE_OK;
}
