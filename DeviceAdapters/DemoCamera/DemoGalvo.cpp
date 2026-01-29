///////////////////////////////////////////////////////////////////////////////
// FILE:          DemoGalvo.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   Galvo device implementation for the DemoCamera adapter
//                Simulates a galvo device with image callback functionality
//
// AUTHOR:        Nenad Amodaj, nenad@amodaj.com, 06/08/2005
//
// COPYRIGHT:     University of California, San Francisco, 2006-2015
//                100X Imaging Inc, 2008
//
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
#include <sstream>

extern const char* g_GalvoDeviceName;
extern const char* NoHubError;

///////////////////////////////////////////////////////////////////////////////
// DemoGalvo implementation
///////////////////////////////////////////////////////////////////////////////

DemoGalvo::DemoGalvo() {}


DemoGalvo::~DemoGalvo()
{
   Shutdown();
}

void DemoGalvo::GetName(char* pName) const
{
   CDeviceUtils::CopyLimitedString(pName, g_GalvoDeviceName);
}

int DemoGalvo::Initialize()
{
   // generate Gaussian kernal
   // Size is determined in the header file
   int xSize = sizeof(gaussianMask_) / sizeof(gaussianMask_[0]);
   int ySize = sizeof(gaussianMask_[0]) / 2;
   for (int x = 0; x < xSize; x++)
   {
      for (int y =0; y < ySize; y++)
      {
         gaussianMask_[x][y] =(unsigned short) GaussValue(41, 0.5, 0.5, xSize / 2, ySize / 2, x, y);
      }
   }

   DemoHub* pHub = static_cast<DemoHub*>(GetParentHub());
   if (!pHub)
   {
      LogMessage(NoHubError);
   }
   else {
      char deviceName[MM::MaxStrLength];
      unsigned int deviceIterator = 0;
      for (;;)
      {
         GetLoadedDeviceOfType(MM::CameraDevice, deviceName, deviceIterator);
         if (0 < strlen(deviceName))
         {
            std::ostringstream os;
            os << "Galvo detected: " << deviceName;
            LogMessage(os.str().c_str());
            MM::Camera* camera = (MM::Camera*) GetDevice(deviceName);
            MM::Hub* cHub = GetCoreCallback()->GetParentHub(camera);
            if (cHub == pHub)
            {
               demoCamera_ = (CDemoCamera*) camera;
               demoCamera_->RegisterImgManipulatorCallBack(this);
               LogMessage("DemoGalvo registered as callback");
               break;
            }
         }
         else
         {
            LogMessage("Galvo detected no camera devices");
            break;
         }
         deviceIterator++;
      }
   }
   return DEVICE_OK;
}

int DemoGalvo::PointAndFire(double x, double y, double pulseTime_us)
{
   SetPosition(x, y);
   MM::MMTime offset(pulseTime_us);
   pfExpirationTime_ = GetCurrentMMTime() + offset;
   pointAndFire_ = true;
   //std::ostringstream os;
   //os << "PointAndFire set galvo to : " << x << " - " << y;
   //LogMessage(os.str().c_str());
   return DEVICE_OK;
}

int DemoGalvo::SetSpotInterval(double pulseTime_Us)
{
   pulseTime_Us_ = pulseTime_Us;
   return DEVICE_OK;
}

int DemoGalvo::SetPosition(double x, double y)
{
   currentX_ = x;
   currentY_ = y;
   return DEVICE_OK;
}

int DemoGalvo::GetPosition(double& x, double& y)
{
   x = currentX_;
   y = currentY_;
   return DEVICE_OK;
}

int DemoGalvo::SetIlluminationState(bool on)
{
   illuminationState_ = on;
   return DEVICE_OK;
}

int DemoGalvo::AddPolygonVertex(int polygonIndex, double x, double y)
{
   std::vector<PointD> vertex = vertices_[polygonIndex];
   vertices_[polygonIndex].push_back(PointD(x, y));
   //std::ostringstream os;
   //os << "Adding point to polygon " << polygonIndex << ", x: " << x  <<
   //   ", y: " << y;
   //LogMessage(os.str().c_str());

   return DEVICE_OK;
}

int DemoGalvo::DeletePolygons()
{
   vertices_.clear();
   return DEVICE_OK;
}

/**
 * This is to load the polygons into the device
 * Since we are virtual, there is nothing to do here
 */
int DemoGalvo::LoadPolygons()
{
   return DEVICE_OK;
}

int DemoGalvo::SetPolygonRepetitions(int /* repetitions */)
{
   return DEVICE_OK;
}

int DemoGalvo::RunPolygons()
{
   std::ostringstream os;
   os << "# of polygons: " << vertices_.size() << std::endl;
   for (std::map<int, std::vector<PointD> >::iterator it = vertices_.begin();
         it != vertices_.end(); ++it)
   {
      os << "ROI " << it->first << " has " << it->second.size() << " points" << std::endl;
      // illuminate just the first point
      this->PointAndFire(it->second.at(0).x, it->second.at(0).y, pulseTime_Us_);
      CDeviceUtils::SleepMs((long) (pulseTime_Us_ / 1000.0));
   }
   LogMessage(os.str().c_str());

   //runROIS_ = true;
   return DEVICE_OK;
}

int DemoGalvo::RunSequence()
{
   return DEVICE_OK;
}

int DemoGalvo::StopSequence()
{
   return DEVICE_OK;
}

// What can this function be doing?
// A channel is never set, so how come we can return one????
// Documentation of the Galvo interface is severely lacking!!!!
int DemoGalvo::GetChannel (char* /* channelName */)
{
   return DEVICE_OK;
}

double DemoGalvo::GetXRange()
{
   return xRange_;
}

double DemoGalvo::GetYRange()
{
   return yRange_;
}


/**
 * Callback function that will be called by DemoCamera everytime
 * a new image is generated.
 * We insert a Gaussian spot if the state of our device suggests to do so
 * The position of the spot is set by the relation defined in the function
 * GalvoToCameraPoint
 * Also will draw ROIs when requested
 */
int DemoGalvo::ChangePixels(ImgBuffer& img)
{
   if (!illuminationState_ && !pointAndFire_ && !runROIS_)
   {
      //std::ostringstream os;
      //os << "No action requested in ChangePixels";
      //LogMessage(os.str().c_str());
      return DEVICE_OK;
   }

   if (runROIS_)
   {
      // establish the bounding boxes around the ROIs in image coordinates
      std::vector<std::vector<Point> > bBoxes = std::vector<std::vector<
         Point> >();
      for (unsigned int i = 0; i < vertices_.size(); i++) {
         std::vector<Point> vertex;
         for (std::vector<PointD>::iterator it = vertices_[i].begin();
               it != vertices_[i].end(); ++it)
         {
            Point p = GalvoToCameraPoint(*it, img);
            vertex.push_back(p);
         }
         std::vector<Point> bBox;
         GetBoundingBox(vertex, bBox);
         bBoxes.push_back(bBox);
         //std::ostringstream os;
         //os << "BBox: " << bBox[0].x << ", " << bBox[0].y << ", " <<
         //  bBox[1].x << ", " << bBox[1].y;
         //LogMessage(os.str().c_str());
      }
      if (img.Depth() == 1)
      {
         const unsigned char highValue = 240;
         unsigned char* pBuf = (unsigned char*) const_cast<unsigned char*>(img.GetPixels());

         // now iterate through the image pixels and set high
         // if they are within a bounding box
         for (unsigned int x = 0; x < img.Width(); x++)
         {
            for (unsigned int y = 0; y < img.Height(); y++)
            {
               bool inROI = false;
               for (unsigned int i = 0; i < bBoxes.size(); i++)
               {
                  if (InBoundingBox(bBoxes[i], Point(x, y)))
                     inROI = true;
               }
               if (inROI)
               {
                  long count = y * img.Width() + x;
                  *(pBuf + count) = *(pBuf + count) + highValue;
               }
            }
         }
         img.SetPixels(pBuf);
      }
      else if (img.Depth() == 2)
      {
         const unsigned short highValue = 2048;
         unsigned short* pBuf = (unsigned short*) const_cast<unsigned char*>(img.GetPixels());

         // now iterate through the image pixels and set high
         // if they are within a bounding box
         for (unsigned int x = 0; x < img.Width(); x++)
         {
            for (unsigned int y = 0; y < img.Height(); y++)
            {
               bool inROI = false;
               for (unsigned int i = 0; i < bBoxes.size(); i++)
               {
                  if (InBoundingBox(bBoxes[i], Point(x, y)))
                     inROI = true;
               }
               if (inROI)
               {
                  long count = y * img.Width() + x;
                  *(pBuf + count) = *(pBuf + count) + highValue;
               }
            }
         }
         img.SetPixels(pBuf);
      }
      runROIS_ = false;
   } else
   {
      Point cp = GalvoToCameraPoint(PointD(currentX_, currentY_), img);
      int xPos = cp.x; int yPos = cp.y;

      std::ostringstream os;
      os << "XPos: " << xPos << ", YPos: " << yPos;
      LogMessage(os.str().c_str());
      int xSpotSize = sizeof(gaussianMask_) / sizeof(gaussianMask_[0]);
      int ySpotSize = sizeof(gaussianMask_[0]) / 2;

      if (xPos > xSpotSize && xPos < (int) (img.Width() - xSpotSize - 1)  &&
         yPos > ySpotSize && yPos < (int) (img.Height() - ySpotSize - 1) )
      {
         if (img.Depth() == 1)
         {
            unsigned char* pBuf = (unsigned char*) const_cast<unsigned char*>(img.GetPixels());
            for (int x = 0; x < xSpotSize; x++)
            {
               for (int y = 0; y < ySpotSize; y++)
               {
                  int w = xPos + x;
                  int h = yPos + y;
                  long count = h * img.Width() + w;
                  *(pBuf + count) = *(pBuf + count) + 5 * (unsigned char) gaussianMask_[x][y];
               }
            }
            img.SetPixels(pBuf);
         }
         else if (img.Depth() == 2)
         {
            unsigned short* pBuf = (unsigned short*) const_cast<unsigned char*>(img.GetPixels());
            for (int x = 0; x < xSpotSize; x++)
            {
               for (int y = 0; y < ySpotSize; y++)
               {
                  int w = xPos + x;
                  int h = yPos + y;
                  long count = h * img.Width() + w;
                  *(pBuf + count) = *(pBuf + count) + 30 * (unsigned short) gaussianMask_[x][y];
               }
            }
            img.SetPixels(pBuf);
         }
      }
      if (pointAndFire_)
      {
         if (GetCurrentMMTime() > pfExpirationTime_)
         {
            pointAndFire_ = false;
         }
      }
   }

   return DEVICE_OK;
}

/**
 * Function that converts between the Galvo and Camera coordinate system
 * There is a bit of a conundrum, since we do not know what ROI and binning were
 * used when calibration took place. Let's assume 1x binning and full frame
 * (and hope that is the same full frame as the camera is set to now).
 * Note: ImgBuffer is not really needed as an input
 * Returns point in coordinates suitable for the given ImgBuffer
 * assuming that it has the ROI and binning as we get from the camera
 */
Point DemoGalvo::GalvoToCameraPoint(PointD galvoPoint, ImgBuffer& img)
{
   long width = img.Width();
   long height = img.Height();
   int binning = 1;
   unsigned x = 0, y = 0, xSize, ySize;
   if (demoCamera_ != 0)
   {
      width = demoCamera_->GetCCDXSize();
      height = demoCamera_->GetCCDYSize();
      binning = demoCamera_->GetBinning();
      // Note: ROI is in units of binned pixels
      demoCamera_->GetROI(x, y, xSize, ySize);
   }
   // Get the position on the unbinned, full CCD
   int xPos = (int) ((double) offsetX_ + (double) (galvoPoint.x / vMaxX_) *
                                 ((double) width - (double) offsetX_) );
   int yPos = (int) ((double) offsetY_ + (double) (galvoPoint.y / vMaxY_) *
                                 ((double) height - (double) offsetY_));

   return Point( (xPos/binning) - x, (yPos/binning) -y);
}

/**
 * Utility function to calculate a 2D Gaussian
 * Used in the initialize function to get a 10x10 2D Gaussian
 */
double DemoGalvo::GaussValue(double amplitude, double sigmaX, double sigmaY, int muX, int muY, int x, int y)
{
   double factor = - ( ((double)(x - muX) * (double)(x - muX) / 2 * sigmaX * sigmaX) +
         (double)(y - muY) * (double)(y - muY) / 2 * sigmaY * sigmaY);

   double result = amplitude * exp(factor);
   std::ostringstream os;
   os << "x: " << x << ", y: " << y << ", value: " << result;
   LogMessage(os.str().c_str());
   return result;

}
/**
 * Returns the bounding box around the points defined in vertex
 * bBox is a vector with 2 points
 */
void DemoGalvo::GetBoundingBox(std::vector<Point>& vertex, std::vector<Point>& bBox)
{
   if (vertex.size() < 1)
   {
      return;
   }
   int minX = vertex[0].x;
   int maxX = minX;
   int minY = vertex[0].y;
   int maxY = minY;
   for (unsigned int i = 1; i < vertex.size(); i++)
   {
      if (vertex[i].x < minX)
         minX = vertex[i].x;
      if (vertex[i].x > maxX)
         maxX = vertex[i].x;
      if (vertex[i].y < minY)
         minY = vertex[i].y;
      if (vertex[i].y > maxY)
         maxY = vertex[i].y;
   }
   bBox.push_back(Point(minX, minY));
   bBox.push_back(Point(maxX, maxY));
}

/**
 * Determines whether the given point is in the boundingBox
 * boundingBox should have two members, one with the minimum x, y position,
 * the second with the maximum x, y positions
 */
bool DemoGalvo::InBoundingBox(std::vector<Point> boundingBox, Point testPoint)
{
   if (testPoint.x >= boundingBox[0].x && testPoint.x <= boundingBox[1].x &&
         testPoint.y >= boundingBox[0].y && testPoint.y <= boundingBox[1].y)
      return true;
   return false;
}

/**
 * Not used (yet), intent was to use this to determine whether
 * a point is within the ROI, rather than drawing a bounding box
 */
bool DemoGalvo::PointInTriangle(Point p, Point p0, Point p1, Point p2)
{
    long s = (long) p0.y * p2.x - p0.x * p2.y + (p2.y - p0.y) * p.x + (p0.x - p2.x) * p.y;
    long t = (long) p0.x * p1.y - p0.y * p1.x + (p0.y - p1.y) * p.x + (p1.x - p0.x) * p.y;

    if ((s < 0) != (t < 0))
        return false;

    long A = (long) -p1.y * p2.x + p0.y * (p2.x - p1.x) + p0.x * (p1.y - p2.y) + p1.x * p2.y;
    if (A < 0.0)
    {
        s = -s;
        t = -t;
        A = -A;
    }
    return s > 0 && t > 0 && (s + t) < A;
}
