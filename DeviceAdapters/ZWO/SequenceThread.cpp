///////////////////////////////////////////////////////////////////////////////
// FILE:          SequenceThread.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters/MMCamera
//-----------------------------------------------------------------------------
// DESCRIPTION:   Impelements sequence thread for rendering live video.
//                Part of the skeleton code for the micro-manager camera adapter.
//                Use it as starting point for writing custom device adapters.
//                
// AUTHOR:        Nenad Amodaj, http://nenad.amodaj.com
//                
// COPYRIGHT:     University of California, San Francisco, 2011
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
//

#include "MyASICam2.h"

inline static void OutputDbgPrint(const char* strOutPutString, ...)
{
#ifdef _DEBUG
	char strBuf[128] = {0};
	sprintf(strBuf, "<%s> ", "MM_ASI");
	va_list vlArgs;
	va_start(vlArgs, strOutPutString);
	vsnprintf((char*)(strBuf+strlen(strBuf)), sizeof(strBuf)-strlen(strBuf), strOutPutString, vlArgs);
	va_end(vlArgs);

#ifdef _WINDOWS
	OutputDebugStringA(strBuf);
#elif defined _LIN
	printf("%s",strBuf);
#endif

#endif
}
SequenceThread::SequenceThread(CMyASICam* pCam)
   :intervalMs_(100.0),
   numImages_(0),
   imageCounter_(0),
   stop_(true),
   camera_(pCam)
{};

SequenceThread::~SequenceThread() {};

void SequenceThread::Stop() {
   stop_=true;
}

void SequenceThread::Start(long numImages, double intervalMs)
{
   numImages_= numImages;
   intervalMs_=intervalMs;
   imageCounter_=0;
   stop_ = false;
   OutputDbgPrint("bf act\n");
   activate();//开始线程
   OutputDbgPrint("af act\n");
}

bool SequenceThread::IsStopped(){
   return stop_;
}


int SequenceThread::svc(void) throw()
{
   int ret=DEVICE_ERR;
   if(camera_->uc_pImg == 0)
   {
	   	camera_->iBufSize = camera_->GetImageBufferSize();
		camera_->uc_pImg = new unsigned char[camera_->iBufSize];
		OutputDbgPrint("buf %d\n", camera_->iBufSize);
   }

 
	 
      do
      {  
         ret = camera_->RunSequenceOnThread(MM::MMTime{});//startTime_
      } while (!IsStopped() );//DEVICE_OK == ret &&           && imageCounter_++ < numImages_-1
	  ASIStopVideoCapture(camera_->ASICameraInfo.CameraID);
   camera_->OnThreadExiting();
   Stop();
   return ret;
}