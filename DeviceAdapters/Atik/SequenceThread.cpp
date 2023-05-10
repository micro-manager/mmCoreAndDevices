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

#include "Atik.h"

SequenceThread::SequenceThread(Atik* pCam)
   :intervalMs_(100.0),
   numImages_(0),
   imageCounter_(0),
   stop_(true),
   camera_(pCam)
{
//LOG("");
};

SequenceThread::~SequenceThread() {};

void SequenceThread::Stop() {
	//LOG("");
   stop_=true;
}

void SequenceThread::Start(long numImages, double intervalMs)
{
	//LOG("");
   numImages_=numImages;
   intervalMs_=intervalMs;
   imageCounter_=0;
   stop_ = false;
   activate();
}

bool SequenceThread::IsStopped(){
	//LOG("");
   return stop_;
}

int SequenceThread::svc(void) throw()
{
//	LOG("");
   int ret=DEVICE_ERR;
   return ret;
}
