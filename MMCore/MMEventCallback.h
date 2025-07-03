///////////////////////////////////////////////////////////////////////////////
// FILE:          MMEventCallback.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     MMCore
//-----------------------------------------------------------------------------
// DESCRIPTION:   Callback class used to send notifications from MMCore to
//                higher levels (such as GUI)
//
// AUTHOR:        Nenad Amodaj, nenad@amodaj.com, 12/10/2007
// COPYRIGHT:     University of California, San Francisco, 2007
//
// LICENSE:       This file is distributed under the "Lesser GPL" (LGPL) license.
//                License text is included with the source distribution.
//
//                This file is distributed in the hope that it will be useful,
//                but WITHOUT ANY WARRANTY; without even the implied warranty
//                of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
//
//                IN NO EVENT SHALL THE COPYRIGHT OWNER OR
//                CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
//                INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES.
// CVS:           $Id: Configuration.h 2 2007-02-27 23:33:17Z nenad $
//
#pragma once
#include <iostream>

class MMEventCallback
{
public:
   MMEventCallback() {}
   virtual ~MMEventCallback() {}

   virtual void onPropertiesChanged()
   {
      std::cout << "onPropertiesChanged()\n";
   }

   virtual void onPropertyChanged(const char* name, const char* propName, const char* propValue)
   {
      std::cout << "onPropertyChanged() " << name << " " << propName << " " << propValue << '\n';
   }

   virtual void onChannelGroupChanged(const char* newChannelGroupName)
   {
      std::cout << "onChannelGroupChanged() " << newChannelGroupName << '\n';
   }

   virtual void onConfigGroupChanged(const char* groupName, const char* newConfigName)
   {
      std::cout << "onConfigGroupChanged() " << groupName << " " << newConfigName << '\n';
   }

   /**
    * \brief Called when the system configuration has changed.
    * 
    * "Changed" includes when a configuration was unloaded, for example because
    * loading a new config file failed.
    */
   virtual void onSystemConfigurationLoaded()
   {
      std::cout << "onSystemConfigurationLoaded()\n";
   }

   virtual void onPixelSizeChanged(double newPixelSizeUm)
   {
      std::cout << "onPixelSizeChanged() " << newPixelSizeUm << '\n';
   }

   virtual void onPixelSizeAffineChanged(double v0, double v1, double v2, double v3, double v4, double v5)
   {
      std::cout << "onPixelSizeAffineChanged() " << v0 << "-" << v1 << "-" << v2 << "-" << v3 << "-" << v4 << "-" << v5 << '\n';
   }

   virtual void onStagePositionChanged(const char* name, double pos)
   {
      std::cout << "onStagePositionChanged()" << name << " " << pos  << '\n';
   }

   virtual void onXYStagePositionChanged(const char* name, double xpos, double ypos)
   {
      std::cout << "onXYStagePositionChanged()" << name << " " << xpos;
      std::cout << " " <<  ypos << '\n';
   }

   virtual void onExposureChanged(const char* name, double newExposure)
   {
      std::cout << "onExposureChanged()" << name << " " << newExposure << '\n';
   }

   virtual void onSLMExposureChanged(const char* name, double newExposure)
   {
      std::cout << "onSLMExposureChanged()" << name << " " << newExposure << '\n';
   }

   virtual void onImageSnapped(const char* cameraLabel)
   {
      std::cout << "onImageSnapped() " << cameraLabel << '\n';
   }

   virtual void onSequenceAcquisitionStarted(const char* cameraLabel)
   {
      std::cout << "onSequenceAcquisitionStarted() " << cameraLabel << '\n';
   }

   virtual void onSequenceAcquisitionStopped(const char* cameraLabel)
   {
      std::cout << "onSequenceAcquisitionStopped() " << cameraLabel << '\n';
   }

};
