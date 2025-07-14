///////////////////////////////////////////////////////////////////////////////
// FILE:          CoreUtils.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     MMCore
//-----------------------------------------------------------------------------
// DESCRIPTION:   Utility classes and functions for use in MMCore
//              
// AUTHOR:        Nenad Amodaj, nenad@amodaj.com, 09/27/2005
//
// COPYRIGHT:     University of California, San Francisco, 2006
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
//
// CVS:           $Id: CoreUtils.h 16845 2018-11-30 02:19:11Z nico $
//

#pragma once

#include "../MMDevice/MMDevice.h"

#include <string>


inline std::string ToString(int d) { return std::to_string(d); }
inline std::string ToString(long d) { return std::to_string(d); }
inline std::string ToString(long long d) { return std::to_string(d); }
inline std::string ToString(unsigned d) { return std::to_string(d); }
inline std::string ToString(unsigned long d) { return std::to_string(d); }
inline std::string ToString(unsigned long long d) { return std::to_string(d); }
inline std::string ToString(float d) { return std::to_string(d); }
inline std::string ToString(double d) { return std::to_string(d); }
inline std::string ToString(long double d) { return std::to_string(d); }

inline std::string ToString(const std::string& d) { return d; }

inline std::string ToString(const char* d)
{
   if (!d) 
      return "(null)";
   return d;
}

inline std::string ToString(const MM::DeviceType d)
{
   // TODO Any good way to ensure this doesn't get out of sync with the enum
   // definition?
   switch (d)
   {
      case MM::UnknownType: return "Unknown";
      case MM::AnyType: return "Any";
      case MM::CameraDevice: return "Camera";
      case MM::ShutterDevice: return "Shutter";
      case MM::StateDevice: return "State";
      case MM::StageDevice: return "Stage";
      case MM::XYStageDevice: return "XYStageDevice";
      case MM::SerialDevice: return "Serial";
      case MM::GenericDevice: return "Generic";
      case MM::AutoFocusDevice: return "Autofocus";
      case MM::CoreDevice: return "Core";
      case MM::ImageProcessorDevice: return "ImageProcessor";
      case MM::SignalIODevice: return "SignalIO";
      case MM::MagnifierDevice: return "Magnifier";
      case MM::SLMDevice: return "SLM";
      case MM::HubDevice: return "Hub";
      case MM::GalvoDevice: return "Galvo";
      case MM::PressurePumpDevice: return "PressurePump";
      case MM::VolumetricPumpDevice: return "VolumetricPump";
   }
   return "Invalid";
}

template <typename T>
inline std::string ToQuotedString(const T& d)
{ return "\"" + ToString(d) + "\""; }

template <>
inline std::string ToQuotedString<const char*>(char const* const& d)
{
   if (!d) // Don't quote if null
      return ToString(d);
   return "\"" + ToString(d) + "\"";
}