/////////////////////////////////////////////////////////////////////////////
// FILE:       DiskoveryModel.cpp
// PROJECT:    MicroManage
// SUBSYSTEM:  DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:
// Andor/Spectral Diskovery Device adapter
//                
// AUTHOR: Nico Stuurman, 06/31/2015
//
// COPYRIGHT:  Regents of the University of California, 2015
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

#ifdef WIN32
   #include <windows.h>
#endif

#include "DiskoveryModel.h"
#include "Diskovery.h"

// Motor Running
void DiskoveryModel::SetMotorRunningSD(const bool p)
{
   MMThreadGuard g(lock_);
   motorRunningSD_ = p;
   std::ostringstream oss;
   oss << p;
   core_.OnPropertyChanged(hubDevice_, motorRunningProp_, oss.str().c_str());
}

// Preset SD
void DiskoveryModel::SetPresetSD(const uint16_t p)
{
   MMThreadGuard g(lock_);
   presetSD_ = p;
   if (sdDevice_ != 0)
   {
      sdDevice_->OnStateChanged(p - 1);
   }
}

void DiskoveryModel::SetPresetWF(const uint16_t p)
{
   MMThreadGuard g(lock_);
   presetWF_ = p;
   if (wfDevice_ != 0)
   {
      wfDevice_->OnStateChanged(p - 1);
   }                                                                   
}

// Preset Iris                                                         
void DiskoveryModel::SetPresetIris(const uint16_t p)                         
{
   MMThreadGuard g(lock_);
   presetIris_ = p;
   if (irisDevice_ != 0)
   {
      irisDevice_->OnStateChanged(p - 1);
   }                                                                   
}

// Preset TIRF                                                         
void DiskoveryModel::SetPresetTIRF(const uint16_t p)                         
{
   MMThreadGuard g(lock_);
   presetPX_ = p;
   if (tirfDevice_ != 0)
   {
      tirfDevice_->OnStateChanged(p);
   }
}

// Preset Filter W
void DiskoveryModel::SetPresetFilterW(const uint16_t p) 
{  
   MMThreadGuard g(lock_);
   presetFilterW_ = p;
   if (filterWDevice_ != 0)
   {
      filterWDevice_->OnStateChanged(p - 1);
   }                                                                   
}

// Preset Filter T                                                     
void DiskoveryModel::SetPresetFilterT(const uint16_t p) 
{ 
   MMThreadGuard g(lock_);
   presetFilterT_ = p;                                       
   if (filterTDevice_ != 0)
   {
      filterTDevice_->OnStateChanged(p - 1);
   }                                                                   
}

// TIRF slider Rot
void DiskoveryModel::SetPositionRot(const int32_t p)
{
   MMThreadGuard g(lock_);
   tirfRotPos_ = p;
   if (tirfDevice_ != 0)
   {
      tirfDevice_->SignalPropChanged("PositionRot", CDeviceUtils::ConvertToString((int) p));
   }
}

// TIRF slider Lin
void DiskoveryModel::SetPositionLin(const int32_t p)
{
   MMThreadGuard g(lock_);
   tirfLinPos_ = p;
   if (tirfDevice_ != 0)
   {
      tirfDevice_->SignalPropChanged("PositionLin", CDeviceUtils::ConvertToString((int) p));
   }
}

// Extract the number of the Iris label and use it as objective magnification
uint16_t DiskoveryModel::GetOM()
{
   uint16_t pos = GetPresetIris();
   const char* label = GetButtonIrisLabel(pos);
   std::string strLabel(label);
   std::stringstream ss(strLabel.substr(0, strLabel.size() -1));
   uint16_t val;
   ss >> val;
   return val;
}

