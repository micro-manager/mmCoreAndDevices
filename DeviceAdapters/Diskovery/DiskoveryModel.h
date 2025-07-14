///////////////////////////////////////////////////////////////////////////////
// FILE:       DiskoveryModel.h
// PROJECT:    MicroManager
// SUBSYSTEM:  DeviceAdapters
// AUTHOR:     Nico Stuurman
// COPYRIGHT:  Regents of the University of California, 2015
// 
//-----------------------------------------------------------------------------
// DESCRIPTION:
// Adapter for the Spectral/Andor/Oxford Instruments Diskovery 1 spinning disk confocal
// microscope system
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

#ifndef _DiskoveryModel_H_
#define _DiskoveryModel_H_

#include "MMDevice.h"
#include "DeviceBase.h"
#include <string>
#include <map>
#include <sstream>
#include <vector>
#include <boost/thread/condition_variable.hpp>

class DiskoveryStateDev;

class DiskoveryModel
{
   public:
      DiskoveryModel(MM::Device* device, MM::Core& core) :
         lock_(),
         deviceBusy_(false),
         logicalBusy_(false),
         presetSD_(0),
         presetWF_(0),
         presetIris_(0),
         presetPX_(0),
         presetFilterT_(0),
         presetFilterW_(0),
         tirfLinPos_(0), 
         tirfRotPos_(0),
         hasWFX_(false),
         hasWFY_(false),
         hasSD_(false),
         hasROT_(false),
         hasLIN_(false),
         hasP1_(false),
         hasP2_(false),
         hasIRIS_(false),
         hasFilterW_(false),
         hasFilterT_(false),
         tirfFocalLength_(150),
         resolutionRotation_(0.0),
         minLinear_(0),
         maxLinear_(0),
         minRotation_(0),
         maxRotation_(0),
         offsetLinear_ (0), 
         offsetRotation_(0),
         wavelength1_(405), wavelength2_(0),
         tubeLensFocalLength_(150), 
         om_(100),
         depth_(100.0), na_(1.45), ri_(1.36),
         exitTIRF_(false),
         hubDevice_(device),
         sdDevice_(0),
         wfDevice_(0),
         tirfDevice_(0),
         irisDevice_(0),
         filterWDevice_(0),
         filterTDevice_(0),
         core_(core)
      {
         // propertyname are stored in the model so that we can generate
         // callbacks indicating properties have changed
         hardwareVersionProp_ = "HardwareVersion";
         firmwareVersionProp_ = "FirmwareVersion";
         manufacturingDateProp_ = "ManufacturingDate";
         serialNumberProp_ = "SerialNumber";
         spinningDiskPositionProp_ = "Spinning Disk Position";
         wideFieldPositionProp_ = "Wide Field Position";
         filterPositionProp_ = "Filter Position";
         irisPositionProp_ = "Iris Position";
         tirfPositionProp_ = "TIRF Position";
         motorRunningProp_ = "Motor Running";
         for (int i = 0; i < 4; i++) {
            std::ostringstream os;
            os << "Preset-" << (i + 1);
            buttonWFLabels_[i] = os.str();
            buttonIrisLabels_[i] = os.str();
            buttonFilterWLabels_[i] = os.str();
            buttonFilterTLabels_[i] = os.str();
            diskLabels_[i] = os.str();
         }
         for (int i = 0; i < 7; i++) {
            lineWavelength_[i] = 0;
            lineEnabled_[i] = false;
         }
      };
      ~DiskoveryModel() 
      {
         varCondition_.notify_one();
      };

      void RegisterSDDevice(DiskoveryStateDev* device) {  MMThreadGuard g(lock_); sdDevice_ = device; };
      void RegisterWFDevice(DiskoveryStateDev* device) {  MMThreadGuard g(lock_); wfDevice_ = device; };
      void RegisterTIRFDevice(DiskoveryStateDev* device) {  MMThreadGuard g(lock_); tirfDevice_ = device; };
      void RegisterIRISDevice(DiskoveryStateDev* device) {  MMThreadGuard g(lock_); irisDevice_ = device; };
      void RegisterFILTERWDevice(DiskoveryStateDev* device) {  MMThreadGuard g(lock_); filterWDevice_ = device; };
      void RegisterFILTERTDevice(DiskoveryStateDev* device) {  MMThreadGuard g(lock_); filterTDevice_ = device; };

      // Hardware version
      std::string GetHardwareVersion() {  MMThreadGuard g(lock_); return hardwareVersion_; };
      void SetHardwareVersionMajor(const uint16_t major)
      {
          MMThreadGuard g(lock_);
          hwmajor_ = major;
          MakeVersionString(hardwareVersion_, hwmajor_, hwminor_, hwrevision_);
      };
      uint16_t GetHardwareVersionMajor() {  MMThreadGuard g(lock_); return hwmajor_; }
      void SetHardwareVersionMinor(const uint16_t minor)
      {
          MMThreadGuard g(lock_);
          hwminor_ = minor;
          MakeVersionString(hardwareVersion_, hwmajor_, hwminor_, hwrevision_);
      };
      uint16_t GetHardwareVersionMinor() {  MMThreadGuard g(lock_); return hwminor_; }
      void SetHardwareVersionRevision(const uint16_t revision)
      {
          MMThreadGuard g(lock_);
          hwrevision_ = revision;
          MakeVersionString(hardwareVersion_, hwmajor_, hwminor_, hwrevision_);
      };
      uint16_t GetHardwareVersionRevision() {  MMThreadGuard g(lock_); return hwrevision_; }

      // Firmware version
      std::string GetFirmwareVersion() {  MMThreadGuard g(lock_); return firmwareVersion_; };
      void SetFirmwareVersionMajor(const uint16_t major)
      {
          MMThreadGuard g(lock_);
          fwmajor_ = major;
          MakeVersionString(firmwareVersion_, fwmajor_, fwminor_, fwrevision_);
      };
      uint16_t GetFirmwareVersionMajor() {  MMThreadGuard g(lock_); return fwmajor_; };
      void SetFirmwareVersionMinor(uint16_t minor)
      {
          MMThreadGuard g(lock_);
          fwminor_ = minor;
          MakeVersionString(firmwareVersion_, fwmajor_, fwminor_, fwrevision_);
      };
      uint16_t GetFirmwareVersionMinor() {  MMThreadGuard g(lock_); return fwminor_; };
      void SetFirmwareVersionRevision(const uint16_t revision)
      {
          MMThreadGuard g(lock_);
          fwrevision_ = revision;
          MakeVersionString(firmwareVersion_, fwmajor_, fwminor_, fwrevision_);
      };
      uint16_t GetFirmwareVersionRevision() {  MMThreadGuard g(lock_); return fwrevision_; };

      // Manufacturing date
      uint16_t GetManufactureYear() {  MMThreadGuard g(lock_); return manYear_; };
      void SetManufactureYear(const uint16_t year) {   MMThreadGuard g(lock_); manYear_ = year; };
      uint16_t GetManufactureMonth() {  MMThreadGuard g(lock_); return manMonth_; };
      void SetManufactureMonth(const uint16_t month) {  MMThreadGuard g(lock_); manMonth_ = month; };
      uint16_t GetManufactureDay() {  MMThreadGuard g(lock_); return manDay_; };
      void SetManufactureDay(const uint16_t day) {  MMThreadGuard g(lock_); manDay_ = day; };

      // Serial number
      std::string GetSerialNumber() {  MMThreadGuard g(lock_); return serialNumber_; };
      void SetSerialNumber(const std::string serialNumber) {  MMThreadGuard g(lock_); serialNumber_ = serialNumber; };

      // TIRF Focal Length
      uint16_t GetTIRFFocalLength() { MMThreadGuard g(lock_); return tirfFocalLength_;};
      void SetTIRFFocalLength(const uint16_t val) { MMThreadGuard g(lock_); tirfFocalLength_ = val;};

      // resolution Rotation
      double GetResolutionRotation() { MMThreadGuard g(lock_); return resolutionRotation_;};
      void SetResolutionRotation(double val) { MMThreadGuard g(lock_); resolutionRotation_ = val; };

      // min positionlinear
      int32_t GetMinLinear() { MMThreadGuard g(lock_); return minLinear_;};
      void SetMinLinear(int32_t val) { MMThreadGuard g(lock_); minLinear_ = val; };

      // max positionlinear
      int32_t GetMaxLinear() { MMThreadGuard g(lock_); return maxLinear_;};
      void SetMaxLinear(int32_t val) { MMThreadGuard g(lock_); maxLinear_ = val; };

      // offset linear
      int32_t GetOffsetLinear() { MMThreadGuard g(lock_); return offsetLinear_;};
      void SetOffsetLinear(int32_t val) { MMThreadGuard g(lock_); offsetLinear_ = val; };

      // min Rotation
      int32_t GetMinRotation() { MMThreadGuard g(lock_); return minRotation_;};
      void SetMinRotation(int32_t val) { MMThreadGuard g(lock_); minRotation_ = val; };

      // max Rotation
      int32_t GetMaxRotation() { MMThreadGuard g(lock_); return maxRotation_;};
      void SetMaxRotation(int32_t val) { MMThreadGuard g(lock_); maxRotation_ = val; };

      // offset Rotation
      int32_t GetOffsetRotation() { MMThreadGuard g(lock_); return offsetRotation_;};
      void SetOffsetRotation(int32_t val) { MMThreadGuard g(lock_); offsetRotation_ = val; };

      // Line Wavelength
      uint16_t GetLineWavelength(uint16_t line) { MMThreadGuard g(lock_); return lineWavelength_[line]; };
      void SetLineWavelength(uint16_t line, uint16_t val) { MMThreadGuard g(lock_); lineWavelength_[line] = val; };

      // Line enabled
      bool GetLineEnabled(uint16_t line) { MMThreadGuard g(lock_); return lineEnabled_[line]; };
      void SetLineEnabled(uint16_t line, uint16_t val) { MMThreadGuard g(lock_); lineEnabled_[line] = (val != 0); };

      // Busy
      // deviceBusy_ keeps track of the busy state as signalled by the
      // controller.  
      // The WaitForDeviceBusy function is used by the MessageSender
      // which only sends comamnds to the controller when it is not busy
      bool GetBusy() 
      { 
         MMThreadGuard g(lock_); 
         return deviceBusy_ || logicalBusy_; 
      };
      void SetDeviceBusy(const bool deviceBusy) 
      {  
         MMThreadGuard g(lock_); 
         deviceBusy_ = deviceBusy; 
         // notify waiting threads and also clear the logical busy
         if (!deviceBusy_)
         {
            logicalBusy_ = false;
            varCondition_.notify_one();
         }
      };
      void WaitForDeviceBusy() 
      {
         boost::mutex::scoped_lock bLock(mutex_);
         while (deviceBusy_) {
            varCondition_.wait(bLock);
         }
      };
      void SetLogicalBusy(const bool busy) 
      {  
         MMThreadGuard g(lock_); 
         logicalBusy_ = busy; 
      };

      // Preset SD
      void SetPresetSD(const uint16_t p);
      uint16_t GetPresetSD() {  MMThreadGuard g(lock_); return presetSD_; };

      // Preset WF
      void SetPresetWF(const uint16_t p); 
      uint16_t GetPresetWF() {  MMThreadGuard g(lock_); return presetWF_; };

      // Preset Iris
      void SetPresetIris(const uint16_t p);
      uint16_t GetPresetIris() {  MMThreadGuard g(lock_); return presetIris_; };

      // Preset TIRF 
      void SetPresetTIRF(const uint16_t p);
      uint16_t GetPresetTIRF() {  MMThreadGuard g(lock_); return presetPX_; };

      // Preset Filter W
      void SetPresetFilterW(const uint16_t p); 
      uint16_t GetPresetFilterW() {  MMThreadGuard g(lock_); return presetFilterW_; };

      // Preset Filter T
      void SetPresetFilterT(const uint16_t p);
      uint16_t GetPresetFilterT() {  MMThreadGuard g(lock_); return presetFilterT_; };

      // TIRF positioner Rot
      void SetPositionRot(const int32_t p);
      int32_t GetPositionRot() {  MMThreadGuard g(lock_); return tirfRotPos_; };

      // TIRF positioner Lin
      void SetPositionLin(const int32_t p);
      int32_t GetPositionLin() {  MMThreadGuard g(lock_); return tirfLinPos_; };

      // Motor Running
      void SetMotorRunningSD(const bool p); 
      bool GetMotorRunningSD() {  MMThreadGuard g(lock_); return motorRunningSD_; };


      // values for TIRF calculations, mainly set by the application
      void SetTIRFWavelength1(long val) { MMThreadGuard g(lock_); wavelength1_ = val;};
      long GetWavelength1() { MMThreadGuard g(lock_); return wavelength1_; };
      void SetTIRFWavelength2(long val) { MMThreadGuard g(lock_); wavelength2_ = val;};
      long GetWavelength2() { MMThreadGuard g(lock_); return wavelength2_; };
      void SetTubeLensFocalLength(double val) { MMThreadGuard g(lock_); tubeLensFocalLength_ = (uint16_t) val; };
      uint16_t  GetTubeLensFocalLength() { MMThreadGuard g(lock_); return tubeLensFocalLength_; };
      uint16_t GetOM(); // { MMThreadGuard g(lock_); return om_; };
      void SetDepth(double val) { MMThreadGuard g(lock_); depth_ = val; };
      double GetDepth() { MMThreadGuard g(lock_); return depth_; };
      void SetNA(double val) { MMThreadGuard g(lock_); na_ = val; };
      double GetNA() { MMThreadGuard g(lock_); return na_; };
      void SetRI(double val) { MMThreadGuard g(lock_); ri_ = val; };
      double GetRI() { MMThreadGuard g(lock_); return ri_; };
      void SetExitTIRF(bool doIt) { MMThreadGuard g(lock_); exitTIRF_ = doIt; };
      bool GetExitTIRF() { MMThreadGuard g(lock_); return exitTIRF_; };

      // Whether or not the controller has the specified capabilities
      void SetHasWFX(bool h) {  MMThreadGuard g(lock_); hasWFX_ = h; };
      bool GetHasWFX() {  MMThreadGuard g(lock_); return hasWFX_; };

      void SetHasWFY(bool h) {  MMThreadGuard g(lock_); hasWFY_ = h; };
      bool GetHasWFY() {  MMThreadGuard g(lock_); return hasWFY_; };
      
      void SetHasSD(bool h) {  MMThreadGuard g(lock_); hasSD_ = h; };
      bool GetHasSD() {  MMThreadGuard g(lock_); return hasSD_; };

      void SetHasROT(bool h) {  MMThreadGuard g(lock_); hasROT_ = h; };
      bool GetHasROT() {  MMThreadGuard g(lock_); return hasROT_; };

      void SetHasLIN(bool h) {  MMThreadGuard g(lock_); hasLIN_ = h; };
      bool GetHasLIN() {  MMThreadGuard g(lock_); return hasLIN_; };

      void SetHasP1(bool h) {  MMThreadGuard g(lock_); hasP1_ = h; };
      bool GetHasP1() {  MMThreadGuard g(lock_); return hasP1_; };

      void SetHasP2(bool h) {  MMThreadGuard g(lock_); hasP2_ = h; };
      bool GetHasP2() {  MMThreadGuard g(lock_); return hasP2_; };

      void SetHasIRIS(bool h) {  MMThreadGuard g(lock_); hasIRIS_ = h; };
      bool GetHasIRIS() {  MMThreadGuard g(lock_); return hasIRIS_; };

      void SetHasFilterW(bool h) {  MMThreadGuard g(lock_); hasFilterW_ = h; };
      bool GetHasFilterW() {  MMThreadGuard g(lock_); return hasFilterW_; };

      void SetHasFilterT(bool h) {  MMThreadGuard g(lock_); hasFilterT_ = h; };
      bool GetHasFilterT() {  MMThreadGuard g(lock_); return hasFilterT_; };

      void SetButtonWFLabel(uint16_t pos, const char* label) {  MMThreadGuard g(lock_); buttonWFLabels_[pos - 1] = label; };
      const char* GetButtonWFLabel (uint16_t pos) {  
         MMThreadGuard g(lock_); 
         if (pos > 0 && pos < 5)
            return buttonWFLabels_[pos - 1].c_str(); 
         return 0;
      };

      void SetButtonIrisLabel(uint16_t pos, const char* label) {  
         MMThreadGuard g(lock_); buttonIrisLabels_[pos - 1] = label; };
      const char* GetButtonIrisLabel (uint16_t pos) {  
         MMThreadGuard g(lock_); 
         if (pos > 0 && pos < 5)
            return buttonIrisLabels_[pos - 1].c_str(); 
         return 0;
      };

      void SetButtonFilterWLabel(uint16_t pos, const char* label) {  
         MMThreadGuard g(lock_); buttonFilterWLabels_[pos - 1] = label; };
      const char* GetButtonFilterWLabel (uint16_t pos) {  
         MMThreadGuard g(lock_); 
         if (pos > 0 && pos < 5)
            return buttonFilterWLabels_[pos - 1].c_str(); 
         return 0;
      };

      void SetButtonFilterTLabel(uint16_t pos, const char* label) {  
         MMThreadGuard g(lock_); buttonFilterTLabels_[pos - 1] = label; };
      const char* GetButtonFilterTLabel (uint16_t pos) {  
         MMThreadGuard g(lock_); 
         if (pos > 0 && pos < 5)
            return buttonFilterTLabels_[pos - 1].c_str(); 
         return 0;
      };

      void SetDiskLabel(uint16_t pos, const char* label) {  
         MMThreadGuard g(lock_); diskLabels_[pos - 1] = label; };
      const char* GetDiskLabel (uint16_t pos) {  
         MMThreadGuard g(lock_); 
         if (pos > 0 && pos < 4)
            return diskLabels_[pos - 1].c_str(); 
         return 0;
      };
      const char* hardwareVersionProp_;
      const char* firmwareVersionProp_;
      const char* manufacturingDateProp_;
      const char* serialNumberProp_;
      const char* spinningDiskPositionProp_;
      const char* wideFieldPositionProp_;
      const char* motorRunningProp_;
      const char* filterPositionProp_;
      const char* irisPositionProp_;
      const char* tirfPositionProp_;

    private:
      void MakeVersionString(std::string& it, uint16_t maj, 
            uint16_t min, uint16_t rev) 
      {
         // note: this function should always be called after setting a lock!
         // do not set the lock here, since we can not be sure if double
         // locking is OK or not
         std::ostringstream oss;
         oss << maj << "." << min << "." << rev;
         it  = oss.str();
      };

      MMThreadLock lock_;
      boost::condition_variable varCondition_;
      boost::mutex mutex_;

      std::string hardwareVersion_;
      uint16_t hwmajor_, hwminor_, hwrevision_;
      std::string firmwareVersion_;
      uint16_t fwmajor_, fwminor_, fwrevision_;
      uint16_t manYear_, manMonth_, manDay_;
      int32_t tirfLinPos_, tirfRotPos_;
      std::string serialNumber_;
      bool deviceBusy_, logicalBusy_;
      uint16_t presetSD_, presetWF_, presetIris_, presetPX_, presetFilterT_, presetFilterW_;
      bool hasWFX_, hasWFY_, hasSD_, hasROT_, hasLIN_, hasP1_, hasP2_, 
         hasIRIS_, hasFilterW_, hasFilterT_;
      bool motorRunningSD_;
      uint16_t tirfFocalLength_;
      double resolutionRotation_;
      int32_t minLinear_, maxLinear_, minRotation_, maxRotation_;
      int32_t offsetLinear_, offsetRotation_;
      uint16_t lineWavelength_[7];
      bool lineEnabled_[7];
      long wavelength1_, wavelength2_;
      uint16_t tubeLensFocalLength_, om_;
      double depth_, na_, ri_;
      bool exitTIRF_;

      std::string buttonWFLabels_[4];
      std::string buttonIrisLabels_[4];
      std::string buttonFilterWLabels_[4];
      std::string buttonFilterTLabels_[4];
      // even though we use only 3, it is easier to declare 4
      std::string diskLabels_[4];

      MM::Device* hubDevice_;
      DiskoveryStateDev* sdDevice_;
      DiskoveryStateDev* wfDevice_;
      DiskoveryStateDev* tirfDevice_;
      DiskoveryStateDev* irisDevice_;
      DiskoveryStateDev* filterWDevice_;
      DiskoveryStateDev* filterTDevice_;
      MM::Core& core_;
};

#endif
