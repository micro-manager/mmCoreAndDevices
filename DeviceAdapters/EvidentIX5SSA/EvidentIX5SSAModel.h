///////////////////////////////////////////////////////////////////////////////
// FILE:          EvidentIX5SSAModel.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   State model for Evident IX5-SSA XY Stage
//
// COPYRIGHT:     University of California, San Francisco, 2025
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
// AUTHOR:        Nico Stuurman, 2025

#pragma once

#include <string>
#include <mutex>

namespace IX5SSA {

///////////////////////////////////////////////////////////////////////////////
// StageModel - Thread-safe state management for IX5-SSA XY Stage
///////////////////////////////////////////////////////////////////////////////

class StageModel
{
public:
   StageModel();
   ~StageModel();

   // Position management
   void SetPositionX(long pos);
   void SetPositionY(long pos);
   void SetPosition(long x, long y);
   long GetPositionX() const;
   long GetPositionY() const;
   void GetPosition(long& x, long& y) const;

   // Target position management
   void SetTargetX(long pos);
   void SetTargetY(long pos);
   void SetTarget(long x, long y);
   long GetTargetX() const;
   long GetTargetY() const;
   void GetTarget(long& x, long& y) const;

   // Busy state
   void SetBusy(bool busy);
   bool IsBusy() const;
   bool IsAtTarget(long tolerance) const;

   // Version and device info
   void SetVersion(const std::string& version);
   std::string GetVersion() const;

   // Limits
   void SetLimitsX(long min, long max);
   void SetLimitsY(long min, long max);
   void GetLimitsX(long& min, long& max) const;
   void GetLimitsY(long& min, long& max) const;

   // Speed settings
   void SetSpeed(long initial, long high, long accel);
   void GetSpeed(long& initial, long& high, long& accel) const;

   // Jog settings
   void SetJogEnabled(bool enabled);
   bool IsJogEnabled() const;
   void SetJogSensitivity(int sensitivity);
   int GetJogSensitivity() const;
   void SetJogDirectionX(bool reverse);
   void SetJogDirectionY(bool reverse);
   bool GetJogDirectionX() const;
   bool GetJogDirectionY() const;

   // Encoder settings
   void SetEncoderPosition(int encoder, long position);
   long GetEncoderPosition(int encoder) const;

private:
   mutable std::mutex mutex_;

   // Position state
   long posX_;
   long posY_;
   long targetX_;
   long targetY_;
   bool busy_;

   // Device info
   std::string version_;

   // Limits
   long minX_;
   long maxX_;
   long minY_;
   long maxY_;

   // Speed
   long speedInitial_;
   long speedHigh_;
   long speedAccel_;

   // Jog
   bool jogEnabled_;
   int jogSensitivity_;
   bool jogReverseX_;
   bool jogReverseY_;

   // Encoder positions
   long encoderPos1_;
   long encoderPos2_;
};

} // namespace IX5SSA
