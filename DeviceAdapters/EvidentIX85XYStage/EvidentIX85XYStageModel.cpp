///////////////////////////////////////////////////////////////////////////////
// FILE:          EvidentIX85XYStageModel.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   State model implementation for Evident IX85 XY Stage
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

#include "EvidentIX85XYStageModel.h"
#include "EvidentIX85XYStageProtocol.h"
#include <cmath>

using namespace IX85XYStage;

StageModel::StageModel() :
   posX_(0),
   posY_(0),
   targetX_(-1),
   targetY_(-1),
   busy_(false),
   version_(""),
   minX_(XY_STAGE_MIN_POS_X),
   maxX_(XY_STAGE_MAX_POS_X),
   minY_(XY_STAGE_MIN_POS_Y),
   maxY_(XY_STAGE_MAX_POS_Y),
   speedInitial_(0),
   speedHigh_(256000),
   speedAccel_(2560000),
   jogEnabled_(false),
   jogSensitivity_(8),
   jogReverseX_(false),
   jogReverseY_(false),
   encoderPos1_(0),
   encoderPos2_(0)
{
}

StageModel::~StageModel()
{
}

///////////////////////////////////////////////////////////////////////////////
// Position management
///////////////////////////////////////////////////////////////////////////////

void StageModel::SetPositionX(long pos)
{
   std::lock_guard<std::mutex> lock(mutex_);
   posX_ = pos;
}

void StageModel::SetPositionY(long pos)
{
   std::lock_guard<std::mutex> lock(mutex_);
   posY_ = pos;
}

void StageModel::SetPosition(long x, long y)
{
   std::lock_guard<std::mutex> lock(mutex_);
   posX_ = x;
   posY_ = y;
}

long StageModel::GetPositionX() const
{
   std::lock_guard<std::mutex> lock(mutex_);
   return posX_;
}

long StageModel::GetPositionY() const
{
   std::lock_guard<std::mutex> lock(mutex_);
   return posY_;
}

void StageModel::GetPosition(long& x, long& y) const
{
   std::lock_guard<std::mutex> lock(mutex_);
   x = posX_;
   y = posY_;
}

///////////////////////////////////////////////////////////////////////////////
// Target position management
///////////////////////////////////////////////////////////////////////////////

void StageModel::SetTargetX(long pos)
{
   std::lock_guard<std::mutex> lock(mutex_);
   targetX_ = pos;
}

void StageModel::SetTargetY(long pos)
{
   std::lock_guard<std::mutex> lock(mutex_);
   targetY_ = pos;
}

void StageModel::SetTarget(long x, long y)
{
   std::lock_guard<std::mutex> lock(mutex_);
   targetX_ = x;
   targetY_ = y;
}

long StageModel::GetTargetX() const
{
   std::lock_guard<std::mutex> lock(mutex_);
   return targetX_;
}

long StageModel::GetTargetY() const
{
   std::lock_guard<std::mutex> lock(mutex_);
   return targetY_;
}

void StageModel::GetTarget(long& x, long& y) const
{
   std::lock_guard<std::mutex> lock(mutex_);
   x = targetX_;
   y = targetY_;
}

///////////////////////////////////////////////////////////////////////////////
// Busy state
///////////////////////////////////////////////////////////////////////////////

void StageModel::SetBusy(bool busy)
{
   std::lock_guard<std::mutex> lock(mutex_);
   busy_ = busy;
}

bool StageModel::IsBusy() const
{
   std::lock_guard<std::mutex> lock(mutex_);
   return busy_;
}

bool StageModel::IsAtTarget(long tolerance) const
{
   std::lock_guard<std::mutex> lock(mutex_);

   // Calculate absolute differences
   long diffX = posX_ - targetX_;
   if (diffX < 0)
      diffX = -diffX;

   long diffY = posY_ - targetY_;
   if (diffY < 0)
      diffY = -diffY;

   return (diffX <= tolerance) && (diffY <= tolerance);
}

///////////////////////////////////////////////////////////////////////////////
// Version and device info
///////////////////////////////////////////////////////////////////////////////

void StageModel::SetVersion(const std::string& version)
{
   std::lock_guard<std::mutex> lock(mutex_);
   version_ = version;
}

std::string StageModel::GetVersion() const
{
   std::lock_guard<std::mutex> lock(mutex_);
   return version_;
}

///////////////////////////////////////////////////////////////////////////////
// Limits
///////////////////////////////////////////////////////////////////////////////

void StageModel::SetLimitsX(long min, long max)
{
   std::lock_guard<std::mutex> lock(mutex_);
   minX_ = min;
   maxX_ = max;
}

void StageModel::SetLimitsY(long min, long max)
{
   std::lock_guard<std::mutex> lock(mutex_);
   minY_ = min;
   maxY_ = max;
}

void StageModel::GetLimitsX(long& min, long& max) const
{
   std::lock_guard<std::mutex> lock(mutex_);
   min = minX_;
   max = maxX_;
}

void StageModel::GetLimitsY(long& min, long& max) const
{
   std::lock_guard<std::mutex> lock(mutex_);
   min = minY_;
   max = maxY_;
}

///////////////////////////////////////////////////////////////////////////////
// Speed settings
///////////////////////////////////////////////////////////////////////////////

void StageModel::SetSpeed(long initial, long high, long accel)
{
   std::lock_guard<std::mutex> lock(mutex_);
   speedInitial_ = initial;
   speedHigh_ = high;
   speedAccel_ = accel;
}

void StageModel::GetSpeed(long& initial, long& high, long& accel) const
{
   std::lock_guard<std::mutex> lock(mutex_);
   initial = speedInitial_;
   high = speedHigh_;
   accel = speedAccel_;
}

///////////////////////////////////////////////////////////////////////////////
// Jog settings
///////////////////////////////////////////////////////////////////////////////

void StageModel::SetJogEnabled(bool enabled)
{
   std::lock_guard<std::mutex> lock(mutex_);
   jogEnabled_ = enabled;
}

bool StageModel::IsJogEnabled() const
{
   std::lock_guard<std::mutex> lock(mutex_);
   return jogEnabled_;
}

void StageModel::SetJogSensitivity(int sensitivity)
{
   std::lock_guard<std::mutex> lock(mutex_);
   jogSensitivity_ = sensitivity;
}

int StageModel::GetJogSensitivity() const
{
   std::lock_guard<std::mutex> lock(mutex_);
   return jogSensitivity_;
}

void StageModel::SetJogDirectionX(bool reverse)
{
   std::lock_guard<std::mutex> lock(mutex_);
   jogReverseX_ = reverse;
}

void StageModel::SetJogDirectionY(bool reverse)
{
   std::lock_guard<std::mutex> lock(mutex_);
   jogReverseY_ = reverse;
}

bool StageModel::GetJogDirectionX() const
{
   std::lock_guard<std::mutex> lock(mutex_);
   return jogReverseX_;
}

bool StageModel::GetJogDirectionY() const
{
   std::lock_guard<std::mutex> lock(mutex_);
   return jogReverseY_;
}

///////////////////////////////////////////////////////////////////////////////
// Encoder settings
///////////////////////////////////////////////////////////////////////////////

void StageModel::SetEncoderPosition(int encoder, long position)
{
   std::lock_guard<std::mutex> lock(mutex_);
   if (encoder == 1)
      encoderPos1_ = position;
   else if (encoder == 2)
      encoderPos2_ = position;
}

long StageModel::GetEncoderPosition(int encoder) const
{
   std::lock_guard<std::mutex> lock(mutex_);
   if (encoder == 1)
      return encoderPos1_;
   else if (encoder == 2)
      return encoderPos2_;
   return 0;
}
