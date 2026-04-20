// Mock device adapter for testing of device sequencing
//
// Copyright (C) 2014 University of California, San Francisco.
//               2023 Board of Regents of the University of Wisconsin System
//
// This library is free software; you can redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation.
//
// This library is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License
// for more details.
//
// IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES.
//
// You should have received a copy of the GNU Lesser General Public License
// along with this library; if not, write to the Free Software Foundation,
// Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
//
//
// Author: Mark Tsuchida

#include "SettingLogger.h"

#include <msgpack.hpp>

#include <memory>
#include <string>
#include <vector>


// This file effectively defines the MessagePack wire format for our test
// images. Unfortunately, there is no automatic mechanism to keep the Java (and
// possibly other) decoder in sync, so be careful! Field order is crucial.


void
BoolSettingValue::Write(msgpack::sbuffer& sbuf) const
{
   msgpack::packer<msgpack::sbuffer> pk(&sbuf);
   pk.pack_array(2);
   // type
   pk.pack(std::string("bool"));
   // value
   pk.pack(value_);
}


void
IntegerSettingValue::Write(msgpack::sbuffer& sbuf) const
{
   msgpack::packer<msgpack::sbuffer> pk(&sbuf);
   pk.pack_array(2);
   // type
   pk.pack(std::string("int"));
   // value
   pk.pack(value_);
}


void
FloatSettingValue::Write(msgpack::sbuffer& sbuf) const
{
   msgpack::packer<msgpack::sbuffer> pk(&sbuf);
   pk.pack_array(2);
   // type
   pk.pack(std::string("float"));
   // value
   pk.pack(value_);
}


void
StringSettingValue::Write(msgpack::sbuffer& sbuf) const
{
   msgpack::packer<msgpack::sbuffer> pk(&sbuf);
   pk.pack_array(2);
   // type
   pk.pack(std::string("string"));
   // value
   pk.pack(value_);
}


void
OneShotSettingValue::Write(msgpack::sbuffer& sbuf) const
{
   msgpack::packer<msgpack::sbuffer> pk(&sbuf);
   pk.pack_array(2);
   // type
   pk.pack(std::string("one_shot"));
   // value
   pk.pack_nil();
}


void
SettingKey::Write(msgpack::sbuffer& sbuf) const
{
   msgpack::packer<msgpack::sbuffer> pk(&sbuf);
   pk.pack_array(2);
   // device
   pk.pack(device_);
   // key
   pk.pack(key_);
}


void
SettingEvent::Write(msgpack::sbuffer& sbuf) const
{
   msgpack::packer<msgpack::sbuffer> pk(&sbuf);
   pk.pack_array(3);
   // key
   key_.Write(sbuf);
   // value
   value_->Write(sbuf);
   // count
   pk.pack(count_);
}


std::string
SettingEvent::AsText() const
{
   return "[" + std::to_string(count_) + "]" +
         key_.GetStringRep() + "=" + value_->GetString();
}


void
CameraInfo::Write(msgpack::sbuffer& sbuf) const
{
   msgpack::packer<msgpack::sbuffer> pk(&sbuf);
   pk.pack_array(5);
   // name
   pk.pack(camera_);
   // serialImageNr
   pk.pack(serialNr_);
   // isSequence
   pk.pack(isSequence_);
   // cumulativeImageNr
   pk.pack(cumulativeNr_);
   // frameNr
   pk.pack(frameNr_);
}


std::string
CameraInfo::AsText() const
{
   std::string ret;
   ret.reserve(256);

   ret += "camera,name=";
   ret += camera_;
   ret += ' ';

   ret += "camera,serialImageNr=";
   ret += std::to_string(serialNr_);
   ret += ' ';

   ret += "camera,isSequence=";
   ret += isSequence_ ? "true" : "false";
   ret += ' ';

   ret += isSequence_ ? "camera,sequenceImageNr=" : "camera,snapImageNr=";
   ret += std::to_string(cumulativeNr_);

   if (isSequence_)
   {
      ret += ' ';
      ret += "camera,frameNr=";
      ret += std::to_string(frameNr_);
   }

   return ret;
}


void
SettingLogger::SetBool(const std::string& device, const std::string& key,
      bool value, bool logEvent)
{
   SettingKey keyRecord = SettingKey(device, key);
   std::shared_ptr<SettingValue> valueRecord =
      std::make_shared<BoolSettingValue>(value);
   settingValues_[keyRecord] = valueRecord;

   if (logEvent)
   {
      SettingEvent event =
         SettingEvent(keyRecord, valueRecord, GetNextCount());
      settingEvents_.push_back(event);
   }
}


bool
SettingLogger::GetBool(const std::string& device,
      const std::string& key) const
{
   SettingKey keyRecord = SettingKey(device, key);
   SettingConstIterator found = settingValues_.find(keyRecord);
   if (found == settingValues_.end())
      return false;
   return found->second->GetBool();
}


void
SettingLogger::SetInteger(const std::string& device, const std::string& key,
      long value, bool logEvent)
{
   SettingKey keyRecord = SettingKey(device, key);
   std::shared_ptr<SettingValue> valueRecord =
      std::make_shared<IntegerSettingValue>(value);
   settingValues_[keyRecord] = valueRecord;

   if (logEvent)
   {
      SettingEvent event =
         SettingEvent(keyRecord, valueRecord, GetNextCount());
      settingEvents_.push_back(event);
   }
}


long
SettingLogger::GetInteger(const std::string& device,
      const std::string& key) const
{
   SettingKey keyRecord = SettingKey(device, key);
   SettingConstIterator found = settingValues_.find(keyRecord);
   if (found == settingValues_.end())
      return 0;
   return found->second->GetInteger();
}


void
SettingLogger::SetFloat(const std::string& device, const std::string& key,
      double value, bool logEvent)
{
   SettingKey keyRecord = SettingKey(device, key);
   std::shared_ptr<SettingValue> valueRecord =
      std::make_shared<FloatSettingValue>(value);
   settingValues_[keyRecord] = valueRecord;

   if (logEvent)
   {
      SettingEvent event =
         SettingEvent(keyRecord, valueRecord, GetNextCount());
      settingEvents_.push_back(event);
   }
}


double
SettingLogger::GetFloat(const std::string& device,
      const std::string& key) const
{
   SettingKey keyRecord = SettingKey(device, key);
   SettingConstIterator found = settingValues_.find(keyRecord);
   if (found == settingValues_.end())
      return 0.0;
   return found->second->GetFloat();
}


void
SettingLogger::SetString(const std::string& device, const std::string& key,
      const std::string& value, bool logEvent)
{
   SettingKey keyRecord = SettingKey(device, key);
   std::shared_ptr<SettingValue> valueRecord =
      std::make_shared<StringSettingValue>(value);
   settingValues_[keyRecord] = valueRecord;

   if (logEvent)
   {
      SettingEvent event =
         SettingEvent(keyRecord, valueRecord, GetNextCount());
      settingEvents_.push_back(event);
   }
}


std::string
SettingLogger::GetString(const std::string& device,
      const std::string& key) const
{
   SettingKey keyRecord = SettingKey(device, key);
   SettingConstIterator found = settingValues_.find(keyRecord);
   if (found == settingValues_.end())
      return std::string();
   return found->second->GetString();
}


void
SettingLogger::FireOneShot(const std::string& device, const std::string& key,
      bool logEvent)
{
   SettingKey keyRecord = SettingKey(device, key);
   std::shared_ptr<SettingValue> valueRecord =
      std::make_shared<OneShotSettingValue>();
   settingValues_[keyRecord] = valueRecord;

   if (logEvent)
   {
      SettingEvent event =
         SettingEvent(keyRecord, valueRecord, GetNextCount());
      settingEvents_.push_back(event);
   }
}


bool
SettingLogger::DumpMsgPackToBuffer(char* dest, size_t destSize,
      const std::string& camera, bool isSequenceImage,
      size_t serialImageNr, size_t cumulativeImageNr, size_t frameNr)
{
   CameraInfo cameraInfo(camera, isSequenceImage,
         serialImageNr, cumulativeImageNr, frameNr);

   msgpack::sbuffer sbuf;
   msgpack::packer<msgpack::sbuffer> pk(&sbuf);

   pk.pack_array(7);
   // packetNumber
   pk.pack(GetNextGlobalImageNr());
   // camera
   cameraInfo.Write(sbuf);
   // startCounter
   pk.pack(counterAtLastReset_);
   // currentCounter
   pk.pack(counter_);
   // startState
   WriteSettingMap(sbuf, startingValues_);
   // currentState
   WriteSettingMap(sbuf, settingValues_);
   // history
   WriteHistory(sbuf);

   if (sbuf.size() <= destSize)
   {
      memcpy(dest, sbuf.data(), sbuf.size());
      memset(dest + sbuf.size(), 0, destSize - sbuf.size());
      return true;
   }
   else
   {
      memset(dest, 0, destSize);
      return false;
   }
}


void
SettingLogger::DrawTextToBuffer(char* dest, size_t destWidth,
      size_t destHeight, const std::string& camera, bool isSequenceImage,
      size_t serialImageNr, size_t cumulativeImageNr, size_t frameNr)
{
   std::string text;

   text += "HubGlobalPacketNr=";
   text += std::to_string(GetNextGlobalImageNr());
   text += '\n';

   CameraInfo cameraInfo(camera, isSequenceImage,
         serialImageNr, cumulativeImageNr, frameNr);
   text += cameraInfo.AsText();
   text += "\n\n";

   text += "State\n";
   text += SettingMapAsText(settingValues_);
   text += "\n\n";

   text += "History\n";
   text += HistoryAsText();

   DrawTextImage(text, reinterpret_cast<uint8_t*>(dest), destWidth, destHeight);
}


void
SettingLogger::WriteSettingMap(msgpack::sbuffer& sbuf,
      const SettingMap& values) const
{
   msgpack::packer<msgpack::sbuffer> pk(&sbuf);

   pk.pack_array(static_cast<uint32_t>(values.size()));
   for (SettingConstIterator it = values.begin(), end = values.end();
         it != end; ++it)
   {
      pk.pack_array(2);
      // key
      it->first.Write(sbuf);
      // value
      it->second->Write(sbuf);
   }
}


std::string
SettingLogger::SettingMapAsText(const SettingMap& values) const
{
   std::string ret;
   ret.reserve(20 * values.size());
   bool first = true;
   for (SettingConstIterator it = values.begin(), end = values.end();
         it != end; ++it)
   {
      if (std::dynamic_pointer_cast<OneShotSettingValue>(it->second))
         continue; // Skip one-shot settings

      if (first)
         first = false;
      else
         ret += ' ';
      ret += it->first.GetStringRep();
      ret += '=';
      ret += it->second->GetString();
   }
   return ret;
}


void
SettingLogger::WriteHistory(msgpack::sbuffer& sbuf) const
{
   msgpack::packer<msgpack::sbuffer> pk(&sbuf);

   pk.pack_array(static_cast<uint32_t>(settingEvents_.size()));
   for (std::vector<SettingEvent>::const_iterator it = settingEvents_.begin(),
         end = settingEvents_.end(); it != end; ++it)
   {
      it->Write(sbuf);
   }
}


std::string
SettingLogger::HistoryAsText() const
{
   std::string ret;
   ret.reserve(20 * settingEvents_.size());
   for (const auto& evt : settingEvents_)
   {
      if (!ret.empty())
         ret += ' ';
      ret += evt.AsText();
   }
   return ret;
}
