///////////////////////////////////////////////////////////////////////////////
// PROJECT:       Micro-Manager
// SUBSYSTEM:     MMDevice
//-----------------------------------------------------------------------------
// COPYRIGHT:     2026, Board of Regents of the University of Wisconsin System
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

#pragma once

#include <cassert>
#include <map>
#include <sstream>
#include <string>

namespace MM {

class CameraImageMetadata {
public:
   // The memory warning in the AddTag() docs below doesn't apply to the
   // current std::map implementation but will apply to an intended future
   // implementation that serializes on the fly.

   /**
    * @brief Add a tag.
    * 
    * The key must not contain newlines.
    * 
    * At present, standard and custom keys are not formally distinguished. In
    * custom keys, it is recommended to include a prefix identifying the camera
    * adapter. For example, "AcmeCam-SensorTemperature" where AcmeCam is the
    * name of the camera adapter.
    * 
    * The value should be a string, integer, or floating point number. If it is
    * a string, it must not contain newlines. Very long strings are
    * discouraged.
    * 
    * If a tag with the same key is added more than once, the last value wins.
    * However, doing so may consume memory. Call Clear() to reuse an instance;
    * if some of the tags are constant, keep an instance with just the constant
    * tags and copy-assign it to the reused instance.
    * 
    * @param key the key (must not be null)
    * @param value the value
    */
   template <typename V>
   void AddTag(const char* key, V value) {
      assert(key != nullptr);
      std::ostringstream strm;
      strm << value;
      tags_[key] = strm.str();
   }

   /** @brief Optimized overload for string values. */
   void AddTag(const char* key, const char* value) {
      assert(key != nullptr);
      assert(value != nullptr);
      tags_[key] = value;
   }

   /** @brief Overload for std::string key. */
   template <typename V>
   void AddTag(const std::string& key, V value) {
      // Once we drop C++14 support, we could replace the char* and std::string&
      // overloads with std::string_view.
      AddTag(key.c_str(), value);
   }

   /**
    * @brief Remove all tags.
    */
   void Clear() { tags_.clear(); }

   /**
    * @brief Return this metadata map serialized to string form.
    * 
    * This is the form used to transmit data from a camera device to MMCore via
    * the InsertImage() function. (The exact format may change with the Device
    * Interface Version.)
    *
    * The returned string is valid until this metadata map is mutated or
    * destroyed.
    *
    * @return the serialized metadata map
    */
   const char* Serialize() const {
      // The serialization format used here is part of the versioned Device
      // Interface.
      //
      // Header: <tag_count>\n
      // (Spaces are tolerated before and after the tag count.)
      // For each tag: s\n<name>\n_\n1\n<value>\n
      // (The 's' indicates "single (not array) tag"; arrays are never used.)
      // (The '_' device-label field is always "_".)
      // (The '1' indicates "read-only"; no actual meaning.)
      //
      // Tags must have unique keys as of DIV 74 (otherwise memory may leak on
      // deserialization by Metadata::Restore())

      serialized_.clear();
      serialized_ = std::to_string(tags_.size());
      serialized_ += '\n';
      for (const auto& tag : tags_) {
         serialized_ += "s\n";
         serialized_ += tag.first;
         serialized_ += "\n_\n1\n";
         serialized_ += tag.second;
         serialized_ += '\n';
      }
      return serialized_.c_str();
   }

private:
   std::map<std::string, std::string> tags_;

   mutable std::string serialized_; // Valid after Serialize()
};

} // namespace MM
