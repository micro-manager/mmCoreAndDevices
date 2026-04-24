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

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <limits>
#include <sstream>
#include <string>

namespace MM {

class CameraImageMetadata {
public:
   CameraImageMetadata() { Clear(); }

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
    * If a tag with the same key is added more than once, the last value wins
    * on the MMCore side after deserialization. However, every added tag
    * consumes memory in the serialized buffer. Call Clear() to reuse an
    * instance; if some of the tags are constant, keep an instance with just
    * the constant tags and copy-assign it to the reused instance.
    *
    * @param key the key (must not be null)
    * @param value the value
    */
   template <typename V>
   void AddTag(const char* key, V value) {
      assert(key != nullptr);
      std::ostringstream strm;
      strm << value;
      AppendTag(key, strm.str().c_str());
   }

   /** @brief Optimized overload for string values. */
   void AddTag(const char* key, const char* value) {
      assert(key != nullptr);
      assert(value != nullptr);
      AppendTag(key, value);
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
   void Clear() {
      buffer_.assign(kCountWidth, ' ');
      buffer_.push_back('\n');
      count_ = 0;
   }

   /**
    * @brief Return this metadata map serialized to string form.
    *
    * This is the form used to transmit data from a camera device to MMCore via
    * the InsertImage() function.
    *
    * The returned string is valid until this metadata map is mutated or
    * destroyed.
    * 
    * Device adapters must not depend on the serialization format, which may
    * change in the future.
    *
    * @return the serialized metadata map
    */
   const char* Serialize() const {
      // The serialization format is part of the versioned Device Interface:
      //
      // Header: <tag_count>\n
      // (Spaces are tolerated before and after the tag count; this
      // implementation right-pads the count to a fixed width.)
      // For each tag: s\n<name>\n_\n1\n<value>\n
      // (The 's' indicates "single (not array) tag"; arrays are never used.)
      // (The '_' device-label field is always "_".)
      // (The '1' indicates "read-only"; no actual meaning.)
      //
      // Tags must have unique keys up to DIV 74. In DIV 75+, duplicate keys are
      // permitted on the wire; the last occurrence wins on deserialization.

      WriteCountHeader();
      return buffer_.c_str();
   }

private:
   // Enough to hold any std::size_t in decimal (20 on 64-bit platforms).
   static constexpr std::size_t kCountWidth =
      std::numeric_limits<std::size_t>::digits10 + 1;

   void AppendTag(const char* key, const char* value) {
      buffer_ += "s\n";
      buffer_ += key;
      buffer_ += "\n_\n1\n";
      buffer_ += value;
      buffer_ += '\n';
      ++count_;
   }

   void WriteCountHeader() const {
      const std::string s = std::to_string(count_);
      assert(s.size() <= kCountWidth);
      auto out = std::copy(s.begin(), s.end(), buffer_.begin());
      std::fill(out, buffer_.begin() + kCountWidth, ' ');
   }

   mutable std::string buffer_;
   std::size_t count_ = 0;
};

} // namespace MM
