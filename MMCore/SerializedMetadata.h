// COPYRIGHT:  2026, Board of Regents of the University of Wisconsin System
//
// LICENSE:    This file is distributed under the "Lesser GPL" (LGPL) license.
//             License text is included with the source distribution.
//
//             This file is distributed in the hope that it will be useful,
//             but WITHOUT ANY WARRANTY; without even the implied warranty
//             of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
//
//             IN NO EVENT SHALL THE COPYRIGHT OWNER OR
//             CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
//             INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES.

#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>

namespace mmcore {
namespace internal {

// Append-and-scan buffer that holds image metadata in the wire-serialized
// form documented in MMDevice/CameraImageMetadata.h. Used internally by
// MMCore to keep camera-supplied tags and Core-added tags in their
// already-serialized form until the public retrieval API needs to hand a
// Metadata to the caller.
//
// The serialization format MUST stay byte-compatible with
// MM::CameraImageMetadata (which is part of the versioned Device
// Interface) and must be parseable by Metadata::Restore. If you change the
// format here, change it there too.
//
// This class deliberately does not depend on the Metadata class: callers
// hand the buffer to Metadata::Restore via View() at retrieval time.
class SerializedMetadata {
public:
   SerializedMetadata() { Clear(); }

   // Adopt the contents of an existing serialized blob (as produced by
   // MM::CameraImageMetadata::Serialize(), this class's View(), or
   // Metadata::Serialize()). A null or empty pointer is treated as "empty".
   explicit SerializedMetadata(const char* serialized) {
      AdoptSerialized(serialized);
   }

   void Clear() {
      buffer_.assign(kCountWidth, ' ');
      buffer_.push_back('\n');
      count_ = 0;
   }

   template <typename V>
   void AddTag(const char* key, V value) {
      assert(key != nullptr);
      std::ostringstream strm;
      strm << value;
      AppendTag(key, strm.str().c_str());
   }

   void AddTag(const char* key, const char* value) {
      assert(key != nullptr);
      assert(value != nullptr);
      AppendTag(key, value);
   }

   template <typename V>
   void AddTag(const std::string& key, V value) {
      AddTag(key.c_str(), value);
   }

   bool HasTag(const char* key) const {
      return FindTagValue(key).first != std::string_view::npos;
   }

   // Returns the value of the (last) tag matching key, or std::nullopt.
   // The returned view is valid until the next mutation of this object.
   std::optional<std::string_view> GetTag(const char* key) const {
      auto pos = FindTagValue(key);
      if (pos.first == std::string_view::npos)
         return std::nullopt;
      return std::string_view(buffer_.data() + pos.first,
                              pos.second - pos.first);
   }

   // Append the tag records from another serialized blob (as produced by
   // MM::CameraImageMetadata::Serialize, this class, or Metadata::Serialize).
   // The other blob's count header is parsed and added to this object's
   // count; its records are appended verbatim.
   void AppendSerialized(const char* otherSerialized) {
      if (otherSerialized == nullptr)
         return;
      const char* nl = std::strchr(otherSerialized, '\n');
      if (nl == nullptr)
         return;
      const std::size_t otherCount =
         static_cast<std::size_t>(std::atol(otherSerialized));
      buffer_.append(nl + 1);
      count_ += otherCount;
   }

   // Returns a view of the buffer in the serialized wire format. The
   // returned view is valid until this object is mutated or destroyed.
   std::string_view View() const {
      WriteCountHeader();
      return buffer_;
   }

private:
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

   void AdoptSerialized(const char* serialized) {
      if (serialized == nullptr || serialized[0] == '\0') {
         Clear();
         return;
      }
      const char* nl = std::strchr(serialized, '\n');
      if (nl == nullptr) {
         Clear();
         return;
      }
      count_ = static_cast<std::size_t>(std::atol(serialized));
      buffer_.assign(kCountWidth, ' ');
      buffer_.push_back('\n');
      buffer_.append(nl + 1);
   }

   void WriteCountHeader() const {
      const std::string s = std::to_string(count_);
      assert(s.size() <= kCountWidth);
      auto out = std::copy(s.begin(), s.end(), buffer_.begin());
      std::fill(out, buffer_.begin() + kCountWidth, ' ');
   }

   // Locate the value bytes of the tag with the given key. Returns
   // (begin, end) byte offsets within buffer_, or (npos, npos) if not
   // found. If a key occurs multiple times, returns the last occurrence
   // (matches Metadata::Restore's "last wins" behavior).
   std::pair<std::size_t, std::size_t>
   FindTagValue(const char* key) const {
      const std::size_t keyLen = std::strlen(key);
      const std::string_view buf(buffer_);
      const std::size_t headerEnd = kCountWidth + 1;
      std::size_t pos = headerEnd;
      std::pair<std::size_t, std::size_t> last = {std::string_view::npos,
                                                  std::string_view::npos};
      while (pos < buf.size()) {
         // Each record starts with "s\n".
         if (buf.compare(pos, 2, "s\n") != 0)
            break;
         std::size_t nameStart = pos + 2;
         std::size_t nameEnd = buf.find('\n', nameStart);
         if (nameEnd == std::string_view::npos)
            break;
         std::size_t deviceStart = nameEnd + 1;
         std::size_t deviceEnd = buf.find('\n', deviceStart);
         if (deviceEnd == std::string_view::npos)
            break;
         std::size_t roEnd = buf.find('\n', deviceEnd + 1);
         if (roEnd == std::string_view::npos)
            break;
         std::size_t valueStart = roEnd + 1;
         std::size_t valueEnd = buf.find('\n', valueStart);
         if (valueEnd == std::string_view::npos)
            break;

         const bool nameMatches = (nameEnd - nameStart) == keyLen &&
            buf.compare(nameStart, keyLen, key, keyLen) == 0;
         const bool isImageTag = (deviceEnd - deviceStart) == 1 &&
            buf[deviceStart] == '_';
         if (nameMatches && isImageTag)
            last = {valueStart, valueEnd};

         pos = valueEnd + 1;
      }
      return last;
   }

   mutable std::string buffer_;
   std::size_t count_ = 0;
};

} // namespace internal
} // namespace mmcore
