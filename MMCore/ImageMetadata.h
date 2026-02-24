///////////////////////////////////////////////////////////////////////////////
// FILE:          ImageMetadata.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     MMCore
//-----------------------------------------------------------------------------
// DESCRIPTION:   Metadata associated with the acquired image
//
// AUTHOR:        Nenad Amodaj, nenad@amodaj.com, 06/07/2007
// COPYRIGHT:     University of California, San Francisco, 2007
//                100X Imaging Inc, 2008
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

#pragma once

#include "CoreDeclHelpers.h"

#include <cstddef>
#include <cstdlib>
#include <exception>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
// MetadataError
// -------------
// Micro-Manager metadata error class, used to create exception objects
// 
class MetadataError : public std::exception
{
public:
   MetadataError(const char* msg) :
      message_(msg) {}

   virtual ~MetadataError() {}

   virtual std::string getMsg()
   {
      return message_;
   }

   virtual const char* what() const MMCORE_NOEXCEPT { return message_.c_str(); }

private:
   std::string message_;
};

class MetadataKeyError : public MetadataError
{
public:
   MetadataKeyError() :
      MetadataError("Undefined metadata key") {}
   MetadataKeyError(const char* key) :
      MetadataError(("Undefined metadata key: " + std::string(key)).c_str()) {}
   ~MetadataKeyError() {}
};

class MetadataIndexError : public MetadataError
{
public:
   MetadataIndexError() :
      MetadataError("Metadata array index out of bounds") {}
   ~MetadataIndexError() {}
};


class MetadataSingleTag;
class MetadataArrayTag;

/**
 * Image information tags - metadata.
 */
class MetadataTag
{
public:
   MetadataTag() : name_("undefined"), deviceLabel_("undefined"), readOnly_(false) {}
   MetadataTag(const char* name, const char* device, bool readOnly) :
      name_(name), deviceLabel_(device), readOnly_(readOnly) {}
   virtual ~MetadataTag() {}

   const std::string& GetDevice() const {return deviceLabel_;}
   const std::string& GetName() const {return name_;}
   const std::string GetQualifiedName() const
   {
      std::string str;
      if (deviceLabel_.compare("_") != 0)
      {
         str.append(deviceLabel_).append("-");
      }
      str.append(name_);
      return str;
   }
   bool IsReadOnly() const  {return readOnly_;}

   void SetDevice(const char* device) {deviceLabel_ = device;}
   void SetName(const char* name) {name_ = name;}
   void SetReadOnly(bool ro) {readOnly_ = ro;}

   /**
    * Equivalent of dynamic_cast<MetadataSingleTag*>(this), but does not use
    * RTTI. This makes it safe against multiple definitions when using 
    * dynamic libraries on Linux (original cause: JVM uses 
    * dlopen with RTLD_LOCAL when loading libraries.
    */
   virtual const MetadataSingleTag* ToSingleTag() const { return 0; }
   /**
    * Equivalent of dynamic_cast<MetadataArrayTag*>(this), but does not use
    * RTTI. @see ToSingleTag
    */
   virtual const MetadataArrayTag*  ToArrayTag()  const { return 0; }

   //inline  MetadataSingleTag* ToSingleTag() {
   //   const MetadataTag *p = this;
   //   return const_cast<MetadataSingleTag*>(p->ToSingleTag());
   //  }
   //inline  MetadataArrayTag* ToArrayTag() {
   //   const MetadataTag *p = this;
   //   return const_cast<MetadataArrayTag*>(p->ToArrayTag());
   //}

   virtual MetadataTag* Clone() = 0;
   virtual std::string Serialize() = 0;
   virtual bool Restore(const char* stream) = 0;
   virtual bool Restore(std::istringstream& is) = 0;

   static std::string ReadLine(std::istringstream& is)
   {
      std::string ret;
      std::getline(is, ret);
      return ret;
   }

private:
   std::string name_;
   std::string deviceLabel_;
   bool readOnly_;
};

class MetadataSingleTag : public MetadataTag
{
public:
   MetadataSingleTag() {}
   MetadataSingleTag(const char* name, const char* device, bool readOnly) :
      MetadataTag(name, device, readOnly) {}
   ~MetadataSingleTag() {}

   const std::string& GetValue() const {return value_;}
   void SetValue(const char* val) {value_ = val;}

   virtual const MetadataSingleTag* ToSingleTag() const { return this; }

   MetadataTag* Clone()
   {
      return new MetadataSingleTag(*this);
   }

   std::string Serialize()
   {
      std::string str;

      str.append(GetName()).append("\n");
      str.append(GetDevice()).append("\n");
      str.append(IsReadOnly() ? "1" : "0").append("\n");

      str.append(value_).append("\n");

      return str;
   }

   bool Restore(const char* stream)
   {
      std::istringstream is(stream);
      return Restore(is);
   }

   bool Restore(std::istringstream& is)
   {
      SetName(ReadLine(is).c_str());
      SetDevice(ReadLine(is).c_str());
      SetReadOnly(std::atoi(ReadLine(is).c_str()) != 0);

      value_ = ReadLine(is);

      return true;
   }

private:
   std::string value_;
};

class MetadataArrayTag : public MetadataTag
{
public:
   MetadataArrayTag() {}
   MetadataArrayTag(const char* name, const char* device, bool readOnly) :
      MetadataTag(name, device, readOnly) {}
   ~MetadataArrayTag() {}

   virtual const MetadataArrayTag* ToArrayTag() const { return this; }

   void AddValue(const char* val) {values_.push_back(val);}
   void SetValue(const char* val, std::size_t idx)
   {
      if (values_.size() < idx+1)
         values_.resize(idx+1);
      values_[idx] = val;
   }

   const std::string& GetValue(std::size_t idx) const {
      if (idx >= values_.size())
         throw MetadataIndexError();
      return values_[idx];
   }

   std::size_t GetSize() const {return values_.size();}

   MetadataTag* Clone()
   {
      return new MetadataArrayTag(*this);
   }

   std::string Serialize()
   {
      std::string str;

      str.append(GetName()).append("\n");
      str.append(GetDevice()).append("\n");
      str.append(IsReadOnly() ? "1" : "0").append("\n");

      std::stringstream os;
      os << values_.size();
      str.append(os.str()).append("\n");

      for (std::size_t i = 0; i < values_.size(); i++)
         str.append(values_[i]).append("\n");

      return str;
   }

   bool Restore(const char* stream)
   {
      std::istringstream is(stream);
      return Restore(is);
   }

   bool Restore(std::istringstream& is)
   {
      SetName(ReadLine(is).c_str());
      SetDevice(ReadLine(is).c_str());
      SetReadOnly(atoi(ReadLine(is).c_str()) != 0);

      std::size_t size = std::atol(ReadLine(is).c_str());

      values_.resize(size);

      for (std::size_t i = 0; i < size; i++)
         values_[i] = ReadLine(is);

      return true;
   }

private:
   std::vector<std::string> values_;
};

/**
 * Container for all metadata associated with a single image.
 */
class Metadata
{
public:

   Metadata() {} // empty constructor

   ~Metadata() // destructor
   {
      Clear();
   }

   Metadata(const Metadata& original) // copy constructor
   {
      for (const auto& p : original.tags_)
         SetTag(*p.second);
   }

   void Clear() { tags_.clear(); }

   std::vector<std::string> GetKeys() const
   {
      std::vector<std::string> keyList;
      for (const auto& p : tags_)
         keyList.push_back(p.first);
      return keyList;
   }

   bool HasTag(const char* key)
   {
      return tags_.find(key) != tags_.end();
   }

   MetadataSingleTag GetSingleTag(const char* key) const MMCORE_LEGACY_THROW(MetadataKeyError)
   {
      MetadataTag* tag = FindTag(key);
      const MetadataSingleTag* stag = tag->ToSingleTag();
      return *stag;
   }

   MetadataArrayTag GetArrayTag(const char* key) const MMCORE_LEGACY_THROW(MetadataKeyError)
   {
      MetadataTag* tag = FindTag(key);
      const MetadataArrayTag* atag = tag->ToArrayTag();
      return *atag;
   }

   void SetTag(MetadataTag& tag)
   {
      std::unique_ptr<MetadataTag> newTag(tag.Clone());
      std::string key(tag.GetQualifiedName());
      tags_[key] = std::move(newTag);
   }

   void RemoveTag(const char* key) { tags_.erase(key); }

   /*
    * Convenience method to add a MetadataSingleTag
    */
   template <class anytype>
   void PutTag(std::string key, std::string deviceLabel, anytype value)
   {
      std::stringstream os;
      os << value;
      auto newTag = std::make_unique<MetadataSingleTag>(
          key.c_str(), deviceLabel.c_str(), true);
      newTag->SetValue(os.str().c_str());
      std::string qname(newTag->GetQualifiedName());
      tags_[qname] = std::move(newTag);
   }

   /*
    * Add a tag not associated with any device.
    */
   template <class anytype>
   void PutImageTag(std::string key, anytype value)
   {
      PutTag(key, "_", value);
   }

   /*
    * Deprecated name. Equivalent to PutImageTag.
    */
   template <class anytype>
   MMCORE_DEPRECATED
   void put(std::string key, anytype value)
   {
      PutImageTag(key, value);
   }

#ifndef SWIG
   Metadata& operator=(const Metadata& rhs)
   {
      Clear();
      for (const auto& p : rhs.tags_)
         SetTag(*p.second);
      return *this;
   }
#endif

   void Merge(const Metadata& newTags)
   {
      for (const auto& p : newTags.tags_)
         SetTag(*p.second);
   }

   std::string Serialize() const
   {
      std::string str;

      std::ostringstream os;
      os << tags_.size();
      str.append(os.str()).append("\n");

      for (const auto& p : tags_)
      {
         const std::string id(p.second->ToArrayTag() ? "a" : "s");
         str.append(id).append("\n");
         str.append(p.second->Serialize());
      }

      return str;
   }

   // TODO: Can this be removed?
   std::string readLine(std::istringstream &iss)
   {
      return MetadataTag::ReadLine(iss);
   }

   bool Restore(const char* stream)
   {
      Clear();
      if (stream == nullptr)
      {
         return true;
      }

      std::istringstream is(stream);

      const std::size_t sz = std::atol(readLine(is).c_str());

      for (std::size_t i=0; i<sz; i++)
      {
         const std::string id(readLine(is));

         std::unique_ptr<MetadataTag> newTag;
         if (id.compare("s") == 0)
         {
            newTag = std::make_unique<MetadataSingleTag>();
         }
         else if (id.compare("a") == 0)
         {
            newTag = std::make_unique<MetadataArrayTag>();
         }
         else
         {
            return false;
         }

         newTag->Restore(is);
         tags_[newTag->GetQualifiedName()] = std::move(newTag);
      }
      return true;
   }

   std::string Dump()
   {
      std::ostringstream os;

      os << tags_.size();
      for (const auto& p : tags_)
      {
         std::string id("s");
         if (p.second->ToArrayTag())
            id = "a";
         std::string ser = p.second->Serialize();
         os << id << " : " << ser << '\n';
      }

      return os.str();
   }

private:
   MetadataTag* FindTag(const char* key) const
   {
      auto it = tags_.find(key);
      if (it != tags_.end())
         return it->second.get();
      else
         throw MetadataKeyError(key);
   }

   std::map<std::string, std::unique_ptr<MetadataTag>> tags_;
};
