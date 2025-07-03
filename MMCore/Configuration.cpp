///////////////////////////////////////////////////////////////////////////////
// FILE:          Configuration.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     Core
//-----------------------------------------------------------------------------
// DESCRIPTION:   Set of properties defined as a high level command
// AUTHOR:        Nenad Amodaj, nenad@amodaj.com, 09/08/2005
// COPYRIGHT:     University of California, San Francisco, 2006
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
// CVS:           $Id: Configuration.cpp 13763 2014-07-01 00:43:11Z mark $
//
#include "Configuration.h"
#include "../MMDevice/MMDevice.h"
#include "Error.h"

#include <cstring>
#include <fstream>
#include <sstream>
#include <string>

#ifdef _MSC_VER
#pragma warning(disable: 4290) // 'C++ exception specification ignored'
#endif

#if defined(__GNUC__) && !defined(__clang__)
// 'dynamic exception specifications are deprecated in C++11 [-Wdeprecated]'
#pragma GCC diagnostic ignored "-Wdeprecated"
#endif

std::string PropertySetting::generateKey(const char* device, const char* prop)
{
   std::string key(device);
   key += "-";
   key += prop;
   return key;
}

/**
 * Returns verbose description of the object's contents.
 */
std::string PropertySetting::getVerbose() const
{
   std::ostringstream txt;
   txt << deviceLabel_ << ":" << propertyName_ << "=" << value_;
   return txt.str();
}

bool PropertySetting::isEqualTo(const PropertySetting& ps)
{
   if (ps.deviceLabel_.compare(deviceLabel_) == 0 &&
      ps.propertyName_.compare(propertyName_) == 0 &&
      ps.value_.compare(value_) == 0)
      return true;
   else
      return false;
}


/**
  * Returns verbose description of the object's contents.
  */
std::string Configuration::getVerbose() const
{
   std::ostringstream txt;
   std::vector<PropertySetting>::const_iterator it;
   txt << "<html>";
   for (it=settings_.begin(); it!=settings_.end(); it++)
      txt << it->getVerbose() << "<br>";
   txt << "</html>";

   return txt.str();
}

/**
 * Returns the setting with specified index.
 */
PropertySetting Configuration::getSetting(size_t index) const throw (CMMError)
{
   if (index >= settings_.size())
   {
      std::ostringstream errTxt;
      errTxt << (unsigned int)index << " - invalid configuration setting index";
      throw CMMError(errTxt.str().c_str(), MMERR_DEVICE_GENERIC);
   }
   return settings_[index];
}

/**
  * Checks whether the property is included in the  configuration.
  */

bool Configuration::isPropertyIncluded(const char* device, const char* prop)
{
   std::map<std::string, int>::iterator it = index_.find(PropertySetting::generateKey(device, prop));
   if (it != index_.end())
      return true;
   else
      return false;
}

/**
  * Get the setting with specified device name and property name.
  */

PropertySetting Configuration::getSetting(const char* device, const char* prop)
{
   std::map<std::string, int>::iterator it = index_.find(PropertySetting::generateKey(device, prop));
   if (it == index_.end())
   {
      std::ostringstream errTxt;
      errTxt << "Property " << prop << " not found in device " << device << ".";
      throw CMMError(errTxt.str().c_str(), MMERR_DEVICE_GENERIC);
   }
   if (((unsigned int) it->second) >= settings_.size()) {
      std::ostringstream errTxt;
      errTxt << "Internal Error locating Property " << prop << " in device " << device << ".";
      throw CMMError(errTxt.str().c_str(), MMERR_DEVICE_GENERIC);
   }

   return settings_[it->second];
}

/**
  * Checks whether the setting is included in the  configuration.
  */

bool Configuration::isSettingIncluded(const PropertySetting& ps)
{
   std::map<std::string, int>::iterator it = index_.find(ps.getKey());
   if (it != index_.end() && settings_[it->second].getPropertyValue().compare(ps.getPropertyValue()) == 0)
      return true;
   else
      return false;
}

/**
  * Checks whether a configuration is included.
  * Included means that all devices from the operand configuration are
  * included and that settings match
  */

bool Configuration::isConfigurationIncluded(const Configuration& cfg)
{
   std::vector<PropertySetting>::const_iterator it;
   for (it=cfg.settings_.begin(); it!=cfg.settings_.end(); ++it)
      if (!isSettingIncluded(*it))
         return false;
   
   return true;
}

/**
 * Adds new property setting to the existing contents.
 */
void Configuration::addSetting(const PropertySetting& setting)
{
   std::map<std::string, int>::iterator it = index_.find(setting.getKey());
   if (it != index_.end())
   {
      // replace
      settings_[it->second] = setting;
   }
   else
   {
      // add new
      index_[setting.getKey()] = (int)settings_.size();
      settings_.push_back(setting);
   }
}

/**
 * Removes property setting, specified by device and property names, from the configuration.
 */
void Configuration::deleteSetting(const char* device, const char* prop)
{
   std::map<std::string, int>::iterator it = index_.find(PropertySetting::generateKey(device, prop));
   if (it == index_.end())
   {
      std::ostringstream errTxt;
      errTxt << "Property " << prop << " not found in device " << device << ".";
      throw CMMError(errTxt.str().c_str(), MMERR_DEVICE_GENERIC);
   }

   settings_.erase(settings_.begin() + it->second); // The argument of erase produces an iterator at the desired position.

   // Re-index 
   index_.clear();
   for (unsigned int i = 0; i < settings_.size(); i++) 
   {
      index_[settings_[i].getKey()] = i;
   }

}