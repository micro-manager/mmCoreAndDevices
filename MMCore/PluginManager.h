///////////////////////////////////////////////////////////////////////////////
// FILE:          PluginManager.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     MMCore
//-----------------------------------------------------------------------------
// DESCRIPTION:   Loading/unloading of device adapter modules
//              
// COPYRIGHT:     University of California, San Francisco, 2006-2014
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
//
// AUTHOR:        Nenad Amodaj, nenad@amodaj.com, 08/10/2005

#pragma once

#include "MockDeviceAdapter.h"
#include "../MMDevice/DeviceThreads.h"

#include <map>
#include <memory>
#include <string>
#include <vector>

class LoadedDeviceAdapter;


class CPluginManager /* final */
{
public:
   CPluginManager();
   ~CPluginManager();

   void UnloadPluginLibrary(const char* moduleName);

   // Device adapter search paths
   template <typename TStringIter>
   void SetSearchPaths(TStringIter begin, TStringIter end)
   { searchPaths_.assign(begin, end); }
   std::vector<std::string> GetSearchPaths() const { return searchPaths_; }
   std::vector<std::string> GetAvailableDeviceAdapters();

   /**
    * Return a device adapter module, loading it if necessary
    */
   std::shared_ptr<LoadedDeviceAdapter>
   GetDeviceAdapter(const std::string& moduleName);
   std::shared_ptr<LoadedDeviceAdapter>
   GetDeviceAdapter(const char* moduleName);

   void LoadMockAdapter(const std::string& name, MockDeviceAdapter* impl);

private:
   static std::vector<std::string> GetDefaultSearchPaths();
   static void GetModules(std::vector<std::string> &modules, const char *path);
   std::string FindInSearchPath(std::string filename);

   std::vector<std::string> searchPaths_;

   std::map< std::string, std::shared_ptr<LoadedDeviceAdapter> > moduleMap_;
};
