///////////////////////////////////////////////////////////////////////////////
// FILE:          Go2Scope.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   Go2Scope devices. Includes the experimental StorageDevice
//
// AUTHOR:        Nenad Amodaj
//
// COPYRIGHT:     Nenad Amodaj, Chan Zuckerberg Initiative, 2024
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
// NOTE:          Storage Device development is supported in part by
//                Chan Zuckerberg Initiative (CZI)
// 
///////////////////////////////////////////////////////////////////////////////
#include "go2scope.h"
#include "ModuleInterface.h"

using namespace std;


///////////////////////////////////////////////////////////////////////////////
// Exported MMDevice API
///////////////////////////////////////////////////////////////////////////////
MODULE_API void InitializeModuleData()
{
   RegisterDevice(g_MMV1Storage, MM::StorageDevice, "Storage for old MM format");
}

MODULE_API MM::Device* CreateDevice(const char* deviceName)
{
   if (deviceName == 0)
      return 0;

   if (strcmp(deviceName, g_MMV1Storage) == 0)
   {
      return new MMV1Storage();
   }

   return 0;
}

MODULE_API void DeleteDevice(MM::Device* pDevice)
{
   delete pDevice;
}


///////////////////////////////////////////////////////////////////////////////
// Lida

MMV1Storage::MMV1Storage() :
   initialized(false)
{
   InitializeDefaultErrorMessages();

	// set device specific error messages
   SetErrorText(ERR_INTERNAL, "Internal driver error, see log file for details");

                                                                             
   // create pre-initialization properties                                   
   // ------------------------------------
   //
                                                                          
   // Name                                                                   
   CreateProperty(MM::g_Keyword_Name, g_MMV1Storage, MM::String, true);
   //
   // Description                                                            
   CreateProperty(MM::g_Keyword_Description, "Storage for old MM format", MM::String, true);
}                                                                            
                                                                             
MMV1Storage::~MMV1Storage()                                                            
{                                                                            
   Shutdown();                                                               
} 

void MMV1Storage::GetName(char* Name) const
{
   CDeviceUtils::CopyLimitedString(Name, g_MMV1Storage);
}  

int MMV1Storage::Initialize()
{
   if (initialized)
      return DEVICE_OK;

	int ret(DEVICE_OK);

   UpdateStatus();

   initialized = true;
   return DEVICE_OK;
}

int MMV1Storage::Shutdown()
{
   if (initialized)
   {
      initialized = false;
   }
   return DEVICE_OK;
}

// Never busy because all commands block
bool MMV1Storage::Busy()
{
   return false;
}

int MMV1Storage::Create(const char* path, const char* name, int numberOfDimensions, const int shape[], const char* meta, char* handle)
{
   return 0;
}

int MMV1Storage::ConfigureDimension(const char* handle, int dimension, const char* name, const char* meaning)
{
   return 0;
}

int MMV1Storage::ConfigureCoordinate(const char* handle, int dimension, int coordinate, const char* name)
{
   return 0;
}

int MMV1Storage::Close(const char* handle)
{
   return 0;
}

int MMV1Storage::Load(const char* path, const char* name, char* handle)
{
   return 0;
}

int MMV1Storage::Delete(char* handle)
{
   return 0;
}

int MMV1Storage::List(const char* path, char** listOfDatasets, int maxItems, int maxItemLength)
{
   return 0;
}

int MMV1Storage::AddImage(const char* handle, unsigned char* pixels, int width, int height, int depth, int coordinates[], int numCoordinates, const char* imageMeta)
{
   return 0;
}

int MMV1Storage::GetSummaryMeta(const char* handle, char* meta)
{
   return 0;
}

int MMV1Storage::GetImageMeta(const char* handle, int coordinates[], int numCoordinates, char* meta)
{
   return 0;
}

const unsigned char* MMV1Storage::GetImage(const char* handle, int coordinates[], int numCoordinates)
{
   return nullptr;
}

int MMV1Storage::GetNumberOfDimensions(const char* handle, int& numDimensions)
{
   return 0;
}

int MMV1Storage::GetDimension(const char* handle, int dimension, char* name, int nameLength, char* meaning, int meaningLength)
{
   return 0;
}

int MMV1Storage::GetCoordinate(const char* handle, int dimension, int coordinate, char* name, int nameLength)
{
   return 0;
}


///////////////////////////////////////////////////////////////////////////////
// Action handlers
///////////////////////////////////////////////////////////////////////////////

