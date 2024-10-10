///////////////////////////////////////////////////////////////////////////////
// FILE:          AcqZarrStorage.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   Zarr writer based on the CZI acquire-zarr library
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
#include "G2SStorage.h"
#include "AcqZarrStorage.h"
#include "zarr.h"
#include "nlohmann/json.hpp"
#include <random>
#include <sstream>
#include <iomanip>

using namespace std;


std::string generate_guid() {
   std::random_device rd;
   std::mt19937 gen(rd());
   std::uniform_int_distribution<> dis(0, 15);
   std::uniform_int_distribution<> dis2(8, 11);

   std::stringstream ss;
   ss << std::hex;

   for (int i = 0; i < 8; i++) {
      ss << dis(gen);
   }
   ss << "-";
   for (int i = 0; i < 4; i++) {
      ss << dis(gen);
   }
   ss << "-4";
   for (int i = 0; i < 3; i++) {
      ss << dis(gen);
   }
   ss << "-" << dis2(gen);
   for (int i = 0; i < 3; i++) {
      ss << dis(gen);
   }
   ss << "-";
   for (int i = 0; i < 12; i++) {
      ss << dis(gen);
   }

   return ss.str();
}

///////////////////////////////////////////////////////////////////////////////
// Zarr storage

AcqZarrStorage::AcqZarrStorage() :
   initialized(false), zarrStream(nullptr)
{
   InitializeDefaultErrorMessages();

	// set device specific error messages
   SetErrorText(ERR_INTERNAL, "Internal driver error, see log file for details");

   auto ver = Zarr_get_api_version();
                                                                             
   // create pre-initialization properties                                   
   // ------------------------------------
   //
                                                                          
   // Name                                                                   
   CreateProperty(MM::g_Keyword_Name, g_AcqZarrStorage, MM::String, true);
   //
   // Description
   ostringstream os;
   os << "Acquire Zarr Storage v" << ver;
   CreateProperty(MM::g_Keyword_Description, os.str().c_str(), MM::String, true);
}                                                                            
                                                                             
AcqZarrStorage::~AcqZarrStorage()                                                            
{                                                                            
   Shutdown();
} 

void AcqZarrStorage::GetName(char* Name) const
{
   CDeviceUtils::CopyLimitedString(Name, g_AcqZarrStorage);
}  

int AcqZarrStorage::Initialize()
{
   if (initialized)
      return DEVICE_OK;

	int ret(DEVICE_OK);

   UpdateStatus();

   initialized = true;
   return DEVICE_OK;
}

int AcqZarrStorage::Shutdown()
{
   if (initialized)
   {
      initialized = false;
   }
   destroyStream();

   return DEVICE_OK;
}

// Never busy because all commands block
bool AcqZarrStorage::Busy()
{
   return false;
}

int AcqZarrStorage::Create(const char* path, const char* name, int numberOfDimensions, const int shape[], MM::StorageDataType pixType, const char* meta, char* handle)
{
   if (zarrStream)
   {
      LogMessage("Another stream is already open. Currently this device supports only one stream.");
      return ERR_ZARR_NUMDIMS;
   }

   if (numberOfDimensions < 3)
   {
      LogMessage("Number of dimensions is lower than 3.");
      return ERR_ZARR_NUMDIMS;
   }

   auto settings = ZarrStreamSettings_create();
   if (!settings)
   {
      LogMessage("Failed creating Zarr stream settings.");
      return ERR_ZARR_SETTINGS;
   }

   // set store
   ostringstream osZarrStreamPath;
   osZarrStreamPath << path << "/" << name;
   ZarrStatus status = ZarrStreamSettings_set_store(settings,
                                                    osZarrStreamPath.str().c_str(),
                                                    osZarrStreamPath.str().size(),
                                                    nullptr);
   if (status != ZarrStatus_Success)
   {
      LogMessage(getErrorMessage(status));
      ZarrStreamSettings_destroy(settings);
      return ERR_ZARR_SETTINGS;
   }

   // set data type
   status = ZarrStreamSettings_set_data_type(settings, (ZarrDataType)pixType);
   if (status != ZarrStatus_Success)
   {
      LogMessage(getErrorMessage(status));
      ZarrStreamSettings_destroy(settings);
      return ERR_ZARR_SETTINGS;
   }

   status = ZarrStreamSettings_reserve_dimensions(settings, numberOfDimensions);
   if (status != ZarrStatus_Success)
   {
      LogMessage(getErrorMessage(status));
      ZarrStreamSettings_destroy(settings);
      return ERR_ZARR_SETTINGS;
   }

   ZarrDimensionProperties dimPropsX;
   string nameX("x");
   dimPropsX.name = nameX.c_str();
   dimPropsX.bytes_of_name = nameX.size();
   dimPropsX.array_size_px = shape[0];
   dimPropsX.chunk_size_px = dimPropsX.array_size_px;
   dimPropsX.shard_size_chunks = 1;

   status = ZarrStreamSettings_set_dimension(settings, 0, &dimPropsX);
   if (status != ZarrStatus_Success)
   {
      LogMessage(getErrorMessage(status));
      ZarrStreamSettings_destroy(settings);
      return ERR_ZARR_SETTINGS;
   }


   ZarrDimensionProperties dimPropsY;
   string nameY("y");
   dimPropsY.name = nameY.c_str();
   dimPropsY.bytes_of_name = nameY.size();
   dimPropsY.array_size_px = shape[1];
   dimPropsY.chunk_size_px = dimPropsY.array_size_px;
   dimPropsY.shard_size_chunks = 1;

   status = ZarrStreamSettings_set_dimension(settings, 0, &dimPropsX);
   if (status != ZarrStatus_Success)
   {
      LogMessage(getErrorMessage(status));
      ZarrStreamSettings_destroy(settings);
      return ERR_ZARR_SETTINGS;
   }


   for (size_t i = 2; i < numberOfDimensions; i++)
   {
      ZarrDimensionProperties dimProps;
      ostringstream osd;
      osd << "dim-" << 1;
      dimProps.name = osd.str().c_str();
      dimProps.bytes_of_name = osd.str().size();
      dimProps.array_size_px = shape[i];
      dimProps.chunk_size_px = 1;
      dimProps.shard_size_chunks = 1;
      ZarrStatus status = ZarrStreamSettings_set_dimension(settings, i, &dimProps);
      if (status != ZarrStatus_Success)
      {
         LogMessage(getErrorMessage(status));
         ZarrStreamSettings_destroy(settings);
         return ERR_ZARR_SETTINGS;
      }
   }

   status = ZarrStreamSettings_set_custom_metadata(settings, meta, strlen(meta));
   if (status != ZarrStatus_Success)
   {
      LogMessage("Invalid summary metadata.");
      ZarrStreamSettings_destroy(settings);
      return ERR_ZARR_SETTINGS;
   }

   zarrStream = ZarrStream_create(settings, ZarrVersion_2);
   if (zarrStream == nullptr)
   {
      LogMessage("Failed creating Zarr stream: " + osZarrStreamPath.str());
      ZarrStreamSettings_destroy(settings);
      return ERR_ZARR_STREAM_CREATE;
   }

   streamHandle = generate_guid();
   // TODO: allow many streams

   streamDimensions = std::vector<int>(shape, shape + numberOfDimensions);
   currentCoordinate = std::vector<int>(numberOfDimensions, 0);
   currentImageNumber = 0;

   ZarrStreamSettings_destroy(settings);

   return DEVICE_OK;
}

int AcqZarrStorage::ConfigureDimension(const char* handle, int dimension, const char* name, const char* meaning)
{
   return DEVICE_OK;
}

int AcqZarrStorage::ConfigureCoordinate(const char* handle, int dimension, int coordinate, const char* name)
{
   return DEVICE_OK;
}

int AcqZarrStorage::Close(const char* handle)
{
   if (zarrStream == nullptr)
   {
      LogMessage("No stream is currently open.");
      return ERR_ZARR_STREAM_CLOSE;
   }
   if (streamHandle.compare(handle) != 0)
   {
      LogMessage("Handle is not valid.");
      return ERR_ZARR_STREAM_CLOSE;
   }

   destroyStream();

   return DEVICE_OK;
}

int AcqZarrStorage::Load(const char* path, const char* name, char* handle)
{
   return DEVICE_NOT_YET_IMPLEMENTED;
}

int AcqZarrStorage::Delete(char* handle)
{
   return DEVICE_NOT_YET_IMPLEMENTED;
}

int AcqZarrStorage::List(const char* path, char** listOfDatasets, int maxItems, int maxItemLength)
{
   return DEVICE_NOT_YET_IMPLEMENTED;
}

int AcqZarrStorage::AddImage(const char* handle, unsigned char* pixels, int width, int height, int depth, int coordinates[], int numCoordinates, const char* imageMeta)
{
   if (zarrStream == nullptr)
   {
      LogMessage("No stream is currently open.");
      return ERR_ZARR_STREAM_ACCESS;
   }
   if (streamHandle.compare(handle) != 0)
   {
      LogMessage("Handle is not valid.");
      return ERR_ZARR_STREAM_ACCESS;
   }

   if (streamDimensions[0] != width || streamDimensions[1] != height)
   {
      LogMessage("Stream dimensions do not match image size");
      return ERR_ZARR_STREAM_APPEND;
   }

   size_t bytesIn((size_t)width * height * depth);
   size_t bytesOut(0);
   ZarrStatus status = ZarrStream_append(zarrStream, pixels, bytesIn, &bytesOut);
   if (status != ZarrStatus_Success)
   {
      LogMessage(getErrorMessage(status));
      return ERR_ZARR_STREAM_APPEND;
   }

   if (bytesOut != bytesIn)
   {
      ostringstream os;
      os << "Bytes in " << bytesIn << " does not match bytes out " << bytesOut;
      LogMessage(os.str());
      return ERR_ZARR_STREAM_APPEND;
   }
   currentImageNumber++;
   int coordPtr = 2;
   while (coordPtr < streamDimensions.size())
   {
      if (currentCoordinate[coordPtr] < streamDimensions[coordPtr] - 1)
      {
         currentCoordinate[coordPtr]++;
         break;
      }
      else
      {
         coordPtr++;
      }
      ostringstream osc;
      osc << "Current coordinate: ";
      for (auto c : currentCoordinate)
         osc << c << " ";
      LogMessage(osc.str());
   }
   
   if (coordPtr == streamDimensions.size())
   {
      LogMessage("End of sequence.");
   }

   return DEVICE_OK;
}

int AcqZarrStorage::GetSummaryMeta(const char* handle, char* meta, int bufSize)
{
   if (zarrStream == nullptr)
   {
      LogMessage("No stream is currently open.");
      return ERR_ZARR_STREAM_ACCESS;
   }
   if (streamHandle.compare(handle) != 0)
   {
      LogMessage("Handle is not valid.");
      return ERR_ZARR_STREAM_ACCESS;
   }

   if (bufSize > 0)
      meta[0] = 0;

   return 0;
}

int AcqZarrStorage::GetImageMeta(const char* handle, int coordinates[], int numCoordinates, char* meta, int bufSize)
{
   if (zarrStream == nullptr)
   {
      LogMessage("No stream is currently open.");
      return ERR_ZARR_STREAM_ACCESS;
   }
   if (streamHandle.compare(handle) != 0)
   {
      LogMessage("Handle is not valid.");
      return ERR_ZARR_STREAM_ACCESS;
   }

   if (bufSize > 0)
      meta[0] = 0;

   return 0;
}

const unsigned char* AcqZarrStorage::GetImage(const char* handle, int coordinates[], int numCoordinates)
{
   if (zarrStream == nullptr)
   {
      LogMessage("No stream is currently open.");
      return nullptr;
   }
   if (streamHandle.compare(handle) != 0)
   {
      LogMessage("Handle is not valid.");
      return nullptr;
   }

   return nullptr;
}

int AcqZarrStorage::GetNumberOfDimensions(const char* handle, int& numDimensions)
{
   if (streamHandle.compare(handle) != 0)
   {
      LogMessage("Handle is not valid.");
      return ERR_ZARR_STREAM_ACCESS;
   }
   return streamDimensions.size();
}

int AcqZarrStorage::GetDimension(const char* handle, int dimension, char* name, int nameLength, char* meaning, int meaningLength)
{
   return DEVICE_NOT_YET_IMPLEMENTED;
}

int AcqZarrStorage::GetCoordinate(const char* handle, int dimension, int coordinate, char* name, int nameLength)
{
   return DEVICE_NOT_YET_IMPLEMENTED;
}

std::string AcqZarrStorage::getErrorMessage(int code)
{
   return std::string(Zarr_get_error_message((ZarrStatus)code));
}

void AcqZarrStorage::destroyStream()
{
   if (zarrStream)
   {
      ZarrStream_destroy(zarrStream);
      zarrStream = nullptr;
      streamHandle = "";
   }
}


///////////////////////////////////////////////////////////////////////////////
// Action handlers
///////////////////////////////////////////////////////////////////////////////

