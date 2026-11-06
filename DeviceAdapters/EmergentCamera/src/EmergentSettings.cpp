// Copyright (c) 2026, Dr. Volodymyr Cherkas
// SPDX-License-Identifier: BSD-3-Clause

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#elif defined(__linux__)
#include <dlfcn.h>
#include <unistd.h>
#endif

#include "EmergentCamera.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>

namespace
{
using Section = std::map<std::string, std::string>;
using Ini = std::map<std::string, Section>;

std::string Trim(const std::string& text)
{
   const size_t first = text.find_first_not_of(" \t\r\n");
   if (first == std::string::npos)
      return {};
   return text.substr(first, text.find_last_not_of(" \t\r\n") - first + 1);
}

Ini ReadIni(const std::filesystem::path& file)
{
   if (!std::filesystem::exists(file))
      return {};
   if (std::filesystem::file_size(file) > 1024 * 1024)
      throw std::runtime_error("settings file exceeds 1 MiB");
   std::ifstream input(file, std::ios::binary);
   if (!input)
      throw std::runtime_error("cannot open settings file for reading");
   Ini ini;
   std::string section, line;
   unsigned lineNumber = 0;
   while (std::getline(input, line))
   {
      ++lineNumber;
      if (lineNumber == 1 && line.compare(0, 3, "\xef\xbb\xbf") == 0)
         line.erase(0, 3);
      line = Trim(line);
      if (line.empty() || line[0] == ';' || line[0] == '#')
         continue;
      if (line.size() > 4096)
         throw std::runtime_error("settings line is too long");
      if (line.front() == '[' && line.back() == ']')
      {
         section = Trim(line.substr(1, line.size() - 2));
         if (section.empty())
            throw std::runtime_error("empty settings section");
         continue;
      }
      const size_t equals = line.find('=');
      if (section.empty() || equals == std::string::npos)
         throw std::runtime_error("invalid INI syntax on line " + std::to_string(lineNumber));
      const auto key = Trim(line.substr(0, equals));
      const auto value = Trim(line.substr(equals + 1));
      if (key.empty() || key.size() >= MM::MaxStrLength || value.size() >= MM::MaxStrLength)
         throw std::runtime_error("invalid settings key/value length");
      if (!ini[section].emplace(key, value).second)
         throw std::runtime_error("duplicate settings key: " + key);
   }
   if (!input.eof())
      throw std::runtime_error("error reading settings file");
   const auto general = ini.find("Emergent");
   if (!ini.empty() && (general == ini.end() || general->second.count("FormatVersion") == 0 ||
       general->second.at("FormatVersion") != "1"))
      throw std::runtime_error("unsupported or missing FormatVersion");
   return ini;
}

void WriteIni(const std::filesystem::path& file, const Ini& ini)
{
   auto temporary = file;
#ifdef _WIN32
   temporary += ".tmp." + std::to_string(GetCurrentProcessId());
#elif defined(__linux__)
   temporary += ".tmp." + std::to_string(getpid());
#else
   temporary += ".tmp";
#endif
   {
      std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
      if (!output)
         throw std::runtime_error("cannot create temporary settings file; check folder permissions");
      output << "; EmergentCamera settings, UTF-8. Saved on clean shutdown or Settings - Save.\n";
      for (const auto& section : ini)
      {
         output << "\n[" << section.first << "]\n";
         for (const auto& item : section.second)
            output << item.first << '=' << item.second << '\n';
      }
      output.flush();
      if (!output)
         throw std::runtime_error("cannot write settings file");
   }
#ifdef _WIN32
   if (!MoveFileExW(temporary.c_str(), file.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
      throw std::runtime_error("cannot replace settings file (Windows error " + std::to_string(GetLastError()) + ")");
#else
   std::filesystem::rename(temporary, file);
#endif
}

bool StartsWith(const std::string& text, const char* prefix)
{
   return text.compare(0, std::char_traits<char>::length(prefix), prefix) == 0;
}

bool Persistable(const std::string& name)
{
   return !StartsWith(name, "Settings - ") && name != "ROI - Center X" &&
      name != "ROI - Center Y" && name != "Image LUT - Index" && name != "Image LUT - Value";
}

int GroupRank(const std::string& name)
{
   if (StartsWith(name, "Camera Info - ") || name == "Description" || name == "Camera" || name == "Camera IP Address") return 0;
   if (StartsWith(name, "Temperature - ")) return 1;
   if (StartsWith(name, "Acquisition - ")) return 2;
   if (StartsWith(name, "Exposure - ")) return 3;
   if (StartsWith(name, "Gain - ") || StartsWith(name, "Auto Control ROI - ")) return 4;
   if (StartsWith(name, "Readout - ")) return 5;
   if (StartsWith(name, "ROI - ") || name == "Binning") return 6;
   if (StartsWith(name, "Trigger - ")) return 7;
   if (StartsWith(name, "PTP - ")) return 8;
   if (StartsWith(name, "Image ")) return 9;
   if (StartsWith(name, "Settings - ")) return 10;
   return 11;
}
}

bool EmergentCamera::GetPropertyName(unsigned index, char* name) const
{
   std::vector<std::string> names;
   for (unsigned i = 0; i < GetNumberOfProperties(); ++i)
   {
      char propertyName[MM::MaxStrLength] = {};
      if (CCameraBase<EmergentCamera>::GetPropertyName(i, propertyName))
         names.emplace_back(propertyName);
   }
   std::stable_sort(names.begin(), names.end(), [](const std::string& a, const std::string& b) {
      const int aRank = GroupRank(a), bRank = GroupRank(b);
      if (aRank == 4 && bRank == 4 && StartsWith(a, "Gain - ") != StartsWith(b, "Gain - "))
         return StartsWith(a, "Gain - ");
      return aRank == bRank ? a < b : aRank < bRank;
   });
   if (index >= names.size())
      return false;
   CDeviceUtils::CopyLimitedString(name, names[index].c_str());
   return true;
}

std::string EmergentCamera::DefaultSettingsFile()
{
#ifdef _WIN32
   HMODULE module = nullptr;
   if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                         reinterpret_cast<LPCWSTR>(&g_EmergentCameraName), &module))
   {
      wchar_t path[32768] = {};
      const DWORD size = GetModuleFileNameW(module, path, 32768);
      if (size > 0 && size < 32768)
         return (std::filesystem::path(path).parent_path() / "Emergent.ini").u8string();
   }
#elif defined(__linux__)
   Dl_info module{};
   if (dladdr(static_cast<const void*>(&g_EmergentCameraName), &module) != 0 &&
       module.dli_fname != nullptr && module.dli_fname[0] != '\0')
      return (std::filesystem::absolute(module.dli_fname).parent_path() / "Emergent.ini").u8string();
#endif
   return (std::filesystem::current_path() / "Emergent.ini").u8string();
}

int EmergentCamera::OnSettingsFile(MM::PropertyBase* property, MM::ActionType action)
{
   if (action == MM::BeforeGet)
      property->Set(settingsFile_.c_str());
   else if (action == MM::AfterSet)
   {
      std::string path;
      property->Get(path);
      if (path.empty() || !std::filesystem::u8path(path).is_absolute())
         return DEVICE_INVALID_INPUT_PARAM;
      settingsFile_ = path;
   }
   return DEVICE_OK;
}

int EmergentCamera::OnRememberSettings(MM::PropertyBase* property, MM::ActionType action)
{
   if (action == MM::BeforeGet)
      property->Set(rememberSettings_ ? "On" : "Off");
   else if (action == MM::AfterSet)
   {
      std::string value;
      property->Get(value);
      rememberSettings_ = value == "On";
   }
   return DEVICE_OK;
}

int EmergentCamera::OnSettingsStatus(MM::PropertyBase* property, MM::ActionType action)
{
   if (action == MM::BeforeGet)
   {
      MMThreadGuard guard(cameraLock_);
      property->Set(settingsStatus_.c_str());
   }
   return DEVICE_OK;
}

int EmergentCamera::OnSaveSettings(MM::PropertyBase* property, MM::ActionType action)
{
   if (action == MM::BeforeGet)
      property->Set("Idle");
   else if (action == MM::AfterSet)
   {
      std::string command;
      property->Get(command);
      property->Set("Idle");
      if (command == "Save now")
      {
         if (IsCapturing())
            return DEVICE_CAMERA_BUSY_ACQUIRING;
         MMThreadGuard guard(cameraLock_);
         return SaveSettingsLocked();
      }
   }
   return DEVICE_OK;
}

int EmergentCamera::SaveSettingsLocked()
{
   try
   {
      if (serial_.empty())
         throw std::runtime_error("camera serial number unavailable");
      const auto file = std::filesystem::u8path(settingsFile_);
      Ini ini = ReadIni(file); // Preserve sections belonging to other cameras.
      Section section;
      section["Model"] = model_;
      section["RequestedFrameRate"] = std::to_string(requestedFrameRateFps_);
      for (unsigned i = 0; i < GetNumberOfProperties(); ++i)
      {
         char name[MM::MaxStrLength] = {}, value[MM::MaxStrLength] = {};
         bool readOnly = true, preInit = true;
         if (!GetPropertyName(i, name) || !Persistable(name) ||
             GetPropertyReadOnly(name, readOnly) != DEVICE_OK || readOnly ||
             GetPropertyInitStatus(name, preInit) != DEVICE_OK || preInit)
            continue;
         if (GetProperty(name, value) != DEVICE_OK)
            throw std::runtime_error(std::string("cannot read setting ") + name);
         section[name] = value;
      }
      ini["Emergent"]["FormatVersion"] = "1";
      ini["Camera:" + serial_] = std::move(section);
      WriteIni(file, ini);
      settingsFileHealthy_ = true;
      settingsStatus_ = "Saved settings for serial " + serial_;
      LogMessage(settingsStatus_.c_str());
      return DEVICE_OK;
   }
   catch (const std::exception& e)
   {
      settingsStatus_ = std::string("Save failed: ") + e.what();
      LogMessage(settingsStatus_.c_str(), false);
      return ERR_SETTINGS_FILE;
   }
}

int EmergentCamera::RestoreSettingsLocked()
{
   restoringSettings_ = true;
   unsigned restored = 0, skipped = 0;
   try
   {
      const Ini ini = ReadIni(std::filesystem::u8path(settingsFile_));
      settingsFileHealthy_ = true;
      const auto found = ini.find("Camera:" + serial_);
      if (found == ini.end())
      {
         settingsStatus_ = "No saved settings for serial " + serial_ + "; frame-rate limit defaults Off";
         restoringSettings_ = false;
         return DEVICE_OK;
      }
      const Section& section = found->second;
      const auto model = section.find("Model");
      if (model == section.end() || model->second != model_)
         throw std::runtime_error("saved camera model does not match");

      const auto write = [this](const char* name, const char* value) {
         if (!HasProperty(name))
            return DEVICE_UNSUPPORTED_COMMAND;
         MM::PropertyType type;
         GetPropertyType(name, type);
         try
         {
            size_t end = 0;
            if (type == MM::Float)
            {
               const double number = std::stod(value, &end);
               if (!std::isfinite(number) || end != std::char_traits<char>::length(value))
                  return DEVICE_INVALID_INPUT_PARAM;
            }
            else if (type == MM::Integer)
            {
               const auto number = std::stoll(value, &end);
               if (end != std::char_traits<char>::length(value) ||
                   number < std::numeric_limits<long>::min() || number > std::numeric_limits<long>::max())
                  return DEVICE_INVALID_INPUT_PARAM;
            }
         }
         catch (const std::exception&) { return DEVICE_INVALID_INPUT_PARAM; }
         char current[MM::MaxStrLength] = {};
         GetProperty(name, current); // Refresh dynamic limits before validation.
         return SetProperty(name, value);
      };
      const auto apply = [&](const char* name) {
         const auto value = section.find(name);
         if (value == section.end()) return;
         if (write(name, value->second.c_str()) == DEVICE_OK)
            ++restored;
         else
         {
            ++skipped;
            LogMessage((std::string("Skipped invalid/unavailable saved setting: ") + name).c_str(), false);
         }
      };

      write("Trigger - Mode", "Internal");
      write("PTP - Mode", "Off");
      write("Exposure - Auto", "Off");
      write("Gain - Auto", "Off");
      write("Acquisition - Frame Rate Limit", "On");
      write("Acquisition - Frame Rate (fps)", std::to_string(frameRateMinimumFps_).c_str());
      apply("Readout - ADC Mode");
      apply("Readout - Pixel Type");
      apply("Binning");
      ClearROI();

      const char* geometry[] = {"ROI - Offset X", "ROI - Offset Y", "ROI - Width", "ROI - Height"};
      bool completeROI = true;
      unsigned roi[4] = {};
      try
      {
         for (unsigned i = 0; i < 4; ++i)
         {
            const auto value = section.find(geometry[i]);
            if (value == section.end()) { completeROI = false; break; }
            size_t end = 0;
            const auto number = std::stoull(value->second, &end);
            if (end != value->second.size() || number > std::numeric_limits<unsigned>::max())
               throw std::runtime_error("invalid saved ROI");
            roi[i] = static_cast<unsigned>(number);
         }
      }
      catch (const std::exception&)
      {
         completeROI = false;
         skipped += 4;
         LogMessage("Skipped invalid saved ROI numbers", false);
      }
      if (completeROI)
      {
         if (SetROI(roi[0], roi[1], roi[2], roi[3]) == DEVICE_OK) restored += 4;
         else { skipped += 4; LogMessage("Skipped invalid saved ROI", false); }
      }
      write("Auto Control ROI - Offset X", "0");
      write("Auto Control ROI - Offset Y", "0");
      apply("Auto Control ROI - Width");
      apply("Auto Control ROI - Height");
      apply("Auto Control ROI - Offset X");
      apply("Auto Control ROI - Offset Y");

      for (const auto& item : section)
      {
         const std::string& name = item.first;
         if (!Persistable(name) || !HasProperty(name.c_str()) || StartsWith(name, "ROI - ") ||
             StartsWith(name, "Auto Control ROI - ") || StartsWith(name, "Readout - ") ||
             name == "Binning" || name == "Acquisition - Frame Rate Limit" || name == "Acquisition - Frame Rate (fps)" ||
             name == "Exposure - Auto" || name == "Gain - Auto" || name == "Trigger - Mode" || name == "PTP - Mode")
            continue;
         bool readOnly = true, preInit = true;
         if (GetPropertyReadOnly(name.c_str(), readOnly) == DEVICE_OK && !readOnly &&
             GetPropertyInitStatus(name.c_str(), preInit) == DEVICE_OK && !preInit)
            apply(name.c_str());
      }
      const auto requested = section.find("RequestedFrameRate");
      if (requested != section.end())
         write("Acquisition - Frame Rate (fps)", requested->second.c_str());
      else
         apply("Acquisition - Frame Rate (fps)");
      apply("Acquisition - Frame Rate Limit");
      apply("Exposure - Auto");
      apply("Gain - Auto");
      apply("PTP - Mode");
      apply("Trigger - Mode");
      settingsStatus_ = "Restored " + std::to_string(restored) + " settings for serial " + serial_;
      if (skipped) settingsStatus_ += "; skipped " + std::to_string(skipped);
      LogMessage(settingsStatus_.c_str());
      restoringSettings_ = false;
      return DEVICE_OK;
   }
   catch (const std::exception& e)
   {
      settingsStatus_ = std::string("Restore failed: ") + e.what();
      settingsFileHealthy_ = false; // Do not overwrite a malformed/future file on close.
      restoringSettings_ = false;
      LogMessage(settingsStatus_.c_str(), false);
      return ERR_SETTINGS_FILE;
   }
}
