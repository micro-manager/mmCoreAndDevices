#include "FileRotation.h"

#include <algorithm>
#include <chrono>
#include <ctime>
#include <filesystem>
#include <string>
#include <vector>

namespace mmcore {
namespace internal {
namespace logging {

namespace {

// Split filename into (stem, extension) where extension includes the dot.
// "CoreLog.log" -> ("CoreLog", ".log")
// "CoreLog" -> ("CoreLog", "")
// "/path/to/CoreLog.log" -> ("/path/to/CoreLog", ".log")
std::pair<std::string, std::string>
SplitFilename(const std::string& filename)
{
   namespace fs = std::filesystem;
   fs::path p(filename);
   std::string ext = p.extension().string();
   std::string stem = filename.substr(0, filename.size() - ext.size());
   return {stem, ext};
}

std::string FormatLocalTimeForFilename()
{
   auto now = std::chrono::system_clock::now();
   auto secs = std::chrono::duration_cast<std::chrono::seconds>(
         now.time_since_epoch());

   std::time_t t(secs.count());
   std::tm* ptm;
#ifdef _WIN32
   ptm = std::localtime(&t);
#else
   std::tm tmstruct;
   ptm = localtime_r(&t, &tmstruct);
#endif

   char buf[32];
   std::strftime(buf, sizeof(buf), "%Y%m%dT%H%M%S", ptm);
   return buf;
}

bool IsRotatedFile(const std::string& name, const std::string& stem,
      const std::string& ext)
{
   // Expected: {stem}_{YYYYMMDD}T{HHMMSS}{ext}
   // The timestamp portion is exactly 15 characters: 8 date + T + 6 time
   const std::size_t timestampLen = 15;
   const std::size_t expectedLen =
         stem.size() + 1 + timestampLen + ext.size(); // +1 for underscore
   if (name.size() != expectedLen)
      return false;
   if (name.compare(0, stem.size(), stem) != 0)
      return false;
   if (name[stem.size()] != '_')
      return false;
   // Verify timestamp portion is digits and 'T'
   std::size_t tsStart = stem.size() + 1;
   for (std::size_t i = 0; i < timestampLen; ++i)
   {
      char c = name[tsStart + i];
      if (i == 8) // Position of 'T'
      {
         if (c != 'T')
            return false;
      }
      else
      {
         if (c < '0' || c > '9')
            return false;
      }
   }
   if (!ext.empty() &&
         name.compare(tsStart + timestampLen, ext.size(), ext) != 0)
      return false;
   return true;
}

} // anonymous namespace

std::string
MakeRotatedFilename(const std::string& filename)
{
   auto [stem, ext] = SplitFilename(filename);
   return stem + "_" + FormatLocalTimeForFilename() + ext;
}

void
DeleteExcessRotatedFiles(const std::string& filename, int maxBackupFiles)
{
   namespace fs = std::filesystem;

   if (maxBackupFiles <= 0)
      return;

   fs::path filePath(filename);
   fs::path dir = filePath.parent_path();
   if (dir.empty())
      dir = ".";

   auto [stem, ext] = SplitFilename(filePath.filename().string());

   std::vector<std::string> rotatedFiles;
   std::error_code ec;
   for (const auto& entry : fs::directory_iterator(dir, ec))
   {
      if (!entry.is_regular_file(ec))
         continue;
      std::string name = entry.path().filename().string();
      if (IsRotatedFile(name, stem, ext))
         rotatedFiles.push_back(name);
   }

   if (static_cast<int>(rotatedFiles.size()) <= maxBackupFiles)
      return;

   // Sort ascending by name (chronological due to fixed-width timestamp)
   std::sort(rotatedFiles.begin(), rotatedFiles.end());

   int toDelete = static_cast<int>(rotatedFiles.size()) - maxBackupFiles;
   for (int i = 0; i < toDelete; ++i)
   {
      fs::remove(dir / rotatedFiles[i], ec);
   }
}

} // namespace logging
} // namespace internal
} // namespace mmcore
