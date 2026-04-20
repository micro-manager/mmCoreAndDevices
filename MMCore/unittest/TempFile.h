#pragma once

#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>

#ifdef _WIN32
#include <Windows.h>
#else
#include <unistd.h>
#endif

class TempFile {
public:
   explicit TempFile(const std::string& contents) {
#ifdef _WIN32
      char dir[MAX_PATH];
      if (GetTempPathA(MAX_PATH, dir) == 0)
         throw std::runtime_error("GetTempPathA failed");
      char path[MAX_PATH];
      if (GetTempFileNameA(dir, "mmc", 0, path) == 0)
         throw std::runtime_error("GetTempFileNameA failed");
      path_ = path;
      std::ofstream ofs(path_, std::ios::binary);
      if (!ofs)
         throw std::runtime_error("Failed to open temp file for writing");
      ofs.write(contents.data(), contents.size());
#else
      std::string tmpl = []() -> std::string {
         const char* dir = std::getenv("TMPDIR");
         if (dir && dir[0] != '\0')
            return std::string(dir) + "/mmcore-test-XXXXXX";
         return "/tmp/mmcore-test-XXXXXX";
      }();
      int fd = mkstemp(tmpl.data());
      if (fd < 0)
         throw std::runtime_error("mkstemp failed");
      auto n = write(fd, contents.data(), contents.size());
      (void)n;
      close(fd);
      path_ = tmpl;
#endif
   }

   ~TempFile() { std::remove(path_.c_str()); }

   TempFile(const TempFile&) = delete;
   TempFile& operator=(const TempFile&) = delete;

   const std::string& getPath() const { return path_; }

private:
   std::string path_;
};
