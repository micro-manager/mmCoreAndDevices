#pragma once

#include <string>

namespace mmcore {
namespace internal {
namespace logging {

std::string MakeRotatedFilename(const std::string& filename);

void DeleteExcessRotatedFiles(const std::string& filename,
      int maxBackupFiles);

} // namespace logging
} // namespace internal
} // namespace mmcore
