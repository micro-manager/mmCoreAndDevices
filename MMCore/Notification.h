// Notification types and dispatch for async notification delivery.
//
// LICENSE:       This file is distributed under the "Lesser GPL" (LGPL)
//                license. License text is included with the source
//                distribution.

#pragma once

#include "MMEventCallback.h"

#include <string>
#include <variant>

namespace mmcore {
namespace internal {

namespace notification {

struct PropertiesChanged {};
struct PropertyChanged {
   std::string deviceLabel;
   std::string propertyName;
   std::string propertyValue;
};
struct ConfigGroupChanged {
   std::string groupName;
   std::string configName;
};
struct PixelSizeChanged {
   double pixelSizeUm;
};
struct PixelSizeAffineChanged {
   double v0, v1, v2, v3, v4, v5;
};
struct StagePositionChanged {
   std::string deviceLabel;
   double position;
};
struct XYStagePositionChanged {
   std::string deviceLabel;
   double x, y;
};
struct ExposureChanged {
   std::string deviceLabel;
   double exposure;
};
struct SLMExposureChanged {
   std::string deviceLabel;
   double exposure;
};
struct ShutterOpenChanged {
   std::string deviceLabel;
   bool open;
};
struct ImageSnapped {
   std::string cameraLabel;
};
struct SequenceAcquisitionStarted {
   std::string cameraLabel;
};
struct SequenceAcquisitionStopped {
   std::string cameraLabel;
};
struct SystemConfigurationLoaded {};
struct ChannelGroupChanged {
   std::string channelGroupName;
};
struct ImageAddedToBuffer {
   std::string cameraLabel;
};

} // namespace notification

using Notification = std::variant<
   notification::PropertiesChanged,
   notification::PropertyChanged,
   notification::ConfigGroupChanged,
   notification::PixelSizeChanged,
   notification::PixelSizeAffineChanged,
   notification::StagePositionChanged,
   notification::XYStagePositionChanged,
   notification::ExposureChanged,
   notification::SLMExposureChanged,
   notification::ShutterOpenChanged,
   notification::ImageSnapped,
   notification::SequenceAcquisitionStarted,
   notification::SequenceAcquisitionStopped,
   notification::SystemConfigurationLoaded,
   notification::ChannelGroupChanged,
   notification::ImageAddedToBuffer
>;

namespace detail {
template <class... Ts> struct overloaded : Ts... { using Ts::operator()...; };
template <class... Ts> overloaded(Ts...) -> overloaded<Ts...>;
} // namespace detail

inline void DispatchNotification(const Notification& notification,
      MMEventCallback& cb) {
   std::visit(detail::overloaded{
      [&](const notification::PropertiesChanged&) {
         cb.onPropertiesChanged();
      },
      [&](const notification::PropertyChanged& n) {
         cb.onPropertyChanged(
            n.deviceLabel.c_str(), n.propertyName.c_str(),
            n.propertyValue.c_str());
      },
      [&](const notification::ConfigGroupChanged& n) {
         cb.onConfigGroupChanged(
            n.groupName.c_str(), n.configName.c_str());
      },
      [&](const notification::PixelSizeChanged& n) {
         cb.onPixelSizeChanged(n.pixelSizeUm);
      },
      [&](const notification::PixelSizeAffineChanged& n) {
         cb.onPixelSizeAffineChanged(n.v0, n.v1, n.v2, n.v3, n.v4, n.v5);
      },
      [&](const notification::StagePositionChanged& n) {
         cb.onStagePositionChanged(n.deviceLabel.c_str(), n.position);
      },
      [&](const notification::XYStagePositionChanged& n) {
         cb.onXYStagePositionChanged(
            n.deviceLabel.c_str(), n.x, n.y);
      },
      [&](const notification::ExposureChanged& n) {
         cb.onExposureChanged(n.deviceLabel.c_str(), n.exposure);
      },
      [&](const notification::SLMExposureChanged& n) {
         cb.onSLMExposureChanged(n.deviceLabel.c_str(), n.exposure);
      },
      [&](const notification::ShutterOpenChanged& n) {
         cb.onShutterOpenChanged(n.deviceLabel.c_str(), n.open);
      },
      [&](const notification::ImageSnapped& n) {
         cb.onImageSnapped(n.cameraLabel.c_str());
      },
      [&](const notification::SequenceAcquisitionStarted& n) {
         cb.onSequenceAcquisitionStarted(n.cameraLabel.c_str());
      },
      [&](const notification::SequenceAcquisitionStopped& n) {
         cb.onSequenceAcquisitionStopped(n.cameraLabel.c_str());
      },
      [&](const notification::SystemConfigurationLoaded&) {
         cb.onSystemConfigurationLoaded();
      },
      [&](const notification::ChannelGroupChanged& n) {
         cb.onChannelGroupChanged(n.channelGroupName.c_str());
      },
      [&](const notification::ImageAddedToBuffer& n) {
         cb.onImageAddedToBuffer(n.cameraLabel.c_str());
      },
   }, notification);
}

} // namespace internal
} // namespace mmcore
