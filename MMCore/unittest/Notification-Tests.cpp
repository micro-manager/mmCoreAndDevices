#include <catch2/catch_all.hpp>

#include "MMCore.h"
#include "MMEventCallback.h"
#include "MockDeviceUtils.h"
#include "Notification.h"
#include "NotificationQueue.h"
#include "StubDevices.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace mmi = mmcore::internal;
namespace notif = mmcore::internal::notification;

// --- NotificationQueue tests ---

TEST_CASE("NotificationQueue: Push and WaitAndPop", "[NotificationQueue]")
{
   mmi::NotificationQueue queue;
   queue.Push(notif::PropertiesChanged{});
   auto n = queue.WaitAndPop();
   REQUIRE(n.has_value());
   CHECK(std::holds_alternative<notif::PropertiesChanged>(*n));
}

TEST_CASE("NotificationQueue: WaitAndPop blocks until Push",
   "[NotificationQueue]")
{
   mmi::NotificationQueue queue;
   std::atomic<bool> popped{false};

   std::thread consumer([&] {
      auto n = queue.WaitAndPop();
      REQUIRE(n.has_value());
      CHECK(std::holds_alternative<notif::PropertyChanged>(*n));
      popped = true;
   });

   // Give consumer time to block
   std::this_thread::sleep_for(std::chrono::milliseconds(20));
   CHECK_FALSE(popped.load());

   queue.Push(notif::PropertyChanged{"dev", "prop", "val"});
   consumer.join();
   CHECK(popped.load());
}

TEST_CASE("NotificationQueue: multiple producers",
   "[NotificationQueue]")
{
   mmi::NotificationQueue queue;
   constexpr int numProducers = 4;
   constexpr int numPerProducer = 100;

   std::atomic<int> consumed{0};
   std::thread consumer([&] {
      while (auto n = queue.WaitAndPop())
         ++consumed;
   });

   std::vector<std::thread> producers;
   for (int i = 0; i < numProducers; ++i) {
      producers.emplace_back([&queue] {
         for (int j = 0; j < numPerProducer; ++j)
            queue.Push(notif::PropertiesChanged{});
      });
   }
   for (auto& t : producers)
      t.join();

   // Give consumer time to drain, then interrupt
   std::this_thread::sleep_for(std::chrono::milliseconds(50));
   queue.RequestInterrupt();
   consumer.join();
   CHECK(consumed.load() == numProducers * numPerProducer);
}

TEST_CASE("NotificationQueue: RequestInterrupt unblocks WaitAndPop",
   "[NotificationQueue]")
{
   mmi::NotificationQueue queue;
   std::atomic<bool> gotNullopt{false};

   std::thread consumer([&] {
      auto n = queue.WaitAndPop();
      CHECK_FALSE(n.has_value());
      gotNullopt = true;
   });

   std::this_thread::sleep_for(std::chrono::milliseconds(20));
   queue.RequestInterrupt();
   consumer.join();
   CHECK(gotNullopt.load());

   // Queue is still functional after interrupt
   queue.Push(notif::PropertiesChanged{});
   auto n = queue.WaitAndPop();
   REQUIRE(n.has_value());
   CHECK(std::holds_alternative<notif::PropertiesChanged>(*n));
}

TEST_CASE("NotificationQueue: RequestInterrupt preserves pending items",
   "[NotificationQueue]")
{
   mmi::NotificationQueue queue;
   queue.Push(notif::PropertiesChanged{});
   queue.Push(notif::PropertyChanged{"dev", "prop", "val"});

   queue.RequestInterrupt();

   auto n = queue.WaitAndPop();
   CHECK_FALSE(n.has_value());

   n = queue.WaitAndPop();
   REQUIRE(n.has_value());
   CHECK(std::holds_alternative<notif::PropertiesChanged>(*n));

   n = queue.WaitAndPop();
   REQUIRE(n.has_value());
   CHECK(std::holds_alternative<notif::PropertyChanged>(*n));
}

// --- Notification Dispatch tests ---

namespace {

class RecordingCallback : public MMEventCallback {
public:
   struct Call {
      std::string method;
      std::vector<std::string> stringArgs;
      std::vector<double> doubleArgs;
      bool boolArg = false;
   };

   std::vector<Call> calls;

   void onPropertiesChanged() override {
      calls.push_back({"onPropertiesChanged", {}, {}, false});
   }
   void onPropertyChanged(const char* name, const char* propName,
      const char* propValue) override {
      calls.push_back(
         {"onPropertyChanged", {name, propName, propValue}, {}, false});
   }
   void onConfigGroupChanged(const char* groupName,
      const char* newConfigName) override {
      calls.push_back(
         {"onConfigGroupChanged", {groupName, newConfigName}, {}, false});
   }
   void onSystemConfigurationLoaded() override {
      calls.push_back({"onSystemConfigurationLoaded", {}, {}, false});
   }
   void onPixelSizeChanged(double newPixelSizeUm) override {
      calls.push_back(
         {"onPixelSizeChanged", {}, {newPixelSizeUm}, false});
   }
   void onPixelSizeAffineChanged(double v0, double v1, double v2,
      double v3, double v4, double v5) override {
      calls.push_back(
         {"onPixelSizeAffineChanged", {}, {v0, v1, v2, v3, v4, v5}, false});
   }
   void onStagePositionChanged(const char* name, double pos) override {
      calls.push_back(
         {"onStagePositionChanged", {name}, {pos}, false});
   }
   void onXYStagePositionChanged(const char* name, double xpos,
      double ypos) override {
      calls.push_back(
         {"onXYStagePositionChanged", {name}, {xpos, ypos}, false});
   }
   void onExposureChanged(const char* name, double newExposure) override {
      calls.push_back(
         {"onExposureChanged", {name}, {newExposure}, false});
   }
   void onSLMExposureChanged(const char* name,
      double newExposure) override {
      calls.push_back(
         {"onSLMExposureChanged", {name}, {newExposure}, false});
   }
   void onShutterOpenChanged(const char* name, bool open) override {
      calls.push_back({"onShutterOpenChanged", {name}, {}, open});
   }
   void onImageSnapped(const char* cameraLabel) override {
      calls.push_back({"onImageSnapped", {cameraLabel}, {}, false});
   }
   void onSequenceAcquisitionStarted(const char* cameraLabel) override {
      calls.push_back(
         {"onSequenceAcquisitionStarted", {cameraLabel}, {}, false});
   }
   void onSequenceAcquisitionStopped(const char* cameraLabel) override {
      calls.push_back(
         {"onSequenceAcquisitionStopped", {cameraLabel}, {}, false});
   }
   void onChannelGroupChanged(
      const char* newChannelGroupName) override {
      calls.push_back(
         {"onChannelGroupChanged", {newChannelGroupName}, {}, false});
   }
   void onImageAddedToBuffer(const char* cameraLabel) override {
      calls.push_back(
         {"onImageAddedToBuffer", {cameraLabel}, {}, false});
   }
};

TEST_CASE("Dispatch PropertiesChanged", "[Notification][Dispatch]")
{
   RecordingCallback cb;
   mmi::DispatchNotification(notif::PropertiesChanged{}, cb);
   REQUIRE(cb.calls.size() == 1);
   CHECK(cb.calls[0].method == "onPropertiesChanged");
}

TEST_CASE("Dispatch PropertyChanged", "[Notification][Dispatch]")
{
   RecordingCallback cb;
   mmi::DispatchNotification(
      notif::PropertyChanged{"Camera", "Exposure", "10.0"}, cb);
   REQUIRE(cb.calls.size() == 1);
   CHECK(cb.calls[0].method == "onPropertyChanged");
   REQUIRE(cb.calls[0].stringArgs.size() == 3);
   CHECK(cb.calls[0].stringArgs[0] == "Camera");
   CHECK(cb.calls[0].stringArgs[1] == "Exposure");
   CHECK(cb.calls[0].stringArgs[2] == "10.0");
}

TEST_CASE("Dispatch ConfigGroupChanged", "[Notification][Dispatch]")
{
   RecordingCallback cb;
   mmi::DispatchNotification(notif::ConfigGroupChanged{"Channel", "DAPI"}, cb);
   REQUIRE(cb.calls.size() == 1);
   CHECK(cb.calls[0].method == "onConfigGroupChanged");
   CHECK(cb.calls[0].stringArgs[0] == "Channel");
   CHECK(cb.calls[0].stringArgs[1] == "DAPI");
}

TEST_CASE("Dispatch PixelSizeChanged", "[Notification][Dispatch]")
{
   RecordingCallback cb;
   mmi::DispatchNotification(notif::PixelSizeChanged{0.325}, cb);
   REQUIRE(cb.calls.size() == 1);
   CHECK(cb.calls[0].method == "onPixelSizeChanged");
   CHECK(cb.calls[0].doubleArgs[0] == Catch::Approx(0.325));
}

TEST_CASE("Dispatch PixelSizeAffineChanged", "[Notification][Dispatch]")
{
   RecordingCallback cb;
   mmi::DispatchNotification(
      notif::PixelSizeAffineChanged{1.0, 0.0, 0.0, 0.0, 1.0, 0.0}, cb);
   REQUIRE(cb.calls.size() == 1);
   CHECK(cb.calls[0].method == "onPixelSizeAffineChanged");
   REQUIRE(cb.calls[0].doubleArgs.size() == 6);
   CHECK(cb.calls[0].doubleArgs[0] == Catch::Approx(1.0));
   CHECK(cb.calls[0].doubleArgs[4] == Catch::Approx(1.0));
}

TEST_CASE("Dispatch StagePositionChanged", "[Notification][Dispatch]")
{
   RecordingCallback cb;
   mmi::DispatchNotification(notif::StagePositionChanged{"Z", 42.5}, cb);
   REQUIRE(cb.calls.size() == 1);
   CHECK(cb.calls[0].method == "onStagePositionChanged");
   CHECK(cb.calls[0].stringArgs[0] == "Z");
   CHECK(cb.calls[0].doubleArgs[0] == Catch::Approx(42.5));
}

TEST_CASE("Dispatch XYStagePositionChanged", "[Notification][Dispatch]")
{
   RecordingCallback cb;
   mmi::DispatchNotification(notif::XYStagePositionChanged{"XY", 1.0, 2.0}, cb);
   REQUIRE(cb.calls.size() == 1);
   CHECK(cb.calls[0].method == "onXYStagePositionChanged");
   CHECK(cb.calls[0].stringArgs[0] == "XY");
   CHECK(cb.calls[0].doubleArgs[0] == Catch::Approx(1.0));
   CHECK(cb.calls[0].doubleArgs[1] == Catch::Approx(2.0));
}

TEST_CASE("Dispatch ExposureChanged", "[Notification][Dispatch]")
{
   RecordingCallback cb;
   mmi::DispatchNotification(notif::ExposureChanged{"Camera", 50.0}, cb);
   REQUIRE(cb.calls.size() == 1);
   CHECK(cb.calls[0].method == "onExposureChanged");
   CHECK(cb.calls[0].stringArgs[0] == "Camera");
   CHECK(cb.calls[0].doubleArgs[0] == Catch::Approx(50.0));
}

TEST_CASE("Dispatch SLMExposureChanged", "[Notification][Dispatch]")
{
   RecordingCallback cb;
   mmi::DispatchNotification(notif::SLMExposureChanged{"SLM", 25.0}, cb);
   REQUIRE(cb.calls.size() == 1);
   CHECK(cb.calls[0].method == "onSLMExposureChanged");
   CHECK(cb.calls[0].stringArgs[0] == "SLM");
   CHECK(cb.calls[0].doubleArgs[0] == Catch::Approx(25.0));
}

TEST_CASE("Dispatch ShutterOpenChanged", "[Notification][Dispatch]")
{
   RecordingCallback cb;
   mmi::DispatchNotification(notif::ShutterOpenChanged{"Shutter", true}, cb);
   REQUIRE(cb.calls.size() == 1);
   CHECK(cb.calls[0].method == "onShutterOpenChanged");
   CHECK(cb.calls[0].stringArgs[0] == "Shutter");
   CHECK(cb.calls[0].boolArg == true);
}

TEST_CASE("Dispatch ImageSnapped", "[Notification][Dispatch]")
{
   RecordingCallback cb;
   mmi::DispatchNotification(notif::ImageSnapped{"Camera"}, cb);
   REQUIRE(cb.calls.size() == 1);
   CHECK(cb.calls[0].method == "onImageSnapped");
   CHECK(cb.calls[0].stringArgs[0] == "Camera");
}

TEST_CASE("Dispatch SequenceAcquisitionStarted",
   "[Notification][Dispatch]")
{
   RecordingCallback cb;
   mmi::DispatchNotification(notif::SequenceAcquisitionStarted{"Camera"}, cb);
   REQUIRE(cb.calls.size() == 1);
   CHECK(cb.calls[0].method == "onSequenceAcquisitionStarted");
   CHECK(cb.calls[0].stringArgs[0] == "Camera");
}

TEST_CASE("Dispatch SequenceAcquisitionStopped",
   "[Notification][Dispatch]")
{
   RecordingCallback cb;
   mmi::DispatchNotification(notif::SequenceAcquisitionStopped{"Camera"}, cb);
   REQUIRE(cb.calls.size() == 1);
   CHECK(cb.calls[0].method == "onSequenceAcquisitionStopped");
   CHECK(cb.calls[0].stringArgs[0] == "Camera");
}

TEST_CASE("Dispatch SystemConfigurationLoaded",
   "[Notification][Dispatch]")
{
   RecordingCallback cb;
   mmi::DispatchNotification(notif::SystemConfigurationLoaded{}, cb);
   REQUIRE(cb.calls.size() == 1);
   CHECK(cb.calls[0].method == "onSystemConfigurationLoaded");
}

TEST_CASE("Dispatch ChannelGroupChanged", "[Notification][Dispatch]")
{
   RecordingCallback cb;
   mmi::DispatchNotification(notif::ChannelGroupChanged{"DAPI"}, cb);
   REQUIRE(cb.calls.size() == 1);
   CHECK(cb.calls[0].method == "onChannelGroupChanged");
   CHECK(cb.calls[0].stringArgs[0] == "DAPI");
}
TEST_CASE("Dispatch ImageAddedToBuffer", "[Notification][Dispatch]")
{
   RecordingCallback cb;
   mmi::DispatchNotification(notif::ImageAddedToBuffer{"Camera"}, cb);
   REQUIRE(cb.calls.size() == 1);
   CHECK(cb.calls[0].method == "onImageAddedToBuffer");
   CHECK(cb.calls[0].stringArgs[0] == "Camera");
}

// --- Integration: registerCallback + postNotification ---

class WaitableCallback : public MMEventCallback {
public:
   std::mutex mutex;
   std::condition_variable cv;
   bool systemConfigLoaded = false;

   void onSystemConfigurationLoaded() override {
      std::lock_guard<std::mutex> lock(mutex);
      systemConfigLoaded = true;
      cv.notify_one();
   }

   bool waitForSystemConfigLoaded(std::chrono::milliseconds timeout) {
      std::unique_lock<std::mutex> lock(mutex);
      return cv.wait_for(lock, timeout,
         [this] { return systemConfigLoaded; });
   }
};

} // namespace

TEST_CASE("registerCallback delivers notifications asynchronously",
   "[Notification][Integration]")
{
   WaitableCallback cb;
   CMMCore core;
   core.registerCallback(&cb);

   core.unloadAllDevices();

   CHECK(cb.waitForSystemConfigLoaded(std::chrono::milliseconds(1000)));
   core.registerCallback(nullptr);
}

TEST_CASE("registerCallback(nullptr) stops delivery",
   "[Notification][Integration]")
{
   RecordingCallback cb;
   CMMCore core;
   core.registerCallback(&cb);
   core.registerCallback(nullptr);

   // No crash, no hanging
}

TEST_CASE("registerCallback swap preserves pending notifications",
   "[Notification][Integration]")
{
   struct SlowCallback : public MMEventCallback {
      std::atomic<int> sysConfigCount{0};
      void onSystemConfigurationLoaded() override {
         std::this_thread::sleep_for(std::chrono::milliseconds(200));
         ++sysConfigCount;
      }
   };

   SlowCallback cbA;
   WaitableCallback cbB;
   CMMCore core;

   core.registerCallback(&cbA);

   // Each call posts a SystemConfigurationLoaded notification.
   // cbA sleeps 200ms per delivery, so notifications accumulate.
   core.unloadAllDevices();
   core.unloadAllDevices();
   core.unloadAllDevices();

   // Swap immediately — cbA is still processing the first notification,
   // so at least one notification remains in the queue.
   core.registerCallback(&cbB);

   CHECK(cbB.waitForSystemConfigLoaded(std::chrono::milliseconds(1000)));

   core.registerCallback(nullptr);
}
