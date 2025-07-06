#include "MMCore.h"
#include "AttachedStorageThread.h"

AttachedStorageThread::~AttachedStorageThread() {
   // Ensure thread is properly stopped if not already
   if (running_) {
      stop();
   }

   // Make sure we join the thread if it's joinable
   if (thread_.joinable()) {
      thread_.join();
   }
}

// Start the worker thread
void AttachedStorageThread::start() {
   // Only start if not already running
   if (!running_) {
      shouldStop_ = false;
      thread_ = std::thread(&AttachedStorageThread::workFunction, this);
   }
}

// Gracefully stop the worker thread
void AttachedStorageThread::stop() {
   if (running_) {
      // Set stop flag
      shouldStop_ = true;

      // Notify the condition variable to wake up the thread
      {
         std::lock_guard<std::mutex> lock(mutex_);
         cv_.notify_all();
      }

      // Wait for thread to finish
      if (thread_.joinable()) {
         thread_.join();
      }
   }
}

// Check if thread is running
bool AttachedStorageThread::isRunning() const {
   return running_.load();
}   // The actual work function that runs in the thread


void AttachedStorageThread::workFunction() {
   // Set running flag
   running_ = true;
   errorMessage_.clear();
   
   std::ostringstream os;
   os << "Attached storage thread started on dataset handle " << core_->getAttachedDataset();
   core_->logMessage(os.str().c_str());

   try {
      while (!shouldStop_.load()) {
         // Pick images from storage
         if (core_->getRemainingImageCount() > 0)
         {
            std::vector<long> coordinates;
            core_->appendNextToDataset(core_->getAttachedDataset(), coordinates, "", 0);
         }
         else
         {
            std::this_thread::sleep_for(std::chrono::microseconds(100));
         }
      }
   }
   catch (CMMError& err)
   {
      errorMessage_ = err.getFullMsg();
   }

   // Clear running flag
   running_ = false;
}

