#include "SingleThread.h"

#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <functional>

SingleThread::SingleThread() : running(true) 
{
  thread = std::thread(&SingleThread::run, this);
}

SingleThread::~SingleThread() 
{
   {
      std::unique_lock<std::mutex> lock(mutex);
      running = false;
      condition.notify_one();
   }
   thread.join();
}

void SingleThread::enqueue(std::function<void()> task) {
   {
      std::unique_lock<std::mutex> lock(mutex);
      tasks.push(task);
   }
   condition.notify_one();
}

void SingleThread::run() {
   while (running) 
   {
      std::function<void()> task;
      {
         std::unique_lock<std::mutex> lock(mutex);
         condition.wait(lock, [this] { return !tasks.empty() || !running; });
         if (!tasks.empty()) 
         {
            task = tasks.front();
            tasks.pop();
         }
      }
      if (task) {
         task();
      }
  }
}


