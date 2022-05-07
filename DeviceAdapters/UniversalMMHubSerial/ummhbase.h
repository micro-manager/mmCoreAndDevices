#pragma once

#include "MMDevice.h"

class UmmhDeviceUtilities
{
public:
	UmmhDeviceUtilities() {};
	~UmmhDeviceUtilities() {};

	void SetBusy(bool val) { busy_ = val; }
	bool GetBusy() { return busy_; }
	MM::MMTime GetTimeout() { return timeout_; }
	void SetTimeout(MM::MMTime val) { timeout_ = val; }
	MM::MMTime GetLastCommandTime() { return lastCommandTime_; }
	void SetLastCommandTime(MM::MMTime val) { lastCommandTime_ = val; }
	void SetUpdating(bool val) { updating_ = val; }
	bool IsUpdating() { return updating_; }

private:
	bool busy_;
	MM::MMTime timeout_;
	MM::MMTime lastCommandTime_;
	bool updating_;
};

