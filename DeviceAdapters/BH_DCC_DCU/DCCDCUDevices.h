#pragma once

#include "DCCDCUInterface.h"

#include "DeviceBase.h"

#include <cassert>
#include <cstddef>
#include <exception>
#include <memory>
#include <string>
#include <utility>

template <DCCOrDCU Model>
inline auto ModelName() -> std::string {
	if (Model == DCCOrDCU::DCC) {
		return "DCC";
	} else if (Model == DCCOrDCU::DCU) {
		return "DCU";
	} else {
		assert(false);
	}
}

template <DCCOrDCU Model>
inline auto ModelDescription() -> std::string {
	if (Model == DCCOrDCU::DCC) {
		return "Becker & Hickl DCC-100 detector control cards";
	} else if (Model == DCCOrDCU::DCU) {
		return "Becker & Hickl DCU USB detector control units";
	} else {
		assert(false);
	}
}

enum class Errors {
	INIT_FAILED = 20001,
	CANNOT_GET_MODULE_STATUS_OR_INFO,
	MODULE_IN_USE,
	MODULE_INIT_MISC_FAILURE,
};

template <DCCOrDCU Model, typename SetFunc>
inline void RegisterErrorMessages(SetFunc f) {
	auto f2 = [&](Errors code, std::string const& text) {
		f(static_cast<int>(code), text.c_str());
	};
	const std::string model = ModelName<Model>();
	f2(Errors::INIT_FAILED,
		model + " init failed (see CoreLog for internal details)");
	f2(Errors::CANNOT_GET_MODULE_STATUS_OR_INFO,
		"Failed to get " + model + " init status or module info");
	f2(Errors::MODULE_IN_USE,
		model + " module is in use by another application "
		"(or needs to be reset using the BH " + model + " app)");
	f2(Errors::MODULE_INIT_MISC_FAILURE,
		model + " module init failed (see CoreLog for internal details)");
}

template <DCCOrDCU Model>
inline auto HubDeviceName() -> std::string {
	return ModelName<Model>() + "Hub";
}

template <DCCOrDCU Model>
inline auto ModuleDeviceNamePrefix() -> std::string {
	return ModelName<Model>() + "Module";
}

template <DCCOrDCU Model>
inline auto ModuleDeviceName(short moduleNo) -> std::string {
	return ModuleDeviceNamePrefix<Model>() + std::to_string(moduleNo + 1);
}

// Return module no if name matches (prefix + number); else return -1.
template <DCCOrDCU Model>
inline auto DeviceNameToModuleNo(std::string const &name) -> short {
	const std::string prefix = ModuleDeviceNamePrefix<Model>();
	if (name.rfind(prefix, 0) != 0) {
		return -1;
	}
	const auto index = name.substr(prefix.size());
	std::size_t len = 0;
	std::size_t i = 0;
	try {
		i = std::stoul(index, &len);
	} catch (std::exception const &) {
		return -1;
	}
	if (len < index.size()) {
		return -1;
	}
	if (i == 0 || i > DCCDCUConfig<Model>::MaxNrModules) {
		return -1;
	}
	return static_cast<short>(i - 1);
}

template <DCCOrDCU Model>
class DCCDCUHubDevice : public HubBase<DCCDCUHubDevice<Model>> {
	using Config = DCCDCUConfig<Model>;
	std::shared_ptr<DCCDCUInterface<Model>> interface_;
	std::string deviceName_;

public:
	DCCDCUHubDevice(std::string name) : deviceName_(std::move(name)) {
		RegisterErrorMessages<Model>([this](int code, const char *text) {
			this->SetErrorText(code, text);
		});

		this->CreateStringProperty("SimulateDevice", "No", false, nullptr, true);
		this->AddAllowedValue("SimulateDevice", "No");
		this->AddAllowedValue("SimulateDevice", "Yes");
		for (short i = 0; i < Config::MaxNrModules; ++i) {
			const auto prop = "UseModule" + std::to_string(i + 1);
			this->CreateStringProperty(prop.c_str(), i == 0 ? "Yes" : "No", false, nullptr, true);
			this->AddAllowedValue(prop.c_str(), "No");
			this->AddAllowedValue(prop.c_str(), "Yes");
		}
	}

	auto Initialize() -> int final {
		const bool simulate = RequestedSimulation();
		const auto moduleSet = RequestedModuleSet();
		interface_ = std::make_shared<DCCDCUInterface<Model>>(moduleSet, simulate);
		const auto initErr = interface_->GetInitError();
		// It is not clear whether we will get an error when some, but not all,
		// of the modules failed to initialize. So generally we just log and
		// continue and handle the init status in each module. But let's give
		// up when it is clearly an error not specific to individual modules.
		if (initErr) {
			this->LogMessage(ModelName<Model>() + " init failed with error: " +
				DCCDCUGetErrorString(initErr));
		}
		switch (initErr) {
		case DCC_OPEN_FILE:
		case DCC_FILE_NVALID:
		case DCC_MEM_ALLOC:
		case DCC_READ_STR:
			return static_cast<int>(Errors::INIT_FAILED);
		}

		CreateProperties();
		return DEVICE_OK;
	}

	auto Shutdown() -> int final {
		interface_.reset();
		return DEVICE_OK;
	}

	void GetName(char *name) const final {
		CDeviceUtils::CopyLimitedString(name, deviceName_.c_str());
	}
	auto Busy() -> bool final { return false; }

	auto DetectInstalledDevices() -> int final {
		const auto moduleSet = RequestedModuleSet();
		for (short i = 0; i < Config::MaxNrModules; ++i) {
			if (moduleSet[i]) {
				this->AddInstalledDevice(
					CreateDevice(ModuleDeviceName<Model>(i).c_str()));
			}
		}
		return DEVICE_OK;
	}

	auto GetDCCDCUInterface() -> std::shared_ptr<DCCDCUInterface<Model>> {
		return interface_;
	}

private:
	auto RequestedSimulation() const -> bool {
		std::array<char, MM::MaxStrLength + 1> buf;
		this->GetProperty("SimulateDevice", buf.data());
		const std::string yesno = buf.data();
		return yesno == "Yes";
	}

	auto RequestedModuleSet() const -> std::bitset<Config::MaxNrModules> {
		std::bitset<Config::MaxNrModules> ret;
		for (short i = 0; i < Config::MaxNrModules; ++i) {
			const auto prop = "UseModule" + std::to_string(i + 1);
			std::array<char, MM::MaxStrLength + 1> buf;
			this->GetProperty(prop.c_str(), buf.data());
			const std::string yesno = buf.data();
			if (yesno == "Yes") {
				ret.set(i);
			}
		}
		return ret;
	}

	void CreateProperties() {
		this->CreateStringProperty("Simulated",
			interface_->IsSimulating() ? "Yes" : "No", true);
	}
};

template <DCCOrDCU Model>
class DCCDCUModuleDevice : public CGenericBase<DCCDCUModuleDevice<Model>> {
	using Config = DCCDCUConfig<Model>;
	std::shared_ptr<DCCDCUModule<Model>> module_;
	short moduleNo_;
	std::string deviceName_;

public:
	DCCDCUModuleDevice(std::string name, short moduleNo) :
		moduleNo_(moduleNo),
		deviceName_(std::move(name))
	{
		RegisterErrorMessages<Model>([this](int code, const char *text) {
			this->SetErrorText(code, text);
		});
	}

	auto Initialize() -> int final {
		auto* hub = static_cast<DCCDCUHubDevice<Model>*>(this->GetParentHub());
		auto iface = hub->GetDCCDCUInterface();
		module_ = iface->GetModule(moduleNo_);
		if (not module_->IsUsable()) {
			if (module_->IsActive()) {
				// Active but failed to get init status or module info; this is
				// unexpected.
				return static_cast<int>(Errors::CANNOT_GET_MODULE_STATUS_OR_INFO);
			}

			// The most common error is that the module is in use by another
			// app, so give it a specific error code and message.
			switch (module_->InitStatus()) {
			case INIT_DCC_MOD_IN_USE:
				return static_cast<int>(Errors::MODULE_IN_USE);
			default:
				this->LogMessage(ModelName<Model>() + " module init failed with init status " +
					std::to_string(module_->InitStatus()));
				return static_cast<int>(Errors::MODULE_INIT_MISC_FAILURE);
			}
		}

		CreateModInfoProperties(module_->ModInfo());

		CreateProperties();
		return DEVICE_OK;
	}

	auto Shutdown() -> int final {
		module_->Close();
		return DEVICE_OK;
	}

	void GetName(char *name) const final {
		CDeviceUtils::CopyLimitedString(name, deviceName_.c_str());
	}

	auto Busy() -> bool final { return false; }

private:
	void CreateModInfoProperties(typename Config::ModInfoType modInfo) {
		this->CreateIntegerProperty("ModuleNumber", moduleNo_ + 1, true);
		this->CreateIntegerProperty("ModuleType", modInfo.ModuleType(), true);
		if (Model == DCCOrDCU::DCC) {
			this->CreateIntegerProperty("BusNumber", modInfo.BusNumber(), true);
			this->CreateIntegerProperty("SlotNumber", modInfo.SlotNumber(), true);
			this->CreateIntegerProperty("BaseAddress", modInfo.BaseAdr(), true);
		}
		if (Model == DCCOrDCU::DCU) {
			this->CreateIntegerProperty("ComPortNumber", modInfo.ComPortNo(), true);
		}
		this->CreateStringProperty("SerialNumber", modInfo.SerialNo().c_str(), true);
	}

	void CreateProperties() {
		// TODO
	}
};
