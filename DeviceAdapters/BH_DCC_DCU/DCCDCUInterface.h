#pragma once

#include "DCCDCUCommonFunctions.h"

#include <algorithm>
#include <bitset>
#include <cstdio>
#include <memory>
#include <string>

// Wrap DCC or DCU device control as RAII objects.

// The BH API is structured such that all DCC modules (or all DCU modules) are
// initialized and closed together (selection between hardware and simulation
// is also for all modules simultaneously).
// In order to wrap the category (DCC or DCU) as a HubDevice and each module as
// its peripheral, it is conveneint to also have an object for each module.

// Generate a temporary minimal .ini file
template <DCCOrDCU Model>
class DCCDCUIniFile {
	using Config = DCCDCUConfig<Model>;

	std::string fileName_;
	std::FILE *fp_;

public:
	static constexpr std::size_t MaxNrModules = Config::MaxNrModules;

	explicit DCCDCUIniFile(std::bitset<MaxNrModules> moduleSet, bool simulate) :
		fileName_([] {
			char nm[L_tmpnam];
			std::tmpnam(nm);
			return std::string(nm);
		}()),
		fp_(std::fopen(fileName_.c_str(), "w"))
	{
		using namespace std::string_literals;
		std::string iniContent;
		// Undocumented: The first line must be a comment starting with
		// "DCC100" or "DCCUSB"
		iniContent += "; "s + Config::IniHeaderComment + "\n\n"s;
		iniContent += "["s + Config::IniBaseTag + "]\n\n"s;
		iniContent += Config::IniSimulationKey + " = "s +
			std::to_string(simulate ? Config::SimulationMode : 0) + "\n\n"s;
		for (int i = 0; i < MaxNrModules; ++i) {
			iniContent += "["s + Config::IniModuleTagPrefix +
				std::to_string(i + 1) + "]\n\n"s;
			iniContent += "active = "s + (moduleSet[i] ? "1"s : "0"s) + "\n\n"s;
		}
		std::fwrite(iniContent.c_str(), 1, iniContent.size(), fp_);
		std::fclose(fp_);
	}

	~DCCDCUIniFile() {
		std::remove(fileName_.c_str());
	}

	auto GetFileName() const -> std::string { return fileName_; }
};

template <DCCOrDCU Model>
class DCCDCUInterface;

// Used with std::shared_ptr
template <DCCOrDCU Model>
class DCCDCUModule {
	using ParentType = DCCDCUInterface<Model>;
	using Config = DCCDCUConfig<Model>;
	std::shared_ptr<ParentType> parent_;
	short moduleNo_;

	bool isActive_;
	short errDuringConstruction_ = 0;
	short initStatus_ = 0;
	typename Config::ModInfoType modInfo_{};

public:
	explicit DCCDCUModule(std::shared_ptr<ParentType> parent, short moduleNo) :
		parent_(parent),
		moduleNo_(moduleNo),
		isActive_(Config::TestIfActive(moduleNo_) != 0)
	{
		auto err = Config::GetInitStatus(moduleNo_, &initStatus_);
		if (err) {
			errDuringConstruction_ = err;
			return;
		}

		if (initStatus_ == INIT_DCC_OK) {
			err = Config::GetModuleInfo(moduleNo_, &modInfo_);
			if (err) {
				errDuringConstruction_ = err;
				return;
			}
		}
	}

	void Close() {
		parent_->modules_[moduleNo_].reset();
	}

	auto IsActive() const -> bool { return isActive_; }

	auto IsUsable() const -> bool {
		return isActive_ && not errDuringConstruction_ && initStatus_ == INIT_DCC_OK;
	}

	auto InitStatus() const -> short { return initStatus_; }
	auto ModInfo() const { return modInfo_; }

	// TODO Dynamic features should call DCC API via parent (so that we can
	// add mutex)
};

// Used with std::shared_ptr
template <DCCOrDCU Model>
class DCCDCUInterface : public std::enable_shared_from_this<DCCDCUInterface<Model>> {
	friend class DCCDCUModule<Model>;
	using Config = DCCDCUConfig<Model>;

	short initError_;

	// Module objects are created lazily after construction because they need
	// access to shared_from_this().
	std::array<std::shared_ptr<DCCDCUModule<Model>>, Config::MaxNrModules> modules_;
	bool modulesCreated_ = false;

	void CreateModules() {
		auto shared_me = this->shared_from_this();
		for (short i = 0; i < MaxNrModules; ++i) {
			modules_[i] = std::make_shared<DCCDCUModule<Model>>(shared_me, i);
		}
		modulesCreated_ = true;
	}

public:
	static constexpr std::size_t MaxNrModules = Config::MaxNrModules;

	explicit DCCDCUInterface(std::bitset<MaxNrModules> moduleSet, bool simulate) {
		auto iniFile = DCCDCUIniFile<Model>(moduleSet, simulate);
		initError_ = Config::Init(iniFile.GetFileName().c_str());
	}

	~DCCDCUInterface() {
		if (std::any_of(modules_.begin(), modules_.end(), [](auto pm) { return !!pm; })) {
			assert(false); // Modules must be closed first.
		}
		(void)Config::Close();
	}

	auto IsSimulating() const -> bool {
		return Config::GetMode() != 0;
	}

	auto GetInitError() const -> short {
		return initError_;
	}

	auto GetModule(int moduleNo) -> std::shared_ptr<DCCDCUModule<Model>> {
		if (not modulesCreated_) {
			CreateModules();
		}
		return modules_[moduleNo];
	}

	void CloseAllModules() {
		for (short i = 0; i < MaxNrModules; ++i) {
			modules_[i].reset();
		}
	}
};
