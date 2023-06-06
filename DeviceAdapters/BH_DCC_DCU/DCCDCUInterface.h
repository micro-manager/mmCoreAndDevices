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

public:
	explicit DCCDCUModule(std::shared_ptr<ParentType> parent, short moduleNo) :
		parent_(parent),
		moduleNo_(moduleNo)
	{}

	void Close() {
		parent_->CloseModule(moduleNo_);
	}

	auto IsActive() const -> bool {
		return parent_->IsModuleActive(moduleNo_);
	}

	auto InitStatus(short& error) const -> short {
		return parent_->GetModuleInitStatus(moduleNo_, error);
	}

	auto ModInfo(short& error) const {
		return parent_->GetModuleInfo(moduleNo_, error);;
	}

	auto GetConnectorParameterBool(short connNo, ConnectorFeature feature, short& error) -> bool {
		return parent_->GetConnectorParameter(moduleNo_, connNo, feature, error) != 0.0f;
	}

	auto GetConnectorParameterUInt(short connNo, ConnectorFeature feature, short& error) -> unsigned {
		return static_cast<unsigned>(parent_->GetConnectorParameter(moduleNo_, connNo, feature, error));
	}

	auto GetConnectorParameterFloat(short connNo, ConnectorFeature feature, short& error) -> float {
		return parent_->GetConnectorParameter(moduleNo_, connNo, feature, error);
	}

	void SetConnectorParameterBool(short connNo, ConnectorFeature feature, bool value, short& error) {
		parent_->SetConnectorParameter(moduleNo_, connNo, feature,
			(value ? 1.0f : 0.0f), error);
	}

	void SetConnectorParameterUInt(short connNo, ConnectorFeature feature, unsigned value, short& error) {
		parent_->SetConnectorParameter(moduleNo_, connNo, feature,
			static_cast<float>(value), error);
	}

	void SetConnectorParameterFloat(short connNo, ConnectorFeature feature, float value, short& error) {
		parent_->SetConnectorParameter(moduleNo_, connNo, feature, value, error);
	}

	auto GetGainHVLimit(short connNo, short& error) -> float {
		return parent_->GetGainHVLimit(moduleNo_, connNo, error);
	}

	void EnableAllOutputs(bool enable, short& error) {
		parent_->EnableAllOutputs(moduleNo_, enable, error);
	}

	void EnableConnectorOutputs(short connNo, bool enable, short& error) {
		parent_->EnableConnectorOutputs(moduleNo_, connNo, enable, error);
	}

	void ClearAllOverloads(short& error) {
		parent_->ClearAllOverloads(moduleNo_, error);
	}

	void ClearConnectorOverload(short connNo, short& error) {
		parent_->ClearConnectorOverload(moduleNo_, connNo, error);
	}

	auto IsOverloaded(short connNo, short& error) -> bool {
		return parent_->IsOverloaded(moduleNo_, connNo, error);
	}

	auto IsCoolerCurrentLimitReached(short connNo, short& error) -> bool {
		return parent_->IsCoolerCurrentLimitReached(moduleNo_, connNo, error);
	}
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

	void CloseModule(short moduleNo) {
		modules_[moduleNo].reset();
	}

	void CloseAllModules() {
		for (short i = 0; i < MaxNrModules; ++i) {
			modules_[i].reset();
		}
	}

	auto IsModuleActive(short moduleNo) -> bool {
		return Config::TestIfActive(moduleNo) != 0;
	}

	auto GetModuleInitStatus(short moduleNo, short& error) -> short {
		short ret{};
		error = Config::GetInitStatus(moduleNo, &ret);
		return ret;
	}

	auto GetModuleInfo(short moduleNo, short& error) -> typename Config::ModInfoType {
		typename Config::ModInfoType ret{};
		error = Config::GetModuleInfo(moduleNo, &ret);
		return ret;
	}

	auto GetConnectorParameter(short moduleNo, short connNo, ConnectorFeature feature,
		short& error) -> float {
		float ret{};
		error = Config::GetParameter(moduleNo,
			Config::ConnectorParameterId(connNo, feature),
			&ret);
		return ret;
	}

	void SetConnectorParameter(short moduleNo, short connNo, ConnectorFeature feature,
		float value, short& error) {
		error = Config::SetParameter(moduleNo,
			Config::ConnectorParameterId(connNo, feature),
			true, value);
	}

	auto GetGainHVLimit(short moduleNo, short connNo, short& error) -> float {
		short shortLimit{};
		error = Config::GetGainHVLimit(moduleNo, connNo, &shortLimit);
		return static_cast<float>(shortLimit);
	}

	void EnableAllOutputs(short moduleNo, bool enable, short& error) {
		error = Config::EnableAllOutputs(moduleNo, enable ? 1 : 0);
	}

	void EnableConnectorOutputs(short moduleNo, short connNo, bool enable, short& error) {
		error = Config::EnableOutput(moduleNo, connNo, enable ? 1 : 0);
	}

	void ClearAllOverloads(short moduleNo, short& error) {
		error = Config::ClearAllOverloads(moduleNo);
	}

	void ClearConnectorOverload(short moduleNo, short connNo, short& error) {
		error = Config::ClearOverload(moduleNo, connNo);
	}

	auto IsOverloaded(short moduleNo, short connNo, short& error) -> bool {
		short state{};
		error = Config::GetOverloadState(moduleNo, &state);
		return state & (1 << connNo);
	}

	auto IsCoolerCurrentLimitReached(short moduleNo, short connNo, short& error) -> bool {
		short state{};
		error = Config::GetCurrLmtState(moduleNo, &state);
		return state & (1 << connNo);
	}
};
