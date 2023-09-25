#include "OxxiusShutter.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <map>
#include "../../MMDevice/ModuleInterface.h"
using namespace std;

///////////////////////////////////////////////////////////////////////////////
//
// Oxxius shutter implementation
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
///////////////////////////////////////////////////////////////////////////////

OxxiusShutter::OxxiusShutter(const char* nameAndSlot) : initialized_(false)
{
	isOpen_ = false;
	parentHub_ = 0;

	name_.assign(nameAndSlot);

	std::string strChnl = string(nameAndSlot);
	strChnl = strChnl.substr(strChnl.length() - 1, 1);
	channel_ = (unsigned int)atoi(strChnl.c_str());

	// Set property list
	// -----------------
	// Name (read only)
	CreateProperty(MM::g_Keyword_Name, name_.c_str(), MM::String, true);

	std::ostringstream shutterDesc;
	shutterDesc << "Electro-mechanical shutter on channel " << channel_ << ".";
	CreateProperty(MM::g_Keyword_Description, shutterDesc.str().c_str(), MM::String, true);

	InitializeDefaultErrorMessages();
	SetErrorText(ERR_NO_PORT_SET, "Hub Device not found.  The Laser combiner is needed to create this device");

	// parent ID display
	CreateHubIDProperty();
}


OxxiusShutter::~OxxiusShutter()
{
	Shutdown();
}


int OxxiusShutter::Initialize()
{
	if (!initialized_) {
		parentHub_ = static_cast<OxxiusCombinerHub*>(GetParentHub());
		if (!parentHub_) {
			return DEVICE_COMM_HUB_MISSING;
		}
		char hubLabel[MM::MaxStrLength];
		parentHub_->GetLabel(hubLabel);
		SetParentID(hubLabel); // for backward compatibility

		// Set property list
		// -----------------

		// Open/Close selector (write/read)
		CPropertyAction* pAct = new CPropertyAction(this, &OxxiusShutter::OnState);
		RETURN_ON_MM_ERROR(CreateProperty(MM::g_Keyword_State, "", MM::String, false, pAct));
		AddAllowedValue(MM::g_Keyword_State, "Open");
		AddAllowedValue(MM::g_Keyword_State, "Closed");

		// Closing the shutter on Initialization
		RETURN_ON_MM_ERROR(SetProperty(MM::g_Keyword_State, "Closed"));

		RETURN_ON_MM_ERROR(UpdateStatus());

		initialized_ = true;
	}

	return DEVICE_OK;
}


int OxxiusShutter::Shutdown()
{
	initialized_ = false;
	return DEVICE_OK;
}


void OxxiusShutter::GetName(char* Name) const
{
	CDeviceUtils::CopyLimitedString(Name, name_.c_str());
}


bool OxxiusShutter::Busy()
{
	return false;
}


int OxxiusShutter::SetOpen(bool openCommand)
{
	if (openCommand)
		return SetProperty(MM::g_Keyword_State, "Open");
	else
		return SetProperty(MM::g_Keyword_State, "Closed");
}


int OxxiusShutter::GetOpen(bool& isOpen)
{
	isOpen = isOpen_;
	return DEVICE_OK;
}


int OxxiusShutter::OnState(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet) {
		if (isOpen_) {
			pProp->Set("Open");
		}
		else {
			pProp->Set("Closed");
		}
	}
	else if (eAct == MM::AfterSet) {

		std::string newState = "";
		pProp->Get(newState);

		std::ostringstream newCommand;
		newCommand << "SH" << channel_ << " ";

		if (newState.compare("Open") == 0) {
			newCommand << "1";
			isOpen_ = true;
		}
		else if (newState.compare("Closed") == 0) {
			newCommand << "0";
			isOpen_ = false;
		}

		RETURN_ON_MM_ERROR(parentHub_->QueryCommand(this, GetCoreCallback(), NO_SLOT, newCommand.str().c_str(), false));
	}
	return DEVICE_OK;
}