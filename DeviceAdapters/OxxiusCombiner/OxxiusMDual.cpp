#include "OxxiusMDual.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <map>
#include "../../MMDevice/ModuleInterface.h"
using namespace std;

///////////////////////////////////////////////////////////////////////////////
//
// Oxxius M-Dual implementation
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
///////////////////////////////////////////////////////////////////////////////

OxxiusMDual::OxxiusMDual(const char* nameAndSlot) : initialized_(false)
{
	parentHub_ = 0;
	core_ = GetCoreCallback();

	std::string tSlot = string(nameAndSlot);
	name_.assign(tSlot); // sets MDual name

	slot_ = tSlot.substr(tSlot.length() - 1, 1);

	// Set property list
	// -----------------
	// Name (read only)
	CreateProperty(MM::g_Keyword_Name, name_.c_str(), MM::String, true);

	CreateProperty(MM::g_Keyword_Description, "M-Dual module", MM::String, true);

	InitializeDefaultErrorMessages();
	SetErrorText(ERR_NO_PORT_SET, "Hub Device not found.  The Laser combiner is needed to create this device");

	// parent ID display
	CreateHubIDProperty();
}


OxxiusMDual::~OxxiusMDual()
{
	Shutdown();
}


int OxxiusMDual::Initialize()
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
		CPropertyAction* pAct = new CPropertyAction(this, &OxxiusMDual::OnSetRatio);//setting the possible positions
		RETURN_ON_MM_ERROR(CreateProperty("Split ratio", "0", MM::Float, false, pAct));
		SetPropertyLimits("Split ratio", 0.0, 100.0);

		// Set property list
		// -----------------
		// State
		/*CPropertyAction* pAct = new CPropertyAction (this, &OxxiusMDual::OnState);
		RETURN_ON_MM_ERROR( CreateProperty(MM::g_Keyword_State, "0", MM::Integer, false, pAct) );
		SetPropertyLimits("Set Position", 0, 100);*/

		/*char pos[3];
		for (unsigned int i=0; i<numPos_; i++) {
			sprintf(pos, "%d", i);
			AddAllowedValue(MM::g_Keyword_State, pos);
		}*/

		// Label
		/*pAct = new CPropertyAction (this, &CStateBase::OnLabel);
		RETURN_ON_MM_ERROR( CreateProperty(MM::g_Keyword_Label, "", MM::String, false, pAct) );

		char state[20];
		for (unsigned int i=0; i<numPos_; i++) {
			sprintf(state, "Position-%d", i);
			SetPositionLabel(i,state);
		}*/

		RETURN_ON_MM_ERROR(UpdateStatus());

		initialized_ = true;
	}

	return DEVICE_OK;
}


int OxxiusMDual::Shutdown()
{
	initialized_ = false;
	return DEVICE_OK;
}


void OxxiusMDual::GetName(char* Name) const
{
	CDeviceUtils::CopyLimitedString(Name, name_.c_str());
}


bool OxxiusMDual::Busy()
{
	return false;
}


/*int OxxiusMDual::OnState(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet) {
		unsigned int currentPos = 0;
		std::ostringstream command;
		command << "IP" << slot_;

		RETURN_ON_MM_ERROR( parentHub_->QueryCommand(this, GetCoreCallback(), NO_SLOT, command.str().c_str()) );
		parentHub_->ParseforInteger(currentPos);

		//SetPosition(currentPos);
		pProp->Set((long)currentPos);
	}
	else if (eAct == MM::AfterSet) {
		long newPosition = 0;

		//GetPosition(newPosition);
		pProp->Get(newPosition);

		std::ostringstream newCommand;
		newCommand << "IP" << slot_ << " " << newPosition;

		RETURN_ON_MM_ERROR( parentHub_->QueryCommand(this, GetCoreCallback(), NO_SLOT, newCommand.str().c_str()) );
	}
	return DEVICE_OK;
}*/


int OxxiusMDual::OnSetRatio(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet) {
		double currentRatio = 0.0;
		std::ostringstream command;
		command << "IP" << slot_;

		RETURN_ON_MM_ERROR(parentHub_->QueryCommand(this, GetCoreCallback(), NO_SLOT, command.str().c_str(), true));
		parentHub_->ParseforPercent(currentRatio);

		pProp->Set(currentRatio);
	}

	else if (eAct == MM::AfterSet) {
		double newRatio = 0.0;

		pProp->Get(newRatio);
		if ((newRatio >= 0.0) || (newRatio <= 100.0)) {
			std::ostringstream newCommand;
			newCommand << "IP" << slot_ << " " << newRatio;

			RETURN_ON_MM_ERROR(parentHub_->QueryCommand(this, GetCoreCallback(), NO_SLOT, newCommand.str().c_str(), true));
		}
	}
	return DEVICE_OK;
}