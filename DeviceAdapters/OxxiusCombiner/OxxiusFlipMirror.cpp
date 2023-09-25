#include "OxxiusFlipMirror.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <map>
#include "../../MMDevice/ModuleInterface.h"
using namespace std;


///////////////////////////////////////////////////////////////////////////////
//
// Oxxius Flip-Mirror implementation
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
///////////////////////////////////////////////////////////////////////////////

OxxiusFlipMirror::OxxiusFlipMirror(const char* nameAndSlot) : initialized_(false)
{
	parentHub_ = 0;
	core_ = GetCoreCallback();

	std::string fSlot = string(nameAndSlot);
	nameF_.assign(fSlot);// set laser name
	fSlot = fSlot.substr(fSlot.length() - 1, 1);
	slot_ = (unsigned int)atoi(fSlot.c_str());// set laser slot

	numPos_ = 0;

	// Set property list ///////////////////////////////////////////////////////////////////////////////////////////////////// NOT WORKING? (duplicate property name Name(4))
	// -----------------
	// Name (read only)
	/*CreateProperty(MM::g_Keyword_Name, nameF_.c_str(), MM::String, true);

	CreateProperty(MM::g_Keyword_Description, "Flip-Mirror module", MM::String, true);

	InitializeDefaultErrorMessages();
	SetErrorText(ERR_NO_PORT_SET, "Hub Device not found.  The Laser combiner is needed to create this device");

	// parent ID display
	CreateHubIDProperty();*/
}


OxxiusFlipMirror::~OxxiusFlipMirror()
{
	Shutdown();
}


int OxxiusFlipMirror::Initialize()
{
	if (!initialized_) {
		parentHub_ = static_cast<OxxiusCombinerHub*>(GetParentHub());
		if (!parentHub_) {
			return DEVICE_COMM_HUB_MISSING;
		}
		char hubLabel[MM::MaxStrLength];
		parentHub_->GetLabel(hubLabel);
		SetParentID(hubLabel); // for backward compatibility

		CPropertyAction* pAct = new CPropertyAction(this, &OxxiusFlipMirror::OnSwitchPos);//setting the possible positions
		RETURN_ON_MM_ERROR(CreateProperty("Switch Position", "0", MM::Integer, false, pAct));
		SetPropertyLimits("Switch Position", 0, 1);

		std::ostringstream descriPt2;
		descriPt2 << "";
		RETURN_ON_MM_ERROR(CreateProperty(MM::g_Keyword_Description, descriPt2.str().c_str(), MM::String, true));

		// Gate, or "closed" position
//		RETURN_ON_MM_ERROR( CreateProperty(MM::g_Keyword_Closed_Position, "0", MM::String, false) );

//		isOpen_ = false;		// MDual closed posisiton is

		RETURN_ON_MM_ERROR(UpdateStatus());

		initialized_ = true;
	}

	return DEVICE_OK;
}


int OxxiusFlipMirror::Shutdown()
{
	initialized_ = false;
	return DEVICE_OK;
}


void OxxiusFlipMirror::GetName(char* Name) const
{
	CDeviceUtils::CopyLimitedString(Name, nameF_.c_str());
}


bool OxxiusFlipMirror::Busy()
{
	return false;
}


int OxxiusFlipMirror::OnSwitchPos(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet) {
		unsigned int currentPos = 0;
		std::ostringstream command;
		command << "FM" << slot_;

		RETURN_ON_MM_ERROR(parentHub_->QueryCommand(this, GetCoreCallback(), NO_SLOT, command.str().c_str(), false));
		parentHub_->ParseforInteger(currentPos);

		//SetPosition(currentPos);
		pProp->Set((long)currentPos);
	}
	else if (eAct == MM::AfterSet) {
		long newPosition = 0;

		//GetPosition(newPosition);
		pProp->Get(newPosition);

		std::ostringstream newCommand;
		newCommand << "FM" << slot_ << " " << newPosition;

		RETURN_ON_MM_ERROR(parentHub_->QueryCommand(this, GetCoreCallback(), NO_SLOT, newCommand.str().c_str(), false));
	}
	return DEVICE_OK;
}