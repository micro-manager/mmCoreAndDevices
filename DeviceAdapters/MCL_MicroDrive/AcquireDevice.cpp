/*
File:		AcquireDevice.cpp
Copyright:	Mad City Labs Inc., 2023
License:	Distributed under the BSD license.
*/

// MCL headers
#include "MicroDrive.h"
#include "AcquireDevice.h"
#include "MCL_MicroDrive.h"

// List/heap headers
#include "handle_list_if.h"
#include "HandleListType.h"

#include <vector>
using namespace std;

static int ChooseAvailableXYStageAxes(unsigned short pid, unsigned char axisBitmap, int handle, bool useStrictChoices);

static int ChooseAvailableZStageAxis(unsigned short pid, unsigned char axisBitmap, int handle, bool useStrictChoices);

static int ChooseAvailableMadTweezerStageAxis(unsigned short pid, unsigned char axisBitmap, int handle, bool useStrictChoices);

static bool FindMatchingDevice(int deviceAdapterType, int &deviceAdapterHandle, int &deviceAdapterAxis);

static bool FindMatchingDeviceInList(int deviceAdapterType, int *handles, int handlesCount, bool useStrictMatcingCriteria, int &deviceAdapterHandle, int &deviceAdapterAxis);

static void ReleaseUnusedDevices();


int AcquireDeviceHandle(int deviceType, int &deviceAdapterHandle, int &deviceAdapterAxis)
{
	deviceAdapterAxis = 0;
	deviceAdapterHandle = 0;

	// Attempt to find a device that can perform the deviceType role.
	bool foundDevice = FindMatchingDevice(deviceType, deviceAdapterHandle, deviceAdapterAxis);
	if (foundDevice)
	{
		// If we found a device add it to our list.
		HandleListType newDeviceAdapter(deviceAdapterHandle, deviceType, deviceAdapterAxis, (deviceType == XYSTAGE_TYPE ? (deviceAdapterAxis + 1) : 0));
		HandleListAddToLockedList(newDeviceAdapter);
	}

	// Release devices that are not needed.
	ReleaseUnusedDevices();

	return foundDevice ? MCL_SUCCESS : MCL_INVALID_HANDLE;
}

bool FindMatchingDevice(int deviceAdapterType, int &deviceAdapterHandle, int &deviceAdapterAxis)
{
	bool foundDevice = false;
	deviceAdapterHandle = 0;
	deviceAdapterAxis = 0;

	// First search through our existing devices to find appropriate matching axes.
	int existingHandlesCount = MCL_NumberOfCurrentHandles();
	if (existingHandlesCount != 0)
	{
		int *existingHandles = new int[existingHandlesCount];
		existingHandlesCount = MCL_GetAllHandles(existingHandles, existingHandlesCount);
		foundDevice = FindMatchingDeviceInList(deviceAdapterType, existingHandles, existingHandlesCount, true, deviceAdapterHandle, deviceAdapterAxis);
		delete[] existingHandles;

		if (foundDevice)
			return true;
	}

	// Next search through all available Micro-Drive systems.
	int handlesCount = MCL_GrabAllHandles();
	if (handlesCount == 0)
	{
		return false;
	}
	int* handles = new int[handlesCount];
	handlesCount = MCL_GetAllHandles(handles, handlesCount);
	foundDevice = FindMatchingDeviceInList(deviceAdapterType, handles, handlesCount, true, deviceAdapterHandle, deviceAdapterAxis);

	// Lastly if we have not found the device, search through all available Nano-Drive systems with relaxed criteria.
	if (!foundDevice)
	{
		foundDevice = FindMatchingDeviceInList(deviceAdapterType, handles, handlesCount, false, deviceAdapterHandle, deviceAdapterAxis);
	}
	delete[] handles;

	return foundDevice;
}

bool FindMatchingDeviceInList(int deviceAdapterType, int *handles, int handlesCount, bool useStrictMatcingCriteria, int &deviceAdapterHandle, int &deviceAdapterAxis)
{
	deviceAdapterAxis = 0;
	for (int ii = 0; ii < handlesCount; ii++)
	{
		unsigned short pid = 0;
		unsigned char axisBitmap = 0;

		if (MCL_GetProductID(&pid, handles[ii]) != MCL_SUCCESS)
			continue;
		if (MCL_GetAxisInfo(&axisBitmap, handles[ii]) != MCL_SUCCESS)
			continue;

		if (deviceAdapterType == XYSTAGE_TYPE)
			deviceAdapterAxis = ChooseAvailableXYStageAxes(pid, axisBitmap, handles[ii], useStrictMatcingCriteria);
		else if (deviceAdapterType == STAGE_TYPE)
			deviceAdapterAxis = ChooseAvailableZStageAxis(pid, axisBitmap, handles[ii], useStrictMatcingCriteria);
		else if(deviceAdapterType == MADTWEEZER_TYPE)
			deviceAdapterAxis = ChooseAvailableMadTweezerStageAxis(pid, axisBitmap, handles[ii], useStrictMatcingCriteria);

		if (deviceAdapterAxis != 0)
		{
			deviceAdapterHandle = handles[ii];
			break;
		}
	}
	return deviceAdapterAxis != 0;
}

void ReleaseUnusedDevices()
{
	int existingHandlesCount = MCL_NumberOfCurrentHandles();
	if (existingHandlesCount != 0)
	{
		int *existingHandles = new int[existingHandlesCount];
		existingHandlesCount = MCL_GetAllHandles(existingHandles, existingHandlesCount);

		// Iterate through the devices and release those that are not in use.
		for (int i = 0; i < existingHandlesCount; i++)
		{
			if (!HandleExistsOnLockedList(existingHandles[i]))
			{
				MCL_ReleaseHandle(existingHandles[i]);
			}
		}
		delete[] existingHandles;
	}
}


int ChooseAvailableXYStageAxes(unsigned short pid, unsigned char axisBitmap, int handle, bool useStrictChoices)
{
	int ordersize = 3;
	int order[] = { M1AXIS, M4AXIS};
	int strictOrder[] = { M1AXIS, 0 };

	switch (pid)
	{
	case MADTWEEZER:
		return 0;
	case MICRODRIVE:
	case NC_MICRODRIVE:
	case MICRODRIVE3:
	case MICRODRIVE4:
		order[1] = 0;
		break;
		// Use the standard order.
	default:
		break;
	}

	int *chosenOrder = useStrictChoices ? strictOrder : order;
	int axis = 0;
	for (int ii = 0; ii < ordersize; ii++)
	{
		if (chosenOrder[ii] == 0)
			break;

		// Check that both axes are valid.
		int xBitmap = 0x1 << (chosenOrder[ii] - 1);
		int yBitmap = 0x1 << chosenOrder[ii];
		if (((axisBitmap & xBitmap) != xBitmap) ||
			((axisBitmap & yBitmap) != yBitmap))
			continue;

		HandleListType device(handle, XYSTAGE_TYPE, chosenOrder[ii], chosenOrder[ii] + 1);
		if (HandleExistsOnLockedList(device) == false)
		{
			axis = chosenOrder[ii];
			break;
		}
	}
	return axis;
}


int ChooseAvailableZStageAxis(unsigned short pid, unsigned char axisBitmap, int handle, bool useStrictChoices)
{
	int ordersize = 6;
	int order[] = { M1AXIS, M2AXIS, M3AXIS, M4AXIS, M5AXIS, M6AXIS };
	int strictOrder[] = { M3AXIS, M4AXIS, M5AXIS, M6AXIS, 0, 0 };

	switch (pid)
	{
		// These devices should be used as XY Stage devices.
	case MICRODRIVE:
	case NC_MICRODRIVE:
		return 0;
	case MICRODRIVE3:
	{
		int neworder[] = { M3AXIS, M2AXIS, M1AXIS, 0, 0, 0 };
		copy(neworder, neworder + ordersize, order);
		break;
	}
	// For 4 and 6 axis systems leave M1/M2 for an XY Stage.
	case MICRODRIVE4:
	{
		int neworder[] = { M3AXIS, M4AXIS, 0, 0, 0, 0 };
		copy(neworder, neworder + ordersize, order);
		break;
	}
	case MICRODRIVE6:
	{
		int neworder[] = { M3AXIS, M6AXIS, M4AXIS, M5AXIS, 0, 0 };
		copy(neworder, neworder + ordersize, order);
		break;
	}
	case MICRODRIVE1:
	{
		int neworder[] = { M1AXIS, M2AXIS, M3AXIS, 0, 0, 0 };
		copy(neworder, neworder + ordersize, order);
		break;
	}
	case MADTWEEZER:
	{
		// The strictOrder and order for MadTweezer are identical.
		int neworder[] = { M1AXIS, 0, 0, 0, 0, 0 };
		copy(neworder, neworder + ordersize, order);		
		copy(neworder, neworder + ordersize, strictOrder);
		break;
	}
	// Use the standard order.
	default:
		break;
	}	

	int *chosenOrder = useStrictChoices ? strictOrder : order;
	int axis = 0;
	for (int ii = 0; ii < ordersize; ii++)
	{
		if (chosenOrder[ii] == 0)
			break;

		// Check that the axis is valid.
		int bitmap = 0x1 << (chosenOrder[ii] - 1);
		if ((axisBitmap & bitmap) != bitmap)
			continue;

		// Check if a matching device is already in our list of controlled devices.
		HandleListType device(handle, STAGE_TYPE, chosenOrder[ii], 0);
		if (HandleExistsOnLockedList(device) == false)
		{
			// If there is no conflict we can choose 
			axis = chosenOrder[ii];
			break;
		}
	}
	return axis;
}

int ChooseAvailableMadTweezerStageAxis(unsigned short pid, unsigned char axisBitmap, int handle, bool useStrictChoices)
{
	int ordersize = 1;
	int order[] = { M2AXIS };
	int strictOrder[] = { M2AXIS };

	if (pid != MADTWEEZER)
		return 0;

	int *chosenOrder = useStrictChoices ? strictOrder : order;
	int axis = 0;
	for (int ii = 0; ii < ordersize; ii++)
	{
		if (chosenOrder[ii] == 0)
			break;

		// Check that the axis is valid.
		int bitmap = 0x1 << (chosenOrder[ii] - 1);
		if ((axisBitmap & bitmap) != bitmap)
			continue;

		// Check if a matching device is already in our list of controlled devices.
		HandleListType device(handle, MADTWEEZER_TYPE, chosenOrder[ii], 0);
		if (HandleExistsOnLockedList(device) == false)
		{
			// If there is no conflict we can choose 
			axis = chosenOrder[ii];
			break;
		}
	}
	return axis;
}