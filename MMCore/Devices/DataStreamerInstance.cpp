// PROJECT:       Micro-Manager
// SUBSYSTEM:     MMCore
//
// DESCRIPTION:   Galvo device instance wrapper
//
// COPYRIGHT:     University of California, San Francisco, 2014,
//                All Rights reserved
//
// LICENSE:       This file is distributed under the "Lesser GPL" (LGPL) license.
//                License text is included with the source distribution.
//
//                This file is distributed in the hope that it will be useful,
//                but WITHOUT ANY WARRANTY; without even the implied warranty
//                of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
//
//                IN NO EVENT SHALL THE COPYRIGHT OWNER OR
//                CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
//                INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES.
//
// AUTHOR:        Mark Tsuchida

#include "DataStreamerInstance.h"


int DataStreamerInstance::GetBufferSize(unsigned& dataBufferSiize) { return GetImpl()->GetBufferSize(dataBufferSiize); }
std::unique_ptr<char[]> DataStreamerInstance::GetBuffer(unsigned expectedDataBufferSize, unsigned& actualDataBufferSize) { return GetImpl()->GetBuffer(expectedDataBufferSize,actualDataBufferSize); }
int DataStreamerInstance::ProcessBuffer(std::unique_ptr<char[]>& pDataBuffer, unsigned dataSize) { return GetImpl()->ProcessBuffer(pDataBuffer, dataSize); }
int DataStreamerInstance::StartStream() { return GetImpl()->StartStream(); }
int DataStreamerInstance::StopStream() { return GetImpl()->StopStream(); }
bool DataStreamerInstance::GetOverflowStatus() { return GetImpl()->GetOverflowStatus(); }
int DataStreamerInstance::ResetOverflowStatus() { return GetImpl()->ResetOverflowStatus(); }
bool DataStreamerInstance::IsStreaming() { return GetImpl()->IsStreaming(); }
int DataStreamerInstance::SetStreamParameters(bool stopOnOverflow, int numberOfBuffers, double durationUs, double updatePeriodUs) { return GetImpl()->SetStreamParameters(stopOnOverflow, numberOfBuffers, durationUs, updatePeriodUs); }
int DataStreamerInstance::GetStreamParameters(bool& stopOnOverflow, int& numberOfBuffers, double& durationUs, double& updatePeriodUs) { return GetImpl()->GetStreamParameters(stopOnOverflow,numberOfBuffers,durationUs,updatePeriodUs); }
int DataStreamerInstance::SetCircularAcquisitionBufferCapacity(int capacity) { return GetImpl()->SetCircularAcquisitionBufferCapacity(capacity); }
int DataStreamerInstance::GetCircularAcquisitionBufferCapacity() { return GetImpl()->GetCircularAcquisitionBufferCapacity(); }

