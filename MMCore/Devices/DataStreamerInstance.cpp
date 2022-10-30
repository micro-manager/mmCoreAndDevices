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
int DataStreamerInstance::ProcessBuffer(std::unique_ptr<char[]>& pDataBuffer) { return GetImpl()->ProcessBuffer(pDataBuffer); }
int DataStreamerInstance::SetStreamParameters(bool stopOnOverflow, unsigned numberOfBuffers, double durationUs, double updatePeriodUs) { return GetImpl()->SetStreamParameters(stopOnOverflow, numberOfBuffers, durationUs, updatePeriodUs); }
int DataStreamerInstance::GetStreamParameters(bool& stopOnOverflow, unsigned& numberOfBuffers, double& durationUs, double& updatePeriodUs) { return GetImpl()->GetStreamParameters(stopOnOverflow,numberOfBuffers,durationUs,updatePeriodUs); }
int DataStreamerInstance::StartStream() { return GetImpl()->StartStream(); }
int DataStreamerInstance::StopStream() { return GetImpl()->StopStream(); }
int DataStreamerInstance::IsStreaming(unsigned& isStreaming) { return GetImpl()->IsStreaming(isStreaming); }
int DataStreamerInstance::SetCircularAcquisitionBufferCapacity(unsigned capacity) { return GetImpl()->SetCircularAcquisitionBufferCapacity(capacity); }
int DataStreamerInstance::GetCircularAcquisitionBufferCapacity(unsigned& capacity) { return GetImpl()->GetCircularAcquisitionBufferCapacity(capacity); }

