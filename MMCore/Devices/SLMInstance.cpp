// PROJECT:       Micro-Manager
// SUBSYSTEM:     MMCore
//
// DESCRIPTION:   SLM device instance wrapper
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

#include "SLMInstance.h"


int SLMInstance::SetImage(unsigned char* pixels) { RequireInitialized(); return GetImpl()->SetImage(pixels); }
int SLMInstance::SetImage(unsigned int* pixels) { RequireInitialized(); return GetImpl()->SetImage(pixels); }
int SLMInstance::DisplayImage() { RequireInitialized(); return GetImpl()->DisplayImage(); }
int SLMInstance::SetPixelsTo(unsigned char intensity) { RequireInitialized(); return GetImpl()->SetPixelsTo(intensity); }
int SLMInstance::SetPixelsTo(unsigned char red, unsigned char green, unsigned char blue) { RequireInitialized(); return GetImpl()->SetPixelsTo(red, green, blue); }
int SLMInstance::SetExposure(double interval_ms) { RequireInitialized(); return GetImpl()->SetExposure(interval_ms); }
double SLMInstance::GetExposure() { RequireInitialized(); return GetImpl()->GetExposure(); }
unsigned SLMInstance::GetWidth() { RequireInitialized(); return GetImpl()->GetWidth(); }
unsigned SLMInstance::GetHeight() { RequireInitialized(); return GetImpl()->GetHeight(); }
unsigned SLMInstance::GetNumberOfComponents() { RequireInitialized(); return GetImpl()->GetNumberOfComponents(); }
unsigned SLMInstance::GetBytesPerPixel() { RequireInitialized(); return GetImpl()->GetBytesPerPixel(); }
int SLMInstance::IsSLMSequenceable(bool& isSequenceable)
{ RequireInitialized(); return GetImpl()->IsSLMSequenceable(isSequenceable); }
int SLMInstance::GetSLMSequenceMaxLength(long& nrEvents)
{ RequireInitialized(); return GetImpl()->GetSLMSequenceMaxLength(nrEvents); }
int SLMInstance::StartSLMSequence() { RequireInitialized(); return GetImpl()->StartSLMSequence(); }
int SLMInstance::StopSLMSequence() { RequireInitialized(); return GetImpl()->StopSLMSequence(); }
int SLMInstance::ClearSLMSequence() { RequireInitialized(); return GetImpl()->ClearSLMSequence(); }
int SLMInstance::AddToSLMSequence(const unsigned char * pixels)
{ RequireInitialized(); return GetImpl()->AddToSLMSequence(pixels); }
int SLMInstance::AddToSLMSequence(const unsigned int * pixels)
{ RequireInitialized(); return GetImpl()->AddToSLMSequence(pixels); }
int SLMInstance::SendSLMSequence() { RequireInitialized(); return GetImpl()->SendSLMSequence(); }
