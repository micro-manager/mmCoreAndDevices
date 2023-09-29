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


int SLMInstance::SetImage(unsigned char* pixels) { RequireInitialized(__func__); return GetImpl()->SetImage(pixels); }
int SLMInstance::SetImage(unsigned int* pixels) { RequireInitialized(__func__); return GetImpl()->SetImage(pixels); }
int SLMInstance::DisplayImage() { RequireInitialized(__func__); return GetImpl()->DisplayImage(); }
int SLMInstance::SetPixelsTo(unsigned char intensity) { RequireInitialized(__func__); return GetImpl()->SetPixelsTo(intensity); }
int SLMInstance::SetPixelsTo(unsigned char red, unsigned char green, unsigned char blue) { RequireInitialized(__func__); return GetImpl()->SetPixelsTo(red, green, blue); }
int SLMInstance::SetExposure(double interval_ms) { RequireInitialized(__func__); return GetImpl()->SetExposure(interval_ms); }
double SLMInstance::GetExposure() { RequireInitialized(__func__); return GetImpl()->GetExposure(); }
unsigned SLMInstance::GetWidth() { RequireInitialized(__func__); return GetImpl()->GetWidth(); }
unsigned SLMInstance::GetHeight() { RequireInitialized(__func__); return GetImpl()->GetHeight(); }
unsigned SLMInstance::GetNumberOfComponents() { RequireInitialized(__func__); return GetImpl()->GetNumberOfComponents(); }
unsigned SLMInstance::GetBytesPerPixel() { RequireInitialized(__func__); return GetImpl()->GetBytesPerPixel(); }
int SLMInstance::IsSLMSequenceable(bool& isSequenceable)
{ RequireInitialized(__func__); return GetImpl()->IsSLMSequenceable(isSequenceable); }
int SLMInstance::GetSLMSequenceMaxLength(long& nrEvents)
{ RequireInitialized(__func__); return GetImpl()->GetSLMSequenceMaxLength(nrEvents); }
int SLMInstance::StartSLMSequence() { RequireInitialized(__func__); return GetImpl()->StartSLMSequence(); }
int SLMInstance::StopSLMSequence() { RequireInitialized(__func__); return GetImpl()->StopSLMSequence(); }
int SLMInstance::ClearSLMSequence() { RequireInitialized(__func__); return GetImpl()->ClearSLMSequence(); }
int SLMInstance::AddToSLMSequence(const unsigned char * pixels)
{ RequireInitialized(__func__); return GetImpl()->AddToSLMSequence(pixels); }
int SLMInstance::AddToSLMSequence(const unsigned int * pixels)
{ RequireInitialized(__func__); return GetImpl()->AddToSLMSequence(pixels); }
int SLMInstance::SendSLMSequence() { RequireInitialized(__func__); return GetImpl()->SendSLMSequence(); }
