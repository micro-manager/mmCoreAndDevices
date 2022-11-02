// PROJECT:       Micro-Manager
// SUBSYSTEM:     MMCore
//
// DESCRIPTION:   Camera device instance wrapper
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

#include "CameraInstance.h"


int CameraInstance::SnapImage() { return GetImpl()->SnapImage(); }
const unsigned char* CameraInstance::GetImageBuffer() { return GetImpl()->GetImageBuffer(); }
const unsigned char* CameraInstance::GetImageBuffer(unsigned channelNr) { return GetImpl()->GetImageBuffer(channelNr); }
const unsigned int* CameraInstance::GetImageBufferAsRGB32() { return GetImpl()->GetImageBufferAsRGB32(); }
unsigned CameraInstance::GetNumberOfComponents() const { return GetImpl()->GetNumberOfComponents(); }

std::string CameraInstance::GetComponentName(unsigned component)
{
   DeviceStringBuffer nameBuf(this, "GetComponentName");
   int err = GetImpl()->GetComponentName(component, nameBuf.GetBuffer());
   ThrowIfError(err, "Cannot get component name at index " +
         ToString(component));
   return nameBuf.Get();
}

int unsigned CameraInstance::GetNumberOfChannels() const { return GetImpl()->GetNumberOfChannels(); }

std::string CameraInstance::GetChannelName(unsigned channel)
{
   DeviceStringBuffer nameBuf(this, "GetChannelName");
   int err = GetImpl()->GetChannelName(channel, nameBuf.GetBuffer());
   ThrowIfError(err, "Cannot get channel name at index " + ToString(channel));
   return nameBuf.Get();
}

long CameraInstance::GetImageBufferSize()const { return GetImpl()->GetImageBufferSize(); }
unsigned CameraInstance::GetImageWidth() const { return GetImpl()->GetImageWidth(); }
unsigned CameraInstance::GetImageHeight() const { return GetImpl()->GetImageHeight(); }
unsigned CameraInstance::GetImageBytesPerPixel() const { return GetImpl()->GetImageBytesPerPixel(); }
unsigned CameraInstance::GetBitDepth() const { return GetImpl()->GetBitDepth(); }
double CameraInstance::GetPixelSizeUm() const { return GetImpl()->GetPixelSizeUm(); }
int CameraInstance::GetBinning() const { return GetImpl()->GetBinning(); }
int CameraInstance::SetBinning(int binSize) { return GetImpl()->SetBinning(binSize); }
void CameraInstance::SetExposure(double exp_ms) { return GetImpl()->SetExposure(exp_ms); }
double CameraInstance::GetExposure() const { return GetImpl()->GetExposure(); }
int CameraInstance::SetROI(unsigned x, unsigned y, unsigned xSize, unsigned ySize) { return GetImpl()->SetROI(x, y, xSize, ySize); }
int CameraInstance::GetROI(unsigned& x, unsigned& y, unsigned& xSize, unsigned& ySize) { return GetImpl()->GetROI(x, y, xSize, ySize); }
int CameraInstance::ClearROI() { return GetImpl()->ClearROI(); }

/**
 * Queries if the camera supports multiple simultaneous ROIs.
 */
bool CameraInstance::SupportsMultiROI()
{
   return GetImpl()->SupportsMultiROI();
}

/**
 * Queries if multiple ROIs have been set (via the SetMultiROI method). Must
 * return true even if only one ROI was set via that method, but must return
 * false if an ROI was set via SetROI() or if ROIs have been cleared.
 */
bool CameraInstance::IsMultiROISet()
{
   return GetImpl()->IsMultiROISet();
}

/**
 * Queries for the current set number of ROIs. Must return zero if multiple
 * ROIs are not set (including if an ROI has been set via SetROI).
 */
int CameraInstance::GetMultiROICount(unsigned int& count)
{
   return GetImpl()->GetMultiROICount(count);
}

/**
 * Set multiple ROIs. Replaces any existing ROI settings including ROIs set
 * via SetROI.
 * @param xs Array of X indices of upper-left corner of the ROIs.
 * @param ys Array of Y indices of upper-left corner of the ROIs.
 * @param widths Widths of the ROIs, in pixels.
 * @param heights Heights of the ROIs, in pixels.
 * @param numROIs Length of the arrays.
 */
int CameraInstance::SetMultiROI(const unsigned int* xs, const unsigned int* ys,
      const unsigned* widths, const unsigned int* heights,
      unsigned numROIs)
{
   return GetImpl()->SetMultiROI(xs, ys, widths, heights, numROIs);
}

/**
 * Queries for current multiple-ROI setting. May be called even if no ROIs of
 * any type have been set. Must return length of 0 in that case.
 * @param xs (Return value) X indices of upper-left corner of the ROIs.
 * @param ys (Return value) Y indices of upper-left corner of the ROIs.
 * @param widths (Return value) Widths of the ROIs, in pixels.
 * @param heights (Return value) Heights of the ROIs, in pixels.
 * @param numROIs Length of the input arrays. If there are fewer ROIs than
 *        this, then this value must be updated to reflect the new count.
 */
int CameraInstance::GetMultiROI(unsigned* xs, unsigned* ys, unsigned* widths,
      unsigned* heights, unsigned* length)
{
   return GetImpl()->GetMultiROI(xs, ys, widths, heights, length);
}

int CameraInstance::StartSequenceAcquisition(long numImages, double interval_ms, bool stopOnOverflow) { return GetImpl()->StartSequenceAcquisition(numImages, interval_ms, stopOnOverflow); }
int CameraInstance::StartSequenceAcquisition(double interval_ms) { return GetImpl()->StartSequenceAcquisition(interval_ms); }
int CameraInstance::StopSequenceAcquisition() { return GetImpl()->StopSequenceAcquisition(); }
int CameraInstance::PrepareSequenceAcqusition() { return GetImpl()->PrepareSequenceAcqusition(); }
bool CameraInstance::IsCapturing() { return GetImpl()->IsCapturing(); }

std::string CameraInstance::GetTags()
{
   // TODO Probably makes sense to deserialize here.
   // Also note the danger of limiting serialized metadata to MM::MaxStrLength
   // (CCameraBase takes no precaution to limit string length; it is an
   // interface bug).
   DeviceStringBuffer serializedMetadataBuf(this, "GetTags");
   GetImpl()->GetTags(serializedMetadataBuf.GetBuffer());
   return serializedMetadataBuf.Get();
}

void CameraInstance::AddTag(const char* key, const char* deviceLabel, const char* value) { return GetImpl()->AddTag(key, deviceLabel, value); }
void CameraInstance::RemoveTag(const char* key) { return GetImpl()->RemoveTag(key); }
int CameraInstance::IsExposureSequenceable(bool& isSequenceable) const { return GetImpl()->IsExposureSequenceable(isSequenceable); }
int CameraInstance::GetExposureSequenceMaxLength(long& nrEvents) const { return GetImpl()->GetExposureSequenceMaxLength(nrEvents); }
int CameraInstance::StartExposureSequence() { return GetImpl()->StartExposureSequence(); }
int CameraInstance::StopExposureSequence() { return GetImpl()->StopExposureSequence(); }
int CameraInstance::ClearExposureSequence() { return GetImpl()->ClearExposureSequence(); }
int CameraInstance::AddToExposureSequence(double exposureTime_ms) { return GetImpl()->AddToExposureSequence(exposureTime_ms); }
int CameraInstance::SendExposureSequence() const { return GetImpl()->SendExposureSequence(); }

bool CameraInstance::IsNewAPIImplemented(){ return GetImpl()->IsNewAPIImplemented();}
bool CameraInstance::HasTrigger(const char* triggerSelector) { return GetImpl()->HasTrigger(triggerSelector); }




int CameraInstance::SetTriggerMode(const char* triggerSelector, bool triggerMode) {
   return GetImpl()->SetTriggerMode(triggerSelector, triggerMode); };
int CameraInstance::SetTriggerSource(const char* triggerSelector, const char* triggerSource) {
   return GetImpl()->SetTriggerSource(triggerSelector, triggerSource); };
int CameraInstance::SetTriggerDelay(const char* triggerSelector, int triggerDelay) {
   return GetImpl()->SetTriggerDelay(triggerSelector, triggerDelay); };
int CameraInstance::SetTriggerActivation(const char* triggerSelector, const char* triggerActivation) {
   return GetImpl()->SetTriggerActivation(triggerSelector, triggerActivation); };
// int CameraInstance::SetTriggerOverlap(const char* triggerSelector, const char* triggerOverlap) {
//    return GetImpl()->SetTriggerOverlap(triggerSelector, triggerOverlap); };

bool CameraInstance::GetTriggerMode(const char* triggerSelector) const{
   bool on;
   GetImpl()->GetTriggerMode(triggerSelector, on);
   return on;
};
std::string CameraInstance::GetTriggerSource(const char* triggerSelector) const {
   char triggerSource[MM::MaxStrLength];
   GetImpl()->GetTriggerSource(triggerSelector, triggerSource);
   return std::string(triggerSource);
};
int CameraInstance::GetTriggerDelay(const char* triggerSelector) const {
   int delay;
   GetImpl()->GetTriggerDelay(triggerSelector, delay);
   return delay;
};
// std::string CameraInstance::GetTriggerOverlap(const char* triggerSelector) const {
//    char triggerOverlap[MM::MaxStrLength];
//    GetImpl()->GetTriggerOverlap(triggerSelector, triggerOverlap);
//    return std::string(triggerOverlap);
// };
std::string CameraInstance::GetTriggerActivation(const char* triggerSelector) const {
   char triggerActivation[MM::MaxStrLength];
   GetImpl()->GetTriggerActivation(triggerSelector, triggerActivation);
   return std::string(triggerActivation);
}

int CameraInstance::SetBurstFrameCount(unsigned count) {
   GetImpl()->SetBurstFrameCount(count);
   return DEVICE_OK;
}

unsigned CameraInstance::GetBurstFrameCount() const {
   return GetImpl()->GetBurstFrameCount();
}

bool CameraInstance::HasExposureMode(const char* exposureMode) {
   return GetImpl()->HasExposureMode(exposureMode);
}

int CameraInstance::SetExposureMode(const char* exposureMode) {
   return GetImpl()->SetExposureMode(exposureMode);
}

std::string CameraInstance::GetExposureMode() {
   char expMode[MM::MaxStrLength];
   GetImpl()->GetExposureMode(expMode);
   return std::string(expMode);
};


int CameraInstance::SendSoftwareTrigger(const char* triggerSelector) { return GetImpl()->TriggerSoftware(triggerSelector); }
int CameraInstance::ArmAcquisition(int frameCount) { return GetImpl()->AcquisitionArm(frameCount);}
int CameraInstance::ArmAcquisition() { return GetImpl()->AcquisitionArm(); }
int CameraInstance::StartAcquisition() { return GetImpl()->AcquisitionStart(); }
int CameraInstance::StopAcquisition() { return GetImpl()->AcquisitionStop(); }
int CameraInstance::AbortAcquisition() { return GetImpl()->AcquisitionAbort(); }

int CameraInstance::GetAcquisitionStatus(const char* statusSelector, bool& status) {
   return GetImpl()->GetAcquisitionStatus(statusSelector, status); }

int CameraInstance::SetIOLineInverted(const char* lineSelector, bool invert) {
   return GetImpl()->GetAcquisitionStatus(lineSelector, invert);
}

int CameraInstance::SetLineAsOutput(const char* lineSelector, bool output) {
   return GetImpl()->SetLineAsOutput(lineSelector, output);
}

int CameraInstance::SetOutputLineSource(const char* lineSelector, const char* source) {
   return GetImpl()->SetOutputLineSource(lineSelector, source);
}

int CameraInstance::GetLineStatus(const char* lineSelector, bool& status) {
   return GetImpl()->GetLineStatus(lineSelector, status);
}

double CameraInstance::GetRollingShutterLineOffset() { return GetImpl()->GetRollingShutterLineOffset(); }
int CameraInstance::SetRollingShutterLineOffset(double offset_us) { return GetImpl()->SetRollingShutterLineOffset(offset_us); }
unsigned CameraInstance::GetRollingShutterActiveLines() const { return GetImpl()->GetRollingShutterActiveLines(); }
unsigned CameraInstance::SetRollingShutterActiveLines(unsigned numLines) { return GetImpl()->SetRollingShutterActiveLines(numLines); }

