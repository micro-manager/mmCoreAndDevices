// PROJECT:       Micro-Manager
// SUBSYSTEM:     MMCore
//
// DESCRIPTION:   Autofocus device instance wrapper
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

#include "AutoFocusInstance.h"


int AutoFocusInstance::SetContinuousFocusing(bool state) { RequireInitialized(); return GetImpl()->SetContinuousFocusing(state); }
int AutoFocusInstance::GetContinuousFocusing(bool& state) { RequireInitialized(); return GetImpl()->GetContinuousFocusing(state); }
bool AutoFocusInstance::IsContinuousFocusLocked() { RequireInitialized(); return GetImpl()->IsContinuousFocusLocked(); }
int AutoFocusInstance::FullFocus() { RequireInitialized(); return GetImpl()->FullFocus(); }
int AutoFocusInstance::IncrementalFocus() { RequireInitialized(); return GetImpl()->IncrementalFocus(); }
int AutoFocusInstance::GetLastFocusScore(double& score) { RequireInitialized(); return GetImpl()->GetLastFocusScore(score); }
int AutoFocusInstance::GetCurrentFocusScore(double& score) { RequireInitialized(); return GetImpl()->GetCurrentFocusScore(score); }
int AutoFocusInstance::AutoSetParameters() { RequireInitialized(); return GetImpl()->AutoSetParameters(); }
int AutoFocusInstance::GetOffset(double &offset) { RequireInitialized(); return GetImpl()->GetOffset(offset); }
int AutoFocusInstance::SetOffset(double offset) { RequireInitialized(); return GetImpl()->SetOffset(offset); }
