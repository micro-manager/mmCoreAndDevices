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

#include "GalvoInstance.h"


int GalvoInstance::PointAndFire(double x, double y, double time_us) { RequireInitialized(); return GetImpl()->PointAndFire(x, y, time_us); }
int GalvoInstance::SetSpotInterval(double pulseInterval_us) { RequireInitialized(); return GetImpl()->SetSpotInterval(pulseInterval_us); }
int GalvoInstance::SetPosition(double x, double y) { RequireInitialized(); return GetImpl()->SetPosition(x, y); }
int GalvoInstance::GetPosition(double& x, double& y) { RequireInitialized(); return GetImpl()->GetPosition(x, y); }
int GalvoInstance::SetIlluminationState(bool on) { RequireInitialized(); return GetImpl()->SetIlluminationState(on); }
double GalvoInstance::GetXRange() { RequireInitialized(); return GetImpl()->GetXRange(); }
double GalvoInstance::GetXMinimum() { RequireInitialized(); return GetImpl()->GetXMinimum(); }
double GalvoInstance::GetYRange() { RequireInitialized(); return GetImpl()->GetYRange(); }
double GalvoInstance::GetYMinimum() { RequireInitialized(); return GetImpl()->GetYMinimum(); }
int GalvoInstance::AddPolygonVertex(int polygonIndex, double x, double y) { RequireInitialized(); return GetImpl()->AddPolygonVertex(polygonIndex, x, y); }
int GalvoInstance::DeletePolygons() { RequireInitialized(); return GetImpl()->DeletePolygons(); }
int GalvoInstance::RunSequence() { RequireInitialized(); return GetImpl()->RunSequence(); }
int GalvoInstance::LoadPolygons() { RequireInitialized(); return GetImpl()->LoadPolygons(); }
int GalvoInstance::SetPolygonRepetitions(int repetitions) { RequireInitialized(); return GetImpl()->SetPolygonRepetitions(repetitions); }
int GalvoInstance::RunPolygons() { RequireInitialized(); return GetImpl()->RunPolygons(); }
int GalvoInstance::StopSequence() { RequireInitialized(); return GetImpl()->StopSequence(); }

std::string GalvoInstance::GetChannel()
{
   RequireInitialized();
   DeviceStringBuffer nameBuf(this, "GetChannel");
   int err = GetImpl()->GetChannel(nameBuf.GetBuffer());
   ThrowIfError(err, "Cannot get current channel name");
   return nameBuf.Get();
}
