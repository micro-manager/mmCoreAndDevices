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


int GalvoInstance::PointAndFire(double x, double y, double time_us) { RequireInitialized(__func__); return GetImpl()->PointAndFire(x, y, time_us); }
int GalvoInstance::SetSpotInterval(double pulseInterval_us) { RequireInitialized(__func__); return GetImpl()->SetSpotInterval(pulseInterval_us); }
int GalvoInstance::SetPosition(double x, double y) { RequireInitialized(__func__); return GetImpl()->SetPosition(x, y); }
int GalvoInstance::GetPosition(double& x, double& y) { RequireInitialized(__func__); return GetImpl()->GetPosition(x, y); }
int GalvoInstance::SetIlluminationState(bool on) { RequireInitialized(__func__); return GetImpl()->SetIlluminationState(on); }
double GalvoInstance::GetXRange() { RequireInitialized(__func__); return GetImpl()->GetXRange(); }
double GalvoInstance::GetXMinimum() { RequireInitialized(__func__); return GetImpl()->GetXMinimum(); }
double GalvoInstance::GetYRange() { RequireInitialized(__func__); return GetImpl()->GetYRange(); }
double GalvoInstance::GetYMinimum() { RequireInitialized(__func__); return GetImpl()->GetYMinimum(); }
int GalvoInstance::AddPolygonVertex(int polygonIndex, double x, double y) { RequireInitialized(__func__); return GetImpl()->AddPolygonVertex(polygonIndex, x, y); }
int GalvoInstance::DeletePolygons() { RequireInitialized(__func__); return GetImpl()->DeletePolygons(); }
int GalvoInstance::RunSequence() { RequireInitialized(__func__); return GetImpl()->RunSequence(); }
int GalvoInstance::LoadPolygons() { RequireInitialized(__func__); return GetImpl()->LoadPolygons(); }
int GalvoInstance::SetPolygonRepetitions(int repetitions) { RequireInitialized(__func__); return GetImpl()->SetPolygonRepetitions(repetitions); }
int GalvoInstance::RunPolygons() { RequireInitialized(__func__); return GetImpl()->RunPolygons(); }
int GalvoInstance::StopSequence() { RequireInitialized(__func__); return GetImpl()->StopSequence(); }

std::string GalvoInstance::GetChannel()
{
   RequireInitialized(__func__);
   DeviceStringBuffer nameBuf(this, "GetChannel");
   int err = GetImpl()->GetChannel(nameBuf.GetBuffer());
   ThrowIfError(err, "Cannot get current channel name");
   return nameBuf.Get();
}
