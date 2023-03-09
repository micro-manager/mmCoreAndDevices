///////////////////////////////////////////////////////////////////////////////
// FILE:          PIController.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   PI GCS Controller Driver
//
// AUTHOR:        Nenad Amodaj, nenad@amodaj.com, 08/28/2006
//                Steffen Rau, s.rau@pi.ws, 28/03/2008
// COPYRIGHT:     University of California, San Francisco, 2006
//                Physik Instrumente (PI) GmbH & Co. KG, 2008
// LICENSE:       This file is distributed under the BSD license.
//                License text is included with the source distribution.
//
//                This file is distributed in the hope that it will be useful,
//                but WITHOUT ANY WARRANTY; without even the implied warranty
//                of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
//
//                IN NO EVENT SHALL THE COPYRIGHT OWNER OR
//                CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
//                INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES.
// CVS:           $Id: Controller.h,v 1.17, 2018-09-26 11:11:16Z, Steffen Rau$
//

#ifndef PI_CONTROLLER_OBSERVER_H_INCLUDED
#define PI_CONTROLLER_OBSERVER_H_INCLUDED

class PIController;
class PIControllerObserver
{
public:
   PIControllerObserver () {}
   virtual ~PIControllerObserver () {};

   virtual void OnControllerDeleted () = 0;
};



#endif //PI_CONTROLLER_OBSERVER_H_INCLUDED
