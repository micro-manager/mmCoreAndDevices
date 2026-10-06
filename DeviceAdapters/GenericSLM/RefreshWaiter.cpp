// COPYRIGHT:     (c) 2009-2015 Regents of the University of California
//                (c) 2016 Open Imaging, Inc.
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
//
// AUTHOR:        Mark Tsuchida, 2016
//                Based on older code by Arthur Edelstein, 2009

#include "RefreshWaiter.h"


namespace {

    struct FindGuidContext
    {
        const std::string* monitorName;
        GUID guid;
        bool found;
    };

    // https://learn.microsoft.com/en-us/windows/win32/api/ddraw/nc-ddraw-lpddenumcallbackexa
    BOOL WINAPI FindGuidForMonitor(GUID* pGUID, LPSTR,
        LPSTR, LPVOID pContext, HMONITOR mointorHandle)
    {   
        FindGuidContext* ctx = static_cast<FindGuidContext*>(pContext);
        if (pGUID == 0 || mointorHandle == 0)
            return TRUE; 

        MONITORINFOEXA info;
        info.cbSize = sizeof(info);
        if (GetMonitorInfoA(mointorHandle, &info) && *ctx->monitorName == info.szDevice)
        {
            ctx->guid = *pGUID;
            ctx->found = true;
            return FALSE;
        }
        return TRUE;
    }

}


RefreshWaiter::RefreshWaiter() :
    hDDrawLib_(0),
    directDrawCreate_(0),
    directDrawEnumerateEx_(0),
    directDraw_(0)
{
    hDDrawLib_ = LoadLibraryA("ddraw.dll");
    if (!hDDrawLib_)
        return;

    directDrawCreate_ = (DirectDrawCreateFunc)GetProcAddress(hDDrawLib_,
        "DirectDrawCreate");
    directDrawEnumerateEx_ = (DirectDrawEnumerateExFunc)GetProcAddress(hDDrawLib_,
        "DirectDrawEnumerateExA");
    if (!directDrawCreate_)
        return;

    HRESULT hr = directDrawCreate_(0, &directDraw_, 0);
    if (hr != DD_OK)
    {
        directDraw_ = 0;
        return;
    }
}


RefreshWaiter::~RefreshWaiter()
{
    if (directDraw_)
        directDraw_->Release();

    if (hDDrawLib_)
        FreeLibrary(hDDrawLib_);
}


int RefreshWaiter::SetMonitor(const std::string& monitorName)
{
    if (!directDrawCreate_)
        return 1;

    GUID* pGuid = 0;
    FindGuidContext ctx;
    ctx.monitorName = &monitorName;
    ctx.found = false;

    if (!monitorName.empty())
    {
        if (!directDrawEnumerateEx_)
            return 1;

        directDrawEnumerateEx_(FindGuidForMonitor, &ctx, DDENUM_ATTACHEDSECONDARYDEVICES);
        if (!ctx.found)
            return 1;

        pGuid = &ctx.guid;
    }

    IDirectDraw* newDirectDraw = 0;
    HRESULT hr = directDrawCreate_(pGuid, &newDirectDraw, 0);
    if (hr != DD_OK)
        return (int)hr;

    if (directDraw_)
        directDraw_->Release();
    directDraw_ = newDirectDraw;
    return 0;
}


int RefreshWaiter::WaitForVerticalBlank()
{
    if (!directDraw_)
        return 1;

    HRESULT result =  directDraw_->WaitForVerticalBlank(DDWAITVB_BLOCKBEGIN, 0);

    if (result == DD_OK)
    {
        return 0;
    }
    return (int) result;

}
