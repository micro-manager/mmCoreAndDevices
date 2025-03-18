///////////////////////////////////////////////////////////////////////////////
// FILE:          Util.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     Device Driver Tests
//-----------------------------------------------------------------------------
// DESCRIPTION:   Helper methods
//
// AUTHOR:        Milos Jovanovic <milos@tehnocad.rs>
//
// COPYRIGHT:     Nenad Amodaj, Chan Zuckerberg Initiative, 2024
//
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
// NOTE:          Storage Device development is supported in part by
//                Chan Zuckerberg Initiative (CZI)
// 
///////////////////////////////////////////////////////////////////////////////
#include <sstream>
#include "MMCore.h"

/**
 * Generate image metadata
 * @param core MM Core instance
 * @param imgind Image index
 * @return Image metadata (JSON)
 */
std::string generateImageMeta(CMMCore& core, int imgind)
{
	// Calculate pixel type
	std::string pixtype = "";
	auto depth = core.getBytesPerPixel();
	auto numComponents = core.getNumberOfComponents();
	switch (depth) {
		case 1:
			pixtype = "GRAY8";
			break;
		case 2:
			pixtype = "GRAY16";
			break;
		case 4:
			if(numComponents == 1)
				pixtype = "GRAY32";
			else
				pixtype = "RGB32";
			break;
		case 8:
			pixtype = "RGB64";
			break;
		default:
			break;
	}

	// Calculate ROI
	int x = 0, y = 0, w = 0, h = 0;
	core.getROI(x, y, w, h);
	std::string roi = std::to_string(x) + "-" + std::to_string(y) + "-" + std::to_string(w) + "-" + std::to_string(h);

	// Calculate ROI affine transform
	auto aff = core.getPixelSizeAffine(true);
	std::string psizeaffine = "";
	if(aff.size() == 6) 
	{
		for(int i = 0; i < 5; ++i)
			psizeaffine += std::to_string(aff[i]) + ";";
		psizeaffine += std::to_string(aff[5]);
	}
	
	// Write JSON
	std::stringstream ss;
	ss << "{";
	Configuration config = core.getSystemStateCache();
	for(int i = 0; (long)i < config.size(); ++i) 
	{
		PropertySetting setting = config.getSetting((long)i);
		auto key = setting.getDeviceLabel() + "-" + setting.getPropertyName();
		auto value = setting.getPropertyValue();
		ss << "\"" << key << "\":\"" << value << "\",";
	}
	ss << "\"BitDepth\":" << core.getImageBitDepth() << ",";
	ss << "\"PixelSizeUm\":" << core.getPixelSizeUm(true) << ",";
	ss << "\"PixelSizeAffine\":\"" << psizeaffine << "\",";
	ss << "\"ROI\":\"" << roi << "\",";
	ss << "\"Width\":" << core.getImageWidth() << ",";
	ss << "\"Height\":" << core.getImageHeight() << ",";
	ss << "\"PixelType\":\"" << pixtype << "\",";
	ss << "\"Frame\":0,";
	ss << "\"FrameIndex\":0,";
	ss << "\"Position\":\"Default\",";
	ss << "\"PositionIndex\":0,";
	ss << "\"Slice\":0,";
	ss << "\"SliceIndex\":0,";
	auto channel = core.getCurrentConfigFromCache(core.getPropertyFromCache("Core", "ChannelGroup").c_str());
	if(channel.empty())
		channel = "Default";
	ss << "\"Channel\":\""<< channel << "\",";
	ss << "\"ChannelIndex\":0,";

	try { ss << "\"BitDepth\":\"" << core.getProperty(core.getCameraDevice().c_str(), "Binning") << "\","; } catch(...) { }

	ss << "\"Image-index\":" << imgind;
	ss << "}";
	return ss.str();
}
