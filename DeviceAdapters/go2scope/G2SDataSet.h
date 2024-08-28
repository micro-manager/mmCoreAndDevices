// ==============================================================================================================================
// 
// Go2Scope
// www.go2scope.com
//
// Go2Scope acquisition engine library
// Data set (6D Image)
// 
// Copyright © 2014 by Luminous Point LLC. All rights reserved.
// www.luminous-point.com
//
// ==============================================================================================================================
#pragma once
#include <string>
#include <vector>
#include <map>
#include <set>
#include "G2SData/G2SImage.h"
#include "G2SAcquisitionEngine/G2SSequenceSettings.h"

/**
 * Micro-manage compatible data set on disk 
 * @author Nenad Amodaj
 * @version 1.0
 */
class G2SDataSet
{
public:
   //============================================================================================================================
	// Constructors & destructors
	//============================================================================================================================
	/**
	 * Default class constructor
	 */
	G2SDataSet() noexcept
   {
      maxChannel = -1;
      maxSlice = -1;
      maxFrame = -1;
      maxPosition = -1;
      created = false;
      imgWidth = 0;
      imgHeight = 0;
      pixelDepth = 0;
      bitDepth = 0;
      pixSizeUm = 0.0;
   }
	
public:
	//============================================================================================================================
	// Static methods
	//============================================================================================================================
   static std::string								generateKey(int frame, int channel, int slice, int position) noexcept;
   static std::string								generateReducedKey(int frame, int channel, int slice) noexcept;
   static g2s::G2SConfig							generateSummaryMetadata(int pos, const SequenceSettings& proto, int imgWidth, int imgHeight, int pixelDepth, int bitDepth);

public:
	//============================================================================================================================
	// Public interface
	//============================================================================================================================
   std::string											create(const std::string& root, const std::string& prefix);
   void													insertImage(g2s::data::G2SImage& img);
   void													setSummaryProperty(const std::string& key, const std::string& value) noexcept;
   std::string											getPath() const noexcept;
   void													setChannelNames(const std::vector<std::string>& names) noexcept { channelNames = names; }
   void													setPositionNames(const std::vector<std::string>& names) noexcept { positionNames = names; }
   void													generateMetadataFiles(const SequenceSettings& proto);

private:
	//============================================================================================================================
	// Internal methods
	//============================================================================================================================
   std::string											getImagePath(int frame, int channel, int slice, int pos) noexcept;
   std::string											generateUniqueRootName(const std::string& acqName, const std::string& baseDir) noexcept;
   std::string											getImageDir(int pos) noexcept;
   std::string											getImageFileName(int frame, int channel, int slice) noexcept;
   void													createPosMetadataFile(int pos, const SequenceSettings& proto);
   void													createDisplayAndCommentsFile(const SequenceSettings& proto);
	
private:
	//============================================================================================================================
	// Data members
	//============================================================================================================================
   std::map<std::string, std::string>			summaryProps;								///< Summary properties (metadata)
   std::map<std::string, std::string>			imageMetadata;								///< Index of image file names
   std::vector<std::string>						positionNames;								///< Position names
   std::vector<std::string>						channelNames;								///< Channel names
   std::string											root;											///< Root directory
   std::string											dataSetName;								///< Actual name of the data set	
   std::set<int>										positions;									///< Positions
   int													imgWidth;									///< Image width
   int													imgHeight;									///< Image height
   int													pixelDepth;									///< Pixel depth	
   int													bitDepth;									///< Bit depth
   int													maxSlice;									///< Max slice index
   int													maxPosition;								///< Max position index
   int													maxFrame;									///< Max time frame index
   int													maxChannel;									///< Max channel index
   bool													created;										///< Is data set complete
   double												pixSizeUm;									///< Pixel size (um)
};
