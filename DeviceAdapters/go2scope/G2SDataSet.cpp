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
#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#include <boost/date_time/gregorian/gregorian.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <FreeImage/FreeImage.h>
#include "G2SCore/G2SException.h"
#include "G2SAcquisitionEngine/G2SDataSet.h"
#include "G2SAcquisitionEngine/G2SAcquisitionDefs.h"

//===============================================================================================================================
// Helper methods
//===============================================================================================================================
/**
 * Save 16-bpp image to disk (in TIFF format)
 * @param fname File path
 * @param pImg Image buffer
 * @param width Image width
 * @param height Image height
 * @author Nenad Amodaj
 * @version 1.0
 */
void save16BitImage(const char* fname, unsigned char* pImg, int width, int height)
{
   FIBITMAP* fif = FreeImage_AllocateT(FIT_UINT16, width, height, 16);
   if (fif == nullptr)
	   throw g2s::G2SException("FreeImage Memory allocation error");

   unsigned char* pBuf = reinterpret_cast<unsigned char*>(FreeImage_GetBits(fif));
   for(int i = 0; i < height; i++)
      memcpy(pBuf + i * width * 2, pImg + (height - 1 - i) * width * 2, width * 2);
   if (!FreeImage_Save(FIF_TIFF, fif, fname, 0))
	   throw g2s::G2SException("Failed saving 16-bit image");
   FreeImage_Unload(fif);
}

/**
 * Save 8-bpp image to disk (in TIFF format)
 * @param fname File path
 * @param pImg Image buffer
 * @param width Image width
 * @param height Image height
 * @author Nenad Amodaj
 * @version 1.0
 */
void save8BitImage(const char* fname, unsigned char* pImg, int width, int height)
{
   FIBITMAP* fif = FreeImage_AllocateT(FIT_BITMAP, width, height, 8);
   if (fif == nullptr)
	   throw g2s::G2SException("FreeImage Memory allocation error");
   unsigned char* pBuf = reinterpret_cast<unsigned char*>(FreeImage_GetBits(fif));
   for(int i = 0; i < height; i++)
      memcpy(pBuf + i * width, pImg + (height - 1 - i) * width, width);
   if (!FreeImage_Save(FIF_TIFF, fif, fname, 0))
		throw g2s::G2SException("Failed saving 8-bit image");
   FreeImage_Unload(fif);
}

//===============================================================================================================================
// Class implementation
//===============================================================================================================================
/**
 * Create data set directory. Automatically sufix the dataset name to create a unique name and not overwrite
 * existing data.
 * 
 * @param baseDir Base directory path
 * @param prefix Directory prefix
 * @throws G2SException
 */
std::string G2SDataSet::create(const std::string& baseDir, const std::string& prefix)
{
   if(created)
	{
		std::stringstream params;
		params << "Base dir: " << baseDir << std::endl;
		params << "Prefix: " << prefix << std::endl;
		params << "Data set name: " << dataSetName;
      throw g2s::G2SException("This dataset is already created", "G2SDataSet::create", params.str());
	}

   std::string actualName(prefix); 
   actualName = generateUniqueRootName(prefix, baseDir);

   std::ostringstream os;
   os << baseDir << "/" << actualName;
   root = os.str();

   bool ret = false;
   boost::system::error_code ec;
   ret = boost::filesystem::create_directories(boost::filesystem::path(root), ec);
   if(!ret)
   {
      std::stringstream params;
		params << "Native error code: " << ec.value() << std::endl;
		params << "Native error message: " << ec.message() << std::endl;
		params << "Base dir: " << baseDir << std::endl;
		params << "Prefix: " << prefix << std::endl;
		params << "Root directory: " << root << std::endl;
		params << "Data set name: " << actualName;
		root = "";
      dataSetName = "";
		throw g2s::G2SException("Unable to create root directory", "G2SDataSet::create", params.str());
   }
   dataSetName = actualName;
   created = true;
   return dataSetName;
}

/**
 * Insert single image annotated with coordinate metadata
 * @param img Image object
 * @throws G2SException
 */
void G2SDataSet::insertImage(g2s::data::G2SImage& img)
{
   g2s::G2SConfig md = img.getMetadata();
   g2s::G2SConfig coords;
   try
   {
      coords = md.getElement(acq::COORDINATES);
	}
	catch(g2s::G2SException& e)
	{
		e.addCall("G2SDataSet::insertImage");
		throw;
	}
   if(coords.empty()) 
	{
		std::stringstream params;
		params << "Image metadata: " << md.serialize();
      throw g2s::G2SException("Image metadata does not contain acquisition coordinates", "G2SDataSet::insertImage", params.str());
   }
	std::string imgPath = "";
	int frame = 0;
   int slice = 0;
   int channel = 0;
   int pos = 0;
	try
	{
      frame = coords.getInt(acq::FRAME, 0);
      slice = coords.getInt(acq::SLICE, 0);
      channel = coords.getInt(acq::CHANNEL, 0);
      pos = coords.getInt(acq::POSITION, 0);
      imgPath = getImagePath(frame, channel, slice, pos);
	}
	catch(g2s::G2SException& e)
	{
		e.addCall("G2SDataSet::insertImage");
		throw;
	}

   // create position directory if necessary
   boost::filesystem::path imageDir = boost::filesystem::path(getImageDir(pos));
   if(!boost::filesystem::exists(imageDir))
   {
      // the first time a position directory is created
      boost::system::error_code ec;
      bool ret = boost::filesystem::create_directories(imageDir, ec);
      if(!ret)
      {
			std::stringstream params;
			params << "Native error code: " << ec.value() << std::endl;
			params << "Native error message: " << ec.message() << std::endl;
			params << "Directory path: " << imageDir;
			throw g2s::G2SException("Directory structure creation failed", "G2SDataSet::insertImage", params.str());
      }
   }

   if(img.getBytesPerPixel() == 1)
      save8BitImage(imgPath.c_str(), &img.getPixels()[0], img.getWidth(), img.getHeight());
   else if (img.getBytesPerPixel() == 2)
      save16BitImage(imgPath.c_str(), &img.getPixels()[0], img.getWidth(), img.getHeight());
   else
	{
		std::stringstream params;
		params << "Image width: " << img.getWidth() << std::endl;
		params << "Image height: " << img.getHeight() << std::endl;
		params << "Image bit depth: " << img.getBitDepth() << std::endl;
		params << "Image channels: " << img.getNumberOfComponents() << std::endl;
		params << "Image pixel format: " << (int)img.getPixelFormat() << std::endl;	
		params << "Image pitch: " << img.getPitch() << std::endl;
		params << "Image buffer size: " << img.getPixelDataSize() << std::endl;		
		params << "Image path: " << imgPath;
      throw g2s::G2SException("Unsupported pixel byte depth", "G2SDataSet::insertImage", params.str());
	}
   // update index of saved images
   imageMetadata[generateKey(frame, channel, slice, pos)] = img.getMetadata().serialize();

   if(frame > maxFrame)
      maxFrame = frame;
   if(slice > maxSlice)
      maxSlice = slice;
   if(pos > maxPosition)
      maxPosition = pos;
   if(channel > maxChannel)
      maxChannel = channel;

   // maintain a set of used position indexes
   positions.insert(pos);

   // save dimensions of the first image
   if(imgWidth == 0)
      imgWidth = img.getWidth();
   if(imgHeight == 0)
      imgHeight = img.getHeight();
   if(pixelDepth == 0)
      pixelDepth = img.getBytesPerPixel();
   if(bitDepth == 0)
      bitDepth = img.getBitDepth();
   if(pixSizeUm == 0.0 && img.getMetadata().hasValue(acq::IMAGE_METADATA) && img.getMetadata().getElement(acq::IMAGE_METADATA).hasValue(mm::image::PIXEL_SIZE))
      pixSizeUm = img.getMetadata().getElement(acq::IMAGE_METADATA).getDouble(mm::image::PIXEL_SIZE);
}

/**
 * Generate and write metadata to disk 
 * @param pos Position index
 * @param proto Acquisition protocol
 * @throws G2SException
 */
void G2SDataSet::createPosMetadataFile(int pos, const SequenceSettings& proto)
{
   g2s::G2SConfig md; // complete metadata
   g2s::G2SConfig mdSum;
   mdSum.set(mm::summary::SOURCE, "Go2Scope");
   mdSum.set(mm::summary::MM_APP_VERSION, "1.4.14");
   mdSum.set(mm::summary::METADATA_VERSION, "9");

   mdSum.set(mm::summary::DATE, to_simple_string(boost::gregorian::day_clock::local_day()));
   mdSum.set(mm::summary::TIME, to_simple_string(boost::posix_time::second_clock::local_time()));

   mdSum.set(mm::summary::PREFIX, dataSetName);
   mdSum.set(mm::summary::DIRECTORY, root);

   mdSum.set(mm::summary::FRAMES, maxFrame + 1);
   mdSum.set(mm::summary::CHANNELS, maxChannel + 1);
   mdSum.set(mm::summary::SLICES, maxSlice + 1);
   mdSum.set(mm::summary::POSITIONS, maxPosition + 1);

   mdSum.set(mm::summary::WIDTH, imgWidth);
   mdSum.set(mm::summary::HEIGHT, imgHeight);
   if(pixelDepth == 1)
      mdSum.set(std::string(mm::summary::PIX_TYPE), std::string(mm::values::PIX_TYPE_GRAY_8));
   else if(pixelDepth == 2)
      mdSum.set(std::string(mm::summary::PIX_TYPE), std::string(mm::values::PIX_TYPE_GRAY_16));
   else if(pixelDepth == 4)
      mdSum.set(std::string(mm::summary::PIX_TYPE), std::string(mm::values::PIX_TYPE_RGB_32));
   else
	{
		std::stringstream params;
		params << "Pixel depth: " << pixelDepth << std::endl;
		params << "Position index: " << pos << std::endl;
		params << "Acquisition protocol: " << proto.serializeToString();
      throw g2s::G2SException("Pixel type not supported", "G2SDataSet::createPosMetadataFile", params.str());
	}
   
	mdSum.set(mm::summary::BIT_DEPTH, bitDepth);
   mdSum.set(mm::summary::PIXEL_ASPECT, 1.0);

   mdSum.set(mm::summary::SLICES_FIRST, proto.slicesFirst);
   mdSum.set(mm::summary::TIME_FIRST, proto.timeFirst);

   std::vector<std::string> chNames;
   std::vector<int> chColors;
   for(std::size_t i = 0;  i < proto.channels.size(); i++)
   {
      chNames.push_back(proto.channels[i].channelName);
      chColors.push_back(proto.channels[i].displayColor);
   }
   mdSum.set(mm::summary::CHANNEL_NAMES, chNames);
   mdSum.set(mm::summary::CHANNEL_COLORS, chColors);
   mdSum.set(mm::summary::INTERVAL_MS, proto.intervalMs);
   try
   {
		// TODO: add custom metadata
      md.set(mm::SUMMARY, mdSum); // add summary
      for(int f = 0; f <= maxFrame; f++)
      {
         for(int s = 0; s <= maxSlice; s++)
         {
            for(int c = 0; c <= maxChannel; c++)
            {
               // add image metadata
               std::string attachedMdStream = imageMetadata[generateKey(f, c, s, pos)];
               g2s::G2SConfig attachedMd;
               attachedMd.parse(attachedMdStream);
               g2s::G2SConfig mdImg = attachedMd.getElement(acq::IMAGE_METADATA);
               mdImg.set(mm::image::FILE_NAME, getImageFileName(f, c, s));
               md.set(generateReducedKey(f, c, s), mdImg); // add image metadata
            }
         }
      }
		// save metadata txt file
      md.save(getImageDir(pos) + "/metadata.txt");
   }
   catch(g2s::G2SException& e)
   {
		e.addCall("G2SDataSet::createPosMetadataFile");
   	throw;
   }
}

/**
 * Generate and write display and comments file
 * @param proto Acquisition protocol
 * @throws G2SException
 */
void G2SDataSet::createDisplayAndCommentsFile(const SequenceSettings& proto)
{
   // display_and_comments.txt
	try
	{
      std::vector<g2s::G2SConfig> mdDisp;
      for(std::size_t i = 0;  i < proto.channels.size(); i++)
      {
         g2s::G2SConfig mdChannel;
         mdChannel.set(mm::display::NAME, proto.channels[i].channelName);
         mdChannel.set(mm::display::COLOR, proto.channels[i].displayColor);
         mdChannel.set(mm::display::MIN, 0);
         mdChannel.set(mm::display::MAX, 65535);
         mdChannel.set(mm::display::GAMMA, 1.0);
         mdChannel.set(mm::display::HISTOGRAM_MAX, -1);
         mdChannel.set(mm::display::DISPLAY_MODE, 1);
         mdDisp.push_back(mdChannel);
      }

      g2s::G2SConfig mdSum;
      mdSum.set(mm::display::SUMMARY, proto.comment);

      g2s::G2SConfig md;
      md.set(mm::display::CHANNELS, mdDisp);
      md.set(mm::display::COMMENTS, mdSum);
		md.save(getPath() + "/display_and_comments.txt");
	}
	catch(g2s::G2SException& e)
	{
		e.addCall("G2SDataSet::createDisplayAndCommentsFile");
      throw;
	}
}

/**
 * Create and save micro-manager metadata files
 * @throws G2SException
 */
void G2SDataSet::generateMetadataFiles(const SequenceSettings& proto)
{
	try
	{
		// create metadata.txt for each position
		for(int p = 0; p <= maxPosition; p++)
		{
			if(positions.find(p) != positions.end())
				createPosMetadataFile(p, proto);
		}

		// create display_and_comment.txt file
		createDisplayAndCommentsFile(proto);
	}
	catch(g2s::G2SException& e)
	{
		e.addCall("G2SDataSet::generateMetadataFiles");
      throw;
	}
}

/**
 * Set summary property (metadata entry)
 * @param key Property key
 * @param value Property value
 */
void G2SDataSet::setSummaryProperty(const std::string& key, const std::string& value) noexcept
{
   summaryProps[key] = value;
}

/**
 * Generates a key for storing images in a map
 * @param frame Time frame index
 * @param channel Channel index
 * @param slice Slice index
 * @param position Position index
 * @return Image key
 */
std::string G2SDataSet::generateKey(int frame, int channel, int slice, int position) noexcept
{
   std::ostringstream os;
   os << "FrameKey-" << frame << "-" << channel << "-" << slice << "-" << position;  
   return os.str();
}

/**
 * Generates a key for storing images on disk (inside a position folder)
 * @param frame Time frame index
 * @param channel Channel index
 * @param slice Slice index
 * @return Image key
 */
std::string G2SDataSet::generateReducedKey(int frame, int channel, int slice) noexcept
{
   std::ostringstream os;
   os << "FrameKey-" << frame << "-" << channel << "-" << slice;  
   return os.str();
}

/**
 * Get data set path
 * @return Root directory path
 */
std::string G2SDataSet::getPath() const noexcept
{
   return root;
}

/**
 * Get image path
 * @param frame Time frame index
 * @param channel Channel index
 * @param slice Slice index
 * @param position Position index
 * @return Image path relative to data set root
 */
std::string G2SDataSet::getImagePath(int frame, int channel, int slice, int pos) noexcept
{
   std::ostringstream os;
   os << getImageDir(pos) << "/" << getImageFileName(frame, channel, slice);
   return os.str();
}

/**
 * Get image directory name
 * @param pos Position index
 * @return Position directory name
 */
std::string G2SDataSet::getImageDir(int pos) noexcept
{
   if(pos >= positionNames.size())
      return str(boost::format("%s/Pos_%03d") % getPath() % pos);
   else
      return str(boost::format("%s/%s") % getPath() % positionNames[pos]);
}

/**
 * Get image file name
 * @param frame Time frame index
 * @param channel Channel index
 * @param slice Slice index
 * @return Image file name
 */
std::string G2SDataSet::getImageFileName(int frame, int channel, int slice) noexcept
{
   std::string chName;
   if(channel >= channelNames.size())
      return str(boost::format("img_%09d_channel%d_%03d.tif") % frame % channel % slice);
   else
      return str(boost::format("img_%09d_%s_%03d.tif") % frame % channelNames[channel] % slice);
}

/**
 * Generate unique name for the data set root directory
 * @param acqName Acquisition name
 * @param baseDir Base directory path
 * @return Data set root directory name
 */
std::string G2SDataSet::generateUniqueRootName(const std::string& acqName, const std::string& baseDir) noexcept
{
   // create new acquisition directory
   int suffixCounter = 0;
   std::string testPath;
   boost::filesystem::path testDir;
   std::string testName;
   do 
	{
      std::ostringstream osTestName;
      osTestName << acqName << "_" << suffixCounter;
      testName = osTestName.str();

      std::ostringstream os;
      os << baseDir << "/" << testName;
      testPath = os.str();

      suffixCounter++;
      testDir = boost::filesystem::path(testPath);
   } 
	while(boost::filesystem::exists(testDir));
   return testName;
}

/**
 * Generate JSON object containing summary metadata
 * based on the initial image parameters and acquisition protocol
 * @param pos Position index
 * @param proto Acquisition protocol
 * @param imgWidth Image width
 * @param imgHeight Image height
 * @param pixelDepth Pixel depth
 * @param bitDepth Image bit depth
 * @return Metadata object
 * @throws G2SException
 */
g2s::G2SConfig G2SDataSet::generateSummaryMetadata(int pos, const SequenceSettings& proto, int imgWidth, int imgHeight, int pixelDepth, int bitDepth)
{
   g2s::G2SConfig mdSum;
   mdSum.set(mm::summary::SOURCE, "Go2Scope");
   mdSum.set(mm::summary::MM_APP_VERSION, "1.4.14");
   mdSum.set(mm::summary::METADATA_VERSION, "10");

   mdSum.set(mm::summary::DATE, to_simple_string(boost::gregorian::day_clock::local_day()));
   mdSum.set(mm::summary::TIME, to_simple_string(boost::posix_time::second_clock::local_time()));

   mdSum.set(mm::summary::PREFIX, proto.prefix);
   mdSum.set(mm::summary::DIRECTORY, proto.root);

   mdSum.set(mm::summary::FRAMES, proto.numFrames);
   mdSum.set(mm::summary::CHANNELS, (int)proto.channels.size());
   mdSum.set(mm::summary::SLICES, (int)proto.slices.size());
   mdSum.set(mm::summary::POSITIONS, (int)proto.positions.size());

   mdSum.set(mm::summary::WIDTH, imgWidth);
   mdSum.set(mm::summary::HEIGHT, imgHeight);
   if (pixelDepth == 1)
      mdSum.set(std::string(mm::summary::PIX_TYPE), std::string(mm::values::PIX_TYPE_GRAY_8));
   else if (pixelDepth == 2)
      mdSum.set(std::string(mm::summary::PIX_TYPE), std::string(mm::values::PIX_TYPE_GRAY_16));
   else if (pixelDepth == 4)
      mdSum.set(std::string(mm::summary::PIX_TYPE), std::string(mm::values::PIX_TYPE_RGB_32));
   else
	{
		std::stringstream params;
		params << "Pixel depth: " << pixelDepth << std::endl;
		params << "Bit depth: " << bitDepth << std::endl;
		params << "Position index: " << pos << std::endl;
		params << "Acquisition protocol: " << proto.serializeToString();
      throw g2s::G2SException("Pixel type not supported", "G2SDataSet::generateSummaryMetadata", params.str());
	}

   mdSum.set(mm::summary::BIT_DEPTH, bitDepth);
   mdSum.set(mm::summary::PIXEL_ASPECT, 1.0);

   mdSum.set(mm::summary::SLICES_FIRST, proto.slicesFirst);
   mdSum.set(mm::summary::TIME_FIRST, proto.timeFirst);
   
	std::vector<std::string> chNames;
   std::vector<int> chColors;
   for(int i = 0; i < proto.channels.size(); i++)
   {
      chNames.push_back(proto.channels[i].channelName);
      chColors.push_back(proto.channels[i].displayColor);
   }
   mdSum.set(mm::summary::CHANNEL_NAMES, chNames);
   mdSum.set(mm::summary::CHANNEL_COLORS, chColors);

   mdSum.set(mm::summary::INTERVAL_MS, proto.intervalMs);
   return mdSum;
}
