package mmcorej;

import mmcorej.org.json.JSONObject;

final class TaggedImageCreator {

   static JSONObject metadataToMap(Metadata md) {
      JSONObject tags = new JSONObject();
      for (String key : md.GetKeys()) {
         try {
            tags.put(key, md.GetSingleTag(key).GetValue());
         } catch (Exception e) {
         }
      }
      return tags;
   }

   static String getROITag(CMMCore core) throws java.lang.Exception {
      String roi = "";
      int[] x = new int[1];
      int[] y = new int[1];
      int[] xSize = new int[1];
      int[] ySize = new int[1];
      core.getROI(x, y, xSize, ySize);
      roi += x[0] + "-" + y[0] + "-" + xSize[0] + "-" + ySize[0];
      return roi;
   }

   static String getPixelType(CMMCore core) {
      int depth = (int) core.getBytesPerPixel();
      int numComponents = (int) core.getNumberOfComponents();
      switch (depth) {
         case 1:
            return "GRAY8";
         case 2:
            return "GRAY16";
         case 4: {
            if (numComponents == 1)
               return "GRAY32";
            else
               return "RGB32";
         }
         case 8:
            return "RGB64";
      }
      return "";
   }

   static String getMultiCameraChannel(JSONObject tags, int cameraChannelIndex) {
      try {
         String camera = tags.getString("Core-Camera");
         String physCamKey = camera + "-Physical Camera " + (1 + cameraChannelIndex);
         if (tags.has(physCamKey)) {
            try {
               return tags.getString(physCamKey);
            } catch (Exception e2) {
               return null;
            }
         } else {
            return null;
         }
      } catch (Exception e) {
         return null;
      }
   }

   static TaggedImage createTaggedImage(
         CMMCore core, boolean includeSystemStateCache,
         Object pixels, Metadata md, int cameraChannelIndex) throws java.lang.Exception {
      TaggedImage image = createTaggedImage(core, includeSystemStateCache, pixels, md);
      JSONObject tags = image.tags;

      if (!tags.has("CameraChannelIndex")) {
         tags.put("CameraChannelIndex", cameraChannelIndex);
         tags.put("ChannelIndex", cameraChannelIndex);
      }
      if (!tags.has("Camera")) {
         String physicalCamera = getMultiCameraChannel(tags, cameraChannelIndex);
         if (physicalCamera != null) {
            tags.put("Camera", physicalCamera);
            tags.put("Channel", physicalCamera);
         }
      }
      return image;
   }

   static TaggedImage createTaggedImage(
         CMMCore core, boolean includeSystemStateCache,
         Object pixels, Metadata md) throws java.lang.Exception {
      JSONObject tags = metadataToMap(md);
      PropertySetting setting;
      if (includeSystemStateCache) {
         Configuration config = core.getSystemStateCache();
         for (int i = 0; i < config.size(); ++i) {
            setting = config.getSetting(i);
            String key = setting.getDeviceLabel() + "-" + setting.getPropertyName();
            String value = setting.getPropertyValue();
            tags.put(key, value);
         }
      }
      tags.put("BitDepth", core.getImageBitDepth());
      tags.put("PixelSizeUm", core.getPixelSizeUm(true));
      tags.put("PixelSizeAffine", core.getPixelSizeAffineAsString());
      tags.put("ROI", getROITag(core));
      tags.put("Width", core.getImageWidth());
      tags.put("Height", core.getImageHeight());
      tags.put("PixelType", getPixelType(core));
      tags.put("Frame", 0);
      tags.put("FrameIndex", 0);
      tags.put("Position", "Default");
      tags.put("PositionIndex", 0);
      tags.put("Slice", 0);
      tags.put("SliceIndex", 0);
      String channel = core.getCurrentConfigFromCache(
            core.getPropertyFromCache("Core", "ChannelGroup"));
      if ((channel == null) || (channel.length() == 0)) {
         channel = "Default";
      }
      tags.put("Channel", channel);
      tags.put("ChannelIndex", 0);

      try {
         tags.put("Binning", core.getProperty(core.getCameraDevice(), "Binning"));
      } catch (Exception ex) {
      }

      return new TaggedImage(pixels, tags);
   }

   private TaggedImageCreator() {
   }
}
