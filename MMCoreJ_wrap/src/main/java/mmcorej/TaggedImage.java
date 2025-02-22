package mmcorej;

import mmcorej.org.json.JSONObject;

 /*
 * @author arthur
 */

public class TaggedImage {
   public Object pix;
   public JSONObject tags;

   public TaggedImage(Object pix, JSONObject tags) {
      this.pix = pix;
      this.tags = tags;
   }

   // This is so that this method can be callled on the 
   // TaggedImagePointer subclass, so pixels are loaded lazily.
   // For regular TaggedImage objects, pixels are already loaded.
   public Object getPixels() {
      return pix;
   }

}
