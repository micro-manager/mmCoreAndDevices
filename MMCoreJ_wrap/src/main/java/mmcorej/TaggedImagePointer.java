package mmcorej;

import mmcorej.org.json.JSONObject;

/**
 * TaggedImagePointer is a wrapper around a pointer to an image in the v2 buffer.
 * It provides copy-free access to data in the C++ layer until the data is actually
 * needed. This class implements lazy loading of image data to optimize memory usage
 * and performance.
 * 
 * <p>This class extends TaggedImage and manages the lifecycle of image data stored
 * in native memory. It ensures proper release of resources when the image data is
 * no longer needed.</p>
 */
public class TaggedImagePointer extends TaggedImage {
   
   public JSONObject tags;

   private final long address_;
   private final CMMCore core_;
   private boolean released_ = false;
   private Object copiedPixels_ = null;

   /**
    * Constructs a new TaggedImagePointer.
    * 
    * @param address Memory address of the image data in native code
    * @param tags JSONObject containing metadata associated with the image
    * @param core Reference to the CMMCore instance managing the native resources
    */
   public TaggedImagePointer(long address, JSONObject tags, CMMCore core) {
      super(null, tags);  // Initialize parent with null pix
      this.address_ = address;
      this.core_ = core;
   }  

   /**
    * Retrieves the pixel data associated with this image.
    * 
    * <p>The first call to this method will copy the data from native memory
    * to Java memory and release the native buffer. Subsequent calls will
    * return the cached copy.</p>
    * 
    * @return Object containing the pixel data
    * @throws IllegalStateException if the image has already been released
    */
   public Object get_pixels() throws IllegalStateException {
      if (released_) {
         throw new IllegalStateException("Image has been released");
      }
      
      if (copiedPixels_ == null) {
        try {
            copiedPixels_ = core_.copyDataAtPointer(address_);
        } catch (Exception e) {
            throw new IllegalStateException("Failed to copy data at pointer", e);
        }
         release();
      }
      return copiedPixels_;
   }

   /**
    * Releases the native memory associated with this image.
    * 
    * <p>This method is synchronized to prevent concurrent access to the
    * release mechanism. Once released, the native memory cannot be accessed
    * again.</p>
    * 
    * @throws IllegalStateException if releasing the read access fails
    */
   public synchronized void release() {
      if (!released_) {
         try {
            core_.releaseReadAccess(address_);
         } catch (Exception e) {
            throw new IllegalStateException("Failed to release read access to image buffer", e);
         }
         released_ = true;
      }
   }

   /**
    * Ensures proper cleanup of native resources when this object is garbage collected.
    * 
    * @throws Throwable if an error occurs during finalization
    */
   @Override
   protected void finalize() throws Throwable {
      release();
      super.finalize();
   }
}


// class LazyJSONObject extends JSONObject {
//     private final long metadataPtr_;
//     private final CMMCore core_;
//     private boolean initialized_ = false;

//     public LazyJSONObject(long metadataPtr, CMMCore core) {
//         this.metadataPtr_ = metadataPtr;
//         this.core_ = core;
//     }

//     private synchronized void initializeIfNeeded() {
//         if (!initialized_) {
//             // Use core's existing metadataToMap helper
//             JSONObject tags = core_.metadataToMap(metadataPtr_);
         
//             // Copy all key/value pairs from the metadata map
//             for (String key : tags.keySet()) {
//                 try {
//                     put(key, tags.get(key));
//                 } catch (Exception e) {}
//             }
         
//             initialized_ = true;
//         }
//     }

//     @Override
//     public Object get(String key) {
//         initializeIfNeeded();
//         return super.get(key);
//     }
// }
