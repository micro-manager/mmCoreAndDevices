package mmcorej;

import mmcorej.org.json.JSONObject;
import mmcorej.org.json.JSONException;
import java.util.Iterator;
import java.util.Collections;

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
   
   public LazyJSONObject tags;

   private final BufferDataPointer dataPointer_;
   private boolean released_ = false;

   /**
    * Constructs a new TaggedImagePointer.
    * 
    * @param dataPointer BufferDataPointer to the image data
    */
   public TaggedImagePointer(BufferDataPointer dataPointer) {
      super(null, null);  // Initialize parent with null pix
      this.dataPointer_ = dataPointer;
      this.tags = new LazyJSONObject(dataPointer);
   }  

   /**
    * Retrieves the pixels and metadata associated with this image.
    * 
    * <p>The first call to this method will copy the data from native memory
    * to Java memory and release the native buffer. Subsequent calls will
    * return the cached copy.</p>
    * 
    * @throws IllegalStateException if te image has already been released
    */
   public synchronized void loadData() throws IllegalStateException {
      if (released_) {
         throw new IllegalStateException("Image has been released");
      }
      
      if (this.pix == null) {
        try {
            this.pix = dataPointer_.getData();
            tags.initializeIfNeeded();
        } catch (Exception e) {
            throw new IllegalStateException("Failed to get pixel data", e);
        }
        release();
      }
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
         dataPointer_.release();
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

