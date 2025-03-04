package mmcorej;

import mmcorej.org.json.JSONObject;
import mmcorej.org.json.JSONException;
import java.util.Iterator;
import java.util.Collections;

/**
 * TaggedImagePointer is a wrapper around a pointer to an image in the v2 buffer 
 * (a BufferDataPointer object). It provides copy-free access to data in the 
 * C++ layer until the data is actually needed. This class implements lazy loading of
 * image data to optimize memory usage and performance.
 * 
 * <p>This class extends TaggedImage and manages the lifecycle of image data stored
 * in native memory. It ensures proper release of resources when the image data is
 * no longer needed.</p>
 * 
 * <p>This class implements AutoCloseable, allowing it to be used with try-with-resources
 * statements for automatic resource management.</p>
 */
public class TaggedImagePointer extends TaggedImage implements AutoCloseable {
   
   public LazyJSONObject tags;

   private BufferDataPointer dataPointer_;
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

   public Object getPixels() {
      loadData();
      return pix;
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
   private synchronized void loadData() throws IllegalStateException {
      if (!released_) {
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
         dataPointer_.dispose(); 
         tags.releasePointer();
         released_ = true;
         dataPointer_ = null;
      }
   }

   /**
    * Closes this resource, relinquishing any underlying resources.
    * This method is invoked automatically when used in a try-with-resources statement.
    * 
    * <p>This implementation calls {@link #release()} to free native memory.</p>
    */
   @Override
   public void close() {
      release();
   }

}

