package mmcorej;

import java.util.Iterator;
import java.util.Collections;
import mmcorej.BufferDataPointer;
import mmcorej.Metadata;
import mmcorej.org.json.JSONException;
import mmcorej.org.json.JSONObject;
import mmcorej.CMMCore;

/**
 * A JSONObject that lazily initializes its contents from a BufferDataPointer.
 */
class LazyJSONObject extends JSONObject {
    private BufferDataPointer dataPointer_;
    private boolean initialized_ = false;


    public LazyJSONObject(BufferDataPointer dataPointer) {
        this.dataPointer_ = dataPointer;
    }

    /**
     * Releases the BufferDataPointer associated with this LazyJSONObject.

     */
    public void releasePointer() {
        dataPointer_ = null;
    }

    synchronized void initializeIfNeeded() throws JSONException {
        if (!initialized_) {
            try {
                Metadata md = new Metadata();
                dataPointer_.getMetadata(md);

                // This handles some type conversions
                JSONObject tags = CMMCore.metadataToMap(md);
                Iterator<String> keyIter = tags.keys();
                while (keyIter.hasNext()) {
                    String key = keyIter.next();
                    super.put(key, tags.get(key));
                }
                initialized_ = true;
            } catch (Exception e) {
                throw new JSONException("Failed to initialize metadata");
            }
        }
    }

    @Override
    public Object get(String key) throws JSONException {
        initializeIfNeeded();
        return super.get(key);
    }

    @Override
    public JSONObject put(String key, Object value) throws JSONException {
        initializeIfNeeded();
        return super.put(key, value);
    }

    @Override
    public Object opt(String key) {
        try {
            initializeIfNeeded();
        } catch (JSONException e) {
            return null; // matches parent class behavior for missing keys
        }
        return super.opt(key);
    }

    @Override
    public boolean has(String key) {
        try {
            initializeIfNeeded();
        } catch (JSONException e) {
            return false;
        }
        return super.has(key);
    }

    @Override
    public Iterator<String> keys() {
        try {
            initializeIfNeeded();
        } catch (JSONException e) {
            // Return empty iterator if initialization fails
            return Collections.<String>emptyList().iterator();
        }
        return super.keys();
    }

    @Override
    public int length() {
        try {
            initializeIfNeeded();
        } catch (JSONException e) {
            return 0;
        }
        return super.length();
    }

    @Override
    public Object remove(String key) {
        try {
            initializeIfNeeded();
        } catch (JSONException e) {
            return null;
        }
        return super.remove(key);
    }
}
