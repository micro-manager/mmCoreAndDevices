package mmcorej.org.json;

/**
 * A JSONObject that lazily initializes its contents from a BufferDataPointer.
 */
class LazyJSONObject extends JSONObject {
    private final BufferDataPointer dataPointer_;
    private boolean initialized_ = false;


    public LazyJSONObject(BufferDataPointer dataPointer) {
        this.dataPointer_ = dataPointer;
    }

    synchronized void initializeIfNeeded() throws JSONException {
        if (!initialized_) {
            try {
                Metadata md = new Metadata();
                dataPointer_.getMetadata(md);

                for (String key : md.GetKeys()) {
                    try {
                        put(key, md.GetSingleTag(key).GetValue());
                    } catch (Exception e) {
                        throw new JSONException("Failed to get value for key: " + key, e);
                    }
                }
                initialized_ = true;
            } catch (Exception e) {
                throw new JSONException("Failed to initialize metadata", e);
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
