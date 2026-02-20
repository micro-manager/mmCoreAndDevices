package mmcorej;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.security.SecureRandom;
import java.util.Locale;

/**
 * Handles extraction and loading of native library from JAR resources.
 */
final class NativeLibraryLoader {

    private NativeLibraryLoader() {}

    /**
     * Load a native library.
     *
     * First, try to find the library as a resource. If found, extract and
     * load. This method works even if the same {@code libName} is loaded from
     * multiple class loaders.
     *
     * A single resource path is searched: {@code /natives/<os>/<arch>}, where
     * {@code <os>} is one of {@code linux}, {@code macos}, or {@code windows}
     * and {@code <arch>} is one of {@code arm64} or {@code x86_64}.
     *
     * If the library was not found as a resource or could not be extracted,
     * try to load from {@code java.library.path}. This is intended for testing
     * and development only.
     *
     * (If the library was found as a resource and was successfully extracted
     * but loading failed, then {@code java.library.path} is not tried.)
     *
     * @param libName the name of the native library, without prefix or suffix
     * @throws UnsatisfiedLinkError if the native library could not be loaded
     * @throws NullPointerException if libName is null
     */
    static void load(String libName) {
        if (libName == null) {
            throw new NullPointerException("libName cannot be null");
        }
        try {
            loadPackaged(libName);
        } catch (Exception e1) { // Do not catch UnsatisfiedLinkError
            try {
                System.loadLibrary(libName);
            } catch (UnsatisfiedLinkError e2) {
                e2.addSuppressed(e1);
                throw e2;
            }
        }
    }

    private static void loadPackaged(String libName) throws Exception {
        String os = detectOs();
        String arch = detectArch();
        if (os == null || arch == null) {
            throw new UnsupportedOperationException(
                "Unsupported platform: " + System.getProperty("os.name") +
                "/" + System.getProperty("os.arch"));
        }

        // We use a unique filename for the extracted library, so that the
        // library can be loaded more than once from multiple class loaders.
        String extractedLibName =
            System.mapLibraryName(libName + "-" + randomHex(12));

        String resourceLibName = System.mapLibraryName(libName);
        String resourcePath =
            "/natives/" + os + "/" + arch + "/" + resourceLibName;

        try (InputStream in =
                 NativeLibraryLoader.class.getResourceAsStream(resourcePath)) {
            if (in == null) {
                throw new IOException("Native library resource not found: " +
                                      resourcePath);
            }

            if ("windows".equals(os)) {
                loadPackagedWindows(in, extractedLibName,
                                    libName + "-jnilibs");
            } else {
                loadPackagedUnix(in, extractedLibName, libName + "-jnilib-");
            }
        }
    }

    private static String detectOs() {
        String os = System.getProperty("os.name").toLowerCase(Locale.ROOT);
        if (os.contains("linux")) {
            return "linux";
        }
        if (os.contains("mac")) {
            return "macos";
        }
        if (os.contains("win")) {
            return "windows";
        }
        return null;
    }

    private static String detectArch() {
        String arch = System.getProperty("os.arch").toLowerCase(Locale.ROOT);
        if (arch.equals("amd64") || arch.equals("x86_64")) {
            return "x86_64";
        }
        if (arch.equals("aarch64")) {
            return "arm64";
        }
        return null;
    }

    private static String randomHex(int length) {
        byte[] bytes = new byte[(length + 1) / 2];
        new SecureRandom().nextBytes(bytes);
        StringBuilder sb = new StringBuilder();
        for (byte b : bytes) {
            sb.append(String.format("%02x", b));
        }
        return sb.substring(0, length);
    }

    private static void loadPackagedUnix(InputStream in, String libName,
                                         String dirPrefix) throws IOException {
        // On Unix-like systems, the temporary directory might be shared and
        // world-writable. We use a private temporary directory just while we
        // load the library, and remove the extracted library as soon as we're
        // done. (Usually the temporary directory is on a filesystem with POSIX
        // semantics, so we can delete an open file.)

        File nativesDir = Files.createTempDirectory(dirPrefix).toFile();
        File libFile = new File(nativesDir, libName);

        try {
            extractLibrary(in, libFile);
            System.load(libFile.getAbsolutePath());
        } finally {
            libFile.delete();
            nativesDir.delete();
        }
    }

    private static void loadPackagedWindows(InputStream in, String libName,
                                            String dirName)
        throws IOException {
        // On Windows, we cannot remove the extracted library from the current
        // process (not even with deleteOnExit()), because it won't be unloaded
        // once loaded. So we extract to a known directory and defer cleanup to
        // the next time we load. This should be safe because the temporary
        // directory is per-user on Windows.

        File nativesDir = getWindowsNativesDirectory(dirName);
        File libFile = new File(nativesDir, libName);

        // A lock directory is used to prevent races with cleanup
        File lockDir = makeLockPath(libFile);
        if (!lockDir.mkdir()) {
            throw new IOException("Cannot create native library lock: " +
                                  lockDir.getName());
        }

        try {
            extractLibrary(in, libFile);
            try {
                System.load(libFile.getAbsolutePath());
            } catch (UnsatisfiedLinkError e) {
                libFile.delete();
                throw e;
            }
        } finally {
            lockDir.delete();
        }

        cleanupWindows(libFile, nativesDir);
    }

    private static File getWindowsNativesDirectory(String dirName)
        throws IOException {
        String tmpdir = System.getProperty("java.io.tmpdir");
        File nativesDir = new File(tmpdir, dirName);
        if (!nativesDir.isDirectory() && !nativesDir.mkdirs()) {
            throw new IOException("Failed to create natives directory: " +
                                  nativesDir.getAbsolutePath());
        }
        return nativesDir;
    }

    private static File makeLockPath(File libFile) {
        String name = libFile.getName();
        int dot = name.lastIndexOf('.');
        String stem = (dot > 0 ? name.substring(0, dot) : name);
        return new File(libFile.getParentFile(), stem + ".lock");
    }

    private static void extractLibrary(InputStream in, File libFile)
        throws IOException {
        try (FileOutputStream out = new FileOutputStream(libFile)) {
            byte[] buffer = new byte[8192];
            int bytesRead;
            while ((bytesRead = in.read(buffer)) != -1) {
                out.write(buffer, 0, bytesRead);
            }
        }
    }

    private static void cleanupWindows(File libFile, File nativesDir) {
        // If a previously-extracted DLL is still in use (by another JVM or
        // classloader), deletion will silently fail. We skip files with a
        // corresponding lock directory to avoid a race with concurrent
        // extraction.

        File[] files = nativesDir.listFiles();
        if (files == null) {
            return;
        }
        long now = System.currentTimeMillis();
        long lockAgeThreshold = 10 * 60 * 1000;
        for (File file : files) {
            if (file.equals(libFile)) {
                continue;
            }
            // It is safe to delete old locks, because locks are only held
            // while extraction and loading of the library.
            if (file.isDirectory() && file.getName().endsWith(".lock")) {
                if (now - file.lastModified() > lockAgeThreshold) {
                    file.delete(); // Ignore failure
                }
                continue;
            }
            if (file.isFile()) {
                File lockDir = makeLockPath(file);
                if (lockDir.isDirectory()) {
                    continue;
                }
                file.delete(); // Ignore failure
            }
        }
    }
}
