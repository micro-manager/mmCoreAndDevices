package mmcorej;

import java.io.File;
import java.net.URL;
import java.net.URLDecoder;
import java.util.ArrayList;
import java.util.List;

final class LegacyLibraryLoading {

   private static class FijiPaths {
      private static String URLtoFilePath(URL url) throws Exception {
         // We need to get rid of multiple protocols (jar: and file:)
         // and end up with an file path correct on every platform.
         // The following lines seem to work, though it's ugly:
         String url1 = URLDecoder.decode(url.getPath(), "UTF-8");
         String url2 = URLDecoder.decode(new URL(url1).getPath(), "UTF-8");
         return new File(url2).getAbsolutePath();
      }

      private static String getJarPath() {
         String classFile = "/mmcorej/CMMCore.class";
         try {
            String path = URLtoFilePath(CMMCore.class.getResource(classFile));
            int bang = path.indexOf('!');
            if (bang >= 0)
               path = path.substring(0, bang);
            return path;
         } catch (Exception e) {
            return "";
         }
      }

      private static String getPlatformString() {
         String osName = System.getProperty("os.name");
         String osArch = System.getProperty("os.arch");
         return osName.startsWith("Mac") ? "macosx"
               : (osName.startsWith("Win") ? "win" : osName.toLowerCase()) +
                     (osArch.indexOf("64") < 0 ? "32" : "64");
      }

      static List<File> getPaths() {
         // Return these dirs:
         // $MMCOREJ_JAR_DIR
         // $MMCOREJ_JAR_DIR/..
         // $MMCOREJ_JAR_DIR/../mm/$PLATFORM
         // $MMCOREJ_JAR_DIR/../.. # Was used by classic Micro-Manager
         // $MMCOREJ_JAR_DIR/../../mm/$PLATFORM
         // XXX: Which one is used by OpenSPIM?

         final File jarDir = new File(getJarPath()).getParentFile();
         final File jarDirParent = jarDir.getParentFile();
         final File jarDirGrandParent = jarDirParent.getParentFile();

         final String fijiPlatform = getPlatformString();
         final File jarDirParentFiji = new File(new File(jarDirParent, "mm"), fijiPlatform);
         final File jarDirGrandParentFiji = new File(new File(jarDirGrandParent, "mm"), fijiPlatform);

         final List<File> searchPaths = new ArrayList<File>();
         searchPaths.add(jarDir);
         searchPaths.add(jarDirParent);
         searchPaths.add(jarDirParentFiji);
         searchPaths.add(jarDirGrandParent);
         searchPaths.add(jarDirGrandParentFiji);
         return searchPaths;
      }
   }

   private static final String MM_PROPERTY_MMCOREJ_LIB_PATH = "mmcorej.library.path";
   private static final String MM_PROPERTY_MMCOREJ_LIB_STDERR_LOG = "mmcorej.library.loading.stderr.log";
   private static final String NATIVE_LIBRARY_NAME = "MMCoreJ_wrap";

   static void logLibraryLoading(String message) {
      boolean useStdErr = true;

      final String useStdErrProp = System.getProperty(MM_PROPERTY_MMCOREJ_LIB_STDERR_LOG, "0");
      if (useStdErrProp.equals("0") ||
            useStdErrProp.equalsIgnoreCase("false") ||
            useStdErrProp.equalsIgnoreCase("no")) {
         useStdErr = false;
      }

      if (useStdErr) {
         System.err.println("MMCoreJ native library loading: " + message);
      }
   }

   private static File getPreferredLibraryPath() {
      final String path = System.getProperty(MM_PROPERTY_MMCOREJ_LIB_PATH);
      if (path != null && path.length() > 0)
         return new File(path);
      return null;
   }

   private static File getHardCodedLibraryPath() {
      final String path = MMCoreJConstants.LIBRARY_PATH;
      if (path != null && path.length() > 0)
         return new File(path);
      return null;
   }

   private static boolean isLinux() {
      return System.getProperty("os.name").toLowerCase().startsWith("linux");
   }

   private static boolean loadNamedNativeLibrary(File dirPath, String libraryName) {
      final String libraryPath = new File(dirPath, libraryName).getAbsolutePath();
      if (new File(libraryPath).exists()) {
         logLibraryLoading("Try loading: " + libraryPath);
         System.load(libraryPath);
         logLibraryLoading("Successfully loaded: " + libraryPath);
         return true;
      }
      logLibraryLoading("Skipping nonexistent candidate: " + libraryPath);
      return false;
   }

   private static boolean loadNativeLibrary(File dirPath) {
      final String libraryName = System.mapLibraryName(NATIVE_LIBRARY_NAME);

      // On OS X, System.mapLibraryName() can return a name with a .dylib
      // suffix (since Java 7?). But our native library is expected to have a
      // .jnilib suffix (traditional on OS X). Try both to be safe.
      if (libraryName.endsWith(".dylib")) {
         final String altLibraryName = "lib" + NATIVE_LIBRARY_NAME + ".jnilib";
         boolean ret = loadNamedNativeLibrary(dirPath, altLibraryName);
         if (ret) {
            return true;
         }
      }

      return loadNamedNativeLibrary(dirPath, libraryName);
   }

   private static boolean loadFromPathSetBySystemProperty() {
      final File preferredPath = getPreferredLibraryPath();
      if (preferredPath != null) {
         logLibraryLoading("Try path given by " + MM_PROPERTY_MMCOREJ_LIB_PATH);
         loadNativeLibrary(preferredPath);
         return true;
      }
      return false;
   }

   private static boolean loadFromHardCodedPaths() {
      final List<File> searchPaths = new ArrayList<File>();

      // Some relative paths were hard-coded for running Micro-Manager in Fiji
      // and in the classic distribution (in which the native library is
      // located in the grand-parent directory of the directory containing the
      // MMCoreJ JAR). This also allows finding the native library in the same
      // directory as the JAR.
      searchPaths.addAll(FijiPaths.getPaths());

      // On Linux, also search a compile-time hard-coded path (TODO It is odd
      // that this is done only in the case of Linux. The build system should
      // be modified so that it can be enabled or disabled on any Unix.)
      if (isLinux()) {
         final File hardCodedPath = getHardCodedLibraryPath();
         if (hardCodedPath != null)
            searchPaths.add(hardCodedPath);
      }

      logLibraryLoading("Will search in hard-coded paths:");
      for (File path : searchPaths) {
         logLibraryLoading("  " + path.getPath());
      }

      for (File path : searchPaths) {
         if (loadNativeLibrary(path)) {
            return true;
         }
      }
      return false;
   }

   static void load() {
      // The most reliable method for locating (the correct copy of)
      // MMCoreJ_wrap is to look in the single path given as a Java system
      // property. The launcher will typically set this property. If this
      // property is set, other paths will not be considered.
      if (!loadFromPathSetBySystemProperty()) {
         // However, if the system property is not set, we search in some
         // candidate directories in order.
         if (!loadFromHardCodedPaths()) {
            // Finally, if all else fails, try the system default mechanism,
            // which will use java.library.path. This is necessary for
            // backward compatibility, and it is also what people will
            // generally expect.
            logLibraryLoading("Falling back to loading using system default method");
            try {
               System.loadLibrary(NATIVE_LIBRARY_NAME);
            } catch (UnsatisfiedLinkError e) {
               logLibraryLoading("System default loading method failed");
            }
         }
      }
   }

   private LegacyLibraryLoading() {
   }
}
