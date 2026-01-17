package mmcorej;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;

import java.io.File;
import java.lang.reflect.Method;
import java.net.URL;
import java.net.URLClassLoader;
import org.junit.jupiter.api.Test;

class NativeLibraryLoadingIT {

    @Test
    void loadFromDefaultClassLoader() {
        assertDoesNotThrow(CMMCore::noop);
    }

    @Test
    void loadFromIsolatedClassLoader() throws Exception {
        URL[] jarUrls = findAllJars();
        try (URLClassLoader loader = new URLClassLoader(jarUrls, null)) {
            Class<?> cls = Class.forName("mmcorej.CMMCore", true, loader);
            Method noopMethod = cls.getMethod("noop");
            assertDoesNotThrow(() -> noopMethod.invoke(null));
        }
    }

    private static URL[] findAllJars() throws Exception {
        URL codeSource = CMMCore.class.getProtectionDomain()
                             .getCodeSource()
                             .getLocation();
        File jarFile = new File(codeSource.toURI());
        File dir = jarFile.getParentFile();
        File[] jars = dir.listFiles(
            (d, name) -> name.startsWith("MMCoreJ-") && name.endsWith(".jar"));
        if (jars == null) {
            throw new IllegalStateException(
                "Cannot list files in directory: " + dir);
        }
        if (jars.length == 0) {
            throw new IllegalStateException("No MMCoreJ JARs found in " + dir);
        }
        URL[] urls = new URL[jars.length];
        for (int i = 0; i < jars.length; i++) {
            urls[i] = jars[i].toURI().toURL();
        }
        return urls;
    }
}
