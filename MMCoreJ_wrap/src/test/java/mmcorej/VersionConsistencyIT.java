package mmcorej;

import static org.junit.jupiter.api.Assertions.*;

import java.io.InputStream;
import java.util.Properties;
import org.junit.jupiter.api.Test;

class VersionConsistencyIT {

    @Test
    void pomVersionMatchesCMMCoreVersion() throws Exception {

        String path = "/META-INF/maven/org.micro-manager.mmcorej/MMCoreJ/pom.properties";
        Properties props = new Properties();
        try (InputStream is = getClass().getResourceAsStream(path)) {
            assertNotNull(is, "pom.properties not found at " + path);
            props.load(is);
        }

        String version = props.getProperty("version");
        assertNotNull(version, "version property not found in pom.properties");
        String[] parts = version.split("\\.");
        assertEquals(3, parts.length,
            "Expected version format major.minor.patch, got: " + version);

        int pomMajor = Integer.parseInt(parts[0]);
        int pomMinor = Integer.parseInt(parts[1]);
        int pomPatch = Integer.parseInt(parts[2]);

        assertEquals(pomMajor, CMMCore.getMMCoreVersionMajor(),
            "Major version mismatch between CMMCore and pom.xml");
        assertEquals(pomMinor, CMMCore.getMMCoreVersionMinor(),
            "Minor version mismatch between CMMCore and pom.xml");
        assertEquals(pomPatch, CMMCore.getMMCoreVersionPatch(),
            "Patch version mismatch between CMMCore and pom.xml");
    }
}
