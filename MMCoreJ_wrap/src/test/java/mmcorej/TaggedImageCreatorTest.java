/*
 * Unit tests for MMCoreJ metadata handling.
 *
 * Note that these tests were generated with the goal of precisely defining
 * current behavior; not all of that behavior is desirable -- but we need to
 * know what it is in order to make compatible (or deliberately breaking)
 * changes. Some of the odd behavior may be relied upon by the acquisition
 * engines, for example.
 */

package mmcorej;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

import mmcorej.org.json.JSONObject;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import java.util.Arrays;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

@ExtendWith(MockitoExtension.class)
class TaggedImageCreatorTest {

    @Mock
    CMMCore core;

    private void stubCoreDefaults(CMMCore core) throws Exception {
        when(core.getBytesPerPixel()).thenReturn(1L);
        when(core.getNumberOfComponents()).thenReturn(1L);
        when(core.getImageBitDepth()).thenReturn(8L);
        when(core.getPixelSizeUm(true)).thenReturn(0.325);
        when(core.getPixelSizeAffineAsString()).thenReturn(
                "1.0;0.0;0.0;0.0;1.0;0.0");
        doAnswer(invocation -> {
            int[] x = invocation.getArgument(0);
            int[] y = invocation.getArgument(1);
            int[] xSize = invocation.getArgument(2);
            int[] ySize = invocation.getArgument(3);
            x[0] = 0;
            y[0] = 0;
            xSize[0] = 512;
            ySize[0] = 512;
            return null;
        }).when(core).getROI(
                any(int[].class), any(int[].class),
                any(int[].class), any(int[].class));
        when(core.getImageWidth()).thenReturn(512L);
        when(core.getImageHeight()).thenReturn(512L);
        lenient().when(core.getPropertyFromCache("Core", "ChannelGroup"))
                .thenReturn("Channel");
        lenient().when(core.getCurrentConfigFromCache("Channel"))
                .thenReturn("DAPI");
        lenient().when(core.getCameraDevice()).thenReturn("Camera");
        lenient().when(core.getProperty("Camera", "Binning"))
                .thenReturn("1");
    }

    // --- metadataToMap ---

    @Test
    void metadataToMap_emptyMetadata() {
        Metadata md = new Metadata();
        JSONObject result = TaggedImageCreator.metadataToMap(md);
        assertEquals(0, result.length());
    }

    @Test
    void metadataToMap_withSingleTags() throws Exception {
        Metadata md = new Metadata();
        MetadataSingleTag tag1 =
                new MetadataSingleTag("Exposure", "Camera", false);
        tag1.SetValue("100.0");
        md.SetTag(tag1);
        MetadataSingleTag tag2 =
                new MetadataSingleTag("Gain", "Camera", true);
        tag2.SetValue("2");
        md.SetTag(tag2);

        JSONObject result = TaggedImageCreator.metadataToMap(md);
        assertEquals(2, result.length());
        assertEquals("100.0", result.getString("Camera-Exposure"));
        assertEquals("2", result.getString("Camera-Gain"));
    }

    @Test
    void metadataToMap_skipsTagOnException() throws Exception {
        Metadata md = mock(Metadata.class);
        StrVector keys = mock(StrVector.class);
        when(md.GetKeys()).thenReturn(keys);
        when(keys.iterator()).thenReturn(Arrays.asList("good", "bad").iterator());

        MetadataSingleTag goodTag = mock(MetadataSingleTag.class);
        when(goodTag.GetValue()).thenReturn("value");
        when(md.GetSingleTag("good")).thenReturn(goodTag);
        when(md.GetSingleTag("bad")).thenThrow(new Exception("bad tag"));

        JSONObject result = TaggedImageCreator.metadataToMap(md);
        assertEquals(1, result.length());
        assertEquals("value", result.getString("good"));
    }

    // --- getROITag ---

    @Test
    void getROITag_formatsCorrectly() throws Exception {
        doAnswer(invocation -> {
            int[] x = invocation.getArgument(0);
            int[] y = invocation.getArgument(1);
            int[] xSize = invocation.getArgument(2);
            int[] ySize = invocation.getArgument(3);
            x[0] = 10;
            y[0] = 20;
            xSize[0] = 640;
            ySize[0] = 480;
            return null;
        }).when(core).getROI(
                any(int[].class), any(int[].class),
                any(int[].class), any(int[].class));

        assertEquals("10-20-640-480", TaggedImageCreator.getROITag(core));
    }

    @Test
    void getROITag_zeroValues() throws Exception {
        doAnswer(invocation -> {
            int[] x = invocation.getArgument(0);
            int[] y = invocation.getArgument(1);
            int[] xSize = invocation.getArgument(2);
            int[] ySize = invocation.getArgument(3);
            x[0] = 0;
            y[0] = 0;
            xSize[0] = 0;
            ySize[0] = 0;
            return null;
        }).when(core).getROI(
                any(int[].class), any(int[].class),
                any(int[].class), any(int[].class));

        assertEquals("0-0-0-0", TaggedImageCreator.getROITag(core));
    }

    // --- getPixelType ---

    @ParameterizedTest
    @CsvSource({
            "1, 1, GRAY8",
            "2, 1, GRAY16",
            "4, 1, GRAY32",
            "4, 3, RGB32", // TODO likely bug
            "4, 4, RGB32",
            "8, 3, RGB64", // TODO likely bug
            "8, 4, RGB64",
            "3, 3, ''",
    })
    void getPixelType_returnsCorrectType(
            long bytesPerPixel, long numComponents, String expected) {
        when(core.getBytesPerPixel()).thenReturn(bytesPerPixel);
        when(core.getNumberOfComponents()).thenReturn(numComponents);
        assertEquals(expected, TaggedImageCreator.getPixelType(core));
    }

    // --- getMultiCameraChannel ---

    @Test
    void getMultiCameraChannel_found() throws Exception {
        JSONObject tags = new JSONObject();
        tags.put("Core-Camera", "Multi");
        tags.put("Multi-Physical Camera 1", "Cam1");
        assertEquals("Cam1",
                TaggedImageCreator.getMultiCameraChannel(tags, 0));
    }

    @Test
    void getMultiCameraChannel_indexOffset() throws Exception {
        JSONObject tags = new JSONObject();
        tags.put("Core-Camera", "Multi");
        tags.put("Multi-Physical Camera 3", "Cam3");
        assertEquals("Cam3",
                TaggedImageCreator.getMultiCameraChannel(tags, 2));
    }

    @Test
    void getMultiCameraChannel_physCamKeyMissing() throws Exception {
        JSONObject tags = new JSONObject();
        tags.put("Core-Camera", "Multi");
        assertNull(TaggedImageCreator.getMultiCameraChannel(tags, 0));
    }

    @Test
    void getMultiCameraChannel_noCoreCamera() {
        JSONObject tags = new JSONObject();
        assertNull(TaggedImageCreator.getMultiCameraChannel(tags, 0));
    }

    // --- createTaggedImage (4-arg) ---

    @Test
    void createTaggedImage_withSystemStateCache_allTags() throws Exception {
        stubCoreDefaults(core);
        Configuration config = new Configuration();
        config.addSetting(new PropertySetting("Dev1", "Prop1", "Value1"));
        config.addSetting(new PropertySetting("Dev2", "Prop2", "Value2"));
        when(core.getSystemStateCache()).thenReturn(config);

        Object pixels = new byte[512 * 512];
        Metadata md = new Metadata();
        TaggedImage image = TaggedImageCreator.createTaggedImage(
                core, true, pixels, md);

        assertSame(pixels, image.pix);
        JSONObject tags = image.tags;
        assertEquals(18, tags.length());
        assertEquals(8L, tags.getLong("BitDepth"));
        assertEquals(0.325, tags.getDouble("PixelSizeUm"));
        assertEquals("1.0;0.0;0.0;0.0;1.0;0.0",
                tags.getString("PixelSizeAffine"));
        assertEquals("0-0-512-512", tags.getString("ROI"));
        assertEquals(512L, tags.getLong("Width"));
        assertEquals(512L, tags.getLong("Height"));
        assertEquals("GRAY8", tags.getString("PixelType"));
        assertEquals(0, tags.getInt("Frame"));
        assertEquals(0, tags.getInt("FrameIndex"));
        assertEquals("Default", tags.getString("Position"));
        assertEquals(0, tags.getInt("PositionIndex"));
        assertEquals(0, tags.getInt("Slice"));
        assertEquals(0, tags.getInt("SliceIndex"));
        assertEquals("DAPI", tags.getString("Channel"));
        assertEquals(0, tags.getInt("ChannelIndex"));
        assertEquals("1", tags.getString("Binning"));
        assertEquals("Value1", tags.getString("Dev1-Prop1"));
        assertEquals("Value2", tags.getString("Dev2-Prop2"));
    }

    @Test
    void createTaggedImage_withoutSystemStateCache() throws Exception {
        stubCoreDefaults(core);

        TaggedImage image = TaggedImageCreator.createTaggedImage(
                core, false, new byte[0], new Metadata());

        verify(core, never()).getSystemStateCache();
        JSONObject tags = image.tags;
        assertEquals(16, tags.length());
        assertTrue(tags.has("BitDepth"));
        assertTrue(tags.has("Channel"));
        assertFalse(tags.has("Dev1-Prop1"));
    }

    @Test
    void createTaggedImage_metadataMerged() throws Exception {
        stubCoreDefaults(core);
        Metadata md = new Metadata();
        MetadataSingleTag tag =
                new MetadataSingleTag("Exposure", "Camera", false);
        tag.SetValue("50.0");
        md.SetTag(tag);

        TaggedImage image = TaggedImageCreator.createTaggedImage(
                core, false, new byte[0], md);

        assertEquals("50.0", image.tags.getString("Camera-Exposure"));
    }

    @Test
    void createTaggedImage_nullChannel_defaultsToDefault() throws Exception {
        stubCoreDefaults(core);
        when(core.getCurrentConfigFromCache("Channel")).thenReturn(null);

        TaggedImage image = TaggedImageCreator.createTaggedImage(
                core, false, new byte[0], new Metadata());

        assertEquals("Default", image.tags.getString("Channel"));
    }

    @Test
    void createTaggedImage_emptyChannel_defaultsToDefault() throws Exception {
        stubCoreDefaults(core);
        when(core.getCurrentConfigFromCache("Channel")).thenReturn("");

        TaggedImage image = TaggedImageCreator.createTaggedImage(
                core, false, new byte[0], new Metadata());

        assertEquals("Default", image.tags.getString("Channel"));
    }

    @Test
    void createTaggedImage_nonEmptyChannel() throws Exception {
        stubCoreDefaults(core);
        when(core.getCurrentConfigFromCache("Channel")).thenReturn("GFP");

        TaggedImage image = TaggedImageCreator.createTaggedImage(
                core, false, new byte[0], new Metadata());

        assertEquals("GFP", image.tags.getString("Channel"));
    }

    @Test
    void createTaggedImage_widthInMetadata_overwritten() throws Exception {
        stubCoreDefaults(core);
        Metadata md = new Metadata();
        MetadataSingleTag tag = new MetadataSingleTag("Width", "_", false);
        tag.SetValue("999");
        md.SetTag(tag);

        TaggedImage image = TaggedImageCreator.createTaggedImage(
                core, false, new byte[0], md);

        assertEquals(512L, image.tags.getLong("Width"));
    }

    @Test
    void createTaggedImage_heightInMetadata_overwritten() throws Exception {
        stubCoreDefaults(core);
        Metadata md = new Metadata();
        MetadataSingleTag tag = new MetadataSingleTag("Height", "_", false);
        tag.SetValue("999");
        md.SetTag(tag);

        TaggedImage image = TaggedImageCreator.createTaggedImage(
                core, false, new byte[0], md);

        assertEquals(512L, image.tags.getLong("Height"));
    }

    @Test
    void createTaggedImage_pixelTypeInMetadata_overwritten() throws Exception {
        stubCoreDefaults(core);
        Metadata md = new Metadata();
        MetadataSingleTag tag =
                new MetadataSingleTag("PixelType", "_", false);
        tag.SetValue("RGB32");
        md.SetTag(tag);

        TaggedImage image = TaggedImageCreator.createTaggedImage(
                core, false, new byte[0], md);

        assertEquals("GRAY8", image.tags.getString("PixelType"));
    }

    @Test
    void createTaggedImage_binningInMetadata_overwritten() throws Exception {
        stubCoreDefaults(core);
        Metadata md = new Metadata();
        MetadataSingleTag tag =
                new MetadataSingleTag("Binning", "_", false);
        tag.SetValue("4");
        md.SetTag(tag);

        TaggedImage image = TaggedImageCreator.createTaggedImage(
                core, false, new byte[0], md);

        assertEquals("1", image.tags.getString("Binning"));
    }

    @Test
    void createTaggedImage_binningThrows_tagOmitted() throws Exception {
        stubCoreDefaults(core);
        when(core.getProperty("Camera", "Binning"))
                .thenThrow(new Exception("no camera"));

        TaggedImage image = TaggedImageCreator.createTaggedImage(
                core, false, new byte[0], new Metadata());

        assertFalse(image.tags.has("Binning"));
    }

    // --- createTaggedImage (5-arg) ---

    @Test
    void createTaggedImage_addsCameraChannelIndex() throws Exception {
        stubCoreDefaults(core);

        TaggedImage image = TaggedImageCreator.createTaggedImage(
                core, false, new byte[0], new Metadata(), 2);

        assertEquals(2, image.tags.getInt("CameraChannelIndex"));
        assertEquals(2, image.tags.getInt("ChannelIndex"));
    }

    @Test
    void createTaggedImage_physicalCamera_setsCameraAndChannel()
            throws Exception {
        stubCoreDefaults(core);
        Configuration config = new Configuration();
        config.addSetting(
                new PropertySetting("Core", "Camera", "Multi"));
        config.addSetting(
                new PropertySetting("Multi", "Physical Camera 3", "PhysCam3"));
        when(core.getSystemStateCache()).thenReturn(config);

        TaggedImage image = TaggedImageCreator.createTaggedImage(
                core, true, new byte[0], new Metadata(), 2);

        assertEquals("PhysCam3", image.tags.getString("Camera"));
        assertEquals("PhysCam3", image.tags.getString("Channel"));
    }

    @Test
    void createTaggedImage_noPhysicalCamera_noCameraTag() throws Exception {
        stubCoreDefaults(core);

        TaggedImage image = TaggedImageCreator.createTaggedImage(
                core, false, new byte[0], new Metadata(), 0);

        assertFalse(image.tags.has("Camera"));
    }

    @Test
    void createTaggedImage_cameraChannelIndexInMetadata_notOverridden()
            throws Exception {
        stubCoreDefaults(core);
        Metadata md = new Metadata();
        MetadataSingleTag tag =
                new MetadataSingleTag("CameraChannelIndex", "_", false);
        tag.SetValue("5");
        md.SetTag(tag);

        TaggedImage image = TaggedImageCreator.createTaggedImage(
                core, false, new byte[0], md, 2);

        assertEquals("5", image.tags.getString("CameraChannelIndex"));
        assertEquals(0, image.tags.getInt("ChannelIndex"));
    }

    @Test
    void createTaggedImage_cameraInMetadata_notOverridden() throws Exception {
        stubCoreDefaults(core);
        Configuration config = new Configuration();
        config.addSetting(
                new PropertySetting("Core", "Camera", "Multi"));
        config.addSetting(
                new PropertySetting("Multi", "Physical Camera 1", "PhysCam1"));
        when(core.getSystemStateCache()).thenReturn(config);

        Metadata md = new Metadata();
        MetadataSingleTag tag =
                new MetadataSingleTag("Camera", "_", false);
        tag.SetValue("ExistingCam");
        md.SetTag(tag);

        TaggedImage image = TaggedImageCreator.createTaggedImage(
                core, true, new byte[0], md, 0);

        assertEquals("ExistingCam", image.tags.getString("Camera"));
        assertEquals("DAPI", image.tags.getString("Channel"));
    }
}
