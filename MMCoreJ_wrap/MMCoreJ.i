/////////////////////////////////////////////////////////////////////////////////
// FILE:          MMCoreJ.i
// PROJECT:       Micro-Manager
// SUBSYSTEM:     MMCoreJ
//-----------------------------------------------------------------------------
// DESCRIPTION:   SWIG generator for the Java interface wrapper.
//              
// COPYRIGHT:     University of California, San Francisco, 2006,
//                All Rights reserved
//
// LICENSE:       This file is distributed under the "Lesser GPL" (LGPL) license.
//                License text is included with the source distribution.
//
//                This file is distributed in the hope that it will be useful,
//                but WITHOUT ANY WARRANTY; without even the implied warranty
//                of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
//
//                IN NO EVENT SHALL THE COPYRIGHT OWNER OR
//                CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
//                INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES.
//
// AUTHOR:        Nenad Amodaj, nenad@amodaj.com, 06/07/2005

#if SWIG_VERSION < 0x020000 || SWIG_VERSION >= 0x040000
#error SWIG 2.x or 3.x is currently required to build MMCoreJ
#endif

#define MMDEVICE_CLIENT_BUILD

%module (directors="1") MMCoreJ
%feature("director") MMEventCallback;

%include std_string.i
%include std_vector.i
%include std_map.i
%include std_pair.i
%include "typemaps.i"


//
// Native library loading
//

// Pull in the compile-time hard-coded paths (determined by Unix configure script)
#ifndef MMCOREJ_LIBRARY_PATH
%define MMCOREJ_LIBRARY_PATH "" %enddef
#endif
%javaconst(1) LIBRARY_PATH;
%constant char *LIBRARY_PATH = MMCOREJ_LIBRARY_PATH;

%pragma(java) jniclasscode=%{
   private static boolean checkIfAlreadyLoaded() {
      try {
         CMMCore.noop();
      } catch (UnsatisfiedLinkError e) {
         return false;
      }
      return true;
   }

   static {
      LegacyLibraryLoading.logLibraryLoading("Start loading...");

      // New-style loading (extract from JAR, fall back to java.library.path)
      if (!checkIfAlreadyLoaded()) {
         try {
            NativeLibraryLoader.load("mmcorej");
            LegacyLibraryLoading.logLibraryLoading("Loaded 'mmcorej' library");
         }
         catch (UnsatisfiedLinkError e) {
            LegacyLibraryLoading.logLibraryLoading(
               "Falling back to 'MMCoreJ_wrap' loading mode...");
         }
      }

      // Legacy loading ('MMCoreJ_wrap')
      if (!checkIfAlreadyLoaded()) {
         LegacyLibraryLoading.load();
      }
   }
%}



// output arguments
%apply double &OUTPUT { double &x_stage };
%apply double &OUTPUT { double &y_stage };
%apply int &OUTPUT { int &x };
%apply int &OUTPUT { int &y };
%apply int &OUTPUT { int &xSize };
%apply int &OUTPUT { int &ySize };


// Java typemap
// change default SWIG mapping of unsigned char* return values
// to byte[]
//
// Assumes that class has the following method defined:
// long GetImageBufferSize()
//


%typemap(jni) unsigned char*        "jbyteArray"
%typemap(jtype) unsigned char*      "byte[]"
%typemap(jstype) unsigned char*     "byte[]"
%typemap(out) unsigned char*
{
   long lSize = (arg1)->getImageBufferSize();
   
   // create a new byte[] object in Java
   jbyteArray data = JCALL1(NewByteArray, jenv, lSize);
   
   // copy pixels from the image buffer
   JCALL4(SetByteArrayRegion, jenv, data, 0, lSize, (jbyte*)result);

   $result = data;
}

// Map input argument: java byte[] -> C++ unsigned char *
%typemap(in) unsigned char*
{
   // Assume that we are sending an image to an SLM device, one byte per pixel (monochrome grayscale).
   
   long expectedLength = (arg1)->getSLMWidth(arg2) * (arg1)->getSLMHeight(arg2);
   long receivedLength = JCALL1(GetArrayLength, jenv, $input);
   
   if (receivedLength != expectedLength && receivedLength != expectedLength*4)
   {
      jclass excep = jenv->FindClass("java/lang/Exception");
      if (excep)
         jenv->ThrowNew(excep, "Image dimensions are wrong for this SLM.");
      return;
   }
   
   $1 = (unsigned char *) JCALL2(GetByteArrayElements, jenv, $input, 0);
}

%typemap(freearg) unsigned char* {
   // Allow the Java byte array to be garbage collected.
   JCALL3(ReleaseByteArrayElements, jenv, $input, (jbyte *) $1, JNI_ABORT); // JNI_ABORT = Don't alter the original array.
}

// change Java wrapper output mapping for unsigned char*
%typemap(javaout) unsigned char* {
    return $jnicall;
 }

%typemap(javain) unsigned char* "$javainput" 


// Map input argument: java List<byte[]> -> C++ std::vector<unsigned char*>
%typemap(jni) std::vector<unsigned char*>        "jobject"
%typemap(jtype) std::vector<unsigned char*>      "java.util.List<byte[]>"
%typemap(jstype) std::vector<unsigned char*>     "java.util.List<byte[]>"
%typemap(in) std::vector<unsigned char*>
{
   // Assume that we are sending an image to an SLM device, one byte per pixel (monochrome grayscale).
   
   long expectedLength = (arg1)->getSLMWidth(arg2) * (arg1)->getSLMHeight(arg2);
   std::vector<unsigned char*> inputVector;
   jclass clazz = jenv->FindClass("java/util/List");
   jmethodID sizeMethodID = jenv->GetMethodID(clazz, "size", "()I");
   // get JNI ID for java.util.List.get(int i) method.
   // Because of type erasure we specify an "Object" return value,
   // but we expect a byte[] to be returned.
   jmethodID getMethodID = jenv->GetMethodID(clazz, "get", "(I)Ljava/lang/Object;");
   int listSize = jenv->CallIntMethod($input, sizeMethodID);
   
   for (int i = 0; i < listSize; ++i) {
     jbyteArray pixels = (jbyteArray) jenv->CallObjectMethod($input, getMethodID, i);
     long receivedLength = jenv->GetArrayLength(pixels);
   	 if (receivedLength != expectedLength && receivedLength != expectedLength*4)
	 {
	    jclass excep = jenv->FindClass("java/lang/Exception");
	     if (excep)
	        jenv->ThrowNew(excep, "Image dimensions are wrong for this SLM.");
	      return;
	  }
	  inputVector.push_back((unsigned char *) JCALL2(GetByteArrayElements, jenv, pixels, 0));
   }
   $1 = inputVector;
}

%typemap(freearg) std::vector<unsigned char*> {
   // Allow the Java List to be garbage collected.
   // Not sure how to do that here -- may not be necessary.
   //JCALL3(ReleaseByteArrayElements, jenv, $input, (jbyte *) $1, JNI_ABORT); // JNI_ABORT = Don't alter the original array.
}

%typemap(javain) std::vector<unsigned char*> "$javainput" 

// Java typemap
// change default SWIG mapping of void* return values
// to return CObject containing array of pixel values
//
// Assumes that class has the following methods defined:
// unsigned GetImageWidth()
// unsigned GetImageHeight()

%typemap(jni) void*        "jobject"
%typemap(jtype) void*      "Object"
%typemap(jstype) void*     "Object"
%typemap(javaout) void* {
   return $jnicall;
}
%typemap(out) void*
{
   long lSize = (arg1)->getImageWidth() * (arg1)->getImageHeight();
   
   if ((arg1)->getBytesPerPixel() == 1)
   {
      // create a new byte[] object in Java
      jbyteArray data = JCALL1(NewByteArray, jenv, lSize);
      if (data == 0)
      {
         jclass excep = jenv->FindClass("java/lang/OutOfMemoryError");
         if (excep)
            jenv->ThrowNew(excep, "The system ran out of memory!");

         $result = 0;
         return $result;
      }
   
      // copy pixels from the image buffer
      JCALL4(SetByteArrayRegion, jenv, data, 0, lSize, (jbyte*)result);

      $result = data;
   }
   else if ((arg1)->getBytesPerPixel() == 2)
   {
      // create a new short[] object in Java
      jshortArray data = JCALL1(NewShortArray, jenv, lSize);
      if (data == 0)
      {
         jclass excep = jenv->FindClass("java/lang/OutOfMemoryError");
         if (excep)
            jenv->ThrowNew(excep, "The system ran out of memory!");
         $result = 0;
         return $result;
      }
  
      // copy pixels from the image buffer
      JCALL4(SetShortArrayRegion, jenv, data, 0, lSize, (jshort*)result);

      $result = data;
   }
   else if ((arg1)->getBytesPerPixel() == 4)
   {
      if ((arg1)->getNumberOfComponents() == 1)
      {
         // create a new float[] object in Java
         jfloatArray data = JCALL1(NewFloatArray, jenv, lSize);
         if (data == 0)
         {
            jclass excep = jenv->FindClass("java/lang/OutOfMemoryError");
            if (excep)
               jenv->ThrowNew(excep, "The system ran out of memory!");

            $result = 0;
            return $result;
         }

         // copy pixels from the image buffer
         JCALL4(SetFloatArrayRegion, jenv, data, 0, lSize, (jfloat*)result);

         $result = data;
      }
      else
      {
         // create a new byte[] object in Java
         jbyteArray data = JCALL1(NewByteArray, jenv, lSize * 4);
         if (data == 0)
         {
            jclass excep = jenv->FindClass("java/lang/OutOfMemoryError");
            if (excep)
               jenv->ThrowNew(excep, "The system ran out of memory!");

            $result = 0;
            return $result;
         }

         // copy pixels from the image buffer
         JCALL4(SetByteArrayRegion, jenv, data, 0, lSize * 4, (jbyte*)result);

         $result = data;
      }
   }
   else if ((arg1)->getBytesPerPixel() == 8)
   {
      // create a new short[] object in Java
      jshortArray data = JCALL1(NewShortArray, jenv, lSize * 4);
      if (data == 0)
      {
         jclass excep = jenv->FindClass("java/lang/OutOfMemoryError");
         if (excep)
            jenv->ThrowNew(excep, "The system ran out of memory!");
         $result = 0;
         return $result;
      }
  
      // copy pixels from the image buffer
      JCALL4(SetShortArrayRegion, jenv, data, 0, lSize * 4, (jshort*)result);

      $result = data;
   }

   else
   {
      // don't know how to map
      // TODO: throw exception?
      $result = 0;
   }
}

// Java typemap
// change default SWIG mapping of void* return values
// to return CObject containing array of pixel values
//
// Assumes that class has the following methods defined:
// unsigned GetImageWidth()
// unsigned GetImageHeight()
// unsigned GetImageDepth()
// unsigned GetNumberOfComponents()


%typemap(jni) unsigned int* "jobject"
%typemap(jtype) unsigned int*      "Object"
%typemap(jstype) unsigned int*     "Object"
%typemap(javaout) unsigned int* {
   return $jnicall;
}
%typemap(out) unsigned int*
{
   long lSize = (arg1)->getImageWidth() * (arg1)->getImageHeight();
   unsigned numComponents = (arg1)->getNumberOfComponents();
   
   if ((arg1)->getBytesPerPixel() == 1 && numComponents == 4)
   {
      // assuming RGB32 format
      // create a new int[] object in Java
      jintArray data = JCALL1(NewIntArray, jenv, lSize);
      if (data == 0)
      {
         jclass excep = jenv->FindClass("java/lang/OutOfMemoryError");
         if (excep)
            jenv->ThrowNew(excep, "The system ran out of memory!");
         $result = 0;
         return $result;
      }
  
      // copy pixels from the image buffer
      JCALL4(SetIntArrayRegion, jenv, data, 0, lSize, (jint*)result);

      $result = data;
   }
   else
   {
      // don't know how to map
      // TODO: thow exception?
      $result = 0;
   }
}


%typemap(jni) imgRGB32 "jintArray"
%typemap(jtype) imgRGB32      "int[]"
%typemap(jstype) imgRGB32     "int[]"
%typemap(javain) imgRGB32     "$javainput"
%typemap(in) imgRGB32
{
   // Assume that we are sending an image to an SLM device, one int (four bytes) per pixel.
   
   if  ((arg1)->getSLMBytesPerPixel(arg2) != 4)
   {
      jclass excep = jenv->FindClass("java/lang/Exception");
      if (excep)
         jenv->ThrowNew(excep, "32-bit array received but not expected for this SLM.");
      return;
   }
   
   long expectedLength = (arg1)->getSLMWidth(arg2) * (arg1)->getSLMHeight(arg2);
   long receivedLength = JCALL1(GetArrayLength, jenv, (jarray) $input);
   
   if (receivedLength != expectedLength)
   {
      jclass excep = jenv->FindClass("java/lang/Exception");
      if (excep)
         jenv->ThrowNew(excep, "Image dimensions are wrong for this SLM.");
      return;
   }
   
   $1 = (imgRGB32) JCALL2(GetIntArrayElements, jenv, (jintArray) $input, 0);
}

%typemap(freearg) imgRGB32 {
   // Allow the Java int array to be garbage collected.
   JCALL3(ReleaseIntArrayElements, jenv, $input, (jint *) $1, JNI_ABORT); // JNI_ABORT = Don't alter the original array.
}


//
// Map all exception objects coming from C++ level
// generic Java Exception
//
%rename(eql) operator=;

// CMMError used by MMCore
%typemap(throws, throws="java.lang.Exception") CMMError {
   jclass excep = jenv->FindClass("java/lang/Exception");
   if (excep)
     jenv->ThrowNew(excep, $1.getFullMsg().c_str());
   return $null;
}

// MetadataKeyError used by Metadata class
%typemap(throws, throws="java.lang.Exception") MetadataKeyError {
   jclass excep = jenv->FindClass("java/lang/Exception");
   if (excep)
     jenv->ThrowNew(excep, $1.getMsg().c_str());
   return $null;
}

// MetadataIndexError used by Metadata class
%typemap(throws, throws="java.lang.Exception") MetadataIndexError {
   jclass excep = jenv->FindClass("java/lang/Exception");
   if (excep)
     jenv->ThrowNew(excep, $1.getMsg().c_str());
   return $null;
}

// We've translated exceptions to java.lang.Exception, so don't wrap the unused
// C++ exception classes.
%ignore CMMError;
%ignore MetadataKeyError;
%ignore MetadataIndexError;


%typemap(javaimports) CMMCore %{
   import java.awt.geom.Point2D;
   import java.awt.Rectangle;
   import java.util.ArrayList;
   import java.util.List;
%}

%typemap(javacode) CMMCore %{
   private boolean includeSystemStateCache_ = true;

   public boolean getIncludeSystemStateCache() {
      return includeSystemStateCache_;
   }
   public void setIncludeSystemStateCache(boolean state) {
      includeSystemStateCache_ = state;
   }

   public TaggedImage getTaggedImage(int cameraChannelIndex) throws java.lang.Exception {
      Metadata md = new Metadata();
      Object pixels = getImage(cameraChannelIndex);
      return TaggedImageCreator.createTaggedImage(this, includeSystemStateCache_, pixels, md, cameraChannelIndex);
   }

   public TaggedImage getTaggedImage() throws java.lang.Exception {
      return getTaggedImage(0);
   }

   public TaggedImage getLastTaggedImage(int cameraChannelIndex) throws java.lang.Exception {
      Metadata md = new Metadata();
      Object pixels = getLastImageMD(cameraChannelIndex, 0, md);
      return TaggedImageCreator.createTaggedImage(this, includeSystemStateCache_, pixels, md, cameraChannelIndex);
   }

   public TaggedImage getLastTaggedImage() throws java.lang.Exception {
      return getLastTaggedImage(0);
   }

   public TaggedImage getNBeforeLastTaggedImage(long n) throws java.lang.Exception {
      Metadata md = new Metadata();
      Object pixels = getNBeforeLastImageMD(n, md);
      return TaggedImageCreator.createTaggedImage(this, includeSystemStateCache_, pixels, md);
   }

   public TaggedImage popNextTaggedImage(int cameraChannelIndex) throws java.lang.Exception {
      Metadata md = new Metadata();
      Object pixels = popNextImageMD(cameraChannelIndex, 0, md);
      return TaggedImageCreator.createTaggedImage(this, includeSystemStateCache_, pixels, md, cameraChannelIndex);
   }

   public TaggedImage popNextTaggedImage() throws java.lang.Exception {
      return popNextTaggedImage(0);
   }

   // convenience functions follow
   
   /*
    * Convenience function. Returns the ROI of the current camera in a java.awt.Rectangle.
    */
   public Rectangle getROI() throws java.lang.Exception {
      // ROI values are given as x,y,w,h in individual one-member arrays (pointers in C++):
      int[][] a = new int[4][1];
      getROI(a[0], a[1], a[2], a[3]);
      return new Rectangle(a[0][0], a[1][0], a[2][0], a[3][0]);
   }
   
    /*
    * Convenience function. Returns the ROI of specified camera in a java.awt.Rectangle.
    */
   public Rectangle getROI(String label) throws java.lang.Exception {
      // ROI values are given as x,y,w,h in individual one-member arrays (pointers in C++):
      int[][] a = new int[4][1];
      getROI(label, a[0], a[1], a[2], a[3]);
      return new Rectangle(a[0][0], a[1][0], a[2][0], a[3][0]);
   }

   /*
    * Convenience function: returns multiple ROIs of the current camera as a
    * list of java.awt.Rectangles.
    */
   public List<Rectangle> getMultiROI() throws java.lang.Exception {
      UnsignedVector xs = new UnsignedVector();
      UnsignedVector ys = new UnsignedVector();
      UnsignedVector widths = new UnsignedVector();
      UnsignedVector heights = new UnsignedVector();
      getMultiROI(xs, ys, widths, heights);
      ArrayList<Rectangle> result = new ArrayList<Rectangle>();
      for (int i = 0; i < xs.size(); ++i) {
         long x = xs.get(i);
         long y = ys.get(i);
         long w = widths.get(i);
         long h = heights.get(i);
         Rectangle r = new Rectangle((int) x, (int) y, (int) w, (int) h);
         result.add(r);
      }
      return result;
   }

   /*
    * Convenience function: convert incoming list of Rectangles into vectors
    * of ints to set multiple ROIs.
    */
   public void setMultiROI(List<Rectangle> rects) throws java.lang.Exception {
      UnsignedVector xs = new UnsignedVector();
      UnsignedVector ys = new UnsignedVector();
      UnsignedVector widths = new UnsignedVector();
      UnsignedVector heights = new UnsignedVector();
      for (Rectangle r : rects) {
         long x = r.x;
         long y  = r.y;
         long w = r.width;
         long h = r.height;
         xs.add(x);
         ys.add(y);
         widths.add(w);
         heights.add(h);
      }
      setMultiROI(xs, ys, widths, heights);
   }

   /**
    * Convenience function.  Retuns affine transform as a String
    * Used in this class and by the acquisition engine 
    * (rather than duplicating this code there
    */
   public String getPixelSizeAffineAsString() throws java.lang.Exception {
      String pa = "";
      DoubleVector aff = getPixelSizeAffine(true);
      if (aff.size() == 6)  {
         for (int i = 0; i < 5; i++) {
            pa += aff.get(i) + ";";
         }
         pa += aff.get(5);
      }
      return pa;
   }

   /* 
    * Convenience function. Returns the current x,y position of the stage in a Point2D.Double.
    */
   public Point2D.Double getXYStagePosition(String stage) throws java.lang.Exception {
      // stage position is given as x,y in individual one-member arrays (pointers in C++):
      double p[][] = new double[2][1];
      getXYPosition(stage, p[0], p[1]);
      return new Point2D.Double(p[0][0], p[1][0]);
   }

   /**
    * Convenience function: returns the current XY position of the current
    * XY stage device as a Point2D.Double.
    */
   public Point2D.Double getXYStagePosition() throws java.lang.Exception {
      double x[] = new double[1];
      double y[] = new double[1];
      getXYPosition(x, y);
      return new Point2D.Double(x[0], y[0]);
   }
   
   /* 
    * Convenience function. Returns the current x,y position of the galvo in a Point2D.Double.
    */
   public Point2D.Double getGalvoPosition(String galvoDevice) throws java.lang.Exception {
      // stage position is given as x,y in individual one-member arrays (pointers in C++):
      double p[][] = new double[2][1];
      getGalvoPosition(galvoDevice, p[0], p[1]);
      return new Point2D.Double(p[0][0], p[1][0]);
   }
%}


%{
#include "MMDeviceConstants.h"
#include "Error.h"
#include "Configuration.h"
#include "ImageMetadata.h"
#include "MMEventCallback.h"
#include "MMCore.h"
%}


// instantiate STL mappings

namespace std {
	%typemap(javaimports) vector<char> %{
		import java.lang.Iterable;
		import java.util.Iterator;
		import java.util.NoSuchElementException;
		import java.lang.UnsupportedOperationException;
	%}

   %typemap(javainterfaces) vector<char> %{ Iterable<Character>%}

   %typemap(javacode) vector<char> %{
      public Iterator<Character> iterator() {
         return new Iterator<Character>() {

            private int i_=0;

            public boolean hasNext() {
               return (i_<size());
            }

            public Character next() throws NoSuchElementException {
               if (hasNext()) {
                  ++i_;
                  return get(i_-1);
               } else {
                  throw new NoSuchElementException();
               }
            }

            public void remove() throws UnsupportedOperationException {
               throw new UnsupportedOperationException();
            }
         };
      }

      public Character[] toArray() {
         if (0==size())
            return new Character[0];

         Character ints[] = new Character[(int) size()];
         for (int i=0; i<size(); ++i) {
            ints[i] = get(i);
         }
         return ints;
      }
   %}
   
   /* 
   * On most platforms a c++ `long` will be 32bit and therefore should map to a Java `Integer`. However 
   * on some platforms a c++ `long` could be 64bit which could potentially cause issues. Ideally we should just avoid using vector<long> in MMCore interfaces.
   */
   
   %typemap(javaimports) vector<long> %{
		import java.lang.Iterable;
		import java.util.Iterator;
		import java.util.NoSuchElementException;
		import java.lang.UnsupportedOperationException;
	%}

   %typemap(javainterfaces) vector<long> %{ Iterable<Integer>%}

   %typemap(javacode) vector<long> %{
      public Iterator<Integer> iterator() {
         return new Iterator<Integer>() {

            private int i_=0;

            public boolean hasNext() {
               return (i_<size());
            }

            public Integer next() throws NoSuchElementException {
               if (hasNext()) {
                  ++i_;
                  return get(i_-1); 
               } else {
                  throw new NoSuchElementException();
               }
            }

            public void remove() throws UnsupportedOperationException {
               throw new UnsupportedOperationException();
            }
         };
      }

      public Integer[] toArray() {
         if (0==size())
            return new Integer[0];

         Integer ints[] = new Integer[(int) size()];
         for (int i=0; i<size(); ++i) {
            ints[i] = get(i);
         }
         return ints;
      }
   %}
   
   %typemap(javaimports) vector<double> %{
		import java.lang.Iterable;
		import java.util.Iterator;
		import java.util.NoSuchElementException;
		import java.lang.UnsupportedOperationException;
	%}

   %typemap(javainterfaces) vector<double> %{ Iterable<Double>%}

   %typemap(javacode) vector<double> %{
      public Iterator<Double> iterator() {
         return new Iterator<Double>() {

            private int i_=0;

            public boolean hasNext() {
               return (i_<size());
            }

            public Double next() throws NoSuchElementException {
               if (hasNext()) {
                  ++i_;
                  return get(i_-1);
               } else {
                  throw new NoSuchElementException();
               }
            }

            public void remove() throws UnsupportedOperationException {
               throw new UnsupportedOperationException();
            }
         };
      }

      public Double[] toArray() {
         if (0==size())
            return new Double[0];

         Double ints[] = new Double[(int) size()];
         for (int i=0; i<size(); ++i) {
            ints[i] = get(i);
         }
         return ints;
      }
   %}


	%typemap(javaimports) vector<string> %{
		import java.lang.Iterable;
		import java.util.Iterator;
		import java.util.NoSuchElementException;
		import java.lang.UnsupportedOperationException;
	%}
	
	%typemap(javainterfaces) vector<string> %{ Iterable<String>%}
	
	%typemap(javacode) vector<string> %{
	
		public Iterator<String> iterator() {
			return new Iterator<String>() {
			
				private int i_=0;
			
				public boolean hasNext() {
					return (i_<size());
				}
				
				public String next() throws NoSuchElementException {
					if (hasNext()) {
						++i_;
						return get(i_-1);
					} else {
					throw new NoSuchElementException();
					}
				}
					
				public void remove() throws UnsupportedOperationException {
					throw new UnsupportedOperationException();
				}		
			};
		}
		
		public String[] toArray() {
			if (0==size())
				return new String[0];
			
			String strs[] = new String[(int) size()];
			for (int i=0; i<size(); ++i) {
				strs[i] = get(i);
			}
			return strs;
		}
		
	%}
	
   

	%typemap(javaimports) vector<bool> %{
		import java.lang.Iterable;
		import java.util.Iterator;
		import java.util.NoSuchElementException;
		import java.lang.UnsupportedOperationException;
	%}
	
	%typemap(javainterfaces) vector<bool> %{ Iterable<Boolean>%}
	
	%typemap(javacode) vector<bool> %{
	
		public Iterator<Boolean> iterator() {
			return new Iterator<Boolean>() {
			
				private int i_=0;
			
				public boolean hasNext() {
					return (i_<size());
				}
				
				public Boolean next() throws NoSuchElementException {
					if (hasNext()) {
						++i_;
						return get(i_-1);
					} else {
					throw new NoSuchElementException();
					}
				}
					
				public void remove() throws UnsupportedOperationException {
					throw new UnsupportedOperationException();
				}		
			};
		}
		
		public Boolean[] toArray() {
			if (0==size())
				return new Boolean[0];
			
			Boolean strs[] = new Boolean[(int) size()];
			for (int i=0; i<size(); ++i) {
				strs[i] = get(i);
			}
			return strs;
		}
		
	%}
	

	%typemap(javaimports) vector<unsigned> %{
		import java.lang.Iterable;
		import java.util.Iterator;
		import java.util.NoSuchElementException;
		import java.lang.UnsupportedOperationException;
	%}

   %typemap(javainterfaces) vector<unsigned> %{ Iterable<Long>%}

   %typemap(javacode) vector<unsigned> %{
      public Iterator<Long> iterator() {
         return new Iterator<Long>() {

            private int i_=0;

            public boolean hasNext() {
               return (i_<size());
            }

            public Long next() throws NoSuchElementException {
               if (hasNext()) {
                  ++i_;
                  return get(i_-1);
               } else {
                  throw new NoSuchElementException();
               }
            }

            public void remove() throws UnsupportedOperationException {
               throw new UnsupportedOperationException();
            }
         };
      }

      public Long[] toArray() {
         if (0==size())
            return new Long[0];

         Long ints[] = new Long[(int) size()];
         for (int i=0; i<size(); ++i) {
            ints[i] = get(i);
         }
         return ints;
      }
   %}




    %template(CharVector)   vector<char>;
    %template(LongVector)   vector<long>;
    %template(DoubleVector) vector<double>;
    %template(StrVector)    vector<string>;
    %template(BooleanVector)    vector<bool>;
    %template(UnsignedVector) vector<unsigned>;
    %template(pair_ss)      pair<string, string>;
    %template(StrMap)       map<string, string>;




}

%import "CoreDeclHelpers.h"

%include "MMDeviceConstants.h"
%include "Error.h"
%include "Configuration.h"
%include "ImageMetadata.h"
%include "MMEventCallback.h"
%include "MMCore.h"

