/*=============================================================================
  Copyright (C) 2023 Allied Vision Technologies.  All Rights Reserved.

  This file is distributed under the BSD license.
  License text is included with the source distribution.

  THIS SOFTWARE IS PROVIDED BY THE AUTHOR "AS IS" AND ANY EXPRESS OR IMPLIED
  WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF TITLE,
  NON-INFRINGEMENT, MERCHANTABILITY AND FITNESS FOR A PARTICULAR  PURPOSE ARE
  DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,
  INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
  AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
  TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

=============================================================================*/
#ifndef CONSTANTS_H
#define CONSTANTS_H

#ifdef __linux__
static std::string VIMBA_X_LIB_DIR = std::string("");         //<! Vimba X library path within installation directory
static constexpr const char *VIMBA_X_LIB_NAME = "libVmbC.so"; //<! Vimba X library name
static constexpr const char *VIMBA_X_IMAGE_TRANSFORM_NAME = "libVmbImageTransform.so"; //<! Vimba X Image Transform library name
#elif _WIN32
static constexpr const char *VIMBA_X_ENV_VAR = "VIMBA_X_HOME"; //<! Vimba X Environmental variable name
static const char *ENV_VALUE = std::getenv(VIMBA_X_ENV_VAR);   //<! Vimba environment variable value
static std::string VIMBA_X_LIB_DIR =
    std::string(ENV_VALUE != nullptr ? ENV_VALUE : "") + std::string("\\bin");       //<! Vimba X library path within installation directory
static constexpr const char *VIMBA_X_LIB_NAME = "VmbC.dll";                          //<! Vimba X library name
static constexpr const char *VIMBA_X_IMAGE_TRANSFORM_NAME = "VmbImageTransform.dll"; //<! Vimba X Image Transform library name
#else
// nothing
#endif

#endif
