/*=============================================================================
  Copyright (C) 2021 Allied Vision Technologies.  All Rights Reserved.

  Redistribution of this header file, in original or modified form, without
  prior written consent of Allied Vision Technologies is prohibited.

-------------------------------------------------------------------------------
 
  File:        VmbConstants.h

-------------------------------------------------------------------------------

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

/**
 * \file
 * \brief File containing constants used in the Vmb C API.
 */

#ifndef VMBCONSTANTS_H_INCLUDE_
#define VMBCONSTANTS_H_INCLUDE_

#ifdef _WIN32
/**
 * \brief the character used to separate file paths in the parameter of ::VmbStartup 
 */
#define VMB_PATH_SEPARATOR_CHAR L';'

/**
 * \brief the string used to separate file paths in the parameter of ::VmbStartup
 */
#define VMB_PATH_SEPARATOR_STRING L";"
#else
/**
 * \brief the character used to separate file paths in the parameter of ::VmbStartup
 */
#define VMB_PATH_SEPARATOR_CHAR ':'

/**
 * \brief the string used to separate file paths in the parameter of ::VmbStartup
 */
#define VMB_PATH_SEPARATOR_STRING ":"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \defgroup SfncNamespaces Sfnc Namespace Constants
 * \{
 */
 
/**
 * \brief The C string identifying the namespace of features not defined in the SFNC
 *        standard.
 */
#define VMB_SFNC_NAMESPACE_CUSTOM "Custom"

/**
 * \brief The C string identifying the namespace of features defined in the SFNC
 *        standard.
 */
#define VMB_SFNC_NAMESPACE_STANDARD "Standard"

/**
 * \}
 */

#ifdef __cplusplus
}
#endif

#endif // VMBCONSTANTS_H_INCLUDE_
