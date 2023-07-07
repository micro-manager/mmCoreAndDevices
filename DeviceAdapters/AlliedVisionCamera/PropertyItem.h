/*=============================================================================
  Copyright (C) 2012 - 2023 Allied Vision Technologies.  All Rights Reserved.

  Redistribution of this header file, in original or modified form, without
  prior written consent of Allied Vision Technologies is prohibited.

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
#ifndef PROPERTYITEM_H
#define PROPERTYITEM_H

#include <string>

/**
 * @brief
 */
struct PropertyItem {
  PropertyItem(const std::string& name, double min = 0.0, double max = 0.0, double step = 1.0);
  ~PropertyItem() = default;

  bool operator==(PropertyItem& rhs) {
    return rhs.m_name == m_name && rhs.m_min == m_min && rhs.m_max == m_max &&
           rhs.m_step == m_step;
  }
  bool operator!=(PropertyItem& rhs) { return !(rhs == *this); }

  std::string m_name;
  double m_min;
  double m_max;
  double m_step;
};

#endif  // PROPERTYITEM_H
