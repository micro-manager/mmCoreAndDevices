// Mock device adapter for testing of device sequencing
//
// Copyright (C) 2023 Board of Regents of the University of Wisconsin System
//
// This library is free software; you can redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation.
//
// This library is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License
// for more details.
//
// IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES.
//
// You should have received a copy of the GNU Lesser General Public License
// along with this library; if not, write to the Free Software Foundation,
// Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
//
//
// Author: Mark Tsuchida

#include "TextImage.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iterator>
#include <string>
#include <vector>

namespace {

constexpr std::size_t GLYPH_WIDTH = 6;
constexpr std::size_t GLYPH_HEIGHT = 13;
constexpr std::size_t TOP_MARGIN = 4;
constexpr std::size_t LEFT_MARGIN = 4;

// ASCII 0x20-0x7f of the public domain fixed-misc 6x13 font from X.org
// (-Misc-Fixed-Medium-R-SemiCondensed--13-120-75-75-C-60-ISO10646-1)
// with 0x7f (DEL) replaced with gray mesh.
// Glyphs are ended with "------\n" and trailing blank rows are omitted.
constexpr char glyphSrc[] = R"(
------


  *
  *
  *
  *
  *
  *
  *

  *
------


 * *
 * *
 * *
------



 * *
 * *
*****
 * *
*****
 * *
 * *
------


  *
 ****
* *
* *
 ***
  * *
  * *
****
  *
------


 *  *
* * *
 * *
   *
  *
 *
 * *
* * *
*  *
------



 *
* *
* *
 *
* *
*  **
*  *
 ** *
------


  *
  *
  *
------

   *
  *
  *
 *
 *
 *
 *
 *
  *
  *
   *
------

 *
  *
  *
   *
   *
   *
   *
   *
  *
  *
 *
------


  *
* * *
 ***
* * *
  *
------




  *
  *
*****
  *
  *
------









  **
  *
 *
------






*****
------









  *
 ***
  *
------


    *
    *
   *
   *
  *
 *
 *
*
*
------


  *
 * *
*   *
*   *
*   *
*   *
*   *
 * *
  *
------


  *
 **
* *
  *
  *
  *
  *
  *
*****
------


 ***
*   *
*   *
    *
   *
  *
 *
*
*****
------


*****
    *
   *
  *
 ***
    *
    *
*   *
 ***
------


   *
   *
  **
 * *
 * *
*  *
*****
   *
   *
------


*****
*
*
* **
**  *
    *
    *
*   *
 ***
------


 ***
*   *
*
*
****
*   *
*   *
*   *
 ***
------


*****
    *
   *
   *
  *
  *
 *
 *
 *
------


 ***
*   *
*   *
*   *
 ***
*   *
*   *
*   *
 ***
------


 ***
*   *
*   *
*   *
 ****
    *
    *
*   *
 ***
------




  *
 ***
  *


  *
 ***
  *
------




  *
 ***
  *


  **
  *
 *
------


    *
   *
  *
 *
*
 *
  *
   *
    *
------





*****

*****
------


*
 *
  *
   *
    *
   *
  *
 *
*
------


 ***
*   *
*   *
    *
   *
  *
  *

  *
------


 ***
*   *
*   *
*  **
* * *
* * *
* **
*
 ****
------


  *
 * *
*   *
*   *
*   *
*****
*   *
*   *
*   *
------


****
 *  *
 *  *
 *  *
 ***
 *  *
 *  *
 *  *
****
------


 ***
*   *
*
*
*
*
*
*   *
 ***
------


****
 *  *
 *  *
 *  *
 *  *
 *  *
 *  *
 *  *
****
------


*****
*
*
*
****
*
*
*
*****
------


*****
*
*
*
****
*
*
*
*
------


 ***
*   *
*
*
*
*  **
*   *
*   *
 ***
------


*   *
*   *
*   *
*   *
*****
*   *
*   *
*   *
*   *
------


 ***
  *
  *
  *
  *
  *
  *
  *
 ***
------


  ***
   *
   *
   *
   *
   *
   *
*  *
 **
------


*   *
*   *
*  *
* *
**
* *
*  *
*   *
*   *
------


*
*
*
*
*
*
*
*
*****
------


*   *
*   *
** **
* * *
* * *
*   *
*   *
*   *
*   *
------


*   *
**  *
**  *
* * *
* * *
*  **
*  **
*   *
*   *
------


 ***
*   *
*   *
*   *
*   *
*   *
*   *
*   *
 ***
------


****
*   *
*   *
*   *
****
*
*
*
*
------


 ***
*   *
*   *
*   *
*   *
*   *
*   *
* * *
 ***
    *
------


****
*   *
*   *
*   *
****
* *
*  *
*   *
*   *
------


 ***
*   *
*
*
 ***
    *
    *
*   *
 ***
------


*****
  *
  *
  *
  *
  *
  *
  *
  *
------


*   *
*   *
*   *
*   *
*   *
*   *
*   *
*   *
 ***
------


*   *
*   *
*   *
*   *
 * *
 * *
 * *
  *
  *
------


*   *
*   *
*   *
*   *
* * *
* * *
* * *
* * *
 * *
------


*   *
*   *
 * *
 * *
  *
 * *
 * *
*   *
*   *
------


*   *
*   *
 * *
 * *
  *
  *
  *
  *
  *
------


*****
    *
   *
   *
  *
 *
 *
*
*****
------

 ***
 *
 *
 *
 *
 *
 *
 *
 *
 *
 ***
------


*
*
 *
 *
  *
   *
   *
    *
    *
------

 ***
   *
   *
   *
   *
   *
   *
   *
   *
   *
 ***
------


  *
 * *
*   *
------











*****
------

  *
   *
------





 ***
    *
 ****
*   *
*  **
 ** *
------


*
*
*
****
*   *
*   *
*   *
*   *
****
------





 ***
*   *
*
*
*   *
 ***
------


    *
    *
    *
 ****
*   *
*   *
*   *
*   *
 ****
------





 ***
*   *
*****
*
*   *
 ***
------


  **
 *  *
 *
 *
****
 *
 *
 *
 *
------





 ***
*   *
*   *
*   *
 ****
    *
*   *
 ***
------


*
*
*
* **
**  *
*   *
*   *
*   *
*   *
------



  *

 **
  *
  *
  *
  *
 ***
------



   *

  **
   *
   *
   *
   *
*  *
*  *
 **
------


*
*
*
*  *
* *
**
* *
*  *
*   *
------


 **
  *
  *
  *
  *
  *
  *
  *
 ***
------





** *
* * *
* * *
* * *
* * *
*   *
------





* **
**  *
*   *
*   *
*   *
*   *
------





 ***
*   *
*   *
*   *
*   *
 ***
------





****
*   *
*   *
*   *
****
*
*
*
------





 ****
*   *
*   *
*   *
 ****
    *
    *
    *
------





* **
**  *
*
*
*
*
------





 ***
*   *
 **
   *
*   *
 ***
------



 *
 *
****
 *
 *
 *
 *  *
  **
------





*   *
*   *
*   *
*   *
*  **
 ** *
------





*   *
*   *
*   *
 * *
 * *
  *
------





*   *
*   *
* * *
* * *
* * *
 * *
------





*   *
 * *
  *
  *
 * *
*   *
------





*   *
*   *
*   *
*  **
 ** *
    *
*   *
 ***
------





*****
   *
  *
 *
*
*****
------

   **
  *
  *
  *
  *
**
  *
  *
  *
  *
   **
------


  *
  *
  *
  *
  *
  *
  *
  *
  *
------

**
  *
  *
  *
  *
   **
  *
  *
  *
  *
**
------


 *  *
* * *
*  *
------
* * *
 * * *
* * *
 * * *
* * *
 * * *
* * *
 * * *
* * *
 * * *
* * *
 * * *
* * *
------
)";

const std::uint8_t *glyphTable() {
   static const std::vector<std::uint8_t> table = [] {
      std::vector<std::uint8_t> ret(
         (0x80 - 0x20) * GLYPH_HEIGHT * GLYPH_WIDTH);
      const std::string src(glyphSrc);
      std::size_t glyph = 0;
      std::size_t y = 0;
      std::size_t start = 0;
      for (;;) {
         std::size_t newline = src.find('\n', start);
         if (newline == std::string::npos) {
            assert(glyph == 0x80 - 0x20);
            break;
         }
         assert(glyph < 0x80 - 0x20);
         const auto line = src.substr(start, newline - start);
         if (line == "------") {
            ++glyph;
            y = 0;
            start = newline + 1;
            continue;
         }
         assert(y < GLYPH_HEIGHT);
         std::size_t x = 0;
         for (auto ch : line) {
             assert(x < GLYPH_WIDTH);
             assert(ch == ' ' || ch == '*');
             ret[(glyph * GLYPH_HEIGHT + y) * GLYPH_WIDTH + x] =
                (ch == '*' ? 0xff : 0x00);
             ++x;
         }
         ++y;
         start = newline + 1;
      }
      return ret;
   }();
   return table.data();
}

const std::uint8_t *getGlyph(char ascii, std::size_t y = 0) {
   // Widen without sign extension
   auto code = std::size_t(static_cast<unsigned char>(ascii));
   // Map all unknown chars to 0x7f (DEL) where we have mesh glyph
   const bool isCtrl = (code & ~0x1f) == 0;
   const bool isHigh = (code & 0x80) != 0;
   if (isCtrl || isHigh) {
      code = 0x7f;
   }
   return glyphTable() +
      ((code - 0x20) * GLYPH_HEIGHT + y) * GLYPH_WIDTH;
}

std::vector<std::string> WrapLines(const std::string &text,
   std::size_t maxRows, std::size_t maxCols) {
   const auto sMaxCols = static_cast<std::ptrdiff_t>(maxCols);
   std::vector<std::string> ret;
   if (maxCols == 0) {
      return ret;
   }
   ret.reserve(maxRows);
   auto start = text.begin();
   for (;;) {
      auto eol = std::find(start, text.end(), '\n');
      while (std::distance(start, eol) > sMaxCols) {
         const auto rstart = std::make_reverse_iterator(
            std::next(start, maxCols));
         const auto rstop = std::make_reverse_iterator(start);
         const auto rspace = std::find(rstart, rstop, ' ');
         if (rspace == rstop || std::next(rspace) == rstop) {
            // Cannot word-wrap, so hard-wrap
            ret.emplace_back(start, std::next(start, maxCols));
            std::advance(start, maxCols);
         }
         else {
            ret.emplace_back(start, std::next(rspace.base(), -1));
            start = rspace.base();
         }
         if (ret.size() == maxRows) {
            return ret;
         }
      }
      ret.emplace_back(start, eol);
      if (ret.size() == maxRows || eol == text.end()) {
         return ret;
      }
      start = std::next(eol);
   }
}

void DrawRowsOfText(const std::vector<std::string> &rows, std::uint8_t *buf,
   std::size_t width, std::size_t height) {
   // Loop over the pixel buffer exactly once, sequentially, for cache
   // friendliness. This means repeatedly accessing the text line and randomly
   // accessing the glyphs, but the text line and glyph table are much smaller
   // than the image.
   std::fill_n(buf, TOP_MARGIN * width, std::uint8_t(0));
   std::size_t row = 0;
   for (; row < rows.size(); ++row) {
      for (std::size_t glyphY = 0; glyphY < GLYPH_HEIGHT; ++glyphY) {
         const auto y = TOP_MARGIN + row * GLYPH_HEIGHT + glyphY;
         if (y >= height) {
            return;
         }
         std::fill_n(std::next(buf, y * width),
            LEFT_MARGIN, std::uint8_t(0));
         const auto &line = rows[row];
         std::size_t col = 0;
         for (char ch : line) {
            const auto x = LEFT_MARGIN + col * GLYPH_WIDTH;
            const auto glyphWidth = std::min(GLYPH_WIDTH, width - x);
            std::copy_n(getGlyph(ch, glyphY), glyphWidth,
               std::next(buf, y * width + x));
            ++col;
         }
         const auto xEnd = LEFT_MARGIN + col * GLYPH_WIDTH;
         std::fill_n(std::next(buf, y * width + xEnd),
            width - xEnd, std::uint8_t(0));
      }
   }
   const auto yEnd = TOP_MARGIN + row * GLYPH_HEIGHT;
   std::fill_n(std::next(buf, yEnd * width),
      (height - yEnd) * width, std::uint8_t(0));
}

} // namespace

void DrawTextImage(const std::string &text, std::uint8_t *buf,
   std::size_t width, std::size_t height) {
   const auto maxRows = (height - 2 * TOP_MARGIN) / GLYPH_HEIGHT;
   const auto maxCols = (width - 2 * LEFT_MARGIN) / GLYPH_WIDTH;
   auto rows = WrapLines(text, maxRows, maxCols);
   DrawRowsOfText(rows, buf, width, height);
}
