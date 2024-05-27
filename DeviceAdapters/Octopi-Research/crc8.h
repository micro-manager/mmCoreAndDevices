// This code is in the Public Domain
// Source: https://www.3dbrew.org/wiki/CRC-8-CCITT

// crc8.h

#include <stdint.h>
#include <string.h>

uint8_t crc8ccitt(const void * data, size_t size);
