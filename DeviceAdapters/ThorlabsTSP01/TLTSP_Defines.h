#ifndef TLTSP_DEFINES_H
#define TLTSP_DEFINES_H

/*---------------------------------------------------------------------------
 Buffers
---------------------------------------------------------------------------*/
#define TLTSP_BUFFER_SIZE            256      // General buffer size
#define TLTSP_ERR_DESCR_BUFFER_SIZE  512      // Buffer size for error messages

/*---------------------------------------------------------------------------
 Error/Warning Codes
   Note: The instrument returns errors within the range -512 .. +1023. 
   The driver adds the value VI_INSTR_ERROR_OFFSET (0xBFFC0900). So the 
   driver returns instrument errors in the range 0xBFFC0700 .. 0xBFFC0CFF.
---------------------------------------------------------------------------*/
// Offsets
#undef VI_INSTR_WARNING_OFFSET
#undef VI_INSTR_ERROR_OFFSET

#define VI_INSTR_WARNING_OFFSET        (0x3FFC0900L)
#define VI_INSTR_ERROR_OFFSET          (_VI_ERROR + VI_INSTR_WARNING_OFFSET)   //0xBFFC0900

// Driver warnings
#undef VI_INSTR_WARN_OVERFLOW
#undef VI_INSTR_WARN_UNDERRUN
#undef VI_INSTR_WARN_NAN

#define VI_INSTR_WARN_OVERFLOW         (VI_INSTR_WARNING_OFFSET + 0x01L)   //0x3FFC0901
#define VI_INSTR_WARN_UNDERRUN         (VI_INSTR_WARNING_OFFSET + 0x02L)   //0x3FFC0902
#define VI_INSTR_WARN_NAN              (VI_INSTR_WARNING_OFFSET + 0x03L)   //0x3FFC0903

/*---------------------------------------------------------------------------
 Attributes
---------------------------------------------------------------------------*/
#define TLTSP_ATTR_SET_VAL           (0)
#define TLTSP_ATTR_MIN_VAL           (1)
#define TLTSP_ATTR_MAX_VAL           (2)
#define TLTSP_ATTR_DFLT_VAL          (3)

#endif
