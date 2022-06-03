#pragma once

namespace ummhflags {
	const char serial_out = 0;
	const char serial_in = 1;
}

namespace ummhwords {
	const char sepSetup = '|';
	const char sepOut = '>';
	const char sepIn = '<';
	const char sepWithin = ':';
	const char sepEnd = ';';

	const char* const device_list_start = "Start";
	const char* const device_list_continue = "Next";
	const char* const device_list_end = "End";

	const char* const wtrue = "true";
	const char* const wfalse = "false";
	const char* const cashed = "cashed";
	const char* const not_supported = "not supported";
	const char* const timeout = "Timeout";
	const char* const cmd = "Command"; // a key word that identifies device command
	// possible standard commands for a shutter
	const char* const set_open = "SetOpen";
	const char* const get_open = "GetOpen";
	const char* const fire = "fire";
	// possible standard commands for a state device
	const char* const get_num_of_pos = "GetNumberOfPositions";
	// possible standard commands for a stage device
	const char* const set_position_um = "SetPositionUm";
	const char* const get_position_um = "GetPositionUm";
	const char* const home = "Home";
	const char* const stop = "Stop";
	// possible standard commands for a XYstage device
	const char* const position_x = "PositionX";
	const char* const position_y = "PositionY";
	// possible standard commands for a camera
	const char* const snap_image = "SnapImage";
	const char* const get_image_buffer = "GetImageBuffer";
	const char* const set_binning = "SetBinning";
	const char* const set_exposure = "SetExposure";
	const char* const set_roi = "SetROI";
	const char* const get_roi = "GetROI";
	const char* const clear_roi = "ClearROI";
	const char* const pixel_type_8bit = "8bit";
	const char* const pixel_type_16bit = "16bit";
	const char* const bit_depth = "BitDepth";
	const char* const image_max_width = "ImageMaxWidth";
	const char* const image_max_height = "ImageMaxHeight";
	const char* const transfer_timeout = "TransferTimeout";
	const char* const endpoint = "Endpoint";

	const char* const prop = "Property";
	const char* const act = "Action";
	const char* const preini_append = " preset";
	const char* const prop_str = "PropertyString";
	const char* const prop_float = "PropertyFloat";
	const char* const prop_int = "PropertyInteger";
	const char* const prop_str_act = "PropertyStringAction";
	const char* const prop_float_act = "PropertyFloatAction";
	const char* const prop_int_act = "PropertyIntegerAction";

}

namespace ummherrors {
	
	// *** errors generated in the adapter ***
	// unable to connect to the controller
	const int adp_communication_error = 400;
	// controller specified version not supported by adapter
	const int adp_version_mismatch = 401;
	// timed out waiting for controller response
	const int adp_lost_communication = 402;	
	// controller returned a string that could not be parsed
	const int adp_string_not_recognized = 403;
	// controller specified device that adapter does not recognize
	const int adp_device_not_recognized = 404; 
	// contoller specified command that adapter does not recognize
	const int adp_device_command_not_recognized = 405;	
	// contoller specified value that is not allowed
	const int adp_device_command_value_not_allowed = 406;	

	// *** errors generated in the controller ***
	// ok status for the device specified by the controller
	const int ctr_ok = 0;
	// busy status for the device specified by the controller
	const int ctr_busy = 1;
	// adapter sent a string that controller is unable to parse
	const int ctr_string_not_recognized = 500;
	// adapter specified a device that controller does not recognize
	const int ctr_device_not_recognized = 501;
	// adapter specified a command that controller does not recognize
	const int ctr_device_command_not_recognized = 502;
	// adapter specified value that controller does not recognize
	const int ctr_device_command_value_not_allowed = 503;
	// contoller device timed out
	const int ctr_device_timeout = 504;

}

