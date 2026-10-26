# Definitional proc to organize widgets for parameters.
proc init_gui { IPINST } {
  ipgui::add_param $IPINST -name "Component_Name"
  #Adding Page
  set Page_0 [ipgui::add_page $IPINST -name "Page 0"]
  ipgui::add_param $IPINST -name "NUM_CLASSES" -parent ${Page_0}
  ipgui::add_param $IPINST -name "PARAMS_FILE" -parent ${Page_0}
  ipgui::add_param $IPINST -name "PE_DSP" -parent ${Page_0}
  ipgui::add_param $IPINST -name "W" -parent ${Page_0}
  ipgui::add_param $IPINST -name "WEIGHTS_FILE" -parent ${Page_0}


}

proc update_PARAM_VALUE.NUM_CLASSES { PARAM_VALUE.NUM_CLASSES } {
	# Procedure called to update NUM_CLASSES when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.NUM_CLASSES { PARAM_VALUE.NUM_CLASSES } {
	# Procedure called to validate NUM_CLASSES
	return true
}

proc update_PARAM_VALUE.PARAMS_FILE { PARAM_VALUE.PARAMS_FILE } {
	# Procedure called to update PARAMS_FILE when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.PARAMS_FILE { PARAM_VALUE.PARAMS_FILE } {
	# Procedure called to validate PARAMS_FILE
	return true
}

proc update_PARAM_VALUE.PE_DSP { PARAM_VALUE.PE_DSP } {
	# Procedure called to update PE_DSP when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.PE_DSP { PARAM_VALUE.PE_DSP } {
	# Procedure called to validate PE_DSP
	return true
}

proc update_PARAM_VALUE.W { PARAM_VALUE.W } {
	# Procedure called to update W when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.W { PARAM_VALUE.W } {
	# Procedure called to validate W
	return true
}

proc update_PARAM_VALUE.WEIGHTS_FILE { PARAM_VALUE.WEIGHTS_FILE } {
	# Procedure called to update WEIGHTS_FILE when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.WEIGHTS_FILE { PARAM_VALUE.WEIGHTS_FILE } {
	# Procedure called to validate WEIGHTS_FILE
	return true
}


proc update_MODELPARAM_VALUE.W { MODELPARAM_VALUE.W PARAM_VALUE.W } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.W}] ${MODELPARAM_VALUE.W}
}

proc update_MODELPARAM_VALUE.NUM_CLASSES { MODELPARAM_VALUE.NUM_CLASSES PARAM_VALUE.NUM_CLASSES } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.NUM_CLASSES}] ${MODELPARAM_VALUE.NUM_CLASSES}
}

proc update_MODELPARAM_VALUE.PE_DSP { MODELPARAM_VALUE.PE_DSP PARAM_VALUE.PE_DSP } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.PE_DSP}] ${MODELPARAM_VALUE.PE_DSP}
}

proc update_MODELPARAM_VALUE.WEIGHTS_FILE { MODELPARAM_VALUE.WEIGHTS_FILE PARAM_VALUE.WEIGHTS_FILE } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.WEIGHTS_FILE}] ${MODELPARAM_VALUE.WEIGHTS_FILE}
}

proc update_MODELPARAM_VALUE.PARAMS_FILE { MODELPARAM_VALUE.PARAMS_FILE PARAM_VALUE.PARAMS_FILE } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.PARAMS_FILE}] ${MODELPARAM_VALUE.PARAMS_FILE}
}

