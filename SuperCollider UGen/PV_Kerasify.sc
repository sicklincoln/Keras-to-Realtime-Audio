//passing filename as ascii chars trick from NovaDiskOut
// "/Users/ioi/Desktop/data alias".collectAs(_.ascii, Array);

//uses a background thread for all processing by default
//else amortises load over control periods, keeps everything after constructor within realtime thread
PV_Kerasify : PV_ChainUGen {

	*new { arg buffer, amortiseflag = 0, path;

		var args = [buffer,amortiseflag] ++ [path.size]++(path.collectAs(_.ascii, Array));

		^this.multiNew('control', *args)
	}
}

