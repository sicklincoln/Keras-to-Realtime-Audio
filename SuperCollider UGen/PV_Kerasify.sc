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



PV_DNNMorph : PV_ChainUGen {

	*new { arg buffer, interpolation=0, layertointerpolate=1, preorpost=0, path1, path2;

		var args = [buffer,interpolation, layertointerpolate, preorpost] ++ [path1.size]++(path1.collectAs(_.ascii, Array))++ [path2.size]++(path2.collectAs(_.ascii, Array));

		^this.multiNew('control', *args)
	}
}


//assumes activationbuffer large enough to contain data for layertoactivate
PV_KerasifyActivationFromBuffer : PV_ChainUGen {

	*new { arg buffer, activationbuffer, layertoactivate=1, path;

		var args = [buffer,activationbuffer,layertoactivate] ++ [path.size]++(path.collectAs(_.ascii, Array));

		^this.multiNew('control', *args)
	}
}