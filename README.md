# Keras to Realtime Audio


This project demonstrates going from a python keras program to train a deep neural net on audio, to realtime deployment in a SuperCollider plugin written in C++, and Web Audio API javascript for a webpage. The neural net processes spectral data in frames.


The SuperCollider route adapts the [kerasify](https://github.com/moof2k/kerasify) library for file saving and loading, with a streamlined neural net implementation, and the web route uses the [ONNX](https://onnx.ai) file format for saving and loading a model and [MMLL](https://github.com/sicklincoln/MMLL) for audio processing.

The original python code depends upon the [librosa](https://librosa.github.io/librosa/) MIR audio library, [keras](https://keras.io) and the aforementioned kerasify and onnxmltools.

Building the SuperCollider plugin requires the SuperCollider source code ([LINK](https://supercollider.github.io)) and CMake.


## License

The code was developed by [Nick Collins](http://composerprogrammer.com/index.html) as part of the AHRC funded [MIMIC](https://mimicproject.com/about) project (Musically Intelligent Machines Interacting Creatively). It is released under an MIT license (python and javascript parts), excepting that the SuperCollider source code is under GNU GPL and so thus is the plugin.
