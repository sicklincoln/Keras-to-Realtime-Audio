import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import SGD


# load and evaluate a saved model
#from numpy import loadtxt
#from keras.models import load_model


#will train over one or more audio files
paths = [
         #'/Users/ioi/Desktop/testoutput/vocals0.wav',
         #'/Users/ioi/Desktop/testoutput/vocals1.wav',
         #'/Users/ioi/Desktop/testoutput/vocals2.wav'
         #'/data/audio/classical/improv/pianoimprov1.aif',
         #'/data/audio/sicklincoln/curioussamples/ecstasyvocalonly.wav'#
         #"/data/audio/classical/improv/improv1.wav",
         #'/data/audio/field/galesonbrightonbeach/R09_0007.WAV'
         '/data/audio/SCsamp/acapella/thehype.wav'
         ];

fftsize = 4096 #2048
halffftsize = 2048 #1024
numprevframes = 1; #have this working with javascript, not yet implemented for SuperCollider
numbinsused = halffftsize
inputdimension = numbinsused * numprevframes


#to roll own
#https://github.com/CPJKU/madmom
#pyaudio
import librosa

whichfftbins = range(halffftsize);

#subset of fft bins based around 88 piano keys in 12TET and some harmonics; a bin and its neighbour are taken
#whichfftbins = [ 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 65, 67, 68, 69, 71, 72, 73, 75, 76, 77, 78, 80, 81, 82, 85, 86, 87, 90, 91, 92, 95, 96, 97, 98, 101, 102, 103, 104, 107, 108, 109, 110, 113, 114, 115, 116, 120, 121, 122, 123, 127, 128, 129, 130, 135, 136, 137, 138, 143, 144, 145, 146, 151, 152, 153, 154, 155, 160, 161, 162, 163, 164, 170, 171, 172, 173, 174, 180, 181, 182, 183, 184, 190, 191, 192, 193, 194, 195, 202, 203, 204, 205, 206, 207, 214, 215, 216, 217, 218, 219, 227, 228, 229, 230, 231, 232, 240, 241, 242, 243, 244, 245, 246, 254, 255, 257, 258, 259, 260, 270, 271, 272, 273, 274, 275, 276, 286, 287, 288, 289, 291, 292, 303, 304, 306, 307, 308, 309, 321, 322, 324, 325, 326, 327, 328, 340, 341, 343, 344, 346, 347, 360, 361, 364, 365, 366, 367, 368, 381, 382, 385, 386, 388, 389, 390, 404, 405, 408, 409, 411, 412, 413, 428, 429, 432, 433, 436, 437, 454, 455, 458, 459, 462, 463, 481, 482, 485, 486, 489, 490, 491, 509, 510, 514, 515, 518, 519, 520, 540, 541, 545, 546, 549, 550, 551, 572, 573, 577, 578, 582, 583, 584, 606, 607, 612, 613, 617, 618, 642, 643, 648, 649, 653, 654, 655, 680, 681, 687, 688, 692, 693, 694, 720, 721, 728, 729, 733, 734, 735, 763, 764, 771, 772, 777, 778, 779, 809, 810, 817, 818, 823, 824, 825, 857, 858, 865, 866, 872, 873, 874, 908, 909, 917, 918, 924, 925, 926, 962, 963, 971, 972, 979, 980, 981, 1019, 1020, 1029, 1030, 1037, 1038, 1039, 1040, 1080, 1081, 1091, 1092, 1099, 1100, 1101, 1144, 1145, 1155, 1156, 1165, 1166, 1167, 1212, 1213, 1224, 1225, 1234, 1235, 1236, 1284, 1285, 1297, 1298, 1307, 1308, 1309, 1310, 1360, 1361, 1374, 1375, 1385, 1386, 1387, 1388, 1441, 1442, 1456, 1457, 1467, 1468, 1469, 1470 ];

numbinsused = len(whichfftbins); #2048 for all, else 379 subset above
inputdimension = numbinsused * numprevframes



def audiofiletotrainingdata(path, usesubset=True):
    #'/data/audio/littleaudio/numan1.wav'
    yt, sr = librosa.load(path,sr=44100,mono=True)

    inputdimensionnow = inputdimension #numbinsused * numprevframes
    
    #remove any start and end silence
    #https://librosa.github.io/librosa/generated/librosa.effects.trim.html
    y, index = librosa.effects.trim(yt)
    
    #only take louder sections
    splitintervals = librosa.effects.split(y, top_db=50, frame_length=fftsize, hop_length=halffftsize);

    z = y[splitintervals[0][0]:splitintervals[0][1]]
    
    for i in range(len(splitintervals)-1):
        z = np.concatenate((z,y[splitintervals[i+1][0]:splitintervals[i+1][1]]))
    
    y = z
    
    #hann
    D_left = librosa.stft(y, n_fft=fftsize, center=False, hop_length=halffftsize, win_length=fftsize, window='boxcar')

    #0.5*Math.log(power+1)*scalefactor;
    magssource = np.log(np.square(np.abs(D_left))+1)* (0.5 * (1/5.456533600026138));
    #np.transpose(

    #newest to oldest in packing in fft frames  e.g. now, now-1, now-2...now-numpreframes+1
#so reverse order based on last dimension
    magssource = np.flip(magssource,1)

    
    if usesubset :
     mags = np.array([ magssource[i] for i in whichfftbins])
    else :
     mags = magssource[0:halffftsize,:]
     inputdimensionnow = halffftsize * numprevframes;


    numtrainingexamples = len(mags[0])-numprevframes+1;

    #np.empty([numtrainingexamples]); #
    x_trainnow = [None]*numtrainingexamples;

    #np.shape(mags) 1 by 1025, need only 1024 of these 0:1024
    for i in range(numtrainingexamples):
#        print(i,numtrainingexamples,np.shape(mags),np.shape(magssource))
#        print(np.shape(mags[0]),mags[0])
#        print(np.array(mags)[:,0:(0+5)])
        x_trainnow[i] = mags[:,i:(i+numprevframes)]

    #[a[i%3] for i in b]
    
    
    x_trainnow = np.asarray(x_trainnow, dtype=np.float32)

    #shape (number, 1024, 5)
    x_trainnow.shape = (len(x_trainnow), inputdimensionnow); #,1

    #other way to get y_train, same result
    #z_train =  x_train[:,:,0,0];
    #z_train = np.transpose(z_train);
    #######z_train.shape = (halffftsize,numtrainingexamples);

    y_trainnow = np.transpose(magssource[0:halffftsize,0:numtrainingexamples]);

    return x_trainnow, y_trainnow

#paths = ['/data/audio/littleaudio/numan1.wav','/data/audio/littleaudio/galv.wav'];


output = audiofiletotrainingdata(paths[0],True)
x_train =  output[0]
y_train =  output[1]

#print('Assumes length of first audio file less than second')
#
#output = audiofiletotrainingdata(paths[0],true)
#x_train =  output[0]
#print(len(x_train))
#
#output = audiofiletotrainingdata(paths[1],false)
#y_train =  output[1][0:len(x_train),:]
#print(len(y_train))



#for i in range(len(paths)-1):
#    output = audiofiletotrainingdata(paths[i+1])
#    x_train = np.concatenate((x_train,output[0]))
#    y_train = np.concatenate((y_train,output[1]))



model = Sequential()

#model.add(Dense(inputdimension//2, input_dim=inputdimension, activation='relu'))
#model.add(Dropout(0.25))
#
#model.add(Dense(inputdimension//4, activation='relu'))
#model.add(Dropout(0.25))

model.add(Dense(halffftsize//2, input_dim=inputdimension, activation='relu'))
model.add(Dropout(0.25))

model.add(Dense(halffftsize//4, activation='relu'))
model.add(Dropout(0.25))

model.add(Dense(halffftsize//2, activation='relu'))
model.add(Dropout(0.25))

model.add(Dense(halffftsize, activation='linear'))

#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#model.compile(loss='categorical_crossentropy', optimizer=sgd)

model.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])

print(np.shape(x_train))
print(np.shape(y_train))

model.fit(x_train, y_train, batch_size=32, epochs=100)
score = model.evaluate(x_train, y_train, batch_size=32)

print(score)

#model export for SuperCollider
from kerasify import export_model
export_model(model, 'DNN1.model')


#model export for javascript
import onnxmltools
#from keras.models import load_model

# Update the input name and path for your Keras model
#input_keras_model = 'model.h5'

# Change this path to the output name and path for the ONNX model
output_onnx_model = '/Users/ioi/Desktop/onnxoutput/modelcheck.onnx'

# Load your Keras model
#keras_model = load_model(input_keras_model)

# Convert the Keras model into ONNX
onnx_model = onnxmltools.convert_keras(model)

# Save as protobuf
onnxmltools.utils.save_model(onnx_model, output_onnx_model)


#X_test.astype(np.float32)
# Compute the prediction with ONNX Runtime
#import onnxruntime as rt
#sess = rt.InferenceSession(output_onnx_model)
#input_name = sess.get_inputs()[0].name
#label_name = sess.get_outputs()[0].name
#
#print(input_name, label_name)
#
#print(np.shape(x_train[0]))
#
#print(x_train[0])
#
#pred_onx = sess.run(None, {input_name: x_train})[0]
#
#print(pred_onx)

#import tensorflowjs as tfjs
##tfjs.converters.save_keras_model(model, '/Users/ioi/Desktop/tfjsoutput2')
#
##can save memory and file size this way
##https://medium.com/huia/creating-an-interactive-artificial-intelligence-experience-in-the-browser-with-tensorflowjs-ea205ee08c02
#tfjs.converters.save_keras_model(model, '/Users/ioi/Desktop/tfjsoutput2',quantization_dtype=np.uint16);
#

# save model and architecture to single file
#https://machinelearningmastery.com/save-load-keras-deep-learning-models/
#model.save("model.h5")
#print("Saved model to disk")
#
## load model
#model = load_model('model.h5')
## summarize model.
#model.summary()
#
#score = model.evaluate(x_test, y_test, batch_size=32)
#
#print(score)
#
