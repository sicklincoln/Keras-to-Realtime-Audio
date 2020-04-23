/*
 SuperCollider real time audio synthesis system
 Copyright (c) 2002 James McCartney. All rights reserved.
 http://www.audiosynth.com
 
 This program is free software; you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation; either version 2 of the License, or
 (at your option) any later version.
 
 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.
 
 You should have received a copy of the GNU General Public License
 along with this program; if not, write to the Free Software
 Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301  USA
 */

//UGen by Nick Collins
//Released under the GNU GPL as extensions for SuperCollider 3

//cmake -DSC_PATH=/data/gitprojects/SuperCollider -DCMAKE_OSX_ARCHITECTURES='x86_64' ..

//NOTE: there are certain assumptions in this code about the size of the input and output (currently 2048 frames of spectral data). However, more general UGens are possible based on adapting more of Kerasify, etc. The current code should work with deeper nets as long as input and output at size 2048
//It is also worth saying that realtime deep learning with smaller spectral frames gets very intensive; with the NRT thread it will never keep up, but there is a possibility of adapting KerasifyA


//uses Kerasify, thank you Robert W. Rose
//https://github.com/moof2k/kerasify
//note that this dependency could probably be removed, e.g. I've rolled my own neural net here since Kerasify itself wasn't realtime memory management efficient, but use Kerasify for export from python and loading a model in the UGen, to save writing my own export/import too.

//if use NRT via DoAsynchronousCommand, too slow
//if try to call model in RT, too much load for one block

//two solutions are possible
//amortiseflag = 0) PV_Kerasify: otherwise must write own thread, see DiskIO_UGens.cpp, potential issues with setting thread priority, but in practice does work out OK (just don't use DoAsynchronousCommand, that NRT thread on the server is far too slow)
//https://stackoverflow.com/questions/18884510/portable-way-of-setting-stdthread-priority-in-c11
//https://chromium.googlesource.com/chromium/src/base/+/master/threading/platform_thread_mac.mm
//https://stackoverflow.com/questions/10876342/equivalent-of-setthreadpriority-on-linux-pthreads

//amortiseflag = 1) Can only amortise over blocks if write own model calculations (going a layer at a time per block, etc). Did try going a layer at a time in Kerasify itself, but it was still too slow. So wrote own net implementation. First block sets up data, subsequent blocks calculate a layer of the network at a time
//assumes network not so deep that more layers than available blocks...


#include "keras_model.h"

#include "SC_PlugIn.h"
#include <math.h>
#include <stdlib.h>
#include "FFT_UGens.h"
#include <time.h>

//int messagecounter = 0;

clock_t start, end;
double cpu_time_used;


InterfaceTable *ft; 


//those supported by kerasify, keras has a few more
//https://keras.io/activations/
enum {
    activation_linear,
    activation_relu,
    activation_softplus,
    activation_sigmoid,
    activation_tanh,
    activation_hardsigmoid
};

//public by default
struct NCLayer {
    
    int inputsize;
    int numunits;
    float * connectionmatrix;//rows by columns, as flat array
    float * bias;
    int activationtype;
    float * output;
    
};

//assumes sizes all fine
void CalculateNCLayer(NCLayer* layer, float * input) {
    
    int i,j;
    float unitsum;
    
    int inputsize = layer->inputsize;
    float * matrix = layer->connectionmatrix;
    float * bias = layer->bias;
    int activation = layer->activationtype;
    float calc;
    float * output = layer->output;
    
    for (i=0; i<layer->numunits; ++i) {
        
        unitsum = 0.0f;
        float * weightsnow = matrix + (i*inputsize);
        
        for (j=0; j<inputsize; ++j)
        {
            unitsum += weightsnow[j] * input[j];
            
        }
        
        unitsum += bias[i];
        
        //nothing to do for linear?
        //activation function
        switch(activation) {
            
            case activation_relu:
            if(unitsum<0.0) unitsum = 0.0;
            break;
            
            case activation_softplus:
            unitsum = log(1.0 + exp(unitsum));
            break;
            case activation_sigmoid: {
                float x = unitsum;
                
                if (x >= 0) {
                    unitsum = 1.0 / (1.0 + exp(-x));
                } else {
                    float z = exp(x);
                    unitsum = z / (1.0 + z);
                }
            }
            break;
            case activation_tanh:
            unitsum = tanh(unitsum);
            break;
            case activation_hardsigmoid: {
                
                float x = (unitsum * 0.2) + 0.5;
                
                if (x <= 0) {
                    unitsum = 0.0;
                } else if (x >= 1) {
                    unitsum = 1.0;
                } else {
                    unitsum = x;
                }
                
            }
            break;
            
            default:
            break;
        }
        
        output[i] = unitsum;
        
    }
    
    
}



void CalculateNCLayerAmort(NCLayer* layer, float * input, int startunit, int unitstodo) {
    
    int i,j;
    float unitsum;
    
    int inputsize = layer->inputsize;
    float * matrix = layer->connectionmatrix;
    float * bias = layer->bias;
    int activation = layer->activationtype;
    float calc;
    float * output = layer->output;
    
    int endunit = startunit+unitstodo;
    
    if(endunit>layer->numunits) {
        
        endunit = layer->numunits;
    }
    
    for (i=startunit; i<endunit; ++i) {
        
        unitsum = 0.0f;
        float * weightsnow = matrix + (i*inputsize);
        
        for (j=0; j<inputsize; ++j)
        {
            unitsum += weightsnow[j] * input[j];
            
        }
        
        unitsum += bias[i];
        
        //nothing to do for linear?
        //activation function
        switch(activation) {
            
            case activation_relu:
            if(unitsum<0.0) unitsum = 0.0;
            break;
            
            case activation_softplus:
            unitsum = log(1.0 + exp(unitsum));
            break;
            case activation_sigmoid: {
                float x = unitsum;
                
                if (x >= 0) {
                    unitsum = 1.0 / (1.0 + exp(-x));
                } else {
                    float z = exp(x);
                    unitsum = z / (1.0 + z);
                }
            }
            break;
            case activation_tanh:
            unitsum = tanh(unitsum);
            break;
            case activation_hardsigmoid: {
                
                float x = (unitsum * 0.2) + 0.5;
                
                if (x <= 0) {
                    unitsum = 0.0;
                } else if (x >= 1) {
                    unitsum = 1.0;
                } else {
                    unitsum = x;
                }
                
            }
            break;
            
            default:
            break;
        }
        
        output[i] = unitsum;
        
    }
    
    
}




//KerasModel * g_model;

struct PV_Kerasify : public Unit
{
    //int m_n;
    //float * m_topn;
    //int * m_topnindices;
    
    char * path;
    
    KerasModel * model;
    
    float * phases; //[2048];
    float * spectrumnow;
    
    //no longer needed, rolled own network below
    // Create a 1D Tensor on length 10 for input data.
    //Tensor * in; //(2048);
    //float datafortensor[2048];
    // Run prediction.
    //Tensor * out;
    
    bool modelready;
    bool newoutput;
    
    //roll own neural network
    float * input;
    int numlayers;
    NCLayer * layers;
    float * output;
    
    int amortisationflag;
    int amortisationcounter;
    
    float * currentinputpointer; //used in amortisation to shift active input data
    //int amortschedulelayer[100];
    //int amortscheduleindexstart[100];
    //int amortscheduleunitstodo[100];
    
    
    
};


#include "SC_SyncCondition.h"
#include <atomic>
//#include <new>
//#include <functional>
#include <thread>
//#include "SC_Lock.h"
#include <boost/lockfree/queue.hpp>
#include <boost/lockfree/spsc_queue.hpp>

enum {
    kCmd_Ctor,
    kCmd_Run,
    kCmd_Dtor,
};

struct KerasifyMsg
{
    //        World *mWorld;
    int16 mCommand;
    int counter;
    //        int16 mChannels;
    //        int32 mBufNum;
    //        int32 mPos;
    //        int32 mFrames;
    
    PV_Kerasify * unit;
    
    void Perform();
};

void KerasifyMsg::Perform()
{
    //PV_Kerasify* unit = (PV_Kerasify*)unit;
    
    switch (mCommand) {
        case kCmd_Run : {
            
            //printf("before apply \n");
            
            // Run prediction.
            //Tensor out(2048); // &out
            
            //out.PrintShape();
            //printf("after print \n");
            
            //printf("thread which %d model %p unit %p %p %p\n",counter, (void*)g_model, (void*)unit, (void*)unit->in, (void*)unit->out);
            
            //printf("unit %f %f %f\n", unit->spectrumnow[0],unit->spectrumnow[1],unit->spectrumnow[2]);
            
            //unit->in->PrintShape();
            //unit->out->PrintShape();
            
            //start = clock();
            //... /* Do the work. */
            
            //unit->model->Apply(unit->in, unit->out);
            //end = clock();
            //cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
            
            //printf("took %f seconds to execute \n", cpu_time_used);
            
            
            
            float * input = unit->input;
            
            for (int k=0; k<unit->numlayers; ++k) {
                
                CalculateNCLayer(&unit->layers[k],input);
                
                input = unit->layers[k].output;
                
            }
            
            //final input pointer points to final layer output
            for (int i=0; i<2048; ++i) {
                
                unit->spectrumnow[i] = input[i]; //((i*i)%17)/17.0;
            }
            
            //printf("unit %f %f %f\n", unit->spectrumnow[0],unit->spectrumnow[1],unit->spectrumnow[2]);
            
            
            
            
            unit->newoutput = true;
            
            
        }
        break;
        case kCmd_Ctor :
        {
            
            // Initialize model.
            
            //unit->in = new Tensor(2048);
            //unit->out = new Tensor(2048);
            
            // Create a 1D Tensor on length 10 for input data.
            //Tensor in(2048);
            //std::vector<float> data_;
            //float data[2048];
            
            //for (int i=0; i<2048; ++i) {
            
            //  data[i] = 0.0f; //((i*i)%17)/17.0;
            //}
            
            //https://stackoverflow.com/questions/259297/how-do-you-copy-the-contents-of-an-array-to-a-stdvector-in-c-without-looping
            //in.data_.insert(in.data_.end(), &data[0], &data[2048]);
            
            //unit->in->data_.insert(unit->in->data_.end(), &data[0], &data[2048]);
            
            //unit->out->data_.insert(unit->out->data_.end(), &data[0], &data[2048]);
            
            //in.data_ = data;
            //in.data_ = {{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}};
            
            //g_model = new KerasModel();
            //g_model->LoadModel(unit->path); //"DNN1.model");
            
            
            unit->model = new KerasModel();
            
            unit->model->LoadModel(unit->path); //"DNN1.model");
            
            //unit->in.data_.insert(unit->in.data_.end(), &(unit->datafortensor[0]), &(unit->datafortensor[2048]));
            
            printf("loaded %s\n",unit->path);
            
            //copy model data to my own local struct
            
            unit->input = new float[2048];
            
            unit->numlayers = unit->model->layers_.size();
            
            unit->layers = new NCLayer[unit->numlayers];
            
            for (int k=0; k<unit->numlayers; ++k) {
                
                //inputsize  * outputsize
                Tensor * weights_ = &(((KerasLayerDense*)unit->model->layers_[k])->weights_);
                Tensor * biases_ = &(((KerasLayerDense*)unit->model->layers_[k])->biases_);
                
                float * matrix = new float[weights_->dims_[0] * weights_->dims_[1]];
                float * bias = new float[biases_->dims_[0]];
                
                int inputsize = weights_->dims_[0];
                
                unit->layers[k].inputsize = inputsize;
                unit->layers[k].numunits = weights_->dims_[1];
                
                
                for (int i = 0; i < weights_->dims_[1]; i++) {
                    
                    float * target = matrix + (inputsize*i);
                    
                    for (int j = 0; j < weights_->dims_[0]; j++) {
                        target[j] = (*weights_)(j, i);
                    }
                }
                
                for (int i = 0; i < biases_->dims_[0]; i++) {
                    bias[i] = (*biases_)(i);
                }
                
                unit->layers[k].connectionmatrix = matrix;
                unit->layers[k].bias = bias;
                
                unit->layers[k].output = new float[biases_->dims_[0]];
                
                
                int type = ((KerasLayerDense*)unit->model->layers_[k])->activation_.activation_type_;
                //KerasLayerActivation::ActivationType
                
                unit->layers[k].activationtype = activation_linear;
                
                
                if(type == KerasLayerActivation::ActivationType::kRelu) unit->layers[k].activationtype = activation_relu;
                
                if(type == KerasLayerActivation::ActivationType::kSoftPlus) unit->layers[k].activationtype = activation_softplus;
                
                if(type == KerasLayerActivation::ActivationType::kSigmoid) unit->layers[k].activationtype = activation_sigmoid;
                
                if(type == KerasLayerActivation::ActivationType::kTanh) unit->layers[k].activationtype = activation_tanh;
                
                if(type == KerasLayerActivation::ActivationType::kHardSigmoid) unit->layers[k].activationtype = activation_hardsigmoid;
                
                
                
                printf("layer number %d type %d inputsize %d numunits %d \n",k, type, inputsize, unit->layers[k].numunits);
                
            }
            
            
            
            printf("Set up internal model data structure with %d layers\n", unit->numlayers);
            
            // Run prediction.
            //Tensor out;
            
            //            start = clock();
            //            //... /* Do the work. */
            //            unit->model->Apply(unit->in, unit->out);
            //            end = clock();
            //            cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
            //
            //            printf("first time took %f seconds to execute \n", cpu_time_used);
            //
            
            unit-> modelready = true;
            
            
        }
        break;
        case kCmd_Dtor :
        {
            //problem if unit already deallocated by this point
            //delete unit->in;
            //delete unit->out;
            delete unit->model;
            
            for (int k=0; k<unit->numlayers; ++k) {
                
                delete [] unit->layers[k].connectionmatrix;
                delete [] unit->layers[k].bias;
                delete [] unit->layers[k].output;
                
            }
            
            delete [] unit->layers;
            delete [] unit->input;
            
            
        }
        break;
    }
    
}

struct KerasifyThread
{
    SC_SyncCondition mKerasifyFifoHasData;
    
#ifdef SUPERNOVA
    boost::lockfree::queue<KerasifyMsg, boost::lockfree::capacity<256> > mKerasifyFifo;
#else
    boost::lockfree::spsc_queue<KerasifyMsg, boost::lockfree::capacity<256> > mKerasifyFifo;
#endif
    
    std::atomic<bool> mRunning;
    std::thread mThread;
    
    KerasifyThread():
    mRunning(false)
    {}
    
    ~KerasifyThread()
    {
        if (mRunning) {
            mRunning.store(false);
            mKerasifyFifoHasData.Signal();
            mThread.join();
        }
    }
    
    void launchThread()
    {
        using namespace std;
        mRunning.store(true);
        
        mThread = thread( bind(&KerasifyThread::kerasifyThreadFunc, this) ) ;
    }
    
    bool Run(KerasifyMsg& data)
    {
        bool pushSucceeded = mKerasifyFifo.push(data);
        if (pushSucceeded)
        mKerasifyFifoHasData.Signal();
        return pushSucceeded;
    }
    
    void kerasifyThreadFunc()
    {
        while (mRunning.load()) {
            mKerasifyFifoHasData.WaitEach();
            
            KerasifyMsg msg;
            bool popSucceeded = mKerasifyFifo.pop(msg);
            
            if (popSucceeded)
            msg.Perform();
        }
    }
};

KerasifyThread *gKerasify;



extern "C" {  
    
    void PV_Kerasify_next(PV_Kerasify* unit, int inNumSamples);
    void PV_Kerasify_Ctor(PV_Kerasify* unit);
    void PV_Kerasify_Dtor(PV_Kerasify* unit);
    
}




void PV_Kerasify_Ctor( PV_Kerasify* unit ) {
    
    //printf("PV_Kerasify_Ctor /n hello \n");
    
    unit-> amortisationflag = (int)ZIN0(1);
    
    unit-> amortisationcounter = 0;
    
    unit->modelready = false;
    unit->newoutput = false;
    
    World *world = unit->mWorld;
    
    unit->phases = (float * ) RTAlloc(unit->mWorld, 2048*sizeof(float));
    unit->spectrumnow = (float * ) RTAlloc(unit->mWorld, 2048*sizeof(float));
    
    for (int i=0; i<2048; ++i) {
        
        unit->phases[i] = 0.0f;
        unit->spectrumnow[i] = 0.0f;
    }
    
    int pathsize = (int) ZIN0(2);
    
    unit->path = (char *) RTAlloc(unit->mWorld,sizeof(char)*(pathsize+1));
    
    for(int i=0; i<pathsize; ++i) {
        unit->path[i] = (char)ZIN0(3+i);
    }
    
    unit->path[pathsize] = 0;
    
    printf("constructor for PV_Kerasify loading %s\n",unit->path);
    
    // send a message to side thread
    KerasifyMsg msg;
    msg.unit = unit;
    //msg.counter = messagecounter;
    msg.mCommand = kCmd_Ctor;
    gKerasify->Run(msg);
    
    
    SETCALC(PV_Kerasify_next);
    ZOUT0(0) = ZIN0(0);
    
}


void PV_Kerasify_Dtor( PV_Kerasify* unit ) {
    
    RTFree(unit->mWorld, unit->path);
    
    RTFree(unit->mWorld, unit->phases);
    RTFree(unit->mWorld, unit->spectrumnow);
    
    // send a message to side thread
    KerasifyMsg msg;
    msg.unit = unit;
    //msg.counter = messagecounter;
    msg.mCommand = kCmd_Dtor;
    gKerasify->Run(msg);
    
    
}


void PV_Kerasify_next( PV_Kerasify *unit, int inNumSamples ) {
    
    int i,j,k;
    
    if(unit-> amortisationcounter<=0) {
        
        float fbufnum = ZIN0(0);
        
        //if (fbufnum < 0.f) return;
        
        if (fbufnum < 0.f) { ZOUT0(0) = -1.f; return; }
        ZOUT0(0) = fbufnum;
        
        int ibufnum = (uint32)fbufnum;
        
        World *world = unit->mWorld;
        SndBuf *buf;
        
        if (ibufnum >= world->mNumSndBufs) {
            int localBufNum = ibufnum - world->mNumSndBufs;
            Graph *parent = unit->mParent;
            if(localBufNum <= parent->localBufNum) {
                buf = parent->mLocalSndBufs + localBufNum;
            } else {
                buf = world->mSndBufs;
            }
        } else {
            buf = world->mSndBufs + ibufnum;
        }
        
        LOCK_SNDBUF(buf);
        
        
        if(unit-> modelready) {
            
            int numbins = (buf->samples - 2) >> 1;
            
            float * data = buf->data; //just use it, it is in form dc, nyquist then real,imag pairs per ascending band
            
            //SCComplexBuf* complex = ToComplexApx(buf);
            //SCComplex * data = complex->bin;
            //also dc, nyquist
            
            float real, imag;
            
            real = data[0]; //DC
            
            //unit->in->data_[0] = 0.09163326695 * log((real*real) + 1);
            //unit->spectrumnow[0] = 0.09163326695 * log((real*real) + 1);
            unit->input[0] = 0.09163326695 * log((real*real) + 1);
            
            unit->phases[0] = 0.0;
            
            //printf("before prep data \n");
            
            for  (j=1;j<numbins; ++j) {
                
                int index = 2*j;
                
                real = data[index];
                imag = data[index+1];
                
                //0.5*Math.log(power+1)*scalefactor; //(1/5.456533600026138)
                //float ampnow = sqrt((real*real) + (imag*imag));
                
                //relates to scale factor and log power encoding used for training neural net in first place with spectral data in range [0,1]
                
                //unit->in->data_[j] = 0.09163326695 * log((real*real) + (imag*imag) + 1);
                
                //unit->spectrumnow[j] = 0.09163326695 * log((real*real) + (imag*imag) + 1);
                
                unit->input[j] = 0.09163326695 * log((real*real) + (imag*imag) + 1);
                
                unit->phases[j] = atan2(imag, real);
                
            }
            
            unit->currentinputpointer = unit->input;
            
            
            if(unit->newoutput) {
                //musn't be replaced mid use, so need to only transfer over from a completed run
                //previous output data potentially
                //power spectrum back to complex (eg no phase, real only)
                
                float magnitude, phase;
                
                for (i = 0; i < numbins; ++i) {
                    
                    //out.data_[i] unit->out->data_[i] unit->out->data_[i]
                    magnitude = exp((unit->spectrumnow[i])*5.456533600026138)-1;
                    
                    phase = unit->phases[i];
                    
                    //fftdata[2*i] = outputspectra.w[i]; //act.w[i]; //
                    //return to magnitude not power
                    data[2*i] =  magnitude * cos(phase);//Math.sqrt(Math.abs(outputspectra.w[i]));
                    data[2*i+1] = magnitude * sin(phase); //0.0;
                    
                    //if(i<10) fftdata[2*i] = 0.0;
                    
                }
                
                unit->newoutput = false; //or not needed at all?
            }
            
            
            //printf("pre message %d model %p unit %p %p %p\n", messagecounter,(void*)g_model,(void*)unit, (void*)unit->in, (void*)unit->out);
            
            if(unit->amortisationflag==0) {
                
                // send a message to side thread
                KerasifyMsg msg;
                msg.unit = unit;
                //msg.counter = messagecounter;
                msg.mCommand = kCmd_Run;
                
                //++messagecounter;
                
                
                //printf("sendMessage %d  %d %d %d\n", msg.mBufNum, msg.mPos, msg.mFrames, msg.mChannels);
                gKerasify->Run(msg);
                
            } else
            unit->amortisationcounter = 1; //unit->model->layers_.size();
            
            
        }
        
        return;
        
    } else {
        
        //amortise layer by layer
        //could refine to calculate parts of layers for further amortisation
        //can create a schedule based on available blocks spread over calcs per layer and layers
        
        //printf("amortise flag %d counter %d layer %d of %d \n",unit->amortisationflag, unit->amortisationcounter, unit->amortisationcounter-1, unit->numlayers);
        
        //available amort periods before new spectral frame = (2048/64 - 1) = 31
        //assuming 2048 hop, 64 sample block size
        
        int stepsperlayer = unit->amortisationflag; //let user set this
        
        float * input = unit->currentinputpointer;
        
        int layernum = (unit->amortisationcounter-1)/stepsperlayer; //split each layer into stepsperlayer
        NCLayer * layernow = &unit->layers[layernum];
        int layerstep = (unit->amortisationcounter-1)%stepsperlayer;
        
        int unitsavailableinlayer = layernow->numunits;
        int totalperamortstep = unitsavailableinlayer/stepsperlayer; //assumes divisible by stepsperlayer, else add 1 to compensate
        
        if((totalperamortstep*stepsperlayer) < unitsavailableinlayer) {
            totalperamortstep = totalperamortstep + 1;
        }
        
        int startposforamortstep = totalperamortstep * layerstep;
        
        //printf("amortise flag %d counter %d layer %d of %d layernum %d layerste %d  unitsavailableinlayer %d totalperamortstep %d startposforamortstep %d \n",unit->amortisationflag, unit->amortisationcounter, unit->amortisationcounter-1, unit->numlayers, layernum, layerstep, unitsavailableinlayer, totalperamortstep, startposforamortstep);
        
        //all at once too high peak CPU
        //CalculateNCLayer(layernow,input);
        
        CalculateNCLayerAmort(layernow,input,startposforamortstep,totalperamortstep);
        
        //if((startposforamortstep+totalperamortstep)>=unitsavailableinlayer)
        if(layerstep==(stepsperlayer-1)) //just did final step of 4 per layer
        {
            unit->currentinputpointer = layernow->output;
            input = unit->currentinputpointer;
        }
        
        ++unit->amortisationcounter;
        
        //unit->amortisationcounter>unit->numlayers
        if(unit->amortisationcounter>(unit->numlayers*stepsperlayer)) {
            unit->amortisationcounter = 0;
            
            //final input pointer points to final layer output
            for (int i=0; i<2048; ++i) {
                
                unit->spectrumnow[i] = input[i];
            }
            
            unit->newoutput = true;
        }
        
        //safety
        ZOUT0(0) = -1.f;
        return;
        
    }
    
    
    /*
     
     
     //amortise by hacking KerasModel class
     //layers_ made public rather than private
     //call from outside, a step at a time over layers
     
     //bool KerasModel::Apply(Tensor* in, Tensor* out) {
     
     Tensor *temp_in, *temp_out;
     
     //for (unsigned int i = 0; i < layers_.size(); i++) {
     
     if (unit->amortisationcounter == 1) {
     temp_in = unit->in;
     temp_out = unit->out;
     } else {
     
     temp_in = unit->out;
     temp_out = unit->in;
     }
     //printf("Apply layer %d of %d layer(s)\n", unit->amortisationcounter-1, unit->model->layers_.size());
     
     bool result = unit->model->layers_[unit->amortisationcounter-1]->Apply(temp_in, temp_out);
     
     if(!result) printf("Failed to apply layer %d \n", unit->amortisationcounter-1);
     
     *(unit->out) = *temp_out;
     
     
     
     //        Tensor temp_in, temp_out;
     //
     //        //for (unsigned int i = 0; i < layers_.size(); i++) {
     //
     //            if (unit->amortisationcounter == 1) {
     //                temp_in = *(unit->in);
     //            } else {
     //
     //                temp_in = *(unit->out);
     //            }
     //
     //
     //    printf("Apply layer %d of %d layer(s)\n", unit->amortisationcounter-1, unit->model->layers_.size());
     //    bool result = unit->model->layers_[unit->amortisationcounter-1]->Apply(&temp_in, &temp_out);
     //
     //    if(!result) printf("Failed to apply layer %d \n", unit->amortisationcounter-1);
     //
     //            //temp_in = temp_out;
     //        //}
     //
     //        //*out = temp_out;
     //
     //        *(unit->out) = temp_out;
     //
     ++unit->amortisationcounter;
     
     if(unit->amortisationcounter>unit->model->layers_.size()) {
     unit->amortisationcounter = 0;
     unit->newoutput = true;
     }
     
     
     //}
     */
    
    
    //NRT thread too slow
    /*
     //run model on latest data, will lag at least one spectral frame behind, but only way to amortise
     CmdData* cmd = (CmdData*)RTAlloc(unit->mWorld, sizeof(CmdData));
     //cmd->samplingrate_ = unit->mRate->mSampleRate;
     cmd->unit = (Unit *)unit;
     cmd->nrtallocated = NULL; //will be allocated in NRT thread
     cmd->type = CmdData::NRTModelRunPV_Kerasify;
     
     DoAsynchronousCommand(unit->mWorld, 0, "", (void*)cmd,
     (AsyncStageFn)cmdStage2,
     (AsyncStageFn)cmdStage3,
     NULL,
     cmdCleanup,
     0, 0);
     
     */
    
    
    
    
    
}








#define DefinePVUnit(name) \
(*ft->fDefineUnit)(#name, sizeof(PV_Unit), (UnitCtorFunc)&name##_Ctor, 0, 0);

C_LINKAGE SC_API_EXPORT void unload(InterfaceTable *inTable)
{
    delete gKerasify;
}

PluginLoad(PV_Kerasify)
{
    
    init_SCComplex(inTable);
    
    ft = inTable;
    
    gKerasify = new KerasifyThread();
    gKerasify->launchThread();
    
    DefinePVUnit(PV_Kerasify);
    
    
    
}


