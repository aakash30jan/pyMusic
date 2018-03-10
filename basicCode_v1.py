#Basic Code v1

#Python is an interpreted programming language which means you can execute code line by line directly and freely, without previously compiling a program into machine-language instructions(like you do in C++/FORTRAN/C etc.)

# If a line starts with "#", it is a commented line and anything code(instruction) written in that line will not be executed


#First we import libraries which have the pre-written and well-defined functions

import numpy as np  
#numpy is the most useful library with support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays.

import matplotlib.pyplot as plt
#matplotlib is a widely used plotting library and pyplot is a matplotlib module which provides a MATLAB-like interface.

import scipy.io.wavfile
#SciPy is a popular library used for scientific computing and technical computing which contains modules for optimization, linear algebra, integration, interpolation, special functions, FFT, signal and image processing, ODE solvers and other tasks common in science and engineering


from pydub import AudioSegment
#pydub is a library for to manipulate audio with a simple and easy high level interface 
#for more information about this library https://github.com/jiaaro/pydub#installation

#we would be using .wav format of audio for analysis as it is a raw and uncompressed format for audio
#convert .mp3 to .wav if necessary
#example- "mpg123 -w temp_alankar.wav temp_alankar.mp3"

filename="temp_alankar.wav" #we are defining the filename here

#clipping the audio if necessary
        #ORGAudio= AudioSegment.from_wav(filename)
        #t1=00001 #in milliseconds 
        #t2=22000 # the audio would be saved from t1 to t2
        #newAudio=ORGAudio[t1:t2] 
        #newAudio.export(filename+"CLIPPED",format='wav') #save the new audio file
        #filename=filename+"CLIPPED" #open the new audio file


rate,audData=scipy.io.wavfile.read(filename) #create array of data from the audio file

print "rate (samples per second)", rate
print "length of audio in seconds ", audData.shape[0] / rate



#wav number of channels mono/stereo
audData.shape[1]
#if stereo grab both channels
channel1=audData[:,0] #left
channel2=audData[:,1] #right

print "datatype ",audData.dtype #the higher value of int the better audio quality

#save wav the original wav file
#scipy.io.wavfile.write(filename+"_ORG", rate, audData)
#save a file at half and double speed
#scipy.io.wavfile.write(filename+"_HLF", rate/2, audData)
#scipy.io.wavfile.write(filename+"_DBL", rate*2, audData)
#save a single channel
#scipy.io.wavfile.write(filename+"_CH1", rate, channel1)
#scipy.io.wavfile.write(filename+"_CH2", rate, channel2)

#Energy of music
np.sum(channel1.astype(float)**2)

#power - energy per unit of time
1.0/(2*(channel1.size)+1)*np.sum(channel1.astype(float)**2)/rate


#create a time variable in seconds
time = np.arange(0, float(audData.shape[0]), 1) / rate

#plot amplitude (or loudness) over time
plt.figure(1)
plt.subplot(211)
plt.plot(time, channel1,  color='red')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.subplot(212)
plt.plot(time, channel2, color='blue')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.show()


from numpy import fft as fft

fourier=fft.fft(channel1)

plt.plot(fourier, color='green')
plt.xlabel('k')
plt.ylabel('Amplitude')
plt.show()




n = len(channel1)
fourier = fourier[0:(n/2)]

# scale by the number of points so that the magnitude does not depend on the length
fourier = fourier / float(n)

#calculate the frequency at each point in Hz
freqArray = np.arange(0, (n/2), 1.0) * (rate*1.0/n);

plt.plot(freqArray/1000, 10*np.log10(fourier), color='green')
plt.xlabel('Frequency (kHz)')
plt.ylabel('Power (dB)')
plt.show()



plt.figure(2, figsize=(8,6))
plt.subplot(211)
Pxx, freqs, bins, im = plt.specgram(channel1, Fs=rate, NFFT=1024, cmap=plt.get_cmap('autumn_r'))
cbar=plt.colorbar(im)
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
cbar.set_label('Intensity dB')
plt.subplot(212)
Pxx, freqs, bins, im = plt.specgram(channel2, Fs=rate, NFFT=1024, cmap=plt.get_cmap('autumn_r'))
cbar=plt.colorbar(im)
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
cbar.set_label('Intensity (dB)')
plt.show()



#SA RE GA MA PA DHA NI SA SA NI DHA PA MA GA RE SA


#you can refer the article here
#ref="http://myinspirationinformation.com/uncategorized/audio-signals-in-python/"

