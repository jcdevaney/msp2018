# -*- coding: utf-8 -*-
"""
analyzeAuio.py

This function returns a structure of note-wise paramters for the segmented 
note audio that is inputted

"""

def analyzeAudio(file, n_fft=2048, hop_length=512, n_chroma=12, n_mfcc=13):

    import librosa
    import aubio

    win_s = hop_length * 4

    sig , sr = librosa.load(file,mono=True,sr=None)
    aubioSource = aubio.source(file, 0, hop_length)

#    class audioData:
#        pass
#    
#    audioVals = audioData()
    
    audioVals = {}
    
    audioVals['file'] = file
    
    # fundamental frequency
    samplerate = aubioSource.samplerate
    tolerance = 0.1
    pitch_o = aubio.pitch("yin", win_s, hop_length, samplerate) 
    pitch_o.set_tolerance(tolerance)
    pitchesYIN = []
    confidences = [] 
    total_frames = 0
    while True:
        samples, read = aubioSource()
        pitch = pitch_o(samples)[0]
        pitch = int(round(pitch))
        confidence = pitch_o.get_confidence()
        pitchesYIN += [pitch]
        confidences += [confidence]
        total_frames += read
        if read < hop_length:
             break
    audioVals['f0'] = pitchesYIN    
    audioVals['f0confidence'] = confidences
    
    
    audioVals['rms'] = librosa.feature.rmse(y=sig)
                
    # harmonic component of the signal
    sig_harmonic, sig_percussive = librosa.effects.hpss(sig)
    
    #stft chroma
    audioVals['twelveChroma'] = librosa.feature.chroma_cqt(y=sig_harmonic,sr=sr, n_chroma=12)
    
    # mfccs    
    audioVals['mfcc'] = librosa.feature.mfcc(y=sig, sr=sr, hop_length=512, n_mfcc=13)
    
    # spectral centroid
    audioVals['specCent']=librosa.feature.spectral_centroid(y=sig, sr=sr, n_fft=2048, hop_length=512, freq=None)
    
    # spectral bandwidth
    audioVals['specBand']=librosa.feature.spectral_bandwidth(y=sig, sr=sr, n_fft=2048, hop_length=512, freq=None)
    
    # spectral contrast
    audioVals['specContrast']=librosa.feature.spectral_contrast(y=sig, sr=sr, n_fft=2048, hop_length=512, freq=None)
    
    # spectral flatness
    audioVals['specFlatness']=librosa.feature.spectral_flatness(y=sig)

    return audioVals