#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 12:30:06 2018

@author: jcdevaney
"""

import librosa

def beatTrack(filename):
    
    # Load audio
    y, sr = librosa.load(filename)
            
    # Harmonic-percussive source separation
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    
    # Now, let's run the beat tracker.
    # We'll use the percussive component for this part
    tempo, beats = librosa.beat.beat_track(y=y_percussive, sr=sr)
        
    beatTimes = librosa.frames_to_time(beats)
    
    return tempo, beats, beatTimes