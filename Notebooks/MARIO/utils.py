

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions for writing music in Python.
@author: khe
"""
import numpy as np

def get_piano_notes():
    '''
    Get the frequency in hertz for all keys on a standard piano.
    Returns
    -------
    note_freqs : dict
        Mapping between note name and corresponding frequency.
    '''
    
    # White keys are in Uppercase and black keys (sharps) are in lowercase
    octave = ['C', 'c', 'D', 'd', 'E', 'F', 'f', 'G', 'g', 'A', 'a', 'B'] 
    base_freq = 440 #Frequency of Note A4
    keys = np.array([x+str(y) for y in range(0,9) for x in octave])
    # Trim to standard 88 keys
    start = np.where(keys == 'A0')[0][0]
    end = np.where(keys == 'C8')[0][0]
    keys = keys[start:end+1]
    
    note_freqs = dict(zip(keys, [2**((n+1-49)/12)*base_freq for n in range(len(keys))]))
    note_freqs[''] = 0.0 # stop
    return note_freqs

def get_sine_wave(frequency, duration, sample_rate=44100, amplitude=4096):
    '''
    Get pure sine wave. 
    Parameters
    ----------
    frequency : float
        Frequency in hertz.
    duration : float
        Time in seconds.
    sample_rate : int, optional
        Wav file sample rate. The default is 44100.
    amplitude : int, optional
        Peak Amplitude. The default is 4096.
    Returns
    -------
    wave : TYPE
        DESCRIPTION.
    '''
    t = np.linspace(0, duration, int(sample_rate*duration))
    wave = amplitude*np.sin(2*np.pi*frequency*t)
    return wave

def apply_overtones(frequency, duration, factor, sample_rate=44100, amplitude=4096):
    '''
    Return fundamental note with overtones applied. 
    Parameters
    ----------
    frequency : float
        Frequency in hertz.
    duration : float
        Time in seconds.
    factor : list
        List of floats as fraction of the fundamental amplitude for amplitudes 
        of the overtones.
    sample_rate : int, optional
        Wav file sample rate. The default is 44100.
    amplitude : int, optional
        Peak Amplitude. The default is 4096.
    Returns
    -------
    fundamental : ndarray
        Output note of `float` type.
    '''
    assert abs(1-sum(factor)) < 1e-8
    
    frequencies = np.minimum(np.array([frequency*(x+1) for x in range(len(factor))]), sample_rate//2)
    amplitudes = np.array([amplitude*x for x in factor])
    
    fundamental = get_sine_wave(frequencies[0], duration, sample_rate, amplitudes[0])
    for i in range(1, len(factor)):
        overtone = get_sine_wave(frequencies[i], duration, sample_rate, amplitudes[i])
        fundamental += overtone
    return fundamental

def get_adsr_weights(frequency, duration, length, decay, sustain_level, sample_rate=44100):
    '''
    ADSR(attack, decay, sustain, and release) envelop generator with exponential
    weights applied.
    Parameters
    ----------
    frequency : float
        Frequency in hertz.
    duration : float
        Time in seconds.
    length : list
        List of fractions that indicates length of each stage in ADSR.
    decay : list
        List of float for decay factor to be used in each stage for exponential
        weights. 
    sustain_level : float
        Amplitude of `S` stage as a fraction of max amplitude.
    sample_rate : int, optional
        Wav file sample rate. The default is 44100.
    Returns
    -------
    weights : ndarray
    '''
    assert abs(sum(length)-1) < 1e-8
    assert len(length) ==len(decay) == 4
    
    intervals = int(duration*frequency)
    len_A = np.maximum(int(intervals*length[0]),1)
    len_D = np.maximum(int(intervals*length[1]),1)
    len_S = np.maximum(int(intervals*length[2]),1)
    len_R = np.maximum(int(intervals*length[3]),1)
    
    decay_A = decay[0]
    decay_D = decay[1]
    decay_S = decay[2]
    decay_R = decay[3]
    
    A = 1/np.array([(1-decay_A)**n for n in range(len_A)])
    A = A/np.nanmax(A)
    D = np.array([(1-decay_D)**n for n in range(len_D)])
    D = D*(1-sustain_level)+sustain_level
    S = np.array([(1-decay_S)**n for n in range(len_S)])
    S = S*sustain_level
    R = np.array([(1-decay_R)**n for n in range(len_R)])
    R = R*S[-1]
    
    weights = np.concatenate((A,D,S,R))
    smoothing = np.array([0.1*(1-0.1)**n for n in range(5)])
    smoothing = smoothing/np.nansum(smoothing)
    weights = np.convolve(weights, smoothing, mode='same')
    
    weights = np.repeat(weights, int(sample_rate*duration/intervals))
    tail = int(sample_rate*duration-weights.shape[0])
    if tail > 0:
        weights = np.concatenate((weights, weights[-1]-weights[-1]/tail*np.arange(tail)))
    return weights

def apply_pedal(note_values, bar_value):
    '''
    Press and hold the sustain pedal throughout the bar.
    Parameters
    ----------
    note_values : list
        List of note duration.
    bar_value : float
        Duration of a measure in seconds.
    Returns
    -------
    new_values : list
        List of note duration with sustain.
    '''
    assert sum(note_values) % bar_value == 0
    new_values = []
    start = 0
    while True:
        cum_value = np.cumsum(np.array(note_values[start:]))
        end = np.where(cum_value == bar_value)[0][0]
        if end == 0:
            new_values += [note_values[start]]
        else:
            this_bar = np.array(note_values[start:start+end+1])
            new_values += [bar_value-np.sum(this_bar[:i]) for i in range(len(this_bar))]
        start += end+1
        if start == len(note_values):
            break
    return new_values

def get_song_data(music_notes, note_values, bar_value, factor, length,
                  decay, sustain_level, sample_rate=44100, amplitude=4096):
    '''
    Generate song from notes. 
    Parameters
    ----------
    music_notes : list
        List of note names. 
    note_values : list
        List of note duration.
    bar_value: float
        Duration of a bar. 
    factor : list
        Factor to be used to generate overtones.
    length : list
        Stage length to be used to calculate ADSR weights.
    decay : list
        Stage decay to be used to calculate ADSR weights.
    sustain_level : float
        Amplitude of `S` stage as a fraction of max amplitude.
    sample_rate : int, optional
        Wav file sample rate. The default is 44100.
    amplitude : int, optional
        Peak Amplitude. The default is 4096.
    Returns
    -------
    song : ndarray
    '''
    note_freqs = get_piano_notes()
    frequencies = [note_freqs[note] for note in music_notes]
    new_values = apply_pedal(note_values, bar_value)
    duration = int(sum(note_values)*sample_rate)
    end_idx = np.cumsum(np.array(note_values)*sample_rate).astype(int)
    start_idx = np.concatenate(([0], end_idx[:-1]))
    end_idx = np.array([start_idx[i]+new_values[i]*sample_rate for i in range(len(new_values))]).astype(int)
    
    song = np.zeros((duration,))
    for i in range(len(music_notes)):
        this_note = apply_overtones(frequencies[i], new_values[i], factor)
        weights = get_adsr_weights(frequencies[i], new_values[i], length, 
                                   decay, sustain_level)
        song[start_idx[i]:end_idx[i]] += this_note*weights

    song = song*(amplitude/np.max(song))
    return song



## HECHO POR MARIO RICO ##
import numpy as np
from scipy.io import wavfile
import utils
import random


def get_piano_notes():   
    # White keys are in Uppercase and black keys (sharps) are in lowercase
    octave = ['C', 'D', 'E', 'F', 'G', 'A', 'B'] 
    base_freq = 440 #Frequency of Note A4
    keys = np.array([x+str(y) for y in range(0,9) for x in octave])
    
    # Trim to standard 88 keys
    start = np.where(keys == 'A1')[0][0]
    end = np.where(keys == 'G4')[0][0]
    keys = keys[start:end+1]
    
    note_freqs = dict(zip(keys, [2**((n+1-49)/12)*base_freq for n in range(len(keys))]))
    note_freqs[''] = 0.0 # stop
    return note_freqs
    
def get_sine_wave(frequency, duration, sample_rate=44100, amplitude=4096):
    t = np.linspace(0, duration, int(sample_rate*duration)) # Time axis
    wave = amplitude*np.sin(2*np.pi*frequency*t)
    return wave

def get_adsr_weights(frequency, duration, length, decay, sustain_level, sample_rate=44100):

    assert abs(sum(length)-1) < 1e-8
    assert len(length) ==len(decay) == 4
    
    intervals = int(duration*frequency)
    len_A = np.maximum(int(intervals*length[0]),1)
    len_D = np.maximum(int(intervals*length[1]),1)
    len_S = np.maximum(int(intervals*length[2]),1)
    len_R = np.maximum(int(intervals*length[3]),1)
    
    decay_A = decay[0]
    decay_D = decay[1]
    decay_S = decay[2]
    decay_R = decay[3]
    
    A = 1/np.array([(1-decay_A)**n for n in range(len_A)])
    A = A/np.nanmax(A)
    D = np.array([(1-decay_D)**n for n in range(len_D)])
    D = D*(1-sustain_level)+sustain_level
    S = np.array([(1-decay_S)**n for n in range(len_S)])
    S = S*sustain_level
    R = np.array([(1-decay_R)**n for n in range(len_R)])
    R = R*S[-1]
    
    weights = np.concatenate((A,D,S,R))
    smoothing = np.array([0.1*(1-0.1)**n for n in range(5)])
    smoothing = smoothing/np.nansum(smoothing)
    weights = np.convolve(weights, smoothing, mode='same')
    
    weights = np.repeat(weights, int(sample_rate*duration/intervals))
    tail = int(sample_rate*duration-weights.shape[0])
    if tail > 0:
        weights = np.concatenate((weights, weights[-1]-weights[-1]/tail*np.arange(tail)))
    return weights

def apply_overtones(frequency, duration, factor, sample_rate=44100, amplitude=4096):

    assert abs(1-sum(factor)) < 1e-8
    
    frequencies = np.minimum(np.array([frequency*(x+1) for x in range(len(factor))]), sample_rate//2)
    amplitudes = np.array([amplitude*x for x in factor])
    
    fundamental = get_sine_wave(frequencies[0], duration, sample_rate, amplitudes[0])
    for i in range(1, len(factor)):
        overtone = get_sine_wave(frequencies[i], duration, sample_rate, amplitudes[i])
        fundamental += overtone
    return fundamental

# Construct harmonic series
# note = apply_overtones(frequency, duration=2.5, factor=factor)
def get_adsr_weights(frequency, duration, length, decay, sustain_level, sample_rate=44100):

    assert abs(sum(length)-1) < 1e-8
    assert len(length) ==len(decay) == 4
    
    intervals = int(duration*frequency)
    len_A = np.maximum(int(intervals*length[0]),1)
    len_D = np.maximum(int(intervals*length[1]),1)
    len_S = np.maximum(int(intervals*length[2]),1)
    len_R = np.maximum(int(intervals*length[3]),1)
    
    decay_A = decay[0]
    decay_D = decay[1]
    decay_S = decay[2]
    decay_R = decay[3]
    
    A = 1/np.array([(1-decay_A)**n for n in range(len_A)])
    A = A/np.nanmax(A)
    D = np.array([(1-decay_D)**n for n in range(len_D)])
    D = D*(1-sustain_level)+sustain_level
    S = np.array([(1-decay_S)**n for n in range(len_S)])
    S = S*sustain_level
    R = np.array([(1-decay_R)**n for n in range(len_R)])
    R = R*S[-1]
    
    weights = np.concatenate((A,D,S,R))
    smoothing = np.array([0.1*(1-0.1)**n for n in range(5)])
    smoothing = smoothing/np.nansum(smoothing)
    weights = np.convolve(weights, smoothing, mode='same')
    
    weights = np.repeat(weights, int(sample_rate*duration/intervals))
    tail = int(sample_rate*duration-weights.shape[0])
    if tail > 0:
        weights = np.concatenate((weights, weights[-1]-weights[-1]/tail*np.arange(tail)))
    return weights

def apply_pedal(note_values, bar_value):
    # Check that we have whole number of bars
    assert sum(note_values) % bar_value == 0
    new_values = []
    start = 0
    while True:
        # Count total duration from end of last bar
        cum_value = np.cumsum(np.array(note_values[start:]))
        # Find end of this bar
        end = np.where(cum_value == bar_value)[0][0]
        if end == 0: # If the note takes up the whole bar
            new_values += [note_values[start]]
        else:
            this_bar = np.array(note_values[start:start+end+1])
            # New value of note is the remainder of bar = (total duration of bar) - (cumulative duration thus far)
            new_values += [bar_value-np.sum(this_bar[:i]) for i in range(len(this_bar))]
        start += end+1
        if start == len(note_values):
            break
    return new_values

def get_song_data(music_notes, note_values, bar_value, factor, length,
                  decay, sustain_level, sample_rate=44100, amplitude=4096):
    # Get note frequencies
    note_freqs = get_piano_notes()
    frequencies = [note_freqs[note] for note in music_notes]
    # Get new note durations with sustain applied
    new_values = apply_pedal(note_values, bar_value)
    # End of each note without sustain
    end_idx = np.cumsum(np.array(note_values)*sample_rate).astype(int)
    # Start of each note
    start_idx = np.concatenate(([0], end_idx[:-1]))
    # End of note with sustain
    end_idx = np.array([start_idx[i]+new_values[i]*sample_rate for i in range(len(new_values))]).astype(int)
    
    # Total duration of the piece
    duration = int(sum(note_values)*sample_rate)    
    song = np.zeros((duration,))
    for i in range(len(music_notes)):
        # Fundamental + overtones
        this_note = apply_overtones(frequencies[i], new_values[i], factor)
        # ADSR model
        weights = get_adsr_weights(frequencies[i], new_values[i], length, 
                                   decay, sustain_level)
        song[start_idx[i]:end_idx[i]] += this_note*weights

    song = song*(amplitude/np.max(song))
    return song

def create_matrix(columns=['A','B','C','D','E','F'], rows=['1','2','3','4']):
    matriz = []
    for tone_letter in columns:
        for number in rows: 
           matriz.append(tone_letter+number)
    
    return np.matrix(matriz).reshape(6,4).transpose()

def generate_random_numbers():
    random_numbers = []
    for i in range(6):
        random_numbers.append(random.random())
    random_numbers = np.array(random_numbers)
    random_numbers = random_numbers/np.sum(random_numbers)
    return random_numbers
emotions = "surprise,anger,fear,joy,sadness,neutral".split(',')

def get_notes(emotions): #emotions is a vector of probabilities and we have to get the notes, if the probability is less than 0.125 it will not be evaluated
    notes = create_matrix()
    notes_to_play = []
    for i in range(0,len(emotions)):
        p = emotions[i]
        if p > 0.125 and p < 1:
            notes_to_play.append(notes[round(p*4)-1,i])
        else:
            notes_to_play.append([])
    return notes_to_play

#order = SURPRISE, ANGER, FEAR, JOY, SADNESS, NEUTRALITY
def generate_sound(notes):
    factor = [0.68, 0.26, 0.03, 0.  , 0.03]
    length = [0.25, 0.25, 0.25, 0.25]
    decay  = [0,0,0,0]
    sustain_level = 1
    global_duration = [2]

    sound = 0
    for i in notes:
        if i != []:
            sound = sound + utils.get_song_data([i], global_duration, 2,
                                 factor, length, decay, sustain_level) 
    wavfile.write('PRUEBAS2.wav', 44100, data.astype(np.int16))
     