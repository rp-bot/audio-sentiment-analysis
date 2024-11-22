import os
RAVDESS_MUSIC_DIR = "P:\\ML_datasets\\RAVDESS_MUSIC"
RAVDESS_SPEECH_DIR = "P:\\ML_datasets\\RAVDESS_SPEECH"
EMOTIFY_DIR = "P:\\ML_datasets\\EMOTIFY"
CURRENT_DIR = "C:\\Users\\RP Bot\\Documents\\GitHub\\audio-semantic-analysis"

RAVDESS_MUSIC_PROCESSED = "P:\\ML_datasets\\RAVDESS_MUSIC\\processed"
RAVDESS_SPEECH_PROCESSED = "P:\\ML_datasets\\PROCESSED_DATA\\RAVDESS_SPEECH"


modality_map = {
    '01': 'full-AV',
    '02': 'video-only',
    '03': 'audio-only'
}

vocal_channel_map = {
    '01': 'speech',
    '02': 'song'
}

emotion_map = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}


emotion_to_int = {
    'neutral': 1,
    'calm': 2,
    'happy': 3,
    'sad': 4,
    'angry': 5,
    'fearful': 6,
    'disgust': 7,
    'surprised': 8
}

int_to_emotion = {
    1: 'neutral',
    2: 'calm',
    3: 'happy',
    4: 'sad',
    5: 'angry',
    6: 'fearful',
    7: 'disgust',
    8: 'surprised'
}


intensity_map = {
    '01': 'normal',
    '02': 'strong'
}

statement_map = {
    '01': 'Kids are talking by the door',
    '02': 'Dogs are sitting by the door'
}

repetition_map = {
    '01': '1st repetition',
    '02': '2nd repetition'
}
