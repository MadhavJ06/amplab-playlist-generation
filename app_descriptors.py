import os
import json
import random
import streamlit as st
import pandas as pd
import numpy as np

# File paths
m3u_filepaths_file = '/mnt/f/SMC/AMPLAB/assignment_01/playlists/streamlit.m3u8'
ESSENTIA_ANALYSIS_PATH = '/mnt/f/SMC/AMPLAB/assignment_01/analysis_results/analysis_results.json'
GENRE_MODEL_JSON = '/mnt/f/SMC/AMPLAB/assignment_01/models/genre_discogs400-discogs-effnet-1.json'

def load_essentia_analysis():
    df = pd.read_json(ESSENTIA_ANALYSIS_PATH)
    
    # probabilities from music_styles
    style_probabilities = pd.DataFrame([
        {f"style_{i}": style["probability"] 
         for i, style in enumerate(row)} 
        for row in df['music_styles']
    ])

    df = df.drop('music_styles', axis=1)
    df = pd.concat([df, style_probabilities], axis=1)
    
    return df

def load_genre_classes():
    with open(GENRE_MODEL_JSON, 'r') as f:
        data = json.load(f)
    return data["classes"]

# Load analysis data and style classes
audio_analysis = load_essentia_analysis()
genre_classes = load_genre_classes()
genre_options = [f"{i}: {genre}" for i, genre in enumerate(genre_classes)]

# Title and description
st.write('# Audio analysis playlists example')
st.write(f'Using analysis data from `{ESSENTIA_ANALYSIS_PATH}`.')
st.write('Loaded audio analysis for', len(audio_analysis), 'tracks.')

# --- Section: Genre Selection (400 genres) ---
st.write('## 🔍 Select by Genre')
st.write('Audio analysis statistics:')
st.write(audio_analysis.describe())

# user select one or more genres
genre_select = st.multiselect('Select genres:', genre_options)
if genre_select:
    selected_columns = [f"style_{sel.split(':')[0]}" for sel in genre_select]
    st.write('Selected genre activation statistics:')
    st.write(audio_analysis[selected_columns].describe())
    
    genre_select_range = st.slider(
        'Select tracks with selected genres activations within range:',
        value=[0.5, 1.0]
    )


# --- Section: Ranking ---
st.write('## 🔝 Rank')
style_rank = st.multiselect('Rank by genre activations (multiplies activations for selected genres):',
                            genre_options, [])
if style_rank:
    rank_cols = [int(sel.split(":")[0]) for sel in style_rank]

# --- Section: Tempo Filtering ---
st.write('## 🎵 Tempo')
# tempo range based on maximum and minimum values in the dataset
tempo_min = 50.0
tempo_max = 190.0
tempo_range = st.slider('Select tempo range (BPM):',
                         min_value=tempo_min,
                         max_value=tempo_max,
                         value=(tempo_min, tempo_max))

# --- Section: Voice/Instrumental Filter ---
st.write('## 🎤 Voice/Instrumental')
voice_filter = st.radio(
    "Filter by vocals/instrumental:",
    ["All", "Vocal music", "Instrumental music"]
)

# --- Section: Danceability Filter ---
st.write('## 💃 Danceability')
dance_min = 0.0
dance_max = 3.0
dance_range = st.slider(
    'Select danceability range (0-3, higher = more danceable):',
    min_value=dance_min,
    max_value=dance_max,
    value=(dance_min, dance_max)
)

# --- Section: Emotion (Valence/Arousal) Filter ---
st.write('## 😊 Emotion')

# Get actual ranges from data but clip to model's valid range [1, 9]
arousal_min = max(1.0, float(audio_analysis['emotion'].apply(lambda x: x['arousal']).min()))
arousal_max = min(9.0, float(audio_analysis['emotion'].apply(lambda x: x['arousal']).max()))
valence_min = max(1.0, float(audio_analysis['emotion'].apply(lambda x: x['valence']).min()))
valence_max = min(9.0, float(audio_analysis['emotion'].apply(lambda x: x['valence']).max()))

# Arousal range selector
arousal_range = st.slider(
    'Select arousal range (1-9, higher = more energetic):',
    min_value=1.0,
    max_value=9.0,
    value=(arousal_min, arousal_max),
)

# Valence range selector
valence_range = st.slider(
    'Select valence range (1-9, higher = more positive):',
    min_value=1.0,
    max_value=9.0,
    value=(valence_min, valence_max),
)

# --- Section: Key/Scale Filter ---
st.write('## 🎼 Key/Scale')

# Available keys and scales
keys = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
scales = ['major', 'minor']

# Create selectors for key and scale
selected_key = st.selectbox('Select key:', ['Any'] + keys)
selected_scale = st.selectbox('Select scale:', ['Any'] + scales)

# Fixed strength threshold
KEY_STRENGTH_THRESHOLD = 0.7

# --- Section: Post-process ---
st.write('## 🔀 Post-process')
max_tracks = st.number_input('Maximum number of tracks (0 for all):', value=0)
shuffle = st.checkbox('Random shuffle')


# --- Run the processing ---
if st.button("RUN"):
    st.write('## 🔊 Results')
    
    mp3s = list(audio_analysis.index)
    
    # --- Filter based on genre activations ---
    if genre_select:
        # Convert selections to style column names
        sel_cols = [f"style_{sel.split(':')[0]}" for sel in genre_select]
        
        # Filter based on selected genres
        result = audio_analysis[sel_cols].copy()
        for col in sel_cols:
            result = result[result[col] >= genre_select_range[0]]
        
        st.write('Tracks after filtering by selected genre activations:')
        st.write(result)
        mp3s = list(result.index)
    
    
    # --- Ranking if requested ---
    if style_rank:
        rank_cols = [f"style_{sel.split(':')[0]}" for sel in style_rank]
        
        audio_analysis_query = audio_analysis[rank_cols].copy()
        audio_analysis_query['RANK'] = audio_analysis_query[rank_cols[0]]
        for col in rank_cols[1:]:
            audio_analysis_query['RANK'] *= audio_analysis_query[col]

        ranked = audio_analysis_query.sort_values(['RANK'], ascending=False)
        ranked = ranked[['RANK'] + rank_cols]
        mp3s = list(ranked.index)
        st.write('Applied ranking by genre activations:')
        st.write(ranked)

    # --- Filter based on tempo ---
    audio_analysis_query = audio_analysis.loc[mp3s]
    tempo_filtered = audio_analysis_query.loc[
        (audio_analysis_query['tempo'] >= tempo_range[0]) & 
        (audio_analysis_query['tempo'] <= tempo_range[1])
    ]
    st.write(f'Tracks after filtering by tempo (between {tempo_range[0]} and {tempo_range[1]} BPM):')
    st.write(tempo_filtered[['tempo']])
    mp3s = list(tempo_filtered.index)

    # --- Filter based on voice/instrumental ---
    if voice_filter != "All":
        audio_analysis_query = audio_analysis.loc[mp3s]
        if voice_filter == "Vocal music":
            voice_filtered = audio_analysis_query[
                audio_analysis_query['voice_instrumental'].apply(
                    lambda x: x['voice'] > x['instrumental'] and x['instrumental'] < 0.3
                )
            ]
        else:  # Instrumental music
            voice_filtered = audio_analysis_query[
                audio_analysis_query['voice_instrumental'].apply(
                    lambda x: x['instrumental'] > x['voice'] and x['voice'] < 0.3
                )
            ]
        st.write(f'Tracks after filtering by {voice_filter.lower()}:')
        st.write(voice_filtered[['voice_instrumental']])
        mp3s = list(voice_filtered.index)

    # --- Filter based on danceability ---
    audio_analysis_query = audio_analysis.loc[mp3s]
    dance_filtered = audio_analysis_query[
        audio_analysis_query['danceability'].apply(
            lambda x: dance_range[0] <= x['danceability'] <= dance_range[1]
        )
    ]
    st.write(f'Tracks after filtering by danceability (between {dance_range[0]:.2f} and {dance_range[1]:.2f}):')
    st.write(dance_filtered['danceability'])
    mp3s = list(dance_filtered.index)

    # --- Filter based on emotion (arousal/valence) ---
    audio_analysis_query = audio_analysis.loc[mp3s]
    emotion_filtered = audio_analysis_query[
        audio_analysis_query['emotion'].apply(
            lambda x: (arousal_range[0] <= x['arousal'] <= arousal_range[1]) and 
                     (valence_range[0] <= x['valence'] <= valence_range[1])
        )
    ]
    st.write(f'Tracks after filtering by emotion (arousal: {arousal_range[0]:.2f}-{arousal_range[1]:.2f}, ' +
             f'valence: {valence_range[0]:.2f}-{valence_range[1]:.2f}):')
    st.write(emotion_filtered['emotion'])
    mp3s = list(emotion_filtered.index)

    # --- Filter based on key/scale ---
    if selected_key != 'Any' or selected_scale != 'Any':
        audio_analysis_query = audio_analysis.loc[mp3s]
        key_filtered = audio_analysis_query[
            audio_analysis_query['key_analysis'].apply(
                lambda x: (selected_key == 'Any' or x['edma']['key'] == selected_key) and
                         (selected_scale == 'Any' or x['edma']['scale'] == selected_scale) and
                         (x['edma']['strength'] >= KEY_STRENGTH_THRESHOLD)
            )
        ]
        st.write(f'Tracks after filtering by key/scale:')
        if len(key_filtered) > 0:
            key_info = key_filtered['key_analysis'].apply(
                lambda x: f"{x['edma']['key']} {x['edma']['scale']} (strength: {x['edma']['strength']:.2f})"
            )

            st.write(pd.DataFrame({'Key/Scale': key_info}))
        mp3s = list(key_filtered.index)

    # --- Limit and Shuffle ---
    if max_tracks:
        mp3s = mp3s[:max_tracks]
        st.write('Using top', len(mp3s), 'tracks from the results.')
    if shuffle:
        random.shuffle(mp3s)
        st.write('Applied random shuffle.')
    
    # --- Save the playlist (M3U8 file) ---
    with open(m3u_filepaths_file, 'w') as f:
        mp3_paths = [str(audio_analysis.loc[idx, 'path']) for idx in mp3s]
        mp3_paths = [os.path.join('..', path) for path in mp3_paths]
        f.write('\n'.join(mp3_paths))
        st.write(f'Stored M3U playlist (local filepaths) to `{m3u_filepaths_file}`.')

    # --- Audio Previews ---
    st.write('Audio previews for the first 10 results:')
    for idx in mp3s[:10]:
        audio_path = str(audio_analysis.loc[idx, 'path'])
        st.audio(audio_path, format="audio/mp3", start_time=0)