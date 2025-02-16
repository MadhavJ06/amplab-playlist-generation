import os
import json
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# File paths
ESSENTIA_ANALYSIS_PATH = '/mnt/f/SMC/AMPLAB/assignment_01/analysis_results/analysis_results.json'

def load_analysis_data():
    """Load analysis data and extract embeddings"""
    df = pd.read_json(ESSENTIA_ANALYSIS_PATH)
    
    # Extract embeddings into separate DataFrames
    effnet_embeddings = np.array([emb for emb in df['embeddings'].apply(lambda x: x['discogs_effnet'])])
    musicnn_embeddings = np.array([emb for emb in df['embeddings'].apply(lambda x: x['msd_musicnn'])])
    
    return df, effnet_embeddings, musicnn_embeddings

def get_similar_tracks(query_idx, embeddings, n_similar=10):
    """Find n most similar tracks using cosine similarity"""
    query_embedding = embeddings[query_idx].reshape(1, -1)
    similarities = cosine_similarity(query_embedding, embeddings)
    
    # Get indices of most similar tracks (excluding the query track)
    similar_indices = similarities[0].argsort()[::-1][1:n_similar+1]
    similarity_scores = similarities[0][similar_indices]
    
    return similar_indices, similarity_scores

# Load data
audio_analysis, effnet_emb, musicnn_emb = load_analysis_data()

# Title
st.write("# Music Similarity Demo")
st.write("Compare similar tracks using different embedding spaces")

st.write("""
## ℹ️ About
This demo compares music similarity using two different embedding spaces:
- **Effnet-Discogs**: Embeddings from a model trained on Discogs genre tags
- **MSD-MusicNN**: Embeddings from a model trained on Million Song Dataset tags

Similarity is computed using cosine similarity between embeddings.
Higher similarity scores (closer to 1.0) indicate more similar tracks.
""")

# Track selection
st.write("## 🎵 Select Query Track")
track_options = [f"{i}: {os.path.basename(path)}" for i, path in enumerate(audio_analysis['path'])]
selected_track = st.selectbox('Choose a track:', track_options)
query_idx = int(selected_track.split(':')[0])

# Display currently selected track
st.write("### 📌 Query Track selected:")
query_path = str(audio_analysis.loc[query_idx, 'path'])
st.audio(query_path, format="audio/mp3", start_time=0)

if st.button("Find Similar Tracks"):
    st.write("## 🎵 Similar Tracks")
    
    # Create two columns for side-by-side display
    col1, col2 = st.columns(2)
    
    # Get similar tracks for both embeddings
    effnet_indices, effnet_scores = get_similar_tracks(query_idx, effnet_emb)
    musicnn_indices, musicnn_scores = get_similar_tracks(query_idx, musicnn_emb)
    
    # Display Effnet results in left column
    with col1:
        st.write("### 💿 Effnet-Discogs")
        for idx, score in zip(effnet_indices, effnet_scores):
            track_path = str(audio_analysis.loc[idx, 'path'])
            st.write(f"**Similarity: {score:.3f}**")
            st.write(f"*{os.path.basename(track_path)}*")
            st.audio(track_path, format="audio/mp3", start_time=0)
    
    # Display MusicNN results in right column
    with col2:
        st.write("### 🎼 MSD-MusicNN")
        for idx, score in zip(musicnn_indices, musicnn_scores):
            track_path = str(audio_analysis.loc[idx, 'path'])
            st.write(f"**Similarity: {score:.3f}**")
            st.write(f"*{os.path.basename(track_path)}*")
            st.audio(track_path, format="audio/mp3", start_time=0)
