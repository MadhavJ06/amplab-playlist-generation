import os
import json
import numpy as np
import essentia
import essentia.standard as es
from pathlib import Path
from tqdm import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Define paths
DATASET_PATH = "/mnt/f/SMC/AMPLAB/assignment_01/MusAV/audio_chunks"
OUTPUT_PATH = "/mnt/f/SMC/AMPLAB/assignment_01/analysis_results"
ESSENTIA_MODELS = Path("/mnt/f/SMC/AMPLAB/assignment_01/models")

class AudioAnalyzer:

    def __init__(self):
        """Initialize all algorithms and models once"""
        # Verify models directory exists
        if not ESSENTIA_MODELS.exists():
            raise RuntimeError(f"Models directory not found at {ESSENTIA_MODELS}")
            
        # Initialize feature extraction algorithms
        self.rhythm_extractor = es.RhythmExtractor2013()
        self.key_extractor = es.KeyExtractor()
        self.loudness = es.LoudnessEBUR128()
        self.danceability = es.Danceability()
        self.mono_mixer = es.MonoMixer()
        self.resampler = es.Resample(outputSampleRate=16000)
        
        # Initialize TensorFlow models
        self.discogs_effnet = es.TensorflowPredictEffnetDiscogs(
            graphFilename=str(ESSENTIA_MODELS / "discogs-effnet-bs64-1.pb"),
            output="PartitionedCall:1"
        )
        self.musicnn = es.TensorflowPredictMusiCNN(
            graphFilename=str(ESSENTIA_MODELS / "msd-musicnn-1.pb"),
            output="model/dense/BiasAdd"
        )
        self.music_style = es.TensorflowPredict2D(
            graphFilename=str(ESSENTIA_MODELS / "genre_discogs400-discogs-effnet-1.pb"),
            input="serving_default_model_Placeholder",
            output="PartitionedCall:0"  
        )
        self.voice_inst_classifier = es.TensorflowPredict2D(
            graphFilename=str(ESSENTIA_MODELS / "voice_instrumental-discogs-effnet-1.pb"),
            output="model/Softmax"  
        )
        self.emotion_classifier = es.TensorflowPredict2D(
            graphFilename=str(ESSENTIA_MODELS / "emomusic-msd-musicnn-2.pb"),
            output="model/Identity"
        )

    def process_audio(self, audio_path):
        """Process a single audio file and compute all features"""
        try:
            # Load audio
            audio_loader = es.AudioLoader(filename=audio_path)
            audio_stereo, sample_rate, num_channels, bit_depth, _, _ = audio_loader()
            
            # Create mono and resampled versions
            audio_mono = self.mono_mixer(audio_stereo, num_channels)
            audio_16k = self.resampler(audio_mono)
            
            # Compute embeddings
            discogs_embedding = self.discogs_effnet(audio_16k)
            musicnn_embedding = self.musicnn(audio_16k)
            
            # Average embeddings across time
            discogs_mean = np.mean(discogs_embedding.squeeze(), axis=0)
            musicnn_mean = np.mean(musicnn_embedding.squeeze(), axis=0)
            
            # Compute features
            features = {
                'filename': os.path.basename(audio_path),
                'path': str(audio_path),
                'embeddings': {
                    'discogs_effnet': discogs_mean.tolist(),
                    'msd_musicnn': musicnn_mean.tolist()
                }
            }
            
            # 1. Tempo analysis
            bpm, beats, beats_confidence, _, _ = self.rhythm_extractor(audio_mono)
            features['tempo'] = float(bpm)
            
            # 2. Key analysis with three profiles
            key_computed = {}
            for profile in ['temperley', 'krumhansl', 'edma']:
                try:
                    key_extractor = es.KeyExtractor(profileType=profile)
                    key_results = key_extractor(audio_mono)
                    key, scale, strength = key_results[0:3]
                    key_computed[profile] = {
                        'key': key,
                        'scale': scale,
                        'strength': float(strength)
                    }
                except Exception as e:
                    print(f"Error processing key extraction for profile {profile}: {str(e)}")
            features['key_analysis'] = key_computed
            
            # 3. Compute loudness
            _, _, integrated_loudness, _, = self.loudness(audio_stereo)
            features['loudness_integrated'] = float(integrated_loudness)
            
            # 4. Compute danceability
            dance_value, dfa = self.danceability(audio_mono)
            features['danceability'] = {
                'danceability': float(dance_value),
                'dfa': dfa.tolist()
            }
            
            # 5. Music styles using discogs embedding
            style_predictions = self.music_style(discogs_embedding)
            style_mean = np.mean(style_predictions.squeeze(), axis=0)
            features['music_styles'] = [
                {'style': f'style_{idx}', 'probability': float(style_mean[idx])}
                for idx in range(len(style_mean))
            ]
            
            # 6. Voice/instrumental using discogs embedding
            voice_predictions = self.voice_inst_classifier(discogs_embedding)
            voice_mean = np.mean(voice_predictions.squeeze(), axis=0)
            features['voice_instrumental'] = {
                'instrumental': float(voice_mean[0]),
                'voice': float(voice_mean[1])
            }
            
            # 7. Emotion using musicnn embedding
            emotion_predictions = self.emotion_classifier(musicnn_embedding)
            emotion_mean = np.mean(emotion_predictions.squeeze(), axis=0)
            features['emotion'] = {
                'arousal': float(emotion_mean[0]),
                'valence': float(emotion_mean[1])
            }
            
            return features
            
        except Exception as e:
            print(f"Error processing {audio_path}: {str(e)}")
            return None



def analyze_collection(collection_path, output_dir="analysis_results"):
    """
    Analyze all audio files in a collection
    
    Parameters:
        collection_path (str or Path): Path to music collection
        output_dir (str or Path): Directory to save analysis results
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Get list of all MP3 files
    collection_path = Path(collection_path)
    audio_files = list(collection_path.rglob("*.mp3"))
    
    # Load existing results if any
    results_file = output_dir / "analysis_results.json"
    if results_file.exists():
        with open(results_file, 'r') as f:
            results = json.load(f)
            processed_files = {result['filename'] for result in results}
    else:
        results = []
        processed_files = set()
    
    # Process files
    remaining_files = [f for f in audio_files if f.name not in processed_files]
    print(f"Found {len(remaining_files)} files to process")
    
    # Initialize analyzer once outside the loop
    analyzer = AudioAnalyzer()
    
    for audio_file in tqdm(remaining_files, desc="Analyzing audio files"):
        try:
            # Use process_audio directly
            features = analyzer.process_audio(str(audio_file))
            if features:  
                results.append(features)
            
            # Save results periodically
            if len(results) % 10 == 0:
                with open(results_file, 'w') as f:
                    json.dump(results, f, indent=4)
                    
        except Exception as e:
            print(f"\nError processing {audio_file.name}: {str(e)}")
            continue
    
    # Save final results
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    return len(results), len(remaining_files)


def main():
    
    collection_path = Path(DATASET_PATH)
    output_dir = Path(OUTPUT_PATH)
    
    print(f"Starting analysis of audio collection: {collection_path}")
    
    # Run analysis
    processed, total = analyze_collection(collection_path, output_dir)
    
    # Print summary
    print("\nAnalysis complete!")
    print(f"Successfully processed: {processed} files")
    print(f"Failed: {total - processed} files")
    print(f"Results saved to: {output_dir / 'analysis_results.json'}")


if __name__ == "__main__":
    main()