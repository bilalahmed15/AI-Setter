from flask import Flask, render_template, jsonify, request
import threading
import queue
import os
import whisper
import sounddevice as sd
import soundfile as sf
import numpy as np
import wave
from datetime import datetime
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader
from openai import OpenAI
import librosa
from scipy.io import wavfile
from sklearn.preprocessing import StandardScaler
import io
from scipy.signal import find_peaks
from scipy import stats

# Initialize Flask app
app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Audio parameters
RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 1024
SILENCE_THRESHOLD = 0.01
MIN_SENTENCE_DURATION = 0.5  # Minimum duration in seconds
SILENCE_DURATION = 0.8  # Wait for 0.8 seconds of silence to consider end of sentence
FRAMES_PER_SILENCE = int(SILENCE_DURATION * RATE / CHUNK_SIZE)  # Calculate frames needed for silence

# Global variables
audio_queue = queue.Queue()
transcription_queue = queue.Queue()
is_listening = False
training_audio_data = []
is_training = False
TRAINING_DURATION = 10  # seconds for voice training

# Initialize the OpenAI client
client = OpenAI(api_key=' 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')  # Replace with your API key

# Global variables for document storage
document_context = []

def add_to_document_context(text, filename):
    """Add document content to the context with source information"""
    document_context.append({
        'content': text,
        'source': filename,
        'timestamp': datetime.now().isoformat()
    })

class SpeakerDiarizer:
    def __init__(self):
        self.voice_profiles = {}
        self.similarity_threshold = 0.80  # Very high base threshold
        self.first_speaker_threshold = 0.80  # Lower threshold for first speaker
        self.voice_history = []
        self.history_size = 15
        self.first_speaker_samples = []  # Store samples from first speaker
        self.min_samples_for_profile = 3
        self.training_samples = []  # Store all training samples
        self.speaker_profile = None  # Main speaker profile

    def extract_voice_features(self, audio_data, sample_rate):
        try:
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            # Enhanced pitch detection
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio_data.astype(float),
                fmin=50,
                fmax=400,
                sr=sample_rate,
                frame_length=1024,
                win_length=512,
                n_thresholds=500
            )
            
            f0_voiced = f0[voiced_flag]
            if len(f0_voiced) == 0:
                return None
                
            # Extract all voice features
            voice_profile = {
                'pitch': self._analyze_detailed_pitch(f0_voiced),
                'mfcc': self._extract_enhanced_mfcc(audio_data, sample_rate),
                'voice_quality': self._analyze_detailed_voice_quality(audio_data, sample_rate),
                'speaking_style': self._analyze_speaking_style(audio_data, sample_rate)
            }
            
            # Verify all features are present
            if any(v is None for v in voice_profile.values()):
                print("Warning: Some features could not be extracted")
                return None
                
            return voice_profile
            
        except Exception as e:
            print(f"Feature extraction error: {e}")
            return None

    def _extract_enhanced_mfcc(self, audio_data, sample_rate):
        try:
            # Extract MFCCs with fixed length
            mfcc = librosa.feature.mfcc(
                y=audio_data.astype(float),
                sr=sample_rate,
                n_mfcc=20,
                n_fft=2048,
                hop_length=512
            )
            
            # Normalize length to fixed size
            target_length = 100
            if mfcc.shape[1] > target_length:
                start = (mfcc.shape[1] - target_length) // 2
                mfcc = mfcc[:, start:start + target_length]
            else:
                pad_width = ((0, 0), (0, target_length - mfcc.shape[1]))
                mfcc = np.pad(mfcc, pad_width, mode='constant')
            
            return mfcc
            
        except Exception as e:
            print(f"MFCC extraction error: {e}")
            return None

    def _analyze_detailed_pitch(self, f0_voiced):
        try:
            if len(f0_voiced) == 0:
                return None
                
            # Convert to numpy array if not already
            f0_voiced = np.array(f0_voiced)
            
            # Calculate mode more safely
            hist, bin_edges = np.histogram(f0_voiced, bins=50)
            mode_idx = np.argmax(hist)
            mode_value = (bin_edges[mode_idx] + bin_edges[mode_idx + 1]) / 2
            
            return {
                'mean': float(np.mean(f0_voiced)),
                'median': float(np.median(f0_voiced)),
                'std': float(np.std(f0_voiced)),
                'q25': float(np.percentile(f0_voiced, 25)),
                'q75': float(np.percentile(f0_voiced, 75)),
                'mode': float(mode_value),
                'range': float(np.ptp(f0_voiced))
            }
        except Exception as e:
            print(f"Pitch analysis error: {e}")
            return None

    def _analyze_detailed_voice_quality(self, audio_data, sample_rate):
        try:
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sample_rate)[0]
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(audio_data)[0]
            
            return {
                'spectral_centroid': float(np.mean(spectral_centroids)),
                'spectral_rolloff': float(np.mean(spectral_rolloff)),
                'zero_crossing_rate': float(np.mean(zcr)),
                'zcr_std': float(np.std(zcr))
            }
        except Exception as e:
            print(f"Voice quality analysis error: {e}")
            return None

    def _analyze_speaking_style(self, audio_data, sample_rate):
        try:
            # Energy contour
            rms = librosa.feature.rms(y=audio_data)[0]
            
            # Onset strength
            onset_env = librosa.onset.onset_strength(y=audio_data, sr=sample_rate)
            
            return {
                'energy_mean': float(np.mean(rms)),
                'energy_std': float(np.std(rms)),
                'onset_strength': float(np.mean(onset_env))
            }
        except Exception as e:
            print(f"Speaking style analysis error: {e}")
            return None

    def detect_speaker(self, audio_data, sample_rate):
        """Compare audio with trained speaker profile"""
        try:
            if not self.speaker_profile:
                return "Client"  # Default to Client if no profile exists
            
            current_profile = self.extract_voice_features(audio_data, sample_rate)
            if not current_profile:
                return "Client"
            
            # Calculate comprehensive similarity score
            similarity_score = self._calculate_similarity(current_profile, self.speaker_profile)
            
            print(f"Voice similarity score: {similarity_score}")  # Debug print
            
            # Return "Me" for trained voice, "Client" for others
            if similarity_score > 0.75:
                return "Me"
            else:
                return "Client"
            
        except Exception as e:
            print(f"Error in speaker detection: {e}")
            return "Client"  # Default to Client on error

    def _initialize_first_speaker(self, profile):
        self.voice_profiles[1] = {
            'profiles': [profile],
            'confidence': 1.0,
            'total_samples': 1
        }
        self.first_speaker_samples = [profile]
        print("Initialized first speaker")
        return 1

    def _compare_with_first_speaker(self, profile):
        try:
            if not self.first_speaker_samples:
                return 0
            
            # Compare with all stored samples of first speaker
            scores = []
            for ref_profile in self.first_speaker_samples[-5:]:  # Use last 5 samples
                score = self._calculate_similarity(profile, [ref_profile])
                scores.append(score)
            
            # Return weighted average, giving more weight to recent samples
            weights = np.linspace(0.5, 1.0, len(scores))
            return np.average(scores, weights=weights)
            
        except Exception as e:
            print(f"First speaker comparison error: {e}")
            return 0

    def _update_first_speaker_profile(self, profile):
        self.first_speaker_samples.append(profile)
        if len(self.first_speaker_samples) > 10:  # Keep last 10 samples
            self.first_speaker_samples.pop(0)
        
        self.voice_profiles[1]['profiles'].append(profile)
        self.voice_profiles[1]['total_samples'] += 1
        print("Updated first speaker profile")

    def _get_similarity_scores(self, profile):
        scores = {}
        for speaker_id, speaker_data in self.voice_profiles.items():
            if speaker_id != 1:  # Skip first speaker as it's handled separately
                score = self._calculate_similarity(profile, speaker_data['profiles'][-5:])
                scores[speaker_id] = score
        return scores

    def _find_best_match(self, scores):
        if not scores:
            return None, 0
        best_match = max(scores.items(), key=lambda x: x[1])
        return best_match[0], best_match[1]

    def _create_new_speaker(self, profile):
        new_id = max(self.voice_profiles.keys()) + 1
        self.voice_profiles[new_id] = {
            'profiles': [profile],
            'confidence': 1.0,
            'total_samples': 1
        }
        print(f"Created new speaker {new_id}")
        return new_id

    def _update_speaker_profile(self, speaker_id, profile):
        speaker_data = self.voice_profiles[speaker_id]
        speaker_data['profiles'].append(profile)
        speaker_data['total_samples'] += 1
        speaker_data['last_active'] = 0
        
        # Update continuity
        self.speaker_continuity[speaker_id] = self.speaker_continuity.get(speaker_id, 0) + 1
        for other_id in self.speaker_continuity:
            if other_id != speaker_id:
                self.speaker_continuity[other_id] = 0

        # Update recent speakers
        if speaker_id in self.recent_speakers:
            self.recent_speakers.remove(speaker_id)
        self.recent_speakers.append(speaker_id)
        if len(self.recent_speakers) > self.history_size:
            self.recent_speakers.pop(0)

        self.last_speaker = speaker_id
        self.speaker_gaps[speaker_id] = 0

    def _calculate_similarity(self, profile1, profile2):
        """Calculate comprehensive similarity between two voice profiles"""
        try:
            # MFCC similarity (40%)
            mfcc_similarity = np.corrcoef(
                profile1['mfcc'].reshape(-1),
                profile2['mfcc'].reshape(-1)
            )[0, 1]
            
            # Pitch similarity (30%)
            pitch_features = ['mean', 'median', 'std']
            pitch_similarity = np.mean([
                1 - abs(profile1['pitch'][f] - profile2['pitch'][f]) / 
                max(profile1['pitch'][f], profile2['pitch'][f])
                for f in pitch_features
            ])
            
            # Voice quality similarity (20%)
            vq_similarity = np.mean([
                1 - abs(profile1['voice_quality'][f] - profile2['voice_quality'][f]) / 
                max(profile1['voice_quality'][f], profile2['voice_quality'][f])
                for f in profile1['voice_quality'].keys()
            ])
            
            # Speaking style similarity (10%)
            style_similarity = np.mean([
                1 - abs(profile1['speaking_style'][f] - profile2['speaking_style'][f]) / 
                max(profile1['speaking_style'][f], profile2['speaking_style'][f])
                for f in profile1['speaking_style'].keys()
            ])
            
            # Weighted combination
            total_similarity = (
                0.4 * mfcc_similarity +
                0.3 * pitch_similarity +
                0.2 * vq_similarity +
                0.1 * style_similarity
            )
            
            return total_similarity
            
        except Exception as e:
            print(f"Error calculating similarity: {e}")
            return 0.0

    def add_training_sample(self, audio_data, sample_rate):
        try:
            profile = self.extract_voice_features(audio_data, sample_rate)
            if profile:
                self.training_samples.append(profile)
                self._update_speaker_profile()
                print(f"Added training sample. Total samples: {len(self.training_samples)}")
                return True
            else:
                print("Failed to extract features for training sample")
        except Exception as e:
            print(f"Error adding training sample: {e}")
        return False

    def _update_speaker_profile(self):
        """Update speaker profile using all training samples"""
        if not self.training_samples:
            return

        try:
            # Average all features across samples
            mfcc_features = [sample['mfcc'] for sample in self.training_samples]
            pitch_features = [sample['pitch'] for sample in self.training_samples]
            voice_quality = [sample['voice_quality'] for sample in self.training_samples]
            speaking_style = [sample['speaking_style'] for sample in self.training_samples]

            # Create averaged profile
            self.speaker_profile = {
                'mfcc': np.mean(mfcc_features, axis=0),
                'pitch': {
                    key: np.mean([p[key] for p in pitch_features]) 
                    for key in pitch_features[0].keys()
                },
                'voice_quality': {
                    key: np.mean([vq[key] for vq in voice_quality]) 
                    for key in voice_quality[0].keys()
                },
                'speaking_style': {
                    key: np.mean([ss[key] for ss in speaking_style]) 
                    for key in speaking_style[0].keys()
                }
            }
            print(f"Updated speaker profile with {len(self.training_samples)} samples")
        except Exception as e:
            print(f"Error updating speaker profile: {e}")

# Initialize the diarizer
diarizer = SpeakerDiarizer()

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'pdf', 'mp3', 'mp4', 'wav'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(file_path):
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {str(e)}")
        return None

def capture_speaker_audio():
    global is_listening
    try:
        print("Started speaker audio capture")
        with sd.InputStream(samplerate=RATE, channels=CHANNELS, dtype='float32', 
                          blocksize=CHUNK_SIZE, callback=audio_callback):
            while is_listening:
                sd.sleep(100)
    except Exception as e:
        print(f"Error in speaker audio capture: {str(e)}")
        is_listening = False

def audio_callback(indata, frames, time, status):
    if status:
        print(f'Status: {status}')
    if is_listening:
        audio_queue.put(indata.copy())

def get_ai_response(transcription):
    try:
        # Prepare document context string
        context_docs = ""
        for doc in document_context:
            context_docs += f"\nDocument: {doc['source']}\nContent: {doc['content']}\n"
        
        print(f"Generating AI response for: {transcription}")
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-16k",  # Using 16k model for longer context
            messages=[
                {
                    "role": "system",
                    "content": f"""You are Alex, a senior SaaS solutions consultant. You specialize in helping clients find the right solutions to their business challenges. When responding:

1. ALWAYS stay focused on the client's specific question
2. If you don't have relevant information in the reference documents, say: "Let me address your specific question about [topic]..." and provide general best practices
3. If you have relevant information in the reference documents, say: "Based on our solution capabilities..." and reference specific features

Your response structure:
1. Brief acknowledgment of their specific need
2. Your proposed solution with clear examples
3. Quick summary of benefits
4. Clear next steps

Remember to:
- Use natural language ("I understand what you're looking for...")
- Stay focused on their exact question
- Only mention features we actually have
- Be honest about capabilities
- Keep responses concise and relevant

Reference Materials:
{context_docs}

Important: Never suggest solutions or products we don't offer. If unsure, focus on understanding their needs better."""
                },
                {"role": "user", "content": transcription}
            ],
            max_tokens=500,  # Increased token limit for more detailed responses
            temperature=0.7
        )
        ai_response = response.choices[0].message.content
        print(f"AI Response generated: {ai_response}")
        return ai_response
    except Exception as e:
        print(f"Error generating AI response: {str(e)}")
        return None

def process_audio():
    frames = []
    silent_frames = 0
    is_recording = False
    sentence_frames = 0
    
    while is_listening:
        try:
            data = audio_queue.get(timeout=0.1)
            rms = np.sqrt(np.mean(data**2))
            
            if rms > SILENCE_THRESHOLD:
                if not is_recording:
                    print("Started recording segment")
                    is_recording = True
                    frames = []
                    sentence_frames = 0
                silent_frames = 0
                frames.append(data)
                sentence_frames += 1
            elif is_recording:
                silent_frames += 1
                frames.append(data)
                
                min_frames = int(MIN_SENTENCE_DURATION * RATE / CHUNK_SIZE)
                if silent_frames > FRAMES_PER_SILENCE and sentence_frames > min_frames:
                    print("Processing audio segment")
                    audio_data = np.concatenate(frames)
                    
                    try:
                        # Save temporary WAV file
                        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_segment.wav')
                        sf.write(temp_path, audio_data, RATE)
                        
                        # Get transcription
                        result = app.whisper_model.transcribe(temp_path)
                        transcription = result['text'].strip()
                        
                        # Detect speaker
                        speaker_label = diarizer.detect_speaker(audio_data, RATE)
                        print(f"Detected speaker: {speaker_label}")
                        
                        if transcription:
                            print(f"Transcribed: {transcription}")
                            transcription_queue.put({
                                'success': True,
                                'transcription': transcription,
                                'speaker': speaker_label,  # Just use the label directly
                                'type': 'final'
                            })
                        
                        # Clean up
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                            
                    except Exception as e:
                        print(f"Error in transcription: {e}")
                    
                    is_recording = False
                    frames = []
                    sentence_frames = 0
                    silent_frames = 0
                    
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Error in audio processing: {e}")

@app.route('/')
def home():
    global is_listening
    # Stop audio capture if it's running
    is_listening = False
    return render_template('home.html')

@app.route('/index')
def index():
    global is_listening
    try:
        # Test audio device availability
        devices = sd.query_devices()
        input_device = None
        for device in devices:
            if device['max_input_channels'] > 0:
                input_device = device['index']
                break
                
        if input_device is None:
            return jsonify({'error': 'No microphone found'}), 400
            
        # Start audio capture when navigating to index page
        is_listening = True
        start_background_threads()
        return render_template('index.html')
    except Exception as e:
        return jsonify({'error': f'Error accessing microphone: {str(e)}'}), 400

@app.route('/get_transcription')
def get_transcription():
    try:
        if not transcription_queue.empty():
            data = transcription_queue.get_nowait()
            print(f"Sending transcription: {data}")  # Debug log
            return jsonify(data)
        return jsonify({'success': False})
    except Exception as e:
        print(f"Error getting transcription: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'files' not in request.files:
        return jsonify({'success': False, 'error': 'No files uploaded'})
    
    files = request.files.getlist('files')

    try:
        if not hasattr(app, 'whisper_model'):
            print("Loading Whisper model...")
            app.whisper_model = whisper.load_model("base")

        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)

                try:
                    file_extension = filename.rsplit('.', 1)[1].lower()
                    
                    if file_extension == 'pdf':
                        extracted_text = extract_text_from_pdf(file_path)
                        if extracted_text:
                            transcription_queue.put({
                                'success': True,
                                'filename': filename,
                                'transcription': extracted_text,
                                'type': 'pdf'
                            })
                    
                    elif file_extension in ['mp3', 'mp4', 'wav']:
                        result = app.whisper_model.transcribe(file_path)
                        transcription_queue.put({
                            'success': True,
                            'filename': filename,
                            'transcription': result['text'],
                            'type': 'audio'
                        })

                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")
                    transcription_queue.put({
                        'success': False,
                        'filename': filename,
                        'error': str(e)
                    })
                finally:
                    if os.path.exists(file_path):
                        os.remove(file_path)

        return jsonify({
            'success': True,
            'message': 'Files uploaded and processing started',
            'total_files': len(files)
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/get_ai_response', methods=['POST'])
def generate_ai_response():
    try:
        data = request.json
        transcription = data.get('transcription')
        if not transcription:
            return jsonify({'success': False, 'error': 'No transcription provided'})
        
        ai_response = get_ai_response(transcription)
        if ai_response:
            return jsonify({
                'success': True,
                'response': ai_response
            })
        return jsonify({
            'success': False,
            'error': 'Failed to generate AI response'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

def start_background_threads():
    if not hasattr(app, 'recording_thread') or not app.recording_thread.is_alive():
        app.recording_thread = threading.Thread(target=capture_speaker_audio)
        app.recording_thread.daemon = True
        app.recording_thread.start()

    if not hasattr(app, 'processing_thread') or not app.processing_thread.is_alive():
        app.processing_thread = threading.Thread(target=process_audio)
        app.processing_thread.daemon = True
        app.processing_thread.start()

@app.route('/start_voice_training', methods=['POST'])
def start_voice_training():
    global is_training, training_audio_data
    is_training = True
    training_audio_data = []
    start_training_capture()
    return jsonify({'success': True})

@app.route('/stop_voice_training', methods=['POST'])
def stop_voice_training():
    global is_training, training_audio_data
    is_training = False
    
    if len(training_audio_data) > 0:
        try:
            # Concatenate all audio data
            audio_data = np.concatenate(training_audio_data)
            
            # Extract voice profile and set as first speaker
            profile = diarizer.extract_voice_features(audio_data, RATE)
            if profile:
                diarizer._initialize_first_speaker(profile)
                return jsonify({'success': True})
            
        except Exception as e:
            print(f"Error in voice training: {e}")
            
    return jsonify({'success': False, 'error': 'Failed to process voice training data'})

@app.route('/upload_voice_training', methods=['POST'])
def upload_voice_training():
    if 'voice_file' not in request.files:
        return jsonify({'success': False, 'error': 'No file uploaded'})
    
    file = request.files['voice_file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'})
        
    try:
        # Save file temporarily
        filename = secure_filename(file.filename)
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(temp_path)
        
        # Load and process audio file
        try:
            audio_data, sr = sf.read(temp_path)
        except Exception as e:
            # Fallback to librosa if soundfile fails
            audio_data, sr = librosa.load(temp_path, sr=RATE)
        
        # Add to training samples
        success = diarizer.add_training_sample(audio_data, sr)
        
        # Clean up
        os.remove(temp_path)
        
        if success:
            return jsonify({
                'success': True, 
                'samples_count': len(diarizer.training_samples)
            })
            
    except Exception as e:
        print(f"Error processing voice training file: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
    return jsonify({'success': False, 'error': 'Failed to process voice file'})

def start_training_capture():
    def training_callback(indata, frames, time, status):
        if status:
            print(f'Status: {status}')
        if is_training:
            training_audio_data.append(indata.copy())
    
    try:
        with sd.InputStream(samplerate=RATE, channels=CHANNELS, dtype='float32',
                          blocksize=CHUNK_SIZE, callback=training_callback):
            while is_training:
                sd.sleep(100)
    except Exception as e:
        print(f"Error in training capture: {e}")

def load_audio(file_path, sample_rate=16000):
    try:
        # Try soundfile first (faster)
        audio_data, sr = sf.read(file_path)
        if sr != sample_rate:
            # Resample if needed
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=sample_rate)
        return audio_data, sample_rate
    except Exception as e:
        print(f"SoundFile failed, using librosa: {str(e)}")
        # Fallback to librosa
        return librosa.load(file_path, sr=sample_rate, mono=True)

@app.route('/get_training_status')
def get_training_status():
    return jsonify({
        'samples_count': len(diarizer.training_samples),
        'has_profile': diarizer.speaker_profile is not None
    })

@app.route('/upload_document', methods=['POST'])
def upload_document():
    if 'document' not in request.files:
        return jsonify({'success': False, 'error': 'No file uploaded'})
    
    file = request.files['document']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'})
        
    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Extract text based on file type
            if filename.endswith('.pdf'):
                text = extract_text_from_pdf(filepath)
            else:
                # Handle other file types if needed
                text = None
            
            if text:
                # Add to document context
                add_to_document_context(text, filename)
                
            # Clean up
            os.remove(filepath)
            
            return jsonify({
                'success': True,
                'message': f'Successfully processed {filename}'
            })
            
        except Exception as e:
            print(f"Error processing document: {e}")
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'success': False, 'error': str(e)})
            
    return jsonify({'success': False, 'error': 'Invalid file type'})

@app.route('/clear_document_context', methods=['POST'])
def clear_document_context():
    global document_context
    document_context = []
    return jsonify({'success': True, 'message': 'Document context cleared'})

@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint for AWS App Runner and monitoring services.
    Returns:
        JSON response with status and timestamp
    """
    try:
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'service': 'whisper-transcription',
            'version': '1.0.0'  # You can update this based on your versioning
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


if __name__ == '__main__':
    app.run(debug=False, threaded=True)
