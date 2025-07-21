import streamlit as st
import os
import numpy as np
import librosa
import librosa.display
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import io
from lime import lime_image
from skimage.segmentation import mark_boundaries
# import soundfile as sf # Removed as we are no longer creating dummy samples

# --- Configuration Parameters ---
SAMPLE_RATE = 16000
N_FFT = 1024
HOP_LENGTH = 512
N_MELS = 128
MAX_AUDIO_LEN_SECONDS = 3

# --- Load the Trained Model ---
MODEL_PATH = 'best_deepfake_detector_model.h5'

@st.cache_resource
def load_deepfake_model():
    """Loads the pre-trained Keras model."""
    if not os.path.exists(MODEL_PATH):
        st.error(f"Error: Model file not found at '{MODEL_PATH}'. Please ensure the model is saved and accessible.")
        return None
    try:
        model = load_model(MODEL_PATH)
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_deepfake_model()

# --- Preprocessing Function ---
def preprocess_audio(audio_file_path_or_buffer, target_sr=SAMPLE_RATE, max_len_seconds=MAX_AUDIO_LEN_SECONDS):
    """
    Loads, resamples, and normalizes an audio file.
    Pads or truncates audio to a fixed length.
    Accepts a file path or a file-like object (BytesIO).
    """
    try:
        y, sr = librosa.load(audio_file_path_or_buffer, sr=target_sr)
        y = librosa.util.normalize(y)

        target_length = int(target_sr * max_len_seconds)
        if len(y) > target_length:
            y = y[:target_length]
        elif len(y) < target_length:
            padding = np.zeros(target_length - len(y))
            y = np.concatenate((y, padding))
            
        return y
    except Exception as e:
        st.error(f"Error processing audio: {e}")
        return None

# --- Feature Extraction Function ---
def extract_features(audio_waveform, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS):
    """
    Extracts Mel-spectrogram from an audio waveform.
    """
    if audio_waveform is None:
        return None
    
    mel_spectrogram = librosa.feature.melspectrogram(
        y=audio_waveform,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )
    
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    return np.expand_dims(mel_spectrogram_db, axis=-1)

# --- LIME Explanation Function ---
def run_lime_explanation_streamlit(uploaded_file_buffer, model, features, 
                                   target_sr=SAMPLE_RATE, max_len_seconds=MAX_AUDIO_LEN_SECONDS,
                                   n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS, 
                                   num_features=100, num_samples=1000):
    """
    Generates and visualizes a LIME explanation for an audio file's prediction in Streamlit.
    """
    st.subheader("LIME Explanation")
    
    img_for_lime = (features[:, :, 0] - features[:, :, 0].min()) / (features[:, :, 0].max() - features[:, :, 0].min())
    img_for_lime = np.stack([img_for_lime, img_for_lime, img_for_lime], axis=-1)

    def lime_predict_fn(images):
        images_mono = images[:, :, :, 0:1]
        min_val = features[:, :, 0].min()
        max_val = features[:, :, 0].max()
        images_scaled = images_mono * (max_val - min_val) + min_val

        if not model.built:
            dummy_input_shape = (1, N_MELS, images_scaled.shape[2], 1)
            _ = model(tf.zeros(dummy_input_shape))
            
        predictions_fake_prob = model.predict(images_scaled)
        return np.hstack([1 - predictions_fake_prob, predictions_fake_prob])

    explainer = lime_image.LimeImageExplainer()

    with st.spinner("Generating LIME explanation... This might take a moment."):
        explanation = explainer.explain_instance(
            img_for_lime.astype(np.double),
            lime_predict_fn,
            top_labels=1,
            hide_color=0,
            num_samples=num_samples
        )

        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0],
            positive_only=False,
            num_features=num_features,
            hide_rest=False
        )

        fig_lime, ax_lime = plt.subplots(figsize=(10, 8))
        ax_lime.imshow(mark_boundaries(temp / 2 + 0.5, mask))
        ax_lime.set_title(f"LIME Explanation for Predicted Class: {explanation.top_labels[0]} (0: REAL, 1: FAKE)")
        ax_lime.axis('off')
        st.pyplot(fig_lime)
        plt.close(fig_lime)
    st.success("LIME explanation generated.")

# --- Prediction and Visualization Function ---
def visualize_and_predict_pipeline(audio_input_buffer, model):
    """
    Predicts whether an audio file is real or fake using the trained model,
    and visualizes the preprocessing and feature extraction steps.
    Accepts a BytesIO buffer.
    """
    if model is None:
        st.warning("Model not loaded. Cannot perform prediction.")
        return None

    # 1. Load Raw Audio
    audio_input_buffer.seek(0)
    raw_y, raw_sr = librosa.load(audio_input_buffer, sr=None)
    
    # 2. Preprocess Audio
    audio_input_buffer.seek(0)
    preprocessed_y = preprocess_audio(audio_input_buffer)
    
    if preprocessed_y is None:
        return None

    # 3. Extract Features (Mel-Spectrogram)
    features = extract_features(preprocessed_y)
    if features is None:
        return None

    features_for_prediction = np.expand_dims(features, axis=0)

    # 4. Make Prediction
    prediction_proba = model.predict(features_for_prediction)[0][0]
    classification_result = "FAKE" if prediction_proba > 0.5 else "REAL"
    
    st.write(f"**Prediction Probability (Fake):** {prediction_proba:.4f}")
    st.write(f"**Classification:** <span style='font-size: 24px; font-weight: bold; color: {'red' if classification_result == 'FAKE' else 'green'};'>{classification_result}</span>", unsafe_allow_html=True)

    st.subheader("Processing Pipeline Visualization")
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    # Plot 1: Raw Waveform
    axes[0].set_title(f'Raw Audio Waveform (Original SR: {raw_sr/1000:.1f} kHz)')
    librosa.display.waveshow(raw_y, sr=raw_sr, ax=axes[0], color='blue', alpha=0.7)
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True, linestyle='--', alpha=0.6)

    # Plot 2: Preprocessed Waveform
    axes[1].set_title(f'Preprocessed Audio Waveform (Resampled to {SAMPLE_RATE/1000:.1f} kHz, Fixed Length)')
    librosa.display.waveshow(preprocessed_y, sr=SAMPLE_RATE, ax=axes[1], color='green', alpha=0.7)
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Amplitude')
    axes[1].grid(True, linestyle='--', alpha=0.6)

    # Plot 3: Mel-Spectrogram
    axes[2].set_title(f'Extracted Mel-Spectrogram (Prediction: {classification_result}, Prob: {prediction_proba:.2f})')
    librosa.display.specshow(features[:, :, 0], sr=SAMPLE_RATE, x_axis='time', y_axis='mel',
                             cmap='viridis', hop_length=HOP_LENGTH, ax=axes[2])
    fig.colorbar(axes[2].collections[0], format='%+2.0f dB', ax=axes[2])
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Mel Frequency')
    axes[2].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    return features


# --- Streamlit App UI ---
st.set_page_config(
    page_title="Deepfake Audio Detection and Analysis",
    page_icon="🔊",
    layout="wide" # Use wide layout for better visualization
)

st.title("Deepfake Audio Detection and Analysis")
st.markdown("""
    This application helps detect deepfake audio using a Convolutional Neural Network (CNN) 
    and provides explainable AI (XAI) insights.
""")

st.header("Choose Input Method")
input_option = st.radio(
    "Select how you want to provide audio:",
    ("Upload Single Audio File", "Try Sample Audio Files", "Upload Batch of Audio Files (Coming Soon)", "Live Audio Recording (Coming Soon)")
)

# Initialize a variable to hold the audio buffer for processing
audio_to_process_buffer = None
selected_file_name = None

if input_option == "Upload Single Audio File":
    uploaded_file = st.file_uploader("Choose an audio file...", type=["wav", "mp3"])
    if uploaded_file is not None:
        st.audio(uploaded_file, format=uploaded_file.type)
        audio_to_process_buffer = io.BytesIO(uploaded_file.getvalue())
        selected_file_name = uploaded_file.name

elif input_option == "Try Sample Audio Files":
    st.markdown("Choose one of our pre-selected sample audio files to test the detector.")
    sample_files = {
        "Sample Real Audio": "sample_real.wav",
        "Sample Fake Audio": "sample_fake.wav"
    }
    sample_choice = st.selectbox("Select a sample audio:", list(sample_files.keys()))
    
    if sample_choice:
        sample_path = sample_files[sample_choice]
        if os.path.exists(sample_path):
            with open(sample_path, "rb") as f:
                sample_audio_bytes = f.read()
                st.audio(sample_audio_bytes, format="audio/wav")
                audio_to_process_buffer = io.BytesIO(sample_audio_bytes)
                selected_file_name = sample_path
        else:
            st.warning(f"Sample file '{sample_path}' not found. Please ensure it's in the app directory.")

elif input_option == "Upload Batch of Audio Files (Coming Soon)":
    st.info("This feature is under development. Please choose another input method for now.")
    # batch_uploaded_files = st.file_uploader("Upload multiple audio files...", type=["wav", "mp3"], accept_multiple_files=True)
    # if batch_uploaded_files:
    #     st.write("Files uploaded for batch processing:")
    #     for file in batch_uploaded_files:
    #         st.write(f"- {file.name}")
    #     # Placeholder for batch processing logic
    #     st.warning("Batch processing logic not yet implemented.")

elif input_option == "Live Audio Recording (Coming Soon)":
    st.info("This feature is under development and requires `streamlit_webrtc`. Please choose another input method for now.")
    # You'd integrate streamlit_webrtc here. Example:
    # from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
    # class AudioFrameProcessor(AudioProcessorBase):
    #     def recv(self, frame):
    #         # Process audio frame here (e.g., convert to numpy array)
    #         # Then feed to your model
    #         # This is complex as it involves real-time buffering and feature extraction
    #         pass
    #
    # webrtc_ctx = webrtc_streamer(
    #     key="audio",
    #     mode=WebRtcMode.SENDONLY,
    #     audio_processor_factory=AudioFrameProcessor,
    #     media_stream_constraints={"video": False, "audio": True},
    # )
    # if webrtc_ctx.audio_processor:
    #     st.write("Recording live audio...")
    #     # You would trigger prediction from within the AudioFrameProcessor
    # else:
    #     st.warning("WebRTC not active. Allow microphone access.")


# --- Processing and Prediction Section ---
if audio_to_process_buffer is not None:
    st.write(f"Analyzing: **{selected_file_name if selected_file_name else 'Uploaded Audio'}**")
    st.write("Processing audio...")
    
    with st.spinner("Analyzing audio and predicting... This might take a moment."):
        extracted_features = visualize_and_predict_pipeline(audio_to_process_buffer, model)
    
    if extracted_features is not None:
        st.markdown("---")
        # Add a button to generate LIME explanation, only if features were successfully extracted
        if st.button("Generate LIME Explanation"):
            # Ensure the buffer is reset for LIME's internal use if it needs to access the raw audio buffer again.
            audio_to_process_buffer.seek(0)
            run_lime_explanation_streamlit(audio_to_process_buffer, model, extracted_features)
        
        # How to Interpret LIME Section (kept styling for better clarity here)
        with st.expander("❓ How to Interpret the LIME Explanation"):
            st.write("""
            The LIME explanation helps us understand *why* the model made a particular prediction.
            
            * The image shown is a modified version of the Mel-spectrogram, which is the model's input.
            * <span style="color:green;font-weight:bold;">Green regions</span> highlight parts of the audio (in terms of frequency and time) that <span style="color:green;font-weight:bold;">strongly contributed to the predicted class</span>. For example, if the prediction is 'FAKE', green areas are what made the model think it's fake.
            * <span style="color:red;font-weight:bold;">Red regions</span> highlight parts that contributed to the <span style="color:red;font-weight:bold;">opposite class</span>. For example, if the prediction is 'FAKE', red areas are what made the model think it's *not* fake, but rather real.
            * The intensity of the color indicates the strength of the contribution.
            * This helps in identifying characteristic patterns or anomalies in the audio that the model is picking up on.
            """, unsafe_allow_html=True) # Ensure unsafe_allow_html=True is here

else:
    st.info("Please select an audio input method and provide audio to begin detection.")

# --- Sidebar Content ---
st.sidebar.header("About This App")
st.sidebar.markdown("""
This application demonstrates a Deepfake Audio Detection model built using a **Convolutional Neural Network (CNN)**. 
It processes audio by converting it into Mel-spectrograms and then uses the CNN to classify it as 'Real' or 'Fake'.
""")

st.sidebar.markdown("---") # Separator for visual appeal

st.sidebar.header("How it works:") # Changed from h3 to h2 or header for consistent sizing
st.sidebar.markdown("""
1.  **Audio Input:** Provide audio via upload, samples, or live recording (coming soon).
2.  **Preprocessing:** The audio is resampled, normalized, and padded/truncated to a fixed length.
3.  **Feature Extraction:** A Mel-spectrogram is generated from the preprocessed audio, representing the audio's frequency content over time.
4.  **Prediction:** The Mel-spectrogram is fed into the trained CNN model, which outputs a probability score (likelihood of being fake).
5.  **Visualization:** You see plots of the raw waveform, preprocessed waveform, and the Mel-spectrogram, along with the final prediction.
6.  **LIME Explanation:** (Optional) See which parts of the spectrogram were most influential in the model's decision.
""")

st.sidebar.markdown("---")
st.sidebar.markdown("""
Developed with <span style='color:#FF5722;'>Streamlit</span>, 
<span style='color:#FFC107;'>TensorFlow</span>, 
<span style='color:#673AB7;'>Librosa</span>, and 
<span style='color:#00BCD4;'>LIME</span>.
""", unsafe_allow_html=True)
st.sidebar.markdown("Code available on [GitHub](https://github.com/samiksha-bansal1/Deepfake-Audio-Detection-and-Analysis)", unsafe_allow_html=True) # Retained link and made it simpler