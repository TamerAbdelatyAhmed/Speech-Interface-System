import streamlit as st
from streamlit_extras.switch_page_button import switch_page
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, pipeline
import torch
import librosa
from tacotron2.text import text_to_sequence
import numpy as np
from scipy.io.wavfile import write

# Function definitions (assuming you have the required models downloaded)
def transcribe_audio(audio_file):
    """Transcribes the uploaded audio file to text using Wav2Vec2."""
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    audio, rate = librosa.load(audio_file, sr=16000)
    input_values = processor(audio, sampling_rate=rate, return_tensors="pt", padding=True).input_values
    with torch.no_grad():
        logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    return processor.batch_decode(predicted_ids)[0]

def process_text(input_text):
    """Processes the transcribed text using a question-answering model."""
    qa_model = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
    response = qa_model(question=input_text, context="This is a demo context for task processing.")
    return response["answer"]

def generate_audio(text, output_file="output.wav"):
    """Generates audio from text using Tacotron2 and WaveGlow."""
    tacotron2 = torch.hub.load('nvidia/DeepLearningExamples:torchhub', 'nvidia_tacotron2', force_reload=True, map_location=torch.device('cpu')) 
    # Remove the mask_padding argument
    tacotron2.model.mask_padding = None  

    waveglow = torch.hub.load('nvidia/DeepLearningExamples:torchhub', 'nvidia_waveglow', force_reload=True, map_location=torch.device('cpu')) 
    sequence = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
    sequence = torch.from_numpy(sequence).long() 
    mel_outputs, mel_outputs_postnet, _, alignments = tacotron2.infer(sequence)
    audio = waveglow.infer(mel_outputs_postnet, sigma=0.666).cpu().numpy()
    write(output_file, 22050, (audio * 32767).astype("int16"))
    return output_file

st.title("Speech Interface System")

uploaded_file = st.file_uploader("Upload a voice wav file", type=["wav"])

if uploaded_file:
    st.audio(uploaded_file, format="audio/wav")

    # Add buttons for transcription and generation
    if st.button("Transcribe"):
        transcription = transcribe_audio(uploaded_file)
        st.session_state.transcription = transcription
        st.write("Transcription:", transcription)

    if st.button("Generate Response"):
        if "transcription" in st.session_state:
            response = process_text(st.session_state.transcription)
            st.write("Response:", response)
            output_audio = generate_audio(response)
            st.audio(output_audio, format="audio/wav")

# Add a text input box for user-provided text
user_input = st.text_input("Enter text to generate audio:")

if st.button("Generate Audio from Text"):
    if user_input:
        try:
            output_audio = generate_audio(user_input)
            st.audio(output_audio, format="audio/wav")
        except RuntimeError as e:
            st.error(f"Error generating audio: {e}") 
