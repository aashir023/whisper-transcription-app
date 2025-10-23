import streamlit as st
from faster_whisper import WhisperModel
import os
import tempfile

# Title
st.title("Audio Transcription")
st.write("Upload MP3 or WAV → Get transcription")

# File uploader
uploaded_file = st.file_uploader("Choose audio file", type=["mp3", "wav", "m4a"])

if uploaded_file is not None:
    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        audio_path = tmp_file.name

    # Show processing
    with st.spinner("Transcribing"):
        # Load faster-whisper small model (int8 = fast & small)
        model = WhisperModel("small", device="cpu", compute_type="int8")

        # Transcribe — fast settings
        segments, _ = model.transcribe(
            audio_path,
            language="en",
            beam_size=7,
            best_of=5,
            temperature=0.0,
            no_speech_threshold=0.5,
            condition_on_previous_text=True
        )

        # Combine text
        transcription = " ".join(segment.text for segment in segments).strip()

    # Show result
    st.success("Transcription complete!")
    st.write(transcription)

    # Cleanup
    os.unlink(audio_path)
