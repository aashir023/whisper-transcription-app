import streamlit as st
import whisper
import re
import os
from pathlib import Path

# Set up the Streamlit app
st.title("Audio Transcription App")
st.write("Upload an audio file (e.g., .mp3, .wav) to get its transcription.")

# File uploader for audio files
uploaded_file = st.file_uploader("Choose an audio file", type=["mp3", "wav"])

if uploaded_file is not None:
    # Save the uploaded file temporarily
    temp_dir = "./temp"
    os.makedirs(temp_dir, exist_ok=True)
    audio_path = os.path.join(temp_dir, uploaded_file.name)
    with open(audio_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Display processing message
    st.write("Transcribing audio...")
    
    # Load Whisper model
    model = whisper.load_model("medium", device="cpu")
    
    # Transcribe audio with tuned parameters
    result = model.transcribe(
    audio_path,
    task="transcribe",
    language="en",
    temperature=0.0,
    beam_size=15,
    fp16=False,
    condition_on_previous_text=True,
    no_speech_threshold=0.8,
    logprob_threshold=-0.3,
)

    
    # Capture verbose output
    # st.write("Verbose Output:")
    # for segment in result.get("segments", []):
    #     start = segment["start"]
    #     end = segment["end"]
    #     text = segment["text"]
    #     st.write(f"[{start:00.3f} --> {end:00.3f}] {text}")
    
    # Get the raw transcription  for edits
    text = result["text"]
    
    # Split text into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    
    # Remove exact duplicates (general post-processing)
    corrected_sentences = []
    seen_sentences = set()
    for sentence in sentences:
        if sentence not in seen_sentences:
            corrected_sentences.append(sentence)
            seen_sentences.add(sentence)
    
    # Display transcription in the UI
    st.write("\n**Transcription:**")
    for sentence in corrected_sentences:
        st.write(sentence)
    
    # Save transcription to file
    output_dir = "./outputs"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{Path(uploaded_file.name).stem}.txt")
    with open(output_path, "w") as f:
        for sentence in corrected_sentences:
            f.write(sentence + "\n")
    
    st.write(f"\nTranscription saved to {output_path}")
    
    # Clean up temporary file
    os.remove(audio_path)
