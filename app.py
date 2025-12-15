import streamlit as st
import os
import time
from transcriber import extract_audio, transcribe_audio, cleanup_files, list_available_models

st.set_page_config(page_title="Video2SRT Converter", page_icon="🎬")

st.title("🎬 Video to SRT Converter")
st.markdown("Upload a video, and we'll generate subtitles for you using Google Gemini AI.")

# Sidebar for API Key and Model Selection #
with st.sidebar:
    api_key_input = st.text_input("Enter Google Gemini API Key", type="password")
    api_key = api_key_input.strip() if api_key_input else ""
    
    # Model selection
    if api_key:
        st.session_state.model_list = [
            "models/gemini-flash-latest"
        ]
        
        selected_model = st.selectbox("Select Model", st.session_state.model_list, index=0)
        st.caption("Using gemini-flash-latest as default")
    else:
        # Fallback if API key is not provided
        selected_model = "models/gemini-flash-latest" # Default to a common model if no API key
        st.caption("Please enter an API key to select a model.")

    st.info("Get your API key from [Google AI Studio](https://aistudio.google.com/app/apikey)")

uploaded_file = st.file_uploader("Choose a video or audio file", type=["mp4", "mov", "avi", "mkv", "mp3", "wav", "m4a", "flac", "ogg"])

if uploaded_file is not None:
    # Check if it's a video file or audio file based on extension
    file_type = uploaded_file.name.split('.')[-1].lower()
    is_video = file_type in ["mp4", "mov", "avi", "mkv"]
    
    if is_video:
        st.video(uploaded_file)
    else:
        st.audio(uploaded_file)
    
    # Use a unique key for the button to avoid state conflicts if file changes
    generate_btn = st.button("Generate Subtitles")
    
    if generate_btn:
        if not api_key:
            st.error("Please provide a valid API Key in the sidebar.")
        else:
            with st.spinner("Processing file..."):
                info_container = st.container()
                status_text = st.empty()
                def update_status(message):
                    if message.startswith("INFO:"):
                        info_container.info(message.replace("INFO:", "ℹ️").strip())
                    else:
                        status_text.text(f"⏳ {message}")
                
                try:
                    # Save uploaded file temporarily
                    temp_path = f"temp_{uploaded_file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    audio_path = None
                    files_to_cleanup = [temp_path]
                    
                    if is_video:
                        update_status("Extracting audio from video...")
                        audio_path = extract_audio(temp_path)
                        if audio_path:
                            files_to_cleanup.append(audio_path)
                    else:
                        update_status("Processing audio file...")
                        audio_path = temp_path # Direct use
                    
                    if audio_path:
                        update_status(f"Transcribing with {selected_model}...")
                        srt_content = transcribe_audio(api_key, audio_path, model_name=selected_model, status_callback=update_status)
                        
                        st.session_state['srt_content'] = srt_content
                        st.session_state['current_file'] = uploaded_file.name
                        
                        update_status("Finalizing...")
                        time.sleep(1) # Brief pause to let user see the message
                        status_text.empty() # Clear status
                        
                        st.success("Transcription complete!")
                        
                        # Cleanup
                        cleanup_files(files_to_cleanup)
                    else:
                        st.error("Failed to process audio.")
                        cleanup_files(files_to_cleanup)
                        
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    # Attempt cleanup if failed
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

    # Display results if they exist for the current file
    if 'srt_content' in st.session_state and st.session_state.get('current_file') == uploaded_file.name:
        # Display SRT preview
        with st.expander("Preview Subtitles"):
            st.code(st.session_state['srt_content'], language="text")
        
        # Download button
        srt_filename = os.path.splitext(uploaded_file.name)[0] + ".srt"
        
        # Check download feedback
        if st.download_button(
            label="Download SRT File",
            data=st.session_state['srt_content'],
            file_name=srt_filename,
            mime="text/plain"
        ):
            st.success("Download Complete! ✅")
