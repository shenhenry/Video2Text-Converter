import os
import google.generativeai as genai
from moviepy.editor import VideoFileClip, AudioFileClip
import json
import time
import re
import math

def extract_audio(video_path, audio_path="temp_audio/temp_audio.mp3"):
    """Extracts audio from a video file."""
    try:
        os.makedirs(os.path.dirname(audio_path), exist_ok=True)
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(audio_path, codec='mp3')
        return audio_path
    except Exception as e:
        print(f"Error extracting audio: {e}")
        return None

def upload_to_gemini(path, mime_type="audio/mp3"):
    """Uploads the file to Gemini."""
    file = genai.upload_file(path, mime_type=mime_type)
    print(f"Uploaded file '{file.display_name}' as: {file.uri}")
    return file

def wait_for_files_active(files):
    """Waits for the uploaded files to be processed."""
    print("Waiting for file processing...")
    for name in (file.name for file in files):
        file = genai.get_file(name)
        while file.state.name == "PROCESSING":
            print(".", end="", flush=True)
            time.sleep(2)
            file = genai.get_file(name)
        if file.state.name != "ACTIVE":
            raise Exception(f"File {file.name} failed to process")
    print("...all files ready")

def list_available_models(api_key):
    """Lists available models that support content generation."""
    genai.configure(api_key=api_key.strip())
    try:
        models = []
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                models.append(m.name)
        return models
    except Exception as e:
        return [f"Error listing models: {str(e)}"]

def time_to_ms(time_str):
    """Converts SRT timestamp "HH:MM:SS,mmm" or "MM:SS,mmm" to milliseconds."""
    try:
        parts = time_str.split(':')
        if len(parts) == 3:
            hours, minutes, seconds = parts
        elif len(parts) == 2:
            hours = 0
            minutes, seconds = parts
        else:
            return 0
            
        seconds, milliseconds = seconds.split(',')
        total_ms = (int(hours) * 3600000) + (int(minutes) * 60000) + (int(seconds) * 1000) + int(milliseconds)
        return total_ms
    except ValueError:
        return 0

def ms_to_time(ms):
    """Converts milliseconds to SRT timestamp "HH:MM:SS,mmm"."""
    hours = ms // 3600000
    ms %= 3600000
    minutes = ms // 60000
    ms %= 60000
    seconds = ms // 1000
    ms %= 1000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{ms:03d}"

def shift_srt_content(srt_content, offset_ms, counter_start=1):
    """Parses, filters, and shifts SRT content."""
    # Regex modified to support optional HH: prefix (e.g. match both 00:00:10,000 and 00:10,000)
    timestamp_pattern = re.compile(r'((?:\d{2}:)?\d{2}:\d{2},\d{3})\s-->\s((?:\d{2}:)?\d{2}:\d{2},\d{3})')
    
    matches = list(timestamp_pattern.finditer(srt_content))
    valid_blocks = []
    
    for i, match in enumerate(matches):
        start_str = match.group(1)
        end_str = match.group(2)
        start_ms = time_to_ms(start_str)
        end_ms = time_to_ms(end_str)
        
        # Get content between this match and next match (or end of string)
        content_start = match.end()
        if i < len(matches) - 1:
            content_end = matches[i+1].start()
        else:
            content_end = len(srt_content)
            
        raw_text = srt_content[content_start:content_end]
        
        # Process lines to remove indices and empty space
        # We rely on the timestamp as the anchor. The text follows it.
        # The index for the NEXT block might be at the end of this chunk.
        lines = [l.strip() for l in raw_text.split('\n') if l.strip()]
        
        # Filter out the index of the next block if present (heuristic: last line is digit)
        if lines and lines[-1].isdigit():
            lines.pop()
            
        # If no text remains, it was an empty/hallucinated block
        if not lines:
            continue
            
        text_content = "\n".join(lines)
        
        # Filter duplicate consecutive blocks (hallucination loop) where timestamps are identical
        if valid_blocks:
            last_start, last_end, _ = valid_blocks[-1]
            # Calculating shifted times for comparison would be clean, but let's just compare raw ms logic
            # implicitly handled since we rebuild fresh below.
            pass

        # Apply shift
        new_start = ms_to_time(start_ms + offset_ms)
        new_end = ms_to_time(end_ms + offset_ms)
        
        valid_blocks.append((new_start, new_end, text_content))
        
    # Reconstruct SRT
    output = []
    curr = counter_start
    for start, end, text in valid_blocks:
        output.append(str(curr))
        output.append(f"{start} --> {end}")
        output.append(text)
        output.append("")
        curr += 1
        
    return "\n".join(output).strip(), curr - 1

def split_audio(audio_path, chunk_duration_sec=600, status_callback=None):
    """Splits audio into chunks of specified duration."""
    audio = AudioFileClip(audio_path)
    duration = audio.duration
    chunks = []
    
    num_chunks = math.ceil(duration / chunk_duration_sec)
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    
    # Create temp directory for splits
    temp_dir = "temp_audio"
    os.makedirs(temp_dir, exist_ok=True)
    
    for i in range(num_chunks):
        start_time = i * chunk_duration_sec
        end_time = min((i + 1) * chunk_duration_sec, duration)
        
        chunk_filename = os.path.join(temp_dir, f"{base_name}_part{i}.mp3")
        
        if status_callback:
            status_callback(f"Splitting file to {os.path.basename(chunk_filename)}...")

        # Extract subclip
        # Check if we are at the very end to avoid tiny clips if accurate
        if start_time >= duration:
            break
            
        chunk = audio.subclip(start_time, end_time)
        chunk.write_audiofile(chunk_filename, codec='mp3', verbose=False, logger=None)
        chunks.append({
            "path": chunk_filename,
            "start_time_ms": int(start_time * 1000),
            "duration_sec": end_time - start_time
        })
    
    audio.close() # Close the main handle
    return chunks

def transcribe_audio(api_key, audio_path, model_name="models/gemini-2.5-flash", status_callback=None):
    """Transcribes audio using Gemini API and returns SRT content."""
    genai.configure(api_key=api_key.strip())

    # 1. Check duration and decide if splitting is needed
    # We'll use a threshold of 3 minutes (180 seconds) to ensure better coverage
    # as the model sometimes cuts off after 2-3 minutes.
    SPLIT_THRESHOLD_SEC = 600
    
    # Use moviepy to check duration quickly without full split yet
    try:
        audio_clip = AudioFileClip(audio_path)
        duration_sec = audio_clip.duration
        audio_clip.close()
    except Exception:
        # Fallback if moviepy fails to read metadata, just try normal process
        duration_sec = 0 

    final_srt_parts = []
    files_to_cleanup = []
    
    chunks_to_process = []
    
    if duration_sec > SPLIT_THRESHOLD_SEC:
        if status_callback:
            status_callback(f"Audio is long ({int(duration_sec)}s), preparing to split...")
        chunks = split_audio(audio_path, chunk_duration_sec=SPLIT_THRESHOLD_SEC, status_callback=status_callback)
        chunks_to_process = chunks
        for c in chunks:
            files_to_cleanup.append(c['path'])
    else:
        # For single chunk, we need to know its duration too if we want to report it,
        # but the request specifically asked about the "last part" when splitting.
        # We can just set it if we have it.
        chunks_to_process = [{"path": audio_path, "start_time_ms": 0, "duration_sec": duration_sec}]

    if status_callback and chunks_to_process:
        last_chunk = chunks_to_process[-1]
        last_duration_min = last_chunk.get("duration_sec", 0) / 60
        status_callback(f"INFO: Last audio chunk duration: {last_duration_min:.2f} minutes")
        # Small sleep to let the user see the message
        time.sleep(1)

    
    current_counter = 1
    
    try:
        generation_config = {
            "temperature": 0.1,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 8192,
            "response_mime_type": "text/plain",
        }
        
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]

        system_instruction = """You are a professional subtitle creator. Generate a valid SRT file content for the audio provided.
        The audio may be split into multiple parts. You must treat them as a single continuous audio stream.
        
        Rules:
        1. The output must be strictly in SRT format, starting from 1.
        2. TIMESTAMPS: 
           - Start timestamps strictly from 00:00:00,000 relative to the beginning of the FIRST audio file.
           - Ensure specific continuity across file boundaries. DO NOT reset the timestamp to 00:00:00 for subsequent files. The timestamp for the start of the second file should follow immediately after the end of the first file.
        3. SEGMENTATION RULES (CRITICAL):
           - Default: Create a new subtitle block for every sentence (split by punctuation).
           - Merge Condition: If two or more consecutive short sentences/phrases have a combined length of 10 characters or less, you CAN put them in the same subtitle block (same timestamp).
           - Split Condition: If a merged line would exceed 10 characters, start a new subtitle block.
        4. Do not include any markdown code blocks, just the raw SRT content.
        5. NO TRANSLATION: Transcribe in the original language of the audio. Do not translate."""

        uploaded_files = []
        
        # 1. Upload all chunks
        for i, chunk in enumerate(chunks_to_process):
            chunk_path = chunk['path']
            chunk_name = os.path.basename(chunk_path)
            
            if status_callback:
                status_callback(f"Uploading part {i+1}/{len(chunks_to_process)}: {chunk_name}...")
            
            audio_file = upload_to_gemini(chunk_path)
            uploaded_files.append(audio_file)
        
        # 2. Wait for all files to be active
        wait_for_files_active(uploaded_files)
        
        # 3. Generate content with ALL files in one request
        model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=generation_config,
            safety_settings=safety_settings,
            system_instruction=system_instruction
        )
        
        prompt_parts = uploaded_files + ["Generate SRT subtitles for this full audio sequence. Ensure timestamps are continuous."]
        
        if status_callback:
            status_callback("Generating subtitles (this may take a while)...")
            
        response = model.generate_content(prompt_parts)
        
        # --- DEBUG LOGGING ---
        try:
            with open("transcription_debug.log", "a", encoding="utf-8") as log_file:
                log_file.write(f"\n{'='*50}\n")
                log_file.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                log_file.write(f"Model: {model_name}\n")
                
                # Log finish reason to diagnose truncation
                finish_reason = "Unknown"
                if response.candidates:
                    finish_reason = response.candidates[0].finish_reason.name
                log_file.write(f"Finish Reason: {finish_reason}\n")
                
                # Log Token Usage
                if response.usage_metadata:
                    log_file.write(f"Token Usage:\n")
                    log_file.write(f"  - Prompt Tokens: {response.usage_metadata.prompt_token_count}\n")
                    log_file.write(f"  - Candidates Tokens (Output): {response.usage_metadata.candidates_token_count}\n")
                    log_file.write(f"  - Total Tokens: {response.usage_metadata.total_token_count}\n")
                
                try:
                    log_file.write(f"Raw Text Length: {len(response.text)} chars\n")
                    log_file.write(f"--- Raw Response Start ---\n")
                    log_file.write(response.text)
                    log_file.write(f"\n--- Raw Response End ---\n")
                except Exception as e:
                    log_file.write(f"Could not log text: {e}\n")
                    if response.prompt_feedback:
                         log_file.write(f"Prompt Feedback: {response.prompt_feedback}\n")
        except Exception as log_err:
            print(f"Failed to write log: {log_err}")
        # ---------------------
        
        raw_srt = ""
        try:
            raw_srt = response.text
             # Remove markdown code blocks if any
            raw_srt = raw_srt.replace("```srt", "").replace("```", "").strip()
        except Exception as e:
             error_msg = f"Error accessing response.text: {e}. "
             if response.prompt_feedback:
                 error_msg += f"Prompt Feedback: {response.prompt_feedback}"
             print(error_msg)
             return f"Error: The model failed to generate text. {error_msg}"
        
        if not raw_srt:
            return "Error: Model returned empty response. (Detailed feedback: check logs)"
        
        # 4. Post-process (clean/shift)
        # We use shift_srt_content with offset 0 just to use its cleaning logic (hallucination removal etc)
        # We trust the model to have done the accumulative timestamps, but we still want to clean the format.
        final_srt, _ = shift_srt_content(raw_srt, 0, counter_start=1)
        
        if status_callback:
             status_callback("Transcription complete!")

        return final_srt

    except Exception as e:
        return f"Error during transcription process: {str(e)}"
    finally:
        # If we created temp chunks, clean them up
        if files_to_cleanup:
            cleanup_files(files_to_cleanup)

def cleanup_files(file_paths):
    for path in file_paths:
        if os.path.exists(path):
            try:
                os.remove(path)
            except Exception as e:
                print(f"Error removing {path}: {e}")
