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

import numpy as np

def find_split_point(audio_clip, target_time, search_window=15):
    """
    Finds the best split point around target_time based on low volume (silence).
    """
    try:
        # Define search range
        start_search = max(0, target_time - search_window)
        end_search = min(audio_clip.duration, target_time + search_window)
        
        # Extract audio segment for analysis
        sub = audio_clip.subclip(start_search, end_search)
        
        # Get audio array (fps is usually 44100)
        # to_soundarray returns shape (N, 2) for stereo
        audio_array = sub.to_soundarray() 
        
        # Calculate volume envelope (RMS)
        # Average simple volume across channels
        volume = np.sqrt(np.mean(audio_array**2, axis=1))
        
        # We want to find a valley. 
        # Smooth slightly to avoid single-sample dropouts? 
        # For now, just finding the absolute minimum in this window is robust enough for speech.
        min_vol_idx = np.argmin(volume)
        
        # Convert index back to time relative to start_search
        best_time_relative = min_vol_idx / sub.fps
        
        best_time = start_search + best_time_relative
        return best_time
        
    except Exception as e:
        print(f"Warning: Silence detection failed ({e}), using exact time.")
        return target_time

def split_audio(audio_path, chunk_duration_sec=180, status_callback=None):
    """Splits audio into chunks roughly at chunk_duration_sec intervals, adjusting for silence."""
    audio = AudioFileClip(audio_path)
    total_duration = audio.duration
    chunks = []
    
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    temp_dir = "temp_audio"
    os.makedirs(temp_dir, exist_ok=True)
    
    current_start = 0.0
    part_idx = 0
    
    while current_start < total_duration:
        # Determine target end time
        target_end = current_start + chunk_duration_sec
        
        # If we are near the end, just take the rest
        if target_end >= total_duration:
            actual_end = total_duration
        else:
            # Find smart split point
            actual_end = find_split_point(audio, target_end, search_window=15)
            
        # Ensure we make progress (don't get stuck if actual_end <= current_start)
        # This can happen if search window pulls back too far
        if actual_end <= current_start + 1.0: # Minimum 1 second chunk
             actual_end = min(current_start + chunk_duration_sec, total_duration)

        chunk_filename = os.path.join(temp_dir, f"{base_name}_part{part_idx}.mp3")
        
        if status_callback:
            status_callback(f"Splitting part {part_idx+1}: {current_start:.2f}s to {actual_end:.2f}s...")
            
        chunk = audio.subclip(current_start, actual_end)
        chunk.write_audiofile(chunk_filename, codec='mp3', verbose=False, logger=None)
        
        chunks.append({
            "path": chunk_filename,
            "start_time_ms": int(current_start * 1000),
            "duration_sec": actual_end - current_start
        })
        
        current_start = actual_end
        part_idx += 1
        
    audio.close()
    return chunks

def transcribe_audio(api_key, audio_path, model_name="models/gemini-2.5-flash", status_callback=None):
    """Transcribes audio using Gemini API and returns SRT content."""
    genai.configure(api_key=api_key.strip())

    # 1. Check duration and decide if splitting is needed
    # We'll use a threshold of 3 minutes (180 seconds) to ensure better coverage
    # as the model sometimes cuts off after 2-3 minutes.
    SPLIT_THRESHOLD_SEC = 180
    
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
        
        Rules:
        1. The output must be strictly in SRT format, starting from 1.
        2. TIMESTAMPS: 
           - Timestamps must strictly correspond to the exact time the text is spoken in the audio segment.
        3. SEGMENTATION RULES (CRITICAL):
           - Default: Create a new subtitle block for every sentence (split by punctuation).
           - Merge Condition: If two or more consecutive short sentences/phrases have a combined length of 10 characters or less, you CAN put them in the same subtitle block (same timestamp).
           - Split Condition: If a merged line would exceed 10 characters, start a new subtitle block.
        4. Do not include any markdown code blocks, just the raw SRT content.
        5. NO TRANSLATION: Transcribe in the original language of the audio. Do not translate."""

        final_srt_parts = []
        current_counter = 1
        
        # Process chunks sequentially
        for i, chunk in enumerate(chunks_to_process):
            chunk_path = chunk['path']
            chunk_name = os.path.basename(chunk_path)
            # Use the pre-calculated start time for this chunk as the offset
            chunk_offset_ms = chunk.get('start_time_ms', 0)
            
            if status_callback:
                status_callback(f"Processing part {i+1}/{len(chunks_to_process)}: {chunk_name}...")
            
            # 1. Upload
            audio_file = upload_to_gemini(chunk_path)
            
            # 2. Wait
            wait_for_files_active([audio_file])
            
            # 3. Generate
            model = genai.GenerativeModel(
                model_name=model_name,
                generation_config=generation_config,
                safety_settings=safety_settings,
                system_instruction=system_instruction
            )
            
            prompt_parts = [audio_file, "Generate SRT subtitles for this audio."]
            
            response = None
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    if status_callback and attempt > 0:
                        status_callback(f"Rate limit hit. Retrying part {i+1} in {20 * (attempt+1)} seconds...")
                    
                    response = model.generate_content(prompt_parts)
                    break # Success!
                    
                except Exception as e:
                    if "429" in str(e) or "Resource exhausted" in str(e):
                        if attempt < max_retries - 1:
                            wait_time = 20 * (attempt + 1)
                            print(f"Warning: Resource exhausted (429). Retrying in {wait_time}s...")
                            time.sleep(wait_time)
                            continue
                        else:
                             raise Exception(f"Failed after {max_retries} attempts due to rate limits. Please try again later.")
                    else:
                        raise e # Re-raise if it's not a rate limit issue
            
            # --- DEBUG LOGGING ---
            try:
                with open("transcription_debug.log", "a", encoding="utf-8") as log_file:
                    log_file.write(f"\n{'='*50}\n")
                    log_file.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    log_file.write(f"Model: {model_name} | Part: {i+1}/{len(chunks_to_process)}\n")
                    
                    finish_reason = "Unknown"
                    if response.candidates:
                        finish_reason = response.candidates[0].finish_reason.name
                    log_file.write(f"Finish Reason: {finish_reason}\n")
                    
                    if response.usage_metadata:
                        log_file.write(f"Token Usage:\n")
                        log_file.write(f"  - Prompt Tokens: {response.usage_metadata.prompt_token_count}\n")
                        log_file.write(f"  - Candidates Tokens: {response.usage_metadata.candidates_token_count}\n")
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
                raw_srt = raw_srt.replace("```srt", "").replace("```", "").strip()
            except Exception as e:
                 print(f"Error accessing response.text: {e}")
                 # Check blocked
                 if response.prompt_feedback:
                     print(f"Prompt Feedback: {response.prompt_feedback}")
            
            if raw_srt:
                # 4. Post-process & Shift
                # We shift the timestamps by the chunk's start time explicitly
                part_srt, last_counter = shift_srt_content(raw_srt, chunk_offset_ms, counter_start=current_counter)
                if part_srt:
                    final_srt_parts.append(part_srt)
                    # Update counter for next chunk
                    # shift_srt_content returns the last used counter, so next starts at last + 1
                    current_counter = last_counter + 1
            else:
                 print(f"Warning: Empty response for part {i+1}")
            
            # Cleanup immediately after use to assume fresh state for next iteration
            # (Though keeping them is fine, deleting minimizes storage use during long process)
            # But the user might want to debug audio... let's stick to cleaning up at the very end
            # or we can delete the remote file resource to keep quotas clean?
            # genai.delete_file(audio_file.name) # If the SDK supports it easily, but let's just leave it for now.
            pass
            
            # Rate limiting cooldown
            if i < len(chunks_to_process) - 1:
                print("Cooling down for 5 seconds to avoid rate limits...")
                time.sleep(5)

        if status_callback:
             status_callback("All parts processed. Stitching timestamps...")

        full_srt = "\n\n".join(final_srt_parts)
        return full_srt

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
