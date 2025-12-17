import os
import google.generativeai as genai
from moviepy.editor import VideoFileClip, AudioFileClip
import json
import time
import re
import math
import config

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

def find_split_point_gemini(api_key, audio_clip, status_callback=None):
    """
    Uses Gemini API to find the best split point in the provided audio clip.
    Returns: time in seconds relative to the start of the clip.
    """
    temp_clip_path = "temp_split_search.mp3"
    try:
        # Write the search window clip to file
        audio_clip.write_audiofile(temp_clip_path, codec='mp3', verbose=False, logger=None)
        
        # Configure GenAI
        genai.configure(api_key=api_key.strip())
        
        # Upload
        if status_callback:
            print("Uploading search window to Gemini for split analysis...")
            
        file = genai.upload_file(temp_clip_path, mime_type="audio/mp3")
        
        # Wait for processing
        while file.state.name == "PROCESSING":
            time.sleep(1)
            file = genai.get_file(file.name)
            
        if file.state.name != "ACTIVE":
            raise Exception("Gemini file processing failed")

        model = genai.GenerativeModel(model_name=config.DEFAULT_MODEL_NAME)
        
        prompt = (
            "Listen to this audio clip. I need to cut the audio file here. "
            "Find the best timestamp (in seconds) to make a cut, such as a moment of silence or the end of a sentence. "
            "Return ONLY the number (e.g. 5.43). If no good point, return the middle of the duration."
        )
        
        response = model.generate_content([file, prompt])
        
        # Clean up
        # genai.delete_file(file.name) 
        
        # Parse response
        text = response.text.strip()
        # Extract number
        match = re.search(r"(\d+(\.\d+)?)", text)
        if match:
             return float(match.group(1))
        
        return audio_clip.duration / 2

    except Exception as e:
        print(f"Gemini split detection failed: {e}. Falling back to center.")
        return audio_clip.duration / 2
    finally:
        if os.path.exists(temp_clip_path):
            os.remove(temp_clip_path)

def split_audio(audio_path, chunk_duration_sec=config.CHUNK_DURATION_SEC, status_callback=None, api_key=None):
    """
    Splits audio into chunks.
    Phase 1: Calculate split points using Gemini API (if api_key provided) or fallback.
    Phase 2: Save points to file.
    Phase 3: Split audio locally.
    """
    audio = AudioFileClip(audio_path)
    total_duration = audio.duration
    
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    temp_dir = "temp_audio"
    os.makedirs(temp_dir, exist_ok=True)
    
    # --- PHASE 1: Calculate Split Points ---
    split_points = []
    current_scan_time = 0.0
    
    # Adaptive search window: 15s for long chunks, or 20% of duration for short ones
    search_window = min(chunk_duration_sec * 1.2, chunk_duration_sec * 0.2)
    
    print("Calculating split points...")
    if status_callback:
        status_callback("Analyzing audio for split points...")

    while current_scan_time < total_duration:
        target_end = current_scan_time + chunk_duration_sec
        
        if target_end >= total_duration:
            actual_end = total_duration
        else:
            # We define the search window range
            start_search = max(0, target_end - search_window)
            end_search = min(audio.duration, target_end + search_window)
            
            if api_key:
                # Extract the subclip for Gemini to analyze
                # Note: find_split_point_gemini expects just the clip and returns relative time
                sub = audio.subclip(start_search, end_search)
                best_relative = find_split_point_gemini(api_key, sub, status_callback)
                actual_end = start_search + best_relative
            else:
                # Fallback to local numpy
                actual_end = find_split_point(audio, target_end, search_window=search_window)
            
        # Ensure progress
        if actual_end <= current_scan_time + 1.0:
             actual_end = min(current_scan_time + chunk_duration_sec, total_duration)
        
        # Store the END time of this chunk
        split_points.append(actual_end)
        current_scan_time = actual_end
        
        if actual_end >= total_duration:
            break
            
    # --- PHASE 2: Save to File ---
    split_points_file = os.path.join(temp_dir, "split_points.txt")
    try:
        with open(split_points_file, "w", encoding="utf-8") as f:
            f.write(f"Total Duration: {total_duration}\n")
            f.write(f"Chunk Duration Target: {chunk_duration_sec}\n")
            f.write("-" * 20 + "\n")
            for idx, pt in enumerate(split_points):
                f.write(f"Chunk {idx+1} End: {pt:.2f}\n")
        print(f"Split points saved to {split_points_file}")
    except Exception as e:
        print(f"Failed to save split points: {e}")

    # --- PHASE 3: Perform Split ---
    chunks = []
    current_start = 0.0
    
    try:
        for idx, end_time in enumerate(split_points):
            chunk_filename = os.path.join(temp_dir, f"{base_name}_part{idx}.mp3")
            
            if status_callback:
                status_callback(f"Splitting part {idx+1}/{len(split_points)}: {current_start:.2f}s to {end_time:.2f}s...")
                
            chunk = audio.subclip(current_start, end_time)
            chunk.write_audiofile(chunk_filename, codec='mp3', verbose=False, logger=None)
            
            chunks.append({
                "path": chunk_filename,
                "start_time_ms": int(current_start * 1000),
                "duration_sec": end_time - current_start
            })
            
            current_start = end_time
            
        return chunks

    except Exception as e:
        print(f"Error during audio splitting: {e}")
        # Cleanup partial chunks
        for c in chunks:
            if os.path.exists(c['path']):
                os.remove(c['path'])
        raise e
    finally:
        audio.close()


def write_log(message):
    """ Writes a message to the debug log with timestamp and force flush. """
    try:
        log_path = "transcription_debug.log"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")
            f.flush() 
    except Exception as e:
        print(f"Failed to write log: {e}")

def transcribe_audio(api_key, audio_path, model_name=config.DEFAULT_MODEL_NAME, status_callback=None):
    """Transcribes audio using Gemini API and returns SRT content."""
    
    # 0. Immediate Logging
    write_log(f"{'-'*30} NEW SESSION {'-'*30}")
    write_log(f"Processing File: {audio_path}")
    write_log(f"Model: {model_name}")

    genai.configure(api_key=api_key.strip())

    # 1. Check duration and decide if splitting is needed
    # User requested 1 minute chunking for testing
    SPLIT_THRESHOLD_SEC = config.SPLIT_THRESHOLD_SEC
    
    # Use moviepy to check duration quickly without full split yet
    try:
        audio_clip = AudioFileClip(audio_path)
        duration_sec = audio_clip.duration
        audio_clip.close()
    except Exception as e:
        write_log(f"Error reading audio duration: {e}")
        # Fallback if moviepy fails to read metadata, just try normal process
        duration_sec = 0 

    final_srt_parts = []
    files_to_cleanup = []
    
    chunks_to_process = []
    
    if duration_sec > SPLIT_THRESHOLD_SEC:
        try:
            if status_callback:
                status_callback(f"Audio is long ({int(duration_sec)}s), preparing to split...")
            
            write_log(f"Audio duration {duration_sec}s > {SPLIT_THRESHOLD_SEC}s. Splitting...")
            chunks = split_audio(audio_path, chunk_duration_sec=SPLIT_THRESHOLD_SEC, status_callback=status_callback, api_key=api_key)
            chunks_to_process = chunks
            for c in chunks:
                files_to_cleanup.append(c['path'])
            
            # Also clean up the split points file
            split_points_file = os.path.join("temp_audio", "split_points.txt")
            files_to_cleanup.append(split_points_file)
        except Exception as e:
            write_log(f"CRITICAL SPLITTING ERROR: {e}")
            write_log(f"Traceback: {str(e)}") # Ideally use traceback.format_exc() but keeping it simple for now or import traceback
            import traceback
            write_log(traceback.format_exc())
            raise e
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
        generation_config = config.GENERATION_CONFIG
        
        safety_settings = config.SAFETY_SETTINGS

        system_instruction = config.SYSTEM_INSTRUCTION

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
            
            write_log(f"Processing Part {i+1}/{len(chunks_to_process)}: {chunk_name} (Offset: {chunk_offset_ms}ms)")

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
            
            # Use chat session to allow for follow-up corrections
            chat = model.start_chat(history=[])
            
            prompt_content = [audio_file, "Generate SRT subtitles for this audio."]
            
            response = None
            max_retries = config.MAX_RETRIES
            
            for attempt in range(max_retries):
                try:
                    if status_callback and attempt > 0:
                        status_callback(f"Rate limit hit. Retrying part {i+1} in {20 * (attempt+1)} seconds...")
                    
                    # For the first attempt, we send the audio and prompt.
                    # For retry on rate limit (exception), we might need to resend?
                    # Actually, if send_message fails, the history isn't updated, so valid to retry send_message.
                    response = chat.send_message(prompt_content)
                    break 
                    
                except Exception as e:
                    if "429" in str(e) or "Resource exhausted" in str(e):
                        if attempt < max_retries - 1:
                            wait_time = 20 * (attempt + 1)
                            print(f"Warning: Resource exhausted (429). Retrying in {wait_time}s...")
                            
                            write_log(f"RATE LIMIT HIT | Part: {i+1} | Attempt: {attempt+1}")
                            write_log(f"Waiting {wait_time}s due to error: {e}")

                            time.sleep(wait_time)
                            continue
                        else:
                             raise Exception(f"Failed after {max_retries} attempts due to rate limits. Please try again later.")
                    else:
                        raise e 
            
            # --- VALIDATION & CORRECTION ---
            # Check if valid SRT (has timestamps)
            if response and response.text:
                # Basic check: does it contain at least one timestamp arrow?
                if "-->" not in response.text:
                    print(f"Warning: Part {i+1} response missing timestamps. Requesting immediate correction from Gemini...")
                    write_log(f"WARNING: Invalid SRT format (no timestamps) for Part {i+1}. Requesting correction.")

                    if status_callback:
                        status_callback(f"Part {i+1} format invalid (missing timestamps). Asking model to fix...")
                    
                    try:
                        # Send correction request
                        correction_prompt = (
                            "The previous output was incorrect because it is missing the SRT timestamps. "
                            "Please regenerate the ENTIRE output for this audio part in valid SRT format. "
                            "Every block MUST have an index, a timestamp line (00:00:00,000 --> 00:00:00,000), and the text."
                        )
                        response = chat.send_message(correction_prompt)
                        # We trust the retry fixed it. We could loop this but let's do one strong correction.
                        if "-->" not in response.text:
                             print(f"Error: Model failed to fix timestamps even after correction.")
                             write_log("ERROR: Correction failed. Still no timestamps.")
                    except Exception as e:
                        print(f"Failed to send correction request: {e}")
                        write_log(f"ERROR Sending correction request: {e}")

            # --- DEBUG LOGGING for Result ---
            finish_reason = "Unknown"
            if response.candidates:
                 finish_reason = response.candidates[0].finish_reason.name
            
            write_log(f"Generation Complete | Finish Reason: {finish_reason}")
            write_log(f"Is Correction: {'Yes' if len(chat.history) > 2 else 'No'}")
            
            if response.usage_metadata:
                write_log(f"Tokens: Prompt={response.usage_metadata.prompt_token_count}, Cand={response.usage_metadata.candidates_token_count}, Total={response.usage_metadata.total_token_count}")
            
            try:
                # Log a snippet or full text? User wants raw text length at least.
                write_log(f"Raw Text Length: {len(response.text)} chars")
                write_log(f"--- Raw Response Preview ---\n{response.text[:500]}...\n--- End Preview ---")
            except Exception as e:
                write_log(f"Could not log text: {e}")
            # ---------------------
            
            raw_srt = ""
            try:
                raw_srt = response.text
                raw_srt = raw_srt.replace("```srt", "").replace("```", "").strip()
            except Exception as e:
                 print(f"Error accessing response.text: {e}")
                 write_log(f"Error accessing response.text: {e}")
                 # Check blocked
                 if response.prompt_feedback:
                     print(f"Prompt Feedback: {response.prompt_feedback}")
                     write_log(f"Prompt Feedback: {response.prompt_feedback}")
            
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
                 write_log(f"Warning: Empty response for part {i+1}")
            
            # Cleanup immediately after use to assume fresh state for next iteration
            # (Though keeping them is fine, deleting minimizes storage use during long process)
            # But the user might want to debug audio... let's stick to cleaning up at the very end
            # or we can delete the remote file resource to keep quotas clean?
            # genai.delete_file(audio_file.name) # If the SDK supports it easily, but let's just leave it for now.
            pass
            
            # Rate limiting cooldown
            if i < len(chunks_to_process) - 1:
                print("Waiting 0.1 second before processing next chunk...")
                time.sleep(0.1)

        if status_callback:
             status_callback("All parts processed. Stitching timestamps...")

        full_srt = "\n\n".join(final_srt_parts)
        write_log("Transcription process finished successfully.")
        return full_srt

    except Exception as e:
        write_log(f"CRITICAL ERROR: {e}")
        import traceback
        write_log(f"Traceback:\n{traceback.format_exc()}")
        return f"Error during transcription process: {str(e)}"
    finally:
        # If we created temp chunks, clean them up
        if files_to_cleanup:
            cleanup_files(files_to_cleanup)

def cleanup_files(file_paths):
    cleaned_dirs = set()
    for path in file_paths:
        if os.path.exists(path):
            try:
                os.remove(path)
                cleaned_dirs.add(os.path.dirname(path))
            except Exception as e:
                print(f"Error removing {path}: {e}")
    
    # Try to remove empty directories
    for d in cleaned_dirs:
        if os.path.exists(d) and not os.listdir(d):
            try:
                os.rmdir(d)
                print(f"Removed empty directory: {d}")
            except OSError:
                pass # Directory not empty
