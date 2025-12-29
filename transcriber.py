import os, pathlib
import sys
import google.generativeai as genai
from moviepy.editor import VideoFileClip, AudioFileClip
import time
import re
import numpy as np
import librosa
import soundfile
import config

# Define Log File Path Globally
current_dir = os.path.dirname(os.path.abspath(__file__))
LOG_FILE_PATH = os.path.join(current_dir, "transcription_debug.log")

# Redirect stdout and stderr to both terminal and log file
class DualWriter:
    def __init__(self, file_path, original_stream):
        self.file_path = file_path
        self.original_stream = original_stream

    def write(self, message):
        # Write to terminal
        try:
            self.original_stream.write(message)
        except Exception:
            pass # Streamlit or other environment might have issues with original stream
        
        # Write to log file
        if config.SAVE_LOG:
            try:
                with open(self.file_path, "a", encoding="utf-8") as f:
                    f.write(message)
                    f.flush()
                    os.fsync(f.fileno())
            except Exception as e:
                pass 

    def flush(self):
        try:
            self.original_stream.flush()
        except Exception:
            pass

if not hasattr(sys, "_dual_writer_setup"):
    # Ensure we don't wrap if it's already a DualWriter (extra safety)
    if not isinstance(sys.stdout, DualWriter):
        sys.stdout = DualWriter(LOG_FILE_PATH, sys.stdout)
    if not isinstance(sys.stderr, DualWriter):
        sys.stderr = DualWriter(LOG_FILE_PATH, sys.stderr)
    sys._dual_writer_setup = True

def write_log(message):
    """ Writes a message to the debug log with timestamp. Explicitly writes to file to avoid stdout capture issues. """
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    formatted_message = f"[{timestamp}] {message}\n"
    
    sys.stdout.write(formatted_message)
    sys.stdout.flush()

def extract_audio(video_path, audio_path="temp/temp_audio.wav"):
    """Extracts audio from a video file."""
    try:
        os.makedirs(os.path.dirname(audio_path), exist_ok=True)
        video = VideoFileClip(video_path)
        # Use wav (pcm_s16le) and explicit fps to ensure CBR (linear time-sample relationship)
        # pcm_s16le is inherently CBR as it is uncompressed raw PCM.
        video.audio.write_audiofile(audio_path, codec='pcm_s16le', fps=44100)
        return audio_path
    except Exception as e:
        write_log(f"Error extracting audio: {e}")
        return None

def upload_to_gemini(path, mime_type="audio/wav"):
    file = genai.upload_file(path, mime_type=mime_type)
    write_log(f"Uploaded file '{file.display_name}' as: {file.uri}")
    return file

def wait_for_files_active(files):
    """Waits for the uploaded files to be processed."""
    write_log("Waiting for file processing...")
    for name in (file.name for file in files):
        file = genai.get_file(name)
        while file.state.name == "PROCESSING":
            write_log(".", end="", flush=True)
            time.sleep(2)
            file = genai.get_file(name)
        if file.state.name != "ACTIVE":
            raise Exception(f"File {file.name} failed to process")
    write_log("...all files ready")

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
    """Converts SRT timestamp "HH:MM:SS,mmm" or just "milliseconds" to integer milliseconds."""
    try:
        time_str = str(time_str).strip()
        if time_str.isdigit():
            return int(time_str)

        time_str = time_str.replace('.', ',')
        if time_str.count(':') == 3: # 處理像 01:02:03:500 的情況
            last_colon = time_str.rfind(':')
            time_str = time_str[:last_colon] + ',' + time_str[last_colon+1:]
        
        parts = time_str.split(':')
        if len(parts) == 3:
            hours, minutes, seconds = parts
        elif len(parts) == 2:
            hours = 0
            minutes, seconds = parts
        else:
            return 0
            
        if ',' in seconds:
            seconds_val, milliseconds_val = seconds.split(',')
            # Pad or truncate milliseconds to 3 digits conceptually? 
            ms_len = len(milliseconds_val)
            ms_int = int(milliseconds_val)
            if ms_len == 1: ms_int *= 100
            elif ms_len == 2: ms_int *= 10
        else:
            seconds_val = seconds
            ms_int = 0
            
        total_ms = (int(hours) * 3600000) + (int(minutes) * 60000) + (int(seconds_val) * 1000) + ms_int
        return total_ms
    except ValueError:
        write_log(f"time_to_ms(): Invalid time string: {time_str}")
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
    if not srt_content:
        write_log(f'No SRT content provided')
        return None

    # Define Pattern: HH:MM:SS,mmm --> HH:MM:SS,mmm (Optional Text)
    # The regex now captures 3 groups: start_time, end_time, and SAME-LINE text (if any)
    # Updated to support optional hours and [.,:] separator, aligning with organize_srt_format
    timestamp_pattern = re.compile(r'((?:\d{1,2}:){1,2}\d{1,2}(?:[.,:]\d{1,3})?)\s-->\s((?:\d{1,2}:){1,2}\d{1,2}(?:[.,:]\d{1,3})?)(.*)')
    # Find all matches
    matches = list(timestamp_pattern.finditer(srt_content))
    write_log(f'shift_srt_content(): Found {len(matches)} matches in SRT content')
    valid_blocks = []
    
    for i, match in enumerate(matches):
        start_str = match.group(1)
        end_str = match.group(2)
        inline_text = match.group(3).strip() # Capture text on the same line
        
        start_ms = time_to_ms(start_str)
        end_ms = time_to_ms(end_str)
        
        # Get content between this match and next match (or end of string)
        content_start = match.end()
        if i < len(matches) - 1:
            content_end = matches[i+1].start()
        else:
            content_end = len(srt_content)
            
        raw_text = srt_content[content_start:content_end]
        
        # If there was text on the same line as the timestamp, treat it as the first line of content
        if inline_text:
            write_log(f'shift_srt_content(): Inline text found in section {i+1}, inline_text: {inline_text}, raw_text: {raw_text}')
            raw_text = "\n" + raw_text
            write_log(f'shift_srt_content(): Inline text found in section {i+1}, modified raw text: {raw_text}')
        
        # Process lines to remove indices and empty space
        lines = [l.strip() for l in raw_text.split('\n') if l.strip()]
        
        # Filter out the index of the next block if present (heuristic: last line is digit)
        if lines and lines[-1].isdigit():
            write_log(f'shift_srt_content(): Section {i+1} removing index line: {lines[-1]}')
            lines.pop()
            
        # If no text remains, it was an empty/hallucinated block
        if not lines:
            write_log(f'shift_srt_content(): No valid SRT content found in section {i+1}.')
            continue
            
        text_content = "\n".join(lines)
        
        # Apply shift
        new_start = ms_to_time(start_ms + offset_ms)
        new_end = ms_to_time(end_ms + offset_ms)

        # Filter duplicate consecutive blocks (hallucination loop) where timestamps are identical
        if valid_blocks:
            last_start, last_end, _ = valid_blocks[-1]
            
            if new_start == last_start and new_end == last_end:
                write_log(f'shift_srt_content(): bypass duplicate block: {start_str} - {end_str}')
                continue
        
        valid_blocks.append((new_start, new_end, text_content))
        
    if not valid_blocks:
        write_log(f'shift_srt_content(): valid_blocks is empty. No valid blocks to return.')
        return None, 0
    
    # Reconstruct SRT
    output = []
    curr = counter_start
    for start, end, text in valid_blocks:
        output.append(str(curr))
        output.append(f"{start} --> {end}")
        output.append(text)
        output.append("")
        curr += 1
        
    if not output:
        write_log("shift_srt_content(): output is empty. No valid output to return.")
        return None, 0
    else:
        return "\n".join(output).strip(), curr - 1

def load_audio_segment(audio_path):
    try:
        data, samplerate = librosa.load(audio_path, sr=None, mono=True)
        write_log(f"Audio loaded via librosa: {len(data)} samples @ {samplerate}Hz")
        return data, samplerate
    except Exception as e:
        write_log(f"Audio load failed (librosa): {e}")
        return None

def find_split_point(audio_data_tuple, win_start_ms, win_end_ms, sliding_window_ms=config.SLIDING_WINDOW_MS):
    try:
        if audio_data_tuple is None or win_start_ms is None or win_end_ms is None:
            write_log(f"find_split_point(): audio_data_tuple or win_start_ms or win_end_ms is None.")
            return (win_start_ms + win_end_ms) // 2 if (win_start_ms is not None and win_end_ms is not None) else 0
        data, fs = audio_data_tuple
        
        write_log(f"find_split_point(): win_start_ms {win_start_ms}ms, win_end_ms {win_end_ms}ms")
        start_idx = max(0, int(win_start_ms * fs // 1000))
        end_idx = min(len(data), int(win_end_ms * fs // 1000))
        search_slice = data[start_idx:end_idx]
        if search_slice.size == 0:
            write_log(f"find_split_point(): search_slice is empty.")
            return (win_start_ms + win_end_ms) // 2

        window_samples = int(sliding_window_ms * fs // 1000)
        if window_samples > search_slice.size:
            window_samples = search_slice.size

        volume = search_slice**2
        energy_sliding = np.convolve(volume, np.ones(window_samples), mode='valid')
        min_energy = np.min(energy_sliding)
        min_indices = np.where(energy_sliding == min_energy)[0]
        min_idx = int(np.median(min_indices))
        
        best_time_relative_ms = ((min_idx + window_samples // 2) / fs) * 1000
        best_time_ms = (start_idx / fs * 1000) + best_time_relative_ms
        
        write_log(f"Sliding window split found at {int(best_time_ms)}ms (Window: {sliding_window_ms}ms)")
        return int(best_time_ms)
        
    except Exception as e:
        write_log(f"Warning: Sliding window detection failed ({e}), using exact time.")
        return (win_start_ms + win_end_ms) // 2

def find_split_point_gemini(api_key, audio_clip, status_callback=None):
    temp_clip_path = "temp/temp_split_search.wav"
    try:
        # Write the search window clip to file
        audio_clip.write_audiofile(temp_clip_path, codec='pcm_s16le', verbose=False, logger=None)
        
        # Configure GenAI
        genai.configure(api_key=api_key.strip())
        
        # Upload
        if status_callback:
            write_log("Uploading search window to Gemini for split analysis...")
            
        file = genai.upload_file(temp_clip_path, mime_type="audio/wav")
        
        # Wait for processing
        while file.state.name == "PROCESSING":
            time.sleep(1)
            file = genai.get_file(file.name)
            
        if file.state.name != "ACTIVE":
            raise Exception("Gemini file processing failed")

        model = genai.GenerativeModel(model_name=config.DEFAULT_MODEL_NAME)
        
        prompt = config.SPLIT_POINT_PROMPT
        
        response = model.generate_content([file, prompt])
        
        # Parse response
        text = response.text.strip()
        # Extract number (Gemini returns milliseconds as per prompt)
        match = re.search(r"(\d+(\.\d+)?)", text)
        if match:
             # Already in ms
             return int(float(match.group(1)))
        
        return int(audio_clip.duration * 1000 / 2)

    except Exception as e:
        write_log(f"Gemini split detection failed: {e}. Falling back to center.")
        return int(audio_clip.duration * 1000 / 2)
    finally:
        if os.path.exists(temp_clip_path):
            os.remove(temp_clip_path)

def split_audio(audio_path, chunk_duration_ms=config.CHUNK_DURATION_SEC*1000, status_callback=None, api_key=None):
    audio = AudioFileClip(audio_path)
    audio_duration_ms = int(audio.duration * 1000)
    
    try:
        pydub_audio = librosa.load(audio_path, sr=None, mono=True)
        write_log(f"Audio loaded via librosa: {len(pydub_audio[0])} samples @ {pydub_audio[1]}Hz")
    except Exception as e:
        write_log(f"Audio load failed (librosa): {e}")
        pydub_audio = None

    temp_dir = pathlib.Path("temp")
    temp_dir.mkdir(exist_ok=True)

    [f.unlink() for f in temp_dir.glob("*") if f.resolve() != pathlib.Path(audio_path).resolve()]

    split_points_ms, curr_ms = [], 0

    while curr_ms < audio_duration_ms:
        win_start_ms = int(curr_ms + (chunk_duration_ms * 0.8))
        win_end_ms = int(curr_ms + (chunk_duration_ms * 1.2))
        #write_log(f"split_audio(): win_start_ms {win_start_ms}ms, win_end_ms {win_end_ms}ms")
        
        if win_start_ms >= audio_duration_ms:
            actual_end_ms = audio_duration_ms
        else:
            win_end_ms = min(audio_duration_ms, win_end_ms)

            if api_key and config.USE_GEMINI_SPLIT:
                win_start_sec = win_start_ms / 1000.0
                win_end_sec = win_end_ms / 1000.0
                found_relative_ms = find_split_point_gemini(api_key, audio.subclip(win_start_sec, win_end_sec), status_callback)
                actual_end_ms = win_start_ms + found_relative_ms
            else:
                if pydub_audio:
                     actual_end_ms = find_split_point(pydub_audio, win_start_ms, win_end_ms)
                else:
                     actual_end_ms = (win_start_ms + win_end_ms) // 2
        
        if actual_end_ms <= curr_ms + 1000: 
            actual_end_ms = min(curr_ms + chunk_duration_ms, audio_duration_ms)
            
        split_points_ms.append(actual_end_ms)
        curr_ms += chunk_duration_ms
        write_log(f"split_audio(): curr_ms becomes {curr_ms}ms")

    chunks = []
    prev_start_ms = 0
    log_data = [f"Total MS: {audio_duration_ms}\nTarget MS: {chunk_duration_ms}\n" + "-"*20]

    for idx, end_t_ms in enumerate(split_points_ms):
        if status_callback: status_callback(f"Part {idx+1}/{len(split_points_ms)}...")
        
        prev_start_sec = prev_start_ms / 1000.0
        end_t_sec = end_t_ms / 1000.0
        
        # Use "temp_partX.wav" naming pattern
        p = temp_dir / f"temp_part{idx}.wav"
        # Enforce CBR and fixed sample rate for chunks as well
        audio.subclip(prev_start_sec, end_t_sec).write_audiofile(str(p), codec='pcm_s16le', fps=44100, verbose=False, logger=None)
        
        # Store metadata in MS
        chunks.append({"path": str(p), "start_time_ms": prev_start_ms, "duration_ms": end_t_ms - prev_start_ms})
        log_data.append(f"Chunk {idx+1} End: {end_t_ms}ms")
        prev_start_ms = end_t_ms

    (temp_dir / "split_points.txt").write_text("\n".join(log_data))
    audio.close()
    return chunks

def organize_srt_format(file_path, output_path=None):
    path = pathlib.Path(file_path)
    if not path.exists():
        return

    try:
        content = path.read_text(encoding="utf-8")
        lines = content.splitlines()
        header = ""
        body_lines = lines
        
        # Preserve Start time / End time header (or old Start Offset)
        if lines and (lines[0].startswith("Start time:") or lines[0].startswith("Start Offset:")):
            header = lines[0]
            body_lines = lines[1:]
        
        raw_body = "\n".join(body_lines).strip()
        
        # Use regex that matches EITHER standard HH:MM:SS,mmm OR raw integer string
        timestamp_pattern = re.compile(
            r'(\d+(?::\d{1,2})*(?:[.,]\d{1,3})?)\s-->\s(\d+(?::\d{1,2})*(?:[.,]\d{1,3})?)'
        )
        
        matches = list(timestamp_pattern.finditer(raw_body))
        
        if not matches:
            write_log(f"organize_srt_format(): No timestamps found in {file_path}, skipping organization.")
            return

        # Parse chunk limit from header to calculate duration
        limit_ms = 0
        if header.startswith("Start time:"):
            try:
                # Support both HH:MM:SS,mmm AND raw integer (ms)
                start_match = re.search(r"Start time:\s*([0-9:.,]+)", header)
                end_match = re.search(r"End time:\s*([0-9:.,]+)", header)
                if start_match and end_match:
                    s_str, e_str = start_match.group(1), end_match.group(1)
                    s_val = time_to_ms(s_str) if ":" in s_str else int(s_str)
                    e_val = time_to_ms(e_str) if ":" in e_str else int(e_str)
                    limit_ms = e_val - s_val
            except Exception as e:
                write_log(f"organize_srt_format(): Error parsing limit: {e}")

        blocks = []
        for i, match in enumerate(matches):
            # 1. Parse raw ms values
            s_ms = time_to_ms(match.group(1))
            e_ms = time_to_ms(match.group(2))
            
            # 2. If last block, check if it exceeds chunk end time
            if i == len(matches) - 1 and limit_ms > 0:
                if e_ms > limit_ms:
                    write_log(f"organize_srt_format(): Capping last block {e_ms}ms -> {limit_ms}ms")
                    e_ms = limit_ms
            
            # 3. Convert to standardized strings
            start_time_str = ms_to_time(s_ms)
            end_time_str = ms_to_time(e_ms)
            
            # Content is after this match and before the next match
            start_idx = match.end()
            if i < len(matches) - 1:
                raw_segment = raw_body[start_idx:matches[i+1].start()]
            else:
                raw_segment = raw_body[start_idx:]
                
            # Clean the text segment
            segment_lines = raw_segment.split('\n')
            clean_lines = []
            for line in segment_lines:
                line = line.strip()
                if not line: continue
                if line.isdigit(): continue
                # Exclude potential source markers or residual timestamps
                if "-->" in line: continue 
                clean_lines.append(line)
            
            text_content = "\n".join(clean_lines)
            blocks.append((start_time_str, end_time_str, text_content))
            
        # Reconstruct file content
        new_content_parts = []
        if header:
            new_content_parts.append(header)
            new_content_parts.append("") # Empty line separate header
            
        for idx, (start, end, text) in enumerate(blocks, 1):
            new_content_parts.append(str(idx))
            new_content_parts.append(f"{start} --> {end}")
            new_content_parts.append(text)
            new_content_parts.append("") # Blank line
            
        final_content = "\n".join(new_content_parts).strip()
        
        # Determine write target
        target_path = pathlib.Path(output_path) if output_path else path
        
        # Ensure one newline at end? standardize to no newline at very end
        target_path.write_text(final_content, encoding="utf-8")
        write_log(f"Organized SRT format from {file_path} to {target_path}. Total blocks: {len(blocks)}")
        
    except Exception as e:
        write_log(f"Error in organize_srt_format for {file_path}: {e}")

def merge_srt_parts(new_chunks_info, status_callback=None):
    final_srt_path = pathlib.Path("temp/final_context.srt")
    
    # Iterate through the new chunks to merge
    for i, chunk in enumerate(new_chunks_info):
        chunk_path = chunk['path']
        # The path here might be the fake mp3 path we created to trick stem logic, e.g. .../temp_modified_partX.mp3
        chunk_stem = pathlib.Path(chunk_path).stem # temp_modified_partX
        
        # We expect the SRT to be at .../temp_modified_partX.srt
        temp_srt_path = pathlib.Path(chunk_path).parent / f"{chunk_stem}.srt"
        
        if not temp_srt_path.exists():
            write_log(f"Error: Missing temp SRT file: {temp_srt_path}")
            continue
        write_log(f"Merging chunk into context: {temp_srt_path}")

        try:
            # 1. Read the new chunk content
            content = temp_srt_path.read_text(encoding="utf-8")
            if not content.strip():
                write_log(f"Warning: SRT file {temp_srt_path} is empty.")
                continue

            # Parse Offset
            lines = content.split('\n')
            offset_ms = 0
            raw_body = content
            
            if lines:
                if lines[0].startswith("Start time:"):
                    # Handle "Start time: HH:MM:SS,mmm" OR "Start time: 12345"
                    match = re.search(r"Start time:\s*([0-9:.,]+)", lines[0])
                    if match:
                        val_str = match.group(1)
                        offset_ms = time_to_ms(val_str) if ":" in val_str else int(val_str)
                        write_log(f"  > Start time parsed: {val_str} -> {offset_ms}ms")
                    
                    parts = content.split('\n\n', 1)
                    raw_body = parts[1] if len(parts) > 1 else "\n".join(lines[2:]) if len(lines) > 2 else ""
                elif lines[0].startswith("Start Offset:"):
                    offset_str = lines[0].replace("Start Offset:", "").strip()
                    try:
                        offset_ms = int(offset_str)
                        write_log(f"  > Start Offset parsed: {offset_str}ms")
                    except:
                        try:
                            offset_ms = time_to_ms(offset_str)
                            write_log(f"  > Start Offset (time) parsed: {offset_str} -> {offset_ms}ms")
                        except Exception as e:
                            write_log(f"  > Error parsing offset '{offset_str}': {e}")
                    
                    parts = content.split('\n\n', 1)
                    raw_body = parts[1] if len(parts) > 1 else "\n".join(lines[2:]) if len(lines) > 2 else ""
                else:
                    write_log(f"  > No recognized header found. Treating file as raw SRT.")

            # 2. Determine starting counter
            current_counter = 1
            existing_content = ""
            
            if final_srt_path.exists():
                existing_content = final_srt_path.read_text(encoding="utf-8")
                indices = re.findall(r'\n(\d+)\n\d{2}:\d{2}', existing_content)
                if not indices:
                    match = re.match(r'^(\d+)\n', existing_content)
                    if match:
                         indices = [match.group(1)]
                
                if indices:
                    current_counter = int(indices[-1]) + 1
            
            write_log(f"  > Current counter sequence starts at: {current_counter}")

            # 3. Process the new chunk (Shift & Renumber)
            # IMPORTANT: The content in temp_modified matches is ALREADY Standard Format. 
            # shift_srt_content handles Standard Format -> MS shift -> Standard Format.
            write_log(f"  > Shifting content with offset {offset_ms}ms...")
            part_srt, last_counter = shift_srt_content(raw_body, offset_ms, counter_start=current_counter)
            
            # 4. Write back
            if part_srt:
                if existing_content:
                    new_content = existing_content + "\n\n" + part_srt
                else:
                    new_content = part_srt
                
                final_srt_path.write_text(new_content, encoding="utf-8")
                write_log(f"Updated {final_srt_path} with new chunk data. Last counter: {last_counter}")
            else:
                write_log("Warning: No valid SRT content found in chunk to merge.")

        except Exception as e:
            write_log(f"Failed to merge chunk {temp_srt_path}: {e}")

    return final_srt_path

def transcribe_audio(api_key, audio_path, model_name=config.DEFAULT_MODEL_NAME, status_callback=None):
    """Transcribes audio using Gemini API and returns SRT content."""
    
    # 0. Immediate Logging
    write_log(f"{'-'*30} NEW SESSION {'-'*30}")
    write_log(f"Processing File: {audio_path}")
    write_log(f"Model: {model_name}")

    # Initialize Context File Cleanup
    final_context_path = pathlib.Path("temp/final_context.srt")
    if final_context_path.exists():
        final_context_path.unlink()

    genai.configure(api_key=api_key.strip())

    # 1. Check duration and decide if splitting is needed
    SPLIT_THRESHOLD_SEC = config.CHUNK_DURATION_SEC
    
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
            # CALL WITH MS
            chunks = split_audio(audio_path, chunk_duration_ms=SPLIT_THRESHOLD_SEC*1000, status_callback=status_callback, api_key=api_key)
            chunks_to_process = chunks
        except Exception as e:
            write_log(f"CRITICAL SPLITTING ERROR: {e}")
            write_log(f"Traceback: {str(e)}") # Ideally use traceback.format_exc() but keeping it simple for now or import traceback
            import traceback
            write_log(traceback.format_exc())
            raise e
    else:
        # Fallback to single chunk with MS duration
        chunks_to_process = [{"path": audio_path, "start_time_ms": 0, "duration_ms": int(duration_sec * 1000)}]

    if status_callback and chunks_to_process:
        last_chunk = chunks_to_process[-1]
        # Use duration_ms
        last_duration_min = last_chunk.get("duration_ms", 0) / 60000
        status_callback(f"INFO: Last audio chunk duration: {last_duration_min:.2f} minutes")
        # Small sleep to let the user see the message
        time.sleep(1)

    
    current_counter = 1
    
    try:
        final_srt_parts = []
        current_counter = 1
        
        # Process chunks sequentially
        for i, chunk in enumerate(chunks_to_process):
            chunk_path = chunk['path']
            chunk_name = os.path.basename(chunk_path)
            chunk_stem = pathlib.Path(chunk_path).stem
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
                generation_config=config.GENERATION_CONFIG,
                safety_settings=config.SAFETY_SETTINGS,
                system_instruction=config.SYSTEM_INSTRUCTION
            )
            
            # Use chat session to allow for follow-up corrections
            chat = model.start_chat(history=[])
            
            prompt_content = [audio_file, config.SYSTEM_INSTRUCTION]
            
            response = None
            max_retries = config.MAX_RETRIES
            
            for attempt in range(max_retries):
                try:
                    if status_callback and attempt > 0:
                        status_callback(f"Rate limit hit. Retrying part {i+1} in {20 * (attempt+1)} seconds...")

                    response = chat.send_message(prompt_content)
                    break 
                    
                except Exception as e:
                    if "429" in str(e) or "Resource exhausted" in str(e):
                        if attempt < max_retries - 1:
                            wait_time = 20 * (attempt + 1)
                            write_log(f"Warning: Resource exhausted (429). Retrying in {wait_time}s...")
                            
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
                    write_log(f"Warning: Part {i+1} response missing timestamps. Requesting immediate correction from Gemini...")
                    write_log(f"WARNING: Invalid SRT format (no timestamps) for Part {i+1}. Requesting correction.")

                    if status_callback:
                        status_callback(f"Part {i+1} format invalid (missing timestamps). Asking model to fix...")
                    
                    try:
                        # Send correction request
                        correction_prompt = (
                            "The previous output was incorrect because it is missing the SRT timestamps. "
                            "Please regenerate the ENTIRE output for this audio part in valid SRT format. "
                            "Every block MUST have an index, a timestamp line (HH:MM:SS,mmm --> HH:MM:SS,mmm), and the text."
                        )
                        response = chat.send_message(correction_prompt)
                        # We trust the retry fixed it. We could loop this but let's do one strong correction.
                        if "-->" not in response.text:
                            write_log(f"Error: Model failed to fix timestamps even after correction.")
                    except Exception as e:
                        write_log(f"Failed to send correction request: {e}")

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
                 write_log(f"Error accessing response.text: {e}")
                 # Check blocked
                 if response.prompt_feedback:
                     write_log(f"Prompt Feedback: {response.prompt_feedback}")
            
            if raw_srt:
                # Save temp part SRT with offset info locally
                try:
                    header_str = f"Start time: {chunk_offset_ms} End time: {chunk_offset_ms + chunk.get('duration_ms', 0)}"
                    
                    # 1. Save Raw (temp_raw_partX.srt)
                    temp_raw_filename = f"temp_raw_{chunk_stem.replace('temp_', '')}.srt" 
                    temp_raw_path = pathlib.Path(chunk_path).parent / temp_raw_filename
                    
                    with open(temp_raw_path, "w", encoding="utf-8") as f:
                        f.write(f"{header_str}\n\n{raw_srt}")
                        
                    write_log(f"Saved RAW temp SRT chunk to: {temp_raw_path}")
                    
                    # 2. Organize to Modified (temp_modified_partX.srt)
                    temp_modified_filename = f"temp_modified_{chunk_stem.replace('temp_', '')}.srt" 
                    temp_modified_path = pathlib.Path(chunk_path).parent / temp_modified_filename
                    
                    # Organize/Clean SRT format from RAW to MODIFIED
                    write_log(f"Organizing {temp_raw_path} -> {temp_modified_path}")
                    organize_srt_format(str(temp_raw_path), output_path=str(temp_modified_path))

                    # === IMMEDIATE INCREMENTAL MERGE ===
                    # Call merge immediately with just this chunk's Modified version
                    # We create a pseudo-chunk info that points to the modified file's stem equivalent
                    # merge_srt_parts expects 'path' -> gets parent & stem -> finds .srt
                    # So we construct a path that looks like ".../temp_modified_part0.wav"
                    modified_chunk_info = chunk.copy()
                    modified_chunk_info['path'] = str(pathlib.Path(chunk_path).parent / f"temp_modified_{chunk_stem.replace('temp_', '')}.wav")
                    
                    merge_srt_parts([modified_chunk_info], status_callback=status_callback)
                    
                except Exception as e:
                    write_log(f"Failed to save/merge temp SRT chunk: {e}")
                    import traceback
                    write_log(traceback.format_exc())

            else:
                 write_log(f"Warning: Empty response for part {i+1}")
            
            # (No cleanup here, we need the files)
            pass
            
            # Rate limiting cooldown
            if i < len(chunks_to_process) - 1:
                write_log("Waiting 0.1 second before processing next chunk...")
                time.sleep(0.1)

        # --- FINAL READ ---
        final_file = pathlib.Path("temp/final_context.srt")
        if final_file.exists():
            full_srt = final_file.read_text(encoding="utf-8")
        else:
            full_srt = ""
            
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
                write_log(f"Error removing {path}: {e}")
    
    # Try to remove empty directories
    for d in cleaned_dirs:
        if os.path.exists(d) and not os.listdir(d):
            try:
                os.rmdir(d)
                write_log(f"Removed empty directory: {d}")
            except OSError:
                pass # Directory not empty
