# Configuration parameters for Video2Text Converter

# Model Settings
# Default model to use if none is specified
DEFAULT_MODEL_NAME = "models/gemini-flash-latest" 

# List of preferred models for the UI dropdown
PREFERRED_MODELS = [
    "models/gemini-flash-latest",
    "models/gemini-2.5-flash",
    "models/gemini-2.5-pro",
    "mosels/gemini-2.0-flash"
]

# Audio Processing Settings
# Duration in seconds for each audio chunk
CHUNK_DURATION_SEC = 60

# Threshold to trigger splitting. If audio is shorter than this, it won't be split.
SPLIT_THRESHOLD_SEC = 60

# Split Method
# If True, use Gemini to find the optimal split point (more accurate but slower).
# If False, use simple silence detection (RMS).
USE_GEMINI_SPLIT = False

# Prompt for finding the best split point in an audio clip
SPLIT_POINT_PROMPT = (
    "Listen to this audio clip. I need to cut the audio file here. "
    "Find the best timestamp (in milliseconds) to make a cut, such as a moment of silence or the end of a sentence. "
    "Return ONLY the number (e.g. 5430). If no good point, return the middle of the duration."
)

# API Retry Settings
MAX_RETRIES = 3
RETRY_DELAY_BASE_SEC = 20

# Logistics
SAVE_LOG = True  # Whether to save debug logs to a file

# Generation Config
GENERATION_CONFIG = {
    "temperature": 0.1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

# Safety Settings - Block nothing to ensure all content is transcribed
SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

# System Instruction for the Gemini Model
SYSTEM_INSTRUCTION = """You are a professional subtitle creator. Generate a valid SRT file content for the audio provided.
Rules:
1. The output must be strictly in SRT format, starting from 1 (should restrictly follow this format).
   - example:
   1
   0 --> 5000
   This is the first subtitle block.
   
   2
   5000 --> 10000
   This is the second subtitle block.
2. TIMESTAMPS: 
   - Timestamps must be in MILLISECONDS (integer).
   - EXAMPLE: 5000 --> 10000
   - DO NOT use HH:MM:SS,mmm format. Use raw milliseconds.
   - Timestamps must strictly correspond to the exact time the text is spoken in the audio segment.
3. SEGMENTATION RULES (CRITICAL):
   - Strict Split Rule: You MUST start a new subtitle block immediately upon encountering any of these punctuation marks: ",", ".", "!", "?", ":", ";", "，", "。", "！", "？", "：", "；". Do NOT combine text across these punctuation marks into the same block.
   - Exception (Enumeration): The ONLY punctuation mark that allows merging is the enumeration comma "、". If segments are separated by "、", you may keep them in the same block IF AND ONLY IF the total length of the block remains 15 characters or less.
     - Example (Merge allowed): "蘋果、香蕉、梨子" (Short enough -> 1 block)
     - Example (Must split): "這是一個非常非常長的列舉項目一、這是一個非常非常長的列舉項目二" (Too long -> Split at "、")
   - Formatting Constraint: Do NOT use newlines within a single subtitle block.
   - General Length Limit: Even without punctuation, try to keep blocks under 15 characters.
4. Do not include any markdown code blocks, just the raw SRT content.
5. LANGUAGE STRICTNESS:
   - If the audio is in Chinese, you MUST transcribe in Traditional Chinese (繁體中文). Do NOT use Simplified Chinese.
   - If the audio is in English, transcribe in English.
   - Do not translate between languages, just transcribe what is heard, but ensure specific Chinese characters are Traditional."""