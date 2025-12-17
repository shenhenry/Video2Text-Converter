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

# API Retry Settings
MAX_RETRIES = 3
RETRY_DELAY_BASE_SEC = 20

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
1. The output must be strictly in SRT format, starting from 1.
2. TIMESTAMPS: 
   - Timestamps must strictly correspond to the exact time the text is spoken in the audio segment.
3. SEGMENTATION RULES (CRITICAL):
   - Default: Create a new subtitle block for every sentence (split by punctuation).
   - Merge Condition: If two or more consecutive short sentences/phrases have a combined length of 10 characters or less, you CAN put them in the same subtitle block (same timestamp).
   - Split Condition: If a merged line would exceed 10 characters, start a new subtitle block.
4. Do not include any markdown code blocks, just the raw SRT content.
5. LANGUAGE STRICTNESS:
   - If the audio is in Chinese, you MUST transcribe in Traditional Chinese (繁體中文). Do NOT use Simplified Chinese.
   - If the audio is in English, transcribe in English.
   - Do not translate between languages, just transcribe what is heard, but ensure specific Chinese characters are Traditional."""
