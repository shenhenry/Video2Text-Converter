# 🎬 Video to SRT Converter

A Streamlit-based application that extracts audio from video files and uses Google's Gemini AI to generate accurate SRT subtitles.

## 🚀 Features

- **Video & Audio Support**: Upload MP4, MOV, AVI, MP3, WAV, and more.
- **Accurate Transcriptions**: Powered by Google Gemini 1.5/Pro models.
- **Smart Timestamping**: 
  - Automatically splits long videos into 10-minute chunks to maintain timestamp accuracy.
  - Fixes common "drift" issues associated with long-context LLM transcriptions.
- **SRT Export**: Generated subtitles are formatted as standard SRT files ready for download.
- **Interactive UI**: Preview video and subtitles directly in the browser.

## 🛠 Prerequisites

- Python 3.9+
- A Google Gemini API Key. Get one [here](https://aistudio.google.com/app/apikey).
- [FFmpeg](https://ffmpeg.org/) (Required by `moviepy` for audio extraction).

## 📦 Installation

1. **Clone the repository** (or download the files):
   ```bash
   git clone <your-repo-url>
   cd "Video2Text Converter"
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🏃‍♂️ Usage

1. **Run the application**:
   ```bash
   streamlit run app.py
   ```

2. **Open your browser**:
   The app usually opens automatically at `http://localhost:8501`.

3. **Generate Subtitles**:
   - Enter your **Google Gemini API Key** in the sidebar.
   - Select a model (e.g., `gemini-1.5-flash`).
   - Upload your video or audio file.
   - Click **Generate Subtitles**.
   - Download the resulting `.srt` file.

## 📝 Notes

- **Long Files**: For files longer than 10 minutes, the app automatically splits them into segments to ensure high accuracy. This process may take a little longer but guarantees better sync.
- **Privacy**: Files are uploaded to Google Gemini for processing and should be handled according to Google's data privacy policies.

## 📄 License

This project is open source.
