# Deployment Guide for Video2Text Converter

Your app is designed to run on **Streamlit Community Cloud**. Since you effectively already have a repository and a live link, "publishing" simply means updating the code on GitHub.

## 🚀 How to Update Your App

To update the live application at `https://video2text-converter.streamlit.app/`:

1.  **Save your changes** locally.
2.  **Commit and Push** to GitHub:
    ```bash
    git add .
    git commit -m "Update application with latest changes"
    git push origin main
    ```
3.  **Wait**: Streamlit Cloud detects the push and automatically redeploys the app. This usually takes 1-2 minutes.
4.  **Refresh**: Go to your URL and refresh to see the changes.

## 🔑 Application Secrets (API Key)

For your app to work on the web, it needs your **Gemini API Key**. It cannot read from your local `.env` file.

1.  Go to your app dashboard: [share.streamlit.io](https://share.streamlit.io/)
2.  Find your app "Video2Text Converter".
3.  Click the **three dots** (⋮) next to your app -> **Settings**.
4.  Click on **Secrets**.
5.  Paste your secrets in the TOML format:

    ```toml
    GEMINI_API_KEY = "your_actual_api_key_here"
    ```

6.  Click **Save**. The app might restart.

## 📦 Dependencies

The file `packages.txt` ensures `ffmpeg` is installed on the server.
The file `requirements.txt` ensures Python libraries like `moviepy` and `google-generativeai` are installed.
