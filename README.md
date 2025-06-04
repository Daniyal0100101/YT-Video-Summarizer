# YouTube Video Summarizer (Powered by Gemini AI)

A fast, clean, and AI-powered web app that summarizes YouTube videos using Gemini AI. Paste any link, get an instant summary, view the transcript, and ask smart follow-up questions.

---

## Features

- Paste any YouTube video URL
- AI-generated, human-like summary (Gemini API)
- Transcript viewer
- Ask follow-up questions about the content
- Built with FastAPI, TailwindCSS, and vanilla JavaScript

---

## Tech Stack

| Layer       | Stack                        |
|-------------|------------------------------|
| Frontend    | HTML, TailwindCSS, JavaScript |
| Backend     | FastAPI (Python)             |
| AI Model    | Gemini 2.0 Flash (Google AI) |
| Video Tools | YouTubeTranscriptAPI, yt-dlp |
| Deployment  | Optional: Render, Railway, etc. |

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/Daniyal0100101/YT-Video-Summarizer.git
cd YT-Video-Summarizer
```

### 2. (Optional) Create a virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

```bash
copy .env.example .env
```

Edit `.env` and add your actual Gemini API key:

```
GEMINI_API_KEY=your-api-key-here
```

### 5. Run the app

```bash
python app.py
```

Then open your browser and visit:

```
http://localhost:8000
```

---

## Project Structure

```
YT-Video-Summarizer/
│
├── app.py                 # FastAPI backend
├── static/                # Frontend (HTML/CSS/JS)
│   ├── index.html
│   ├── style.css
│   └── script.js
├── .env.example           # Environment variable template
├── .gitignore             # Ignoring .env and .venv
├── requirements.txt       # Python dependencies
└── README.md
```

---

## Environment Variables

| Variable         | Description                 |
|------------------|-----------------------------|
| `GEMINI_API_KEY` | Required for Gemini AI access |

---

## Optional Improvements

- Add summary export (PDF or Markdown)
- Support multilingual transcripts
- CI/CD pipeline for auto-deploy

---

## License

All rights reserved. You must obtain explicit, written permission from the author before using, copying, modifying, or distributing any part of this project.

Created by [@Daniyal0100101](https://github.com/Daniyal0100101)
