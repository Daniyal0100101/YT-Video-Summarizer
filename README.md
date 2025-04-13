# 📽️ YouTube Summarizer (AI-Powered)

A web app that summarizes YouTube videos using the Gemini AI API. Just paste a YouTube link and get an instant, human-like summary with optional follow-up questions.

---

## 🚀 Features

- 🔗 Paste YouTube video link
- 🧠 Summarizes using Gemini AI
- 📄 Displays transcript + summary
- 🤖 Ask follow-up questions
- 🌐 Built with FastAPI + HTML/CSS/JS

---

## 🛠️ Tech Stack

- **Backend**: FastAPI
- **Frontend**: Static HTML/CSS/JS
- **AI**: Gemini API (Google AI)
- **Hosting**: [Render](https://render.com) (free-tier friendly)

---

## 📦 Setup Instructions

```bash
# 1. Clone the repo
git clone https://github.com/your-username/youtube-summarizer.git
cd youtube-summarizer

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your API key
cp .env.example .env
# Then edit .env and add your Gemini API key

# 4. Run the app
python app.py
