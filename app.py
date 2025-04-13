import os
import logging
import asyncio
import google.generativeai as genai
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, HttpUrl, Field, validator
from typing import List, Dict, Any, Optional
from youtube_transcript_api import YouTubeTranscriptApi
from pytube import YouTube
from dotenv import load_dotenv
import time
from cachetools import TTLCache
import re

# ---------------------------------------------------
# LOGGING CONFIGURATION
# ---------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------
# SETUP & CONFIGURATION
# ---------------------------------------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is missing. Please set it in the .env file.")

# Cache configuration: store up to 100 transcript/summary pairs for 1 hour (3600 seconds)
transcript_cache = TTLCache(maxsize=100, ttl=3600)
summary_cache = TTLCache(maxsize=100, ttl=3600)

# ----------------------------
# 1) TWO SEPARATE SYSTEM PROMPTS
# ----------------------------
SUMMARY_SYSTEM_INSTRUCTION = """
You are an AI assistant specialized in generating **detailed, fact-based summaries** of YouTube videos. Your task is to provide a **comprehensive** yet **concise** breakdown of the video's content, ensuring full coverage of all major points.

### **Guidelines for Generating Summaries**  

1. **Complete and Fact-Based Coverage**  
   - Capture **all essential topics, explanations, and details** covered in the video.  
   - Avoid speculation or adding content not explicitly stated.  
   - If specific details are unclear or missing, focus on summarizing what is available without assuming additional information.  

2. **Well-Structured and Readable Format**  
   - **Introduction:** Clearly describe the main topic and objective of the video.  
   - **Key Points:** Break down all discussed sections, including explanations, instructions, or insights shared by the speaker.  
   - **Conclusion:** Summarize the final message, takeaways, or action items mentioned.  

3. **Context-Rich, Clear Summaries**  
   - Present information in a way that fully represents the depth and intent of the content.  
   - Ensure that **every major concept, process, or explanation given by the speaker is reflected** in the summary.  
   - Use structured formatting (headings, bullet points) for better readability.  

4. **No Speculation or Missing Gaps**  
   - If the speaker provides limited information on a topic, state only what is covered.  
   - Do not infer or assume additional details.  

5. **Professional and Informative Tone**  
   - Keep responses neutral, professional, and easy to understand.  
   - Ensure clarity in technical explanations or instructional content.  

Your role is to ensure **the summary fully reflects the video's content**, providing users with an accurate and complete understanding without adding or missing key details.
"""

FOLLOWUP_SYSTEM_INSTRUCTION = """
You are a specialized AI assistant that answers follow-up questions about YouTube videos.
Your primary task is to provide accurate, detailed responses based on the ORIGINAL video content of the video.

FOLLOW-UP PRINCIPLES:

1. ALWAYS prioritize the original video content as your primary source of information.
   - The video content contains the exact words spoken in the video
   - Use it for direct quotes and specific details
   - When answering questions, search the entire video content for relevant information

2. Use the summary only as a secondary reference for context and structure.
   - The summary provides an overview but may not contain all details
   - When the video content contains more specific information than the summary, use the video content

3. Be precise and factual in your responses:
   - Answer directly based on what is explicitly stated in the video content
   - If information is not in the video content, clearly state this limitation
   - Never invent or assume information not present in the source material

4. When making claims or referencing content from the video, include simplified time references like "(~5:30)" or "(early in the video)" or "(near the end)"
   - Convert raw timestamps to more user-friendly MM:SS format
   - Only include references when citing specific information from the video
   - Do not overuse timestamps - only add them when they add value

5. Format your responses for clarity using:
   - Bullet points for lists
   - Bold text for key points
   - Headings for organizing longer answers

Remember: Your goal is to help users understand the video content as accurately as possible by leveraging the complete video content data.
"""

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

# ----------------------------
# 2) INITIALIZE TWO MODELS
# ----------------------------
try:
    summary_model = genai.GenerativeModel(
        "gemini-2.0-flash-lite",
        system_instruction=SUMMARY_SYSTEM_INSTRUCTION
    )
    followup_model = genai.GenerativeModel(
        "gemini-2.0-flash-lite",
        system_instruction=FOLLOWUP_SYSTEM_INSTRUCTION
    )
    logger.info("Successfully initialized both Gemini models")
except Exception as e:
    logger.error(f"Failed to initialize Gemini models: {str(e)}")
    raise

# ---------------------------------------------------
# FASTAPI APP SETUP
# ---------------------------------------------------
app = FastAPI(
    title="YouTube Video Summarizer API",
    description="API for summarizing YouTube videos and providing interactive follow-up capabilities",
    version="1.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Mount static files BEFORE defining routes
app.mount("/static", StaticFiles(directory="static"), name="static")

# ---------------------------------------------------
# DATA MODELS
# ---------------------------------------------------
class VideoRequest(BaseModel):
    youtube_url: HttpUrl = Field(..., description="Full YouTube video URL")

    @validator('youtube_url')
    def validate_youtube_url(cls, url):
        youtube_regex = r'^.*(?:youtu\.be\/|v\/|u\/\w\/|embed\/|watch\?v=|\&v=)([^#\&\?]*).*'
        match = re.search(youtube_regex, str(url))
        if not match:
            raise ValueError("Invalid YouTube URL format")
        return url

class ConversationMessage(BaseModel):
    role: str = Field(..., description="Role of the message sender (user/assistant)")
    content: str = Field(..., description="Content of the message")

class FollowUpRequest(BaseModel):
    question: str = Field(..., description="User's follow-up question")
    history: List[ConversationMessage] = Field(..., description="Conversation history")
    transcript: str = Field(..., description="Original video transcript")

class VideoResponse(BaseModel):
    title: str
    description: str
    transcript: str
    video_id: str
    duration: Optional[int] = None
    author: Optional[str] = None
    published_date: Optional[str] = None

class SummaryResponse(BaseModel):
    summary: str

class FollowUpResponse(BaseModel):
    response: str

# ---------------------------------------------------
# MIDDLEWARE FOR REQUEST TIMING AND LOGGING
# ---------------------------------------------------
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    logger.info(f"Request to {request.url.path} processed in {process_time:.4f} seconds")
    return response

# ---------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------
def extract_youtube_id(url: str) -> str:
    youtube_regex = r"(?:youtube\.com/(?:[^/]+/.+/|(?:v|e(?:mbed)?)/|.*[?&]v=)|youtu\.be/)([^\"&?/\s]{11})"
    match = re.search(youtube_regex, url)
    if match:
        return match.group(1)
    raise HTTPException(status_code=400, detail="Invalid YouTube URL format.")

def format_timestamp(seconds: float) -> str:
    """Convert raw seconds to MM:SS format for user-friendly references."""
    minutes = int(seconds // 60)
    remaining_seconds = int(seconds % 60)
    return f"{minutes}:{remaining_seconds:02d}"

async def get_transcript(video_id: str) -> str:
    """Fetches the YouTube transcript using the transcript API. Uses cache if available."""
    if video_id in transcript_cache:
        logger.info(f"Transcript cache hit for video {video_id}")
        return transcript_cache[video_id]
    try:
        transcript_data = YouTubeTranscriptApi.get_transcript(video_id)
        # Join only the 'text' entries – timestamps from the API are not used here.
        transcript = "\n".join(f"[{round(item['start'], 2)}s] {item['text']}" for item in transcript_data)
        transcript_cache[video_id] = transcript
        return transcript
    except Exception as e:
        logger.warning(f"Failed to fetch transcript for {video_id}: {str(e)}")
        return "Transcript not available."

async def get_video_details(youtube_url: str) -> Dict[str, Any]:
    """Fetches video metadata from YouTube."""
    video_id = extract_youtube_id(youtube_url)
    try:
        yt = YouTube(youtube_url)
        return {
            "title": yt.title or "Title unavailable",
            "description": yt.description or "Description unavailable",
            "video_id": video_id,
            "duration": yt.length,
            "author": yt.author,
            "published_date": yt.publish_date.strftime("%Y-%m-%d") if yt.publish_date else None
        }
    except Exception as e:
        logger.error(f"Error fetching video details: {str(e)}")
        return {
            "title": "Title unavailable",
            "description": "Description unavailable",
            "video_id": video_id,
            "duration": None,
            "author": None,
            "published_date": None
        }

async def generate_summary(transcript: str, video_id: str) -> str:
    """Uses the summary_model to generate a structured summary from the transcript."""
    if video_id in summary_cache:
        logger.info(f"Summary cache hit for video {video_id}")
        return summary_cache[video_id]
    if not transcript or transcript == "Transcript not available." or len(transcript.split()) < 10:
        return "No meaningful video content available for summarization."

    # Create a prompt that instructs the model not to generate fake timestamps
    prompt = f"""
    Summarize the following YouTube video content concisely:

    VIDEO CONTENT:
    {transcript}

    Instructions:
    - Provide an introduction, key points, and conclusion.
    - Do NOT create or include any timestamps unless they are explicitly present in the transcript.
    - Ensure factual accuracy and do not add information not in the transcript.

    Use markdown formatting for readability.
    """

    try:
        response = await asyncio.to_thread(lambda: summary_model.generate_content(prompt))
        summary = response.text if response.text else "Error: No response from Gemini."
        summary_cache[video_id] = summary
        return summary
    except Exception as e:
        error_msg = f"Error generating summary: {str(e)}"
        logger.error(error_msg)
        return error_msg

async def generate_followup_response(question: str, formatted_convo: str, transcript: str) -> str:
    """Uses the followup_model to handle follow-up Q&A based on the conversation and transcript."""
    try:
        # Preprocess the transcript to convert timestamps to user-friendly format
        processed_transcript = transcript
        if transcript and transcript != "Transcript not available.":
            # Look for timestamp patterns like [123.45s] and convert them
            timestamp_pattern = r"\[(\d+\.\d+)s\]"
            def replace_timestamp(match):
                seconds = float(match.group(1))
                return f"[{format_timestamp(seconds)}]"
            processed_transcript = re.sub(timestamp_pattern, replace_timestamp, transcript)
        
        # Create a structured prompt that emphasizes using the transcript and not fabricating timestamps.
        prompt = f"""ORIGINAL VIDEO TRANSCRIPT:
{processed_transcript}

CONVERSATION HISTORY:
{formatted_convo}

USER QUESTION: {question}

Instructions:
1. Answer the question using primarily the ORIGINAL TRANSCRIPT as your source of truth.
2. Be specific and cite information directly from the transcript when possible.
3. If the information isn't in the transcript, clearly state this limitation.
4. Format your response for readability (bullet points, bold for key points, etc.).
5. When referencing specific content from the video:
   - Include a simple time reference in (~MM:SS) format when you quote or reference specific content
   - Only include time references when citing important information
   - References should look like "(~2:45)" or "(at around 15:30)" - keep them simple and user-friendly
   - Use general references like "(early in the video)" when exact timestamps aren't available
   - Whenever you cite important information, include a reference to when it appears in the video

Remember: the transcript contains the exact words spoken in the video and should be your primary source of information.
"""
        response = await followup_model.generate_content_async(prompt)
        return response.text
    except Exception as e:
        logger.error(f"Error generating follow-up response: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate follow-up response: {str(e)}")

# ---------------------------------------------------
# ROUTES
# ---------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def serve_index():
    static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
    file_path = os.path.join(static_dir, "index.html")
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return HTMLResponse(content=content)
    raise HTTPException(status_code=404, detail="Index file not found.")

@app.post("/api/video-data", response_class=JSONResponse)
async def fetch_video_data(request: VideoRequest):
    """Fetches video details (title, description, and transcript) from YouTube."""
    try:
        logger.info(f"Received video-data request for URL: {request.youtube_url}")
        youtube_url = str(request.youtube_url)
        video_id = extract_youtube_id(youtube_url)
        
        video_details, transcript = await asyncio.gather(
            get_video_details(youtube_url),
            get_transcript(video_id)
        )
        
        result = {**video_details, "transcript": transcript}
        logger.info(f"Successfully processed video data for ID: {video_id}")
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error in fetch_video_data: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"detail": f"Failed to process video: {str(e)}"}
        )

@app.post("/api/generate-summary", response_class=JSONResponse)
async def generate_summary_endpoint(request: VideoRequest):
    """Generates a summary for the given YouTube video using the summary_model."""
    try:
        logger.info(f"Received generate-summary request for URL: {request.youtube_url}")
        youtube_url = str(request.youtube_url)
        video_id = extract_youtube_id(youtube_url)
        
        transcript = await get_transcript(video_id)
        if transcript == "Transcript not available.":
            return JSONResponse(content={"summary": "No video content available for this video."})
            
        summary = await generate_summary(transcript, video_id)
        logger.info(f"Successfully generated summary for video ID: {video_id}")
        return JSONResponse(content={"summary": summary})
    except Exception as e:
        logger.error(f"Error in generate_summary_endpoint: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"detail": f"Failed to generate summary: {str(e)}"}
        )

@app.post("/api/follow-up", response_class=JSONResponse)
async def follow_up_questions(request: FollowUpRequest):
    """
    Handles follow-up Q&A based on previous conversation history and the original transcript.
    We use both the conversation content and transcript to provide accurate answers.
    """
    try:
        logger.info(f"Received follow-up request with question: {request.question}")
        formatted_convo = "\n".join([
            f"{msg.role.upper()}: {msg.content}" 
            for msg in request.history
        ])
        
        response = await generate_followup_response(
            request.question, 
            formatted_convo,
            request.transcript
        )
        
        logger.info("Successfully generated follow-up response")
        return JSONResponse(content={"response": response})
    except Exception as e:
        logger.error(f"Error in follow_up_questions: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"detail": f"Failed to generate follow-up response: {str(e)}"}
        )

# ---------------------------------------------------
# STATIC FILES
# ---------------------------------------------------
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# ---------------------------------------------------
# APP STARTUP EVENTS
# ---------------------------------------------------
@app.on_event("startup")
async def startup_event():
    logger.info("Application starting up...")
    if not GEMINI_API_KEY:
        logger.critical("Missing Gemini API key")
        raise ValueError("GEMINI_API_KEY is required")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Application shutting down...")

# ---------------------------------------------------
# MAIN ENTRY POINT
# ---------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)