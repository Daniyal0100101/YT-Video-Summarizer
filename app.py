import os
import logging
import asyncio
import google.generativeai as genai
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, HttpUrl, Field, field_validator
from typing import List, Dict, Any, Optional
from cachetools import TTLCache
import re
import uvicorn

# External libs for transcript and download fallbacks
import httpx
from youtube_transcript_api import YouTubeTranscriptApi
try:
    from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
except Exception:
    class TranscriptsDisabled(Exception): pass
    class NoTranscriptFound(Exception): pass

from yt_dlp import YoutubeDL
from pytube import YouTube as PytubeYouTube
from dotenv import load_dotenv
import xml.etree.ElementTree as ET
from urllib.parse import urljoin

# -----------------------------
# LOGGING
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# -----------------------------
# ENV & CONFIG
# -----------------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PROXY_TRANSCRIPT_URL = os.getenv("PROXY_TRANSCRIPT_URL")  # optional self-hosted proxy
USE_TUBETEXT = os.getenv("USE_TUBETEXT", "false").lower() in ("1", "true", "yes")
TUBETEXT_API = os.getenv("TUBETEXT_API", "https://tubetext.vercel.app/youtube/transcript")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is required in environment")

# -----------------------------
# CACHE & RATE LIMITING
# -----------------------------
TRANSCRIPT_CACHE_SIZE = 200
CACHE_TTL = 3600  # 1 hour
transcript_cache = TTLCache(maxsize=TRANSCRIPT_CACHE_SIZE, ttl=CACHE_TTL)
summary_cache = TTLCache(maxsize=TRANSCRIPT_CACHE_SIZE, ttl=CACHE_TTL)

RATE_LIMIT_REQUESTS = 5
RATE_LIMIT_WINDOW = 60
ai_rate_limit_cache = TTLCache(maxsize=10000, ttl=RATE_LIMIT_WINDOW)

def get_client_ip(request: Request) -> str:
    x_forwarded_for = request.headers.get("x-forwarded-for")
    if x_forwarded_for:
        return x_forwarded_for.split(",")[0].strip()
    return request.client.host if request.client else "unknown"

def check_rate_limit(request: Request):
    ip = get_client_ip(request)
    count = ai_rate_limit_cache.get(ip, 0)
    if count >= RATE_LIMIT_REQUESTS:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                            detail=f"Rate limit exceeded: {RATE_LIMIT_REQUESTS} requests per {RATE_LIMIT_WINDOW} seconds.")
    ai_rate_limit_cache[ip] = count + 1

# -----------------------------
# GEMINI SYSTEM PROMPTS
# -----------------------------
SUMMARY_SYSTEM_INSTRUCTION = """
You are an advanced AI assistant with expertise in generating **thorough, fact-driven summaries** of YouTube videos.

Your primary objective is to deliver a **comprehensive** yet **succinct** synthesis of the video, ensuring no major points or key explanations are omitted.

### Guidelines for Generating Summaries

1. **Thorough and Accurate Coverage**
   - Summarize **all critical topics, explanations, and details** presented in the video.
   - Avoid conjecture; include strictly what is clearly communicated.
   - If details are incomplete or ambiguous, focus solely on the available information, without extrapolating.

2. **Clear and Logical Structure**
   - **Introduction:** Present the central theme and aim of the video.
   - **Key Points:** Organize all major ideas, explanations, steps, or insights shared by the speaker in a logical progression.
   - **Conclusion:** Outline the main takeaways, final thoughts, or actionable items provided in the video.

3. **Contextual and Accessible Summaries**
   - Accurately reflect the scope and intent of the speaker's message.
   - Ensure that **every significant concept, explanation, or process is included**.
   - Utilize structured formatting (headings and bullet points) for clarity and ease of understanding.

4. **Faithful Representation—No Speculation or Gaps**
   - Only report on specifically provided information.
   - Never infer or invent content beyond the explicit statements in the video.

5. **Professional, Clear, and Engaging Tone**
   - Maintain a neutral, informative style that is approachable.
   - Make technical explanations or instructions accessible and understandable.

Your role is to generate summaries that **faithfully represent the video's content**, giving users a precise and thorough understanding without adding or omitting any critical details.
"""
FOLLOWUP_SYSTEM_INSTRUCTION = """
You are a specialized AI assistant dedicated to answering follow-up questions about YouTube videos.

Your key responsibility is to provide clear, thorough answers strictly grounded in the ORIGINAL video content.

## Follow-up Principles

1. **Always rely primarily on the video content for information.**
   - Use the exact spoken words from the video for accuracy.
   - Extract direct quotes and specifics as needed.
   - Search the full video content to address the user's query.

2. **Use the summary only as a supporting context.**
   - Refer to the summary for general structure, but defer to the video content for details.
   - Prefer specific information from the video over general points from the summary.

3. **Respond with precision and factual accuracy.**
   - Base answers entirely on explicit content from the video.
   - If something is not addressed in the video, state explicitly that the information is unavailable.
   - Never create or presume details not found in the source.

4. **Use user-friendly time references when citing video content**, such as "(~5:30)", "(early in the video)", or "(near the end)".
   - Format raw timestamps as MM:SS for readability.
   - Include timestamps only where they enhance clarity.
   - Avoid unnecessary timestamp repetition.

5. **Format answers for maximum clarity and accessibility.**
   - Use bullet points for lists.
   - Highlight key information with bold.
   - Break up longer responses with informative headings.

Remember: Your mission is to ensure users receive fully accurate, well-structured answers by utilizing the entire video content, never inventing or overlooking relevant information.
"""

# -----------------------------
# Gemini model initialization
# -----------------------------
MODEL_NAME = "gemini-2.0-flash-lite"
summary_model = None
followup_model = None
try:
    genai.configure(api_key=GEMINI_API_KEY)
    summary_model = genai.GenerativeModel(MODEL_NAME, system_instruction=SUMMARY_SYSTEM_INSTRUCTION)
    followup_model = genai.GenerativeModel(MODEL_NAME, system_instruction=FOLLOWUP_SYSTEM_INSTRUCTION)
    logger.info("Initialized Gemini models")
except Exception as e:
    logger.error(f"Failed to init Gemini: {e}")
    raise

# -----------------------------
# FastAPI setup
# -----------------------------
app = FastAPI(title="YouTube Video Summarizer API", version="1.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["GET","POST","OPTIONS"], allow_headers=["*"])

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# -----------------------------
# Data models
# -----------------------------
class VideoRequest(BaseModel):
    youtube_url: HttpUrl = Field(..., description="Full YouTube video URL")
    @field_validator('youtube_url')
    @classmethod
    def validate_youtube_url(cls, v):
        youtube_regex = r'^.*(?:youtu\.be\/|v\/|u\/\w\/|embed\/|watch\?v=|\&v=)([^#\&\?]*).*'
        if not re.search(youtube_regex, str(v)):
            raise ValueError("Invalid YouTube URL format")
        return v

class ConversationMessage(BaseModel):
    role: str
    content: str

class FollowUpRequest(BaseModel):
    question: str
    history: List[ConversationMessage]
    transcript: str
    title: Optional[str] = None
    description: Optional[str] = None

# -----------------------------
# Helpers
# -----------------------------
def extract_youtube_id(url: str) -> str:
    youtube_regex = r"(?:youtube\.com/(?:[^/]+/.+/|(?:v|e(?:mbed)?)/|.*[?&]v=)|youtu\.be/)([^\"&?/\s]{11})"
    m = re.search(youtube_regex, url)
    if m:
        return m.group(1)
    raise HTTPException(status_code=400, detail="Invalid YouTube URL format.")

def strip_vtt_srt(text: str) -> str:
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.upper().startswith("WEBVTT"):
            continue
        if re.fullmatch(r'\d+', line):
            continue
        # SRT or VTT timestamps
        if re.match(r'^\d{2}:\d{2}:\d{2}\.\d{3} -->', line) or re.match(r'^\d{2}:\d{2}\.\d{3} -->', line) or re.match(r'^\d{2}:\d{2}:\d{2},\d{3} -->', line):
            continue
        if re.match(r'^\d{2}:\d{2}\.\d{3} -->', line):
            continue
        # remove m3u8 tags if they ever make it here
        if line.startswith("#EXT"):
            continue
        lines.append(line)
    return "\n".join(lines)

def parse_youtube_xml_transcript(xml_text: str) -> str:
    try:
        root = ET.fromstring(xml_text)
        parts = []
        for elem in root.findall('.//text'):
            text = (elem.text or "")
            parts.append(text.strip())
        return "\n".join([p for p in parts if p])
    except Exception:
        cleaned = re.sub(r'<[^>]+>', '', xml_text)
        return strip_vtt_srt(cleaned)

# -----------------------------
# HTTP / HLS helpers
# -----------------------------
async def http_get_text(url: str, timeout: int = 15) -> Optional[str]:
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.get(url)
            if r.status_code == 200:
                return r.text
    except Exception as e:
        logger.debug(f"HTTP fetch failed for caption url: {e}")
    return None

async def fetch_hls_segments_from_manifest(manifest_text: str, manifest_url: str, max_segments: int = 200) -> Optional[str]:
    """
    Given an m3u8 manifest text, fetch referenced segment URLs and assemble their textual content.
    Returns combined text or None.
    """
    lines = manifest_text.splitlines()
    segment_urls = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        # this line should be a URL (absolute or relative)
        seg_url = urljoin(manifest_url, line)
        segment_urls.append(seg_url)
    if not segment_urls:
        # sometimes manifest indexes other m3u8 playlists (master playlist). Try to find first playable sub-playlist
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            seg_url = urljoin(manifest_url, line)
            # recursively fetch first playlist
            text = await http_get_text(seg_url)
            if text and (text.lstrip().startswith("#EXTM3U") or "#EXTINF" in text):
                return await fetch_hls_segments_from_manifest(text, seg_url, max_segments=max_segments)
        return None

    assembled_parts = []
    async with httpx.AsyncClient(timeout=20) as client:
        count = 0
        for seg in segment_urls:
            if count >= max_segments:
                break
            try:
                r = await client.get(seg)
                if r.status_code == 200:
                    assembled_parts.append(r.text)
                    count += 1
            except Exception:
                continue
    combined = "\n".join(assembled_parts)
    if not combined:
        return None
    if combined.lstrip().startswith("<?xml") or "<transcript" in combined:
        return parse_youtube_xml_transcript(combined)
    if "WEBVTT" in combined[:200] or "WEBVTT" in combined:
        return strip_vtt_srt(combined)
    # fallback: remove timestamps / m3u8 tags
    return strip_vtt_srt(combined)

# -----------------------------
# yt-dlp helper with HLS support
# -----------------------------
async def try_yt_dlp_captions(video_id: str) -> Optional[str]:
    ydl_opts = {"quiet": True, "skip_download": True}
    youtube_url = f"https://www.youtube.com/watch?v={video_id}"
    try:
        def _extract():
            with YoutubeDL(ydl_opts) as ydl:
                return ydl.extract_info(youtube_url, download=False)
        info = await asyncio.to_thread(_extract)
        for key in ("automatic_captions", "subtitles"):
            caps = info.get(key) or {}
            if not isinstance(caps, dict):
                continue
            for code in ("en", "a.en", "en-US"):
                formats = caps.get(code) or caps.get(code.lower()) or None
                if formats:
                    for fmt in formats:
                        url = fmt.get("url")
                        if url:
                            logger.info(f"Found caption URL via yt-dlp ({key}/{code}) for {video_id}")
                            text = await http_get_text(url)
                            if not text:
                                continue
                            # handle HLS manifest
                            if text.lstrip().startswith("#EXTM3U") or "#EXTINF" in text:
                                logger.info(f"HLS/m3u8 manifest detected for {video_id}; assembling segments")
                                assembled = await fetch_hls_segments_from_manifest(text, url)
                                if assembled:
                                    return assembled
                                # if assemble failed, continue to other formats
                                continue
                            if text.lstrip().startswith("<?xml") or "<transcript" in text:
                                return parse_youtube_xml_transcript(text)
                            if text.lstrip().startswith("WEBVTT") or "WEBVTT" in text[:200]:
                                return strip_vtt_srt(text)
                            # fallback: strip timestamps
                            return strip_vtt_srt(text)
            # also check generic 'en' key
            if "en" in caps:
                formats = caps.get("en")
                for fmt in formats:
                    url = fmt.get("url")
                    if url:
                        text = await http_get_text(url)
                        if text:
                            if text.lstrip().startswith("#EXTM3U") or "#EXTINF" in text:
                                assembled = await fetch_hls_segments_from_manifest(text, url)
                                if assembled:
                                    return assembled
                            if text.lstrip().startswith("<?xml") or "<transcript" in text:
                                return parse_youtube_xml_transcript(text)
                            return strip_vtt_srt(text)
    except Exception as e:
        logger.debug(f"yt-dlp caption extraction failed: {e}")
    return None

# -----------------------------
# pytube fallback
# -----------------------------
async def try_pytube_captions(video_id: str) -> Optional[str]:
    def _fetch(vid: str) -> Optional[str]:
        try:
            yt = PytubeYouTube(f"https://www.youtube.com/watch?v={vid}")
            if not yt.captions:
                return None
            caption = None
            for code in ("en", "a.en", "en-US"):
                caption = yt.captions.get(code)
                if caption:
                    break
            if not caption:
                caption = next(iter(yt.captions.values()), None)
            if caption:
                return caption.generate_srt_captions()
        except Exception as e:
            logger.debug(f"pytube caption fetch exception: {e}")
            return None
    return await asyncio.to_thread(_fetch, video_id)

# -----------------------------
# youtube-transcript-api wrapper (handles different versions)
# -----------------------------
async def try_youtube_transcript_api(video_id: str) -> Optional[str]:
    # try class-level get_transcript
    try:
        func = getattr(YouTubeTranscriptApi, "get_transcript", None)
        if callable(func):
            try:
                data = await asyncio.to_thread(func, video_id, ["en", "en-US", "a.en"])
                if data:
                    return "\n".join(item.get("text", "") for item in data)
            except TranscriptsDisabled:
                logger.info("TranscriptsDisabled (get_transcript)")
                return None
            except NoTranscriptFound:
                logger.info("NoTranscriptFound (get_transcript)")
            except Exception as e:
                logger.debug(f"get_transcript call failed: {e}")
    except Exception:
        pass

    # try module-level get_transcript
    try:
        import importlib
        mod = importlib.import_module("youtube_transcript_api")
        mod_func = getattr(mod, "get_transcript", None)
        if callable(mod_func):
            try:
                data = await asyncio.to_thread(mod_func, video_id, ["en", "en-US", "a.en"])
                if data:
                    return "\n".join(item.get("text", "") for item in data)
            except (TranscriptsDisabled, NoTranscriptFound):
                return None
            except Exception as e:
                logger.debug(f"module.get_transcript failed: {e}")
    except Exception:
        pass

    # try list_transcripts -> find_transcript -> fetch (1.2.x etc)
    try:
        if hasattr(YouTubeTranscriptApi, "list_transcripts"):
            def _list_fetch(vid: str):
                transcripts = YouTubeTranscriptApi.list_transcripts(vid)
                transcript_obj = transcripts.find_transcript(["en", "en-US", "a.en"])
                return transcript_obj.fetch()
            try:
                data = await asyncio.to_thread(_list_fetch, video_id)
                if data:
                    return "\n".join(item.get("text", "") for item in data)
            except TranscriptsDisabled:
                logger.info("TranscriptsDisabled (list_transcripts)")
                return None
            except NoTranscriptFound:
                logger.info("NoTranscriptFound (list_transcripts)")
            except Exception as e:
                logger.debug(f"list_transcripts pattern failed: {e}")
    except Exception:
        pass

    return None

# -----------------------------
# Main get_transcript with fallbacks & caching
# -----------------------------
async def get_transcript(video_id: str) -> str:
    if video_id in transcript_cache:
        logger.debug(f"Transcript cache hit: {video_id}")
        return transcript_cache[video_id]

    # 1) youtube-transcript-api
    try:
        yt_api_text = await try_youtube_transcript_api(video_id)
        if yt_api_text:
            transcript_cache[video_id] = yt_api_text
            logger.info(f"Transcript from youtube-transcript-api for {video_id}")
            return yt_api_text
    except Exception as e:
        logger.debug(f"youtube-transcript-api wrapper error: {e}")

    # 2) yt-dlp (includes HLS handling)
    try:
        ytdlp_text = await try_yt_dlp_captions(video_id)
        if ytdlp_text:
            transcript_cache[video_id] = ytdlp_text
            logger.info(f"Transcript from yt-dlp captions for {video_id}")
            return ytdlp_text
    except Exception as e:
        logger.debug(f"yt-dlp captions attempt failed: {e}")

    # 3) pytube
    try:
        pytube_text = await try_pytube_captions(video_id)
        if pytube_text:
            plain = strip_vtt_srt(pytube_text)
            transcript_cache[video_id] = plain
            logger.info(f"Transcript from pytube for {video_id}")
            return plain
    except Exception as e:
        logger.debug(f"pytube attempt failed: {e}")

    # 4) optional TubeText
    if USE_TUBETEXT:
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                r = await client.get(TUBETEXT_API, params={"video_id": video_id})
                if r.status_code == 200:
                    data = r.json()
                    if data.get("success") and "data" in data:
                        payload = data["data"]
                        if payload.get("full_text"):
                            transcript_cache[video_id] = payload.get("full_text")
                            logger.info(f"Transcript from TubeText for {video_id}")
                            return payload.get("full_text")
                        if payload.get("transcript") and isinstance(payload.get("transcript"), list):
                            joined = "\n".join(payload.get("transcript"))
                            transcript_cache[video_id] = joined
                            logger.info(f"Transcript list from TubeText for {video_id}")
                            return joined
                else:
                    logger.debug(f"TubeText returned {r.status_code} for {video_id}")
        except Exception as e:
            logger.debug(f"TubeText fetch error: {e}")

    # 5) optional proxy
    if PROXY_TRANSCRIPT_URL:
        try:
            async with httpx.AsyncClient(timeout=20) as client:
                r = await client.get(PROXY_TRANSCRIPT_URL, params={"video_id": video_id})
                if r.status_code == 200:
                    data = r.json()
                    if data.get("full_text"):
                        transcript_cache[video_id] = data["full_text"]
                        logger.info(f"Transcript from proxy for {video_id}")
                        return data["full_text"]
                    if data.get("transcript") and isinstance(data["transcript"], list):
                        joined = "\n".join(data["transcript"])
                        transcript_cache[video_id] = joined
                        logger.info(f"Transcript list from proxy for {video_id}")
                        return joined
        except Exception as e:
            logger.debug(f"Proxy transcript error: {e}")

    # final fallback
    msg = "Transcript is not available for this video via available services."
    transcript_cache[video_id] = msg
    logger.info(f"No transcript available for {video_id}")
    return msg

# -----------------------------
# Video metadata retrieval
# -----------------------------
async def get_video_details(youtube_url: str) -> Dict[str, Any]:
    video_id = extract_youtube_id(youtube_url)
    ydl_opts = {"quiet": True, "skip_download": True}
    try:
        def _extract(url: str):
            with YoutubeDL(ydl_opts) as ydl:
                return ydl.extract_info(url, download=False)
        info = await asyncio.to_thread(_extract, youtube_url)
        upload_date = info.get("upload_date")
        formatted = f"{upload_date[0:4]}-{upload_date[4:6]}-{upload_date[6:8]}" if upload_date else None
        return {
            "title": info.get("title") or "Title unavailable",
            "description": info.get("description") or "Description unavailable",
            "video_id": info.get("id") or video_id,
            "duration": info.get("duration"),
            "author": info.get("uploader"),
            "published_date": formatted,
        }
    except Exception as e:
        logger.debug(f"yt-dlp metadata failed: {e}")

    # noembed fallback
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get("https://noembed.com/embed", params={"url": youtube_url})
            if r.status_code == 200:
                d = r.json()
                return {
                    "title": d.get("title") or "Title unavailable",
                    "description": d.get("description") or "Description unavailable",
                    "video_id": video_id,
                    "duration": None,
                    "author": d.get("author_name"),
                    "published_date": None
                }
    except Exception as e:
        logger.debug(f"NoEmbed failed: {e}")

    return {
        "title": "Title unavailable",
        "description": "Description unavailable",
        "video_id": video_id,
        "duration": None,
        "author": None,
        "published_date": None
    }

# -----------------------------
# Summary & follow-up
# -----------------------------
async def generate_summary(transcript: str, video_id: str) -> str:
    if video_id in summary_cache:
        return summary_cache[video_id]
    if not transcript or len(transcript.split()) < 20 or transcript.startswith("Transcript is not available"):
        return "No meaningful video content available for summarization."
    prompt = f"""
    Summarize the following YouTube video content concisely:

    VIDEO CONTENT:
    {transcript}

    Instructions:
    - Provide an introduction, key points, and conclusion.
    - If timestamps are present in the transcript, include them in the summary for reference.
    - Ensure factual accuracy and do not add information not in the transcript.

    Use markdown formatting for readability.
    """
    try:
        response = await asyncio.to_thread(lambda: summary_model.generate_content(prompt))
        summary = getattr(response, "text", None) or str(response)
        summary_cache[video_id] = summary
        return summary
    except Exception as e:
        logger.error(f"Summary error: {e}")
        return f"Error generating summary: {e}"

async def generate_followup_response(question: str, formatted_convo: str, transcript: str, title: Optional[str], description: Optional[str]) -> str:
    context_parts = []
    if title:
        context_parts.append(f"VIDEO TITLE:\n{title}")
    if description:
        context_parts.append(f"VIDEO DESCRIPTION:\n{description}")
    if transcript:
        context_parts.append(f"ORIGINAL VIDEO TRANSCRIPT:\n{transcript}")
    context_str = "\n\n".join(context_parts)
    if not context_str.strip():
        return "No video data (title, description, or transcript) is available to answer your question."
    prompt = f"""{context_str}

CONVERSATION HISTORY:
{formatted_convo}

USER QUESTION: {question}

Instructions:
1. Answer the question using only the provided video data (title, description, transcript). Do not speculate or use outside knowledge.
2. If the answer is not present in the provided data, clearly state that you cannot answer due to missing information.
3. Be specific and cite information directly from the data when possible.
4. Format your response for readability (bullet points, bold for key points, etc.).
5. When referencing specific content from the transcript, include a simple time reference in (~MM:SS) format if available.
"""
    try:
        try:
            response = await followup_model.generate_content_async(prompt)
            return getattr(response, "text", str(response))
        except AttributeError:
            response = await asyncio.to_thread(lambda: followup_model.generate_content(prompt))
            return getattr(response, "text", str(response))
    except Exception as e:
        logger.error(f"Error generating follow-up response: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate follow-up response: {e}")

# -----------------------------
# Routes
# -----------------------------
@app.get("/", response_class=HTMLResponse)
async def index():
    file_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    raise HTTPException(status_code=404, detail="Index not found.")

@app.get("/favicon.ico")
async def favicon():
    return FileResponse("static/favicon.ico")

@app.post("/api/video-data", response_class=JSONResponse)
async def fetch_video_data(request: VideoRequest):
    try:
        youtube_url = str(request.youtube_url)
        video_id = extract_youtube_id(youtube_url)
        video_details, transcript = await asyncio.gather(get_video_details(youtube_url), get_transcript(video_id))
        result = {**video_details, "transcript": transcript}
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"fetch_video_data error: {e}")
        return JSONResponse(status_code=500, content={"detail": str(e)})

@app.post("/api/generate-summary", response_class=JSONResponse)
async def generate_summary_endpoint(request: VideoRequest, req: Request):
    try:
        check_rate_limit(req)
        youtube_url = str(request.youtube_url)
        video_id = extract_youtube_id(youtube_url)
        transcript = await get_transcript(video_id)
        if transcript.startswith("Transcript is not available"):
            return JSONResponse(content={"summary": transcript})
        summary = await generate_summary(transcript, video_id)
        return JSONResponse(content={"summary": summary})
    except HTTPException as e:
        if e.status_code == status.HTTP_429_TOO_MANY_REQUESTS:
            return JSONResponse(status_code=429, content={"detail": e.detail})
        logger.error(f"generate_summary_endpoint error: {e}")
        return JSONResponse(status_code=500, content={"detail": str(e)})

@app.post("/api/follow-up", response_class=JSONResponse)
async def follow_up_questions(request: FollowUpRequest, req: Request):
    try:
        check_rate_limit(req)
        formatted = "\n".join([f"{m.role.upper()}: {m.content}" for m in request.history])
        response = await generate_followup_response(request.question, formatted, request.transcript, getattr(request, 'title', None), getattr(request, 'description', None))
        return JSONResponse(content={"response": response})
    except HTTPException as e:
        if e.status_code == status.HTTP_429_TOO_MANY_REQUESTS:
            return JSONResponse(status_code=429, content={"detail": e.detail})
        logger.error(f"follow_up_questions error: {e}")
        return JSONResponse(status_code=500, content={"detail": str(e)})

# -----------------------------
# Lifecycle events
# -----------------------------
@app.on_event("startup")
async def startup_event():
    logger.info("Application startup completed.")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("App shutting down...")

# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
