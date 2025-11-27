# # # separate folders for each pdf

import os
import re
import textwrap
import requests
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from PyPDF2 import PdfReader
from gtts import gTTS
from moviepy.editor import AudioFileClip, concatenate_videoclips, ImageClip, VideoFileClip, CompositeVideoClip
from collections import Counter
import concurrent.futures
import functools
from textwrap import shorten
import pysrt
from moviepy.video.tools.subtitles import SubtitlesClip
import os, textwrap, numpy as np
import pyttsx3
import time


from googletrans import Translator
translator = Translator()



import streamlit as st




# Custom CSS to override Streamlit's default styles
st.markdown(
	"""
	<style>
	/* Hide Streamlit's default header */

	/* Hide Streamlit's footer */
	footer { visibility: hidden; }

	/* Override Streamlit default font */
	html, section, body, [class*="css"]  {
		
		background: #fff !important;
		 max-width: 100%;
	}

	/* Hide hamburger menu and Streamlit branding */
	.st-emotion-cache-1v0mbdj { display: none; }  /* Sidebar collapse icon */
	.st-emotion-cache-13ln4jf { display: none; }  /* Main menu */
	.st-emotion-cache-1d391kg { display: none; }  /* Footer */


	.st-emotion-cache-1w723zb {
	width: 100%;
	padding: 6rem 1rem 10rem;
	max-width: 100%;
}

	</style>
	""",
	unsafe_allow_html=True
)






# ─── CONFIG ───────────────────────────────────────────────────────────
TARGET_WORDS = 1800
AUDIO_DIR = "audio"
SRT_DIR   = "srt"
VIDEO_DIR = "videos"
IMG_DIR   = "images"
UNSPLASH_ACCESS_KEY = os.getenv("UNSPLASH_ACCESS_KEY", "")
for d in (AUDIO_DIR, SRT_DIR, VIDEO_DIR, IMG_DIR):
	os.makedirs(d, exist_ok=True)







# Hide Streamlit menu and footer
hide_menu = """
	<style>
		#MainMenu {visibility: hidden;}
		footer {visibility: hidden;}
		.stDeployButton {visibility: hidden;}
		[data-testid="stToolbar"] {visibility: hidden !important;}
		[data-testid="stDecoration"] {display: none;}
	</style>
"""
st.markdown(hide_menu, unsafe_allow_html=True)


import base64

# Encode your logo (SVG, PNG, or JPG)
with open("Logo.svg", "rb") as f:
    encoded_svg = base64.b64encode(f.read()).decode()

st.markdown(f"""
    <!-- Bootstrap CSS -->
    <link
      href="https://cdn.jsdelivr.net/npm/bootstrap@5.0.2/dist/css/bootstrap.min.css"
      rel="stylesheet"
      crossorigin="anonymous"
    >

    <style>
    /* Hide Streamlit's default top header bar (stable selector) */
    header[data-testid="stHeader"] {{
        display: none;
    }}



    /* Your custom header */
    header.header {{
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        padding: 5px 2rem;
        background: #ffffff;
        box-shadow: -3px 3px 6px rgba(0,0,0,0.16);
        z-index: 99999;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }}

    /* Logo styling */
    header.header .logo img {{
        height: 50px !important;   /* fixed height */
        width: auto !important;    /* keep aspect ratio */
        display: inline-block !important;
        visibility: visible !important;
    }}

    /* Optional title styling */
    header.header .title {{
        font-size: 20px;
        font-weight: 500;
        color: #141414;
        margin-left: 15px;
    }}

    /* Spacer so content isn’t hidden behind fixed header */
    .spacer-header {{
        height: 70px;  /* match header’s total height */
    }}
    </style>

    <!-- Sticky Header with Logo + Title -->
    <header class="header">
      <div class="d-flex align-items-center">
        <div class="logo">
          <img
            src="data:image/svg+xml;base64,{encoded_svg}"
            alt="ICFAI Logo"
          >
        </div>
        
    </header>

    <!-- Push content down -->
    <div class="spacer-header"></div>
""", unsafe_allow_html=True)

 

st.markdown("""
<style>
/* Remove extra padding Streamlit adds around markdown blocks */
.block-container {
    padding-top: 0rem !important; 
}

/* Remove extra margins around markdown text */
.stMarkdown {
    margin: 0 !important;
    padding: 0 !important;
}
</style>
""", unsafe_allow_html=True)

 


footer_html = """
<style>
  .footer {
	position: fixed;
	left: 0;
	bottom: 0;
	width: 100%;
	background-color: #b11226;  /* deep red */
	color: white;
	text-align: center;
	padding: 0.75rem 0;
	font-size: 0.9rem;
	z-index: 100;
  }
  /* avoid Streamlit’s bottom bar from overlapping */
  .reportview-container .main footer { visibility: hidden; }
</style>
<div class="footer">
  © 2024 Copyright All Rights Reserved by 
  <strong>The ICFAI Group</strong>.
</div>
"""
st.markdown(footer_html, unsafe_allow_html=True)




# st.set_page_config(page_title="📄→🎥 Headings Only", layout="wide")

# ─── PAGE EXTRACTION & SUMMARIZATION ─────────────────────────────────
@st.cache_data
def extract_pages_from_bytes(pdf_bytes):
	clean = lambda t: re.sub(r'\s+', ' ',
				 re.sub(r'Page \d+|Reprint 2025-26|===== Page \d+ =====', '', t)
			   ).strip()
	reader = PdfReader(pdf_bytes)
	return [clean(page.extract_text() or "") for page in reader.pages]


def summarize(pages, stopwords, target_words=TARGET_WORDS):
	text = " ".join(pages)
	words = re.findall(r'\w+', text.lower())
	freqs = Counter(w for w in words if w not in stopwords)
	sents = re.split(r'(?<=[\.\?\!])\s+', text)
	scored = [(i, sum(freqs[w] for w in re.findall(r'\w+', s.lower()))/max(len(s),1), s)
			  for i, s in enumerate(sents)]
	scored.sort(key=lambda x: x[1], reverse=True)
	sel, wc = [], 0
	for idx, _, s in scored:
		wcount = len(re.findall(r'\w+', s))
		if wc + wcount > target_words and wc > 0:
			break
		sel.append((idx, s))
		wc += wcount
	sel.sort()
	return " ".join(s for _, s in sel)

def split_into_slides(text, max_chars=380):
	chunks, cur = [], ""
	for p in text.split('. '):
		p = p.strip()
		if not p:
			continue
		p += '.' if not p.endswith('.') else ''
		if len(cur) + len(p) < max_chars:
			cur += p + " "
		else:
			chunks.append(cur.strip())
			cur = p + " "
	if cur:
		chunks.append(cur.strip())
	return chunks

# ─── HEADING EXTRACTION ─────────────────────────────────────────────────
def extract_heading(txt: str) -> str:
	"""Return only the first sentence as the slide heading."""
	# split on first period, question, or exclamation
	parts = re.split(r'(?<=[\.\?\!])\s+', txt, maxsplit=1)
	return parts[0].strip()

# ─── IMAGE SEARCH & DOWNLOAD (with caching) ────────────────────────────
@functools.lru_cache(maxsize=256)
@functools.lru_cache(maxsize=256)
def fetch_image_for_slide(text, prefix, idx):
    img_path = os.path.join(IMG_DIR, f"{prefix}_slide_{idx}.jpg")
    
    # If image already exists, return it
    if os.path.exists(img_path):
        return img_path
        
    # If no Unsplash key, return None
    if not UNSPLASH_ACCESS_KEY:
        return None

    query = text if len(text) < 50 else text[:50]
    url = "https://api.unsplash.com/search/photos"
    params = {"query": query, "per_page": 1, "orientation": "landscape"}
    headers = {"Authorization": f"Client-ID {UNSPLASH_ACCESS_KEY}"}
    
    try:
        r = requests.get(url, params=params, headers=headers, timeout=10)
        r.raise_for_status()
        results = r.json().get("results")
        if not results:
            return None
            
        img_url = results[0]["urls"]["regular"]
        resp = requests.get(img_url, stream=True, timeout=10)
        resp.raise_for_status()
        
        with open(img_path, "wb") as out:
            for chunk in resp.iter_content(1024):
                out.write(chunk)
                
        return img_path
    except Exception as e:
        print(f"Error fetching image: {e}")
        # Return the path anyway - it might exist from a previous attempt
        return img_path if os.path.exists(img_path) else None

# ─── at the top of your script, after imports ────────────────────────
VOICE_OPTIONS = {
	
	 "Indian English (co.in)": ("en", "co.in"),
	"US English (com)":        ("en", "com"  ),
	"British English (co.uk)": ("en", "co.uk"),
	# …etc…

	# Indian languages — no TLD override needed
	"Hindi (hi)":    ("hi", None),
	"Telugu (te)":   ("te", None),
	"Kannada(kn)":   ("kn",None),
	"Tamil(ta)":     ("ta",None),
	"Malayalam(ml)": ("ml",None),
	"Marathi(mr)":   ("mr",None),
	"Gujarati(gu)":  ("gu",None),
	"Bengali(bn)":   ("bn",None)
}




# ─── Load Bootstrap CSS ─────────────────────────────────────────────────
# Load Bootstrap CSS once
st.markdown("""
<link
  href="https://cdn.jsdelivr.net/npm/bootstrap@5.0.2/dist/css/bootstrap.min.css"
  rel="stylesheet"
  integrity="sha384-EVSTQN3/azprG1Anm3QDgpJLIm9Nao0Yz1ztcQTwFspd3yD65VohhpuuCOmLASjC"
  crossorigin="anonymous"
/>
<style>
#root{
	background: #2b296a;
	height
}
</style>
""", unsafe_allow_html=True)

# Make a 2-column layout: col1 for the card, col2 for the selectbox
col1, col2, col3 = st.columns([3, 6, 3])


 
with col2:
	st.markdown("""
	<style>
	.upload-container {
		text-align: center;
		# background-color: #f6f6f6;
		padding: 10px	; 
		box-shadow: 0 4px 12px rgba(0,0,0,0.05);
		font-family: 'Segoe UI', sans-serif;
		 
	}

	.upload-title {
		font-size: 32px;
		font-weight: 700;
		margin-bottom: 8px;
		color: #1a1a1a;
	} 
	 
	/* Force center align and style for the file uploader */
	section[data-testid="stFileUploader"] {
		display: flex;
		justify-content: center;
		margin-top: -20px;
		margin-bottom: 10px;
	}

	section[data-testid="stFileUploader"] label {
		background-color: #d32f2f;
		color: white !important;
		padding: 14px 28px;
		font-size: 18px;
		font-weight: bold;
		border-radius: 50px;
		cursor: pointer;
		transition: background-color 0.3s ease;
	}

	section[data-testid="stFileUploader"] label:hover {
		background-color: #b71c1c;
	}
	</style>

	<div class="upload-container">
		<div class="upload-title">Multilingual Video Generator</div>
			
	</div>
	""", unsafe_allow_html=True)

	# Appears just below the container and matches visually

 

 

# ─── TTS & SLIDE CREATION ─────────────────────────────────────────────
def synthesize_slide_gtts(txt, filename, lang, tld=None):
	if tld:
		tts = gTTS(text=txt, lang=lang, tld=tld)
	else:
		tts = gTTS(text=txt, lang=lang)
	tts.save(filename)
	return filename





def synthesize_slides(slides, prefix, lang, tld):
	files = []
	to_say = []

	# include lang/tld in your filename so different languages never collide
	tag = f"{lang}" + (f"_{tld}" if tld else "")
	for i, txt in enumerate(slides):
		out_path = os.path.join(
			AUDIO_DIR,
			f"{prefix}_{tag}_scene_{i:03d}.mp3"
		)
		files.append(out_path)
		if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
			to_say.append((txt, out_path))

	for txt, out_path in to_say:
		synthesize_slide_gtts(txt, out_path, lang, tld)

	return files





def make_slide(txt, duration, bg_image_path=None):
	W, H = 640, 360

	# pick background
	if bg_image_path and os.path.exists(bg_image_path):
		bg = Image.open(bg_image_path).convert("RGB").resize((W, H), Image.LANCZOS)
	else:
		bg = Image.new("RGB", (W, H), (30, 30, 70))

	draw = ImageDraw.Draw(bg)
	font_path = "arial.ttf"
	if selected_lang == "hi" and os.path.exists("NotoSansDevanagari-Regular.ttf"):
	   font = ImageFont.truetype("NotoSansDevanagari-Regular.ttf", 20)

	elif selected_lang == "te":
		font_path = r"Nirmala.ttf"
		font = ImageFont.truetype(font_path, 20) if os.path.exists(font_path) else ImageFont.load_default()
	elif selected_lang == "kn" and os.path.exists("NotoSansKannada-Regular.ttf"):
	   font = ImageFont.truetype("NotoSansKannada-Regular.ttf", 20)
	elif selected_lang == "ta" and os.path.exists("NotoSansTamil-Regular.ttf"):
	   font = ImageFont.truetype("NotoSansTamil-Regular.ttf", 20)
	elif selected_lang == "ml" and os.path.exists("NotoSansMalayalam-Regular.ttf"):
	   font = ImageFont.truetype("NotoSansMalayalam-Regular.ttf", 20)
	elif selected_lang == "mr" and os.path.exists("NotoSansDevanagari-Regular.ttf"):
	   font = ImageFont.truetype("NotoSansDevanagari-Regular.ttf", 20)
	elif selected_lang == "gu" and os.path.exists("NotoSansGujarati-Regular.ttf"):
	   font = ImageFont.truetype("NotoSansGujarati-Regular.ttf", 20)
	elif selected_lang == "bn" and os.path.exists("NotoSansBengali-Regular.ttf"):
	   font = ImageFont.truetype("NotoSansBengali-Regular.ttf", 20)
	
	else:
	   try:
		   font = ImageFont.truetype("arial.ttf", 20)
	   except:
		   font = ImageFont.load_default()
	# only show heading
	heading = extract_heading(txt)
	lines = textwrap.wrap(heading, width=40)

	# draw at fixed top margin
	y = 20
	for line in lines:
		bbox = draw.textbbox((0, 0), line, font=font)
		w = bbox[2] - bbox[0]
		draw.text(((W - w)//2, y), line, font=font, fill="white")
		y += bbox[3] - bbox[1] + 8   # line height + small spacing

	return ImageClip(np.array(bg)).set_duration(duration)



# ─── SRT WRITING ───────────────────────────────────────────────────────
def format_timestamp(seconds):
	ms = int((seconds - int(seconds)) * 1000)
	h = int(seconds // 3600)
	m = int((seconds % 3600) // 60)
	s = int(seconds % 60)
	return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def write_srt(slides, audio_files, prefix):
	subs = pysrt.SubRipFile()
	current = 0.0
	index = 1

	for slide_text, wav in zip(slides, audio_files):
		# load duration
		clip = AudioFileClip(wav)
		dur  = clip.duration
		clip.close()

		# split into sentences
		sents = re.split(r'(?<=[\.!?])\s+', slide_text.strip())
		if not sents:
			continue

		per = dur / len(sents)
		for i, sentence in enumerate(sents):
			start = current + i * per
			end   = start + per
			subs.append(pysrt.SubRipItem(
				index=index,
				start=pysrt.SubRipTime(milliseconds=int(start*1000)),
				end=  pysrt.SubRipTime(milliseconds=int(end*1000)),
				text= sentence.strip()
			))
			index += 1

		current += dur

	out = os.path.join(SRT_DIR, f"{prefix}.srt")
	subs.save(out, encoding="utf-8")
	return out



# ─── 2) BURN with PIL so each sentence appears/fades in sync ────────────
def burn_subtitles_pil(video_path, srt_path, out_path):
	video = VideoFileClip(video_path)
	W, H   = video.size
	fps    = video.fps

	if selected_lang == "hi" and os.path.exists("NotoSansDevanagari-Regular.ttf"):
	   font = ImageFont.truetype("NotoSansDevanagari-Regular.ttf", 20)
	

	elif selected_lang == "te":
		font_path = r"Nirmala.ttf"
		font = ImageFont.truetype(font_path, 20) if os.path.exists(font_path) else ImageFont.load_default()
	elif selected_lang == "kn" and os.path.exists("NotoSansKannada-Regular.ttf"):
	   font = ImageFont.truetype("NotoSansKannada-Regular.ttf", 20)
	elif selected_lang == "ta" and os.path.exists("NotoSansTamil-Regular.ttf"):
	   font = ImageFont.truetype("NotoSansTamil-Regular.ttf", 20)
	elif selected_lang == "ml" and os.path.exists("NotoSansMalayalam-Regular.ttf"):
	   font = ImageFont.truetype("NotoSansMalayalam-Regular.ttf", 20)
	elif selected_lang == "mr" and os.path.exists("NotoSansDevanagari-Regular.ttf"):
	   font = ImageFont.truetype("NotoSansDevanagari-Regular.ttf", 20)
	elif selected_lang == "gu" and os.path.exists("NotoSansGujarati-Regular.ttf"):
	   font = ImageFont.truetype("NotoSansGujarati-Regular.ttf", 20)
	elif selected_lang == "bn" and os.path.exists("NotoSansBengali-Regular.ttf"):
	   font = ImageFont.truetype("NotoSansBengali-Regular.ttf", 20)
	

	
	
	else:
	   # fallback to a system font
	   font_path = r"arial.ttf"
	   font = ImageFont.truetype(font_path, 20) if os.path.exists(font_path) else ImageFont.load_default()


	subs      = pysrt.open(srt_path)
	subtitle_clips = []
	pad_x, pad_y    = 8, 4
	margin_bot      = 30
	fade            = 0.1

	for sub in subs:
		start = sub.start.ordinal/1000.0
		end   = sub.end.ordinal/1000.0
		txt   = sub.text.replace("\n"," ")

		# wrap to max two lines
		lines = textwrap.wrap(txt, width=50)
		if len(lines) > 2:
			lines = [ lines[0], " ".join(lines[1:]) ]

		# measure text
		dummy = Image.new("RGBA", (1,1))
		d     = ImageDraw.Draw(dummy)
		widths  = [d.textbbox((0,0), ln, font=font)[2] for ln in lines]
		heights = [d.textbbox((0,0), ln, font=font)[3] for ln in lines]
		box_w = min(max(widths) + 2*pad_x, int(W*0.8))
		box_h = sum(heights) + (len(lines)+1)*pad_y

		# render subtitle image
		img = Image.new("RGBA", (box_w, box_h), (0,0,0,0))
		dd  = ImageDraw.Draw(img)
		dd.rectangle([(0,0),(box_w,box_h)], fill=(0,0,0,180))
		y = pad_y
		for ln in lines:
			w,h = d.textbbox((0,0), ln, font=font)[2:4]
			x = (box_w - w)//2
			dd.text((x, y), ln, font=font, fill="white")
			y += h + pad_y

		arr = np.array(img)
		clip = (ImageClip(arr, ismask=False)
				.set_start(start).set_end(end)
				.set_position(("center", H - box_h - margin_bot))
				.fadein(fade).fadeout(fade)
			   )
		subtitle_clips.append(clip)

	final = CompositeVideoClip([video, *subtitle_clips])
	final.write_videofile(
		out_path,
		codec="libx264", audio_codec="aac",
		fps=fps, threads=os.cpu_count()
	)





# ─── 3) GENERATE VIDEO: slides + moving subtitles ───────────────────────
def generate_video(slides, prefix, rebuild=False):
    # 1) synthesize audio
    audio_files = synthesize_slides(slides, prefix, selected_lang, selected_tld)
    st.session_state[f"audio_files_{prefix}"] = audio_files

    common_heading = extract_heading(slides[0])

    # 2) fetch images with proper fallback
    img_paths = []
    for i in range(len(slides)):
        found = False
        
        # First check for uploaded media files
        for ext in [".mp4", ".mov", ".jpg", ".jpeg", ".png"]:
            path = os.path.join(IMG_DIR, f"{prefix}_slide_{i}{ext}")
            if os.path.exists(path):
                img_paths.append(path)
                found = True
                break
        
        # If no uploaded media, try Unsplash
        if not found:
            unsplash_path = fetch_image_for_slide(slides[i], prefix, i)
            if unsplash_path and os.path.exists(unsplash_path):
                img_paths.append(unsplash_path)
                found = True
        
        # Final fallback - create a default path that won't break the code
        if not found:
            # Create a default image path that we can check for later
            default_path = os.path.join(IMG_DIR, f"{prefix}_slide_{i}_default.jpg")
            img_paths.append(default_path)

    # 3) build slide clips with safe path handling
    def _build(params):
        txt, aud, img_or_vid_path = params
        
        # Handle None or non-existent paths
        if img_or_vid_path is None or not os.path.exists(img_or_vid_path):
            # Use a solid color background as fallback
            W, H = 640, 360
            bg = Image.new("RGB", (W, H), (30, 30, 70))  # Dark blue background
            bg_clip = ImageClip(np.array(bg)).set_duration(AudioFileClip(aud).duration)
        else:
            audio = AudioFileClip(aud)
            duration = audio.duration
            audio.close()
            
            heading_short = shorten(common_heading, width=60, placeholder="…")
            ext = os.path.splitext(img_or_vid_path)[-1].lower()
            
            if ext in ['.mp4', '.mov']:
                # Use video background
                bg_clip = VideoFileClip(img_or_vid_path).subclip(0, min(duration, VideoFileClip(img_or_vid_path).duration))
                bg_clip = bg_clip.resize((640, 360)).set_duration(duration)
            else:
                # Use image background
                bg_clip = make_slide(heading_short, duration, img_or_vid_path)

        clip = bg_clip.set_audio(AudioFileClip(aud))
        return clip

    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as ex:
        clips = list(ex.map(_build, zip(slides, audio_files, img_paths)))

    # 4) concatenate + write raw
    raw = concatenate_videoclips(clips, method="compose")
    pdf_folder = os.path.join(VIDEO_DIR, prefix)
    os.makedirs(pdf_folder, exist_ok=True)

    tag = f"{selected_lang}" + (f"_{selected_tld}" if selected_tld else "")
    raw_filename    = f"{prefix}_{tag}.mp4"
    subtitled_fname = f"{prefix}_{tag}_subtitled.mp4"

    raw_path = os.path.join(pdf_folder, raw_filename)
    subtitled_out = os.path.join(pdf_folder, subtitled_fname)
    
    # write raw
    raw.write_videofile(raw_path, fps=24, codec="libx264", audio_codec="aac", 
                        threads=os.cpu_count(), ffmpeg_params=["-preset","ultrafast","-crf","30"], logger=None)

    # write SRT
    srt_path = write_srt(slides, audio_files, prefix)

    # burn moving subtitles
    burn_subtitles_pil(raw_path, srt_path, subtitled_out)
    try:
        os.remove(raw_path)
    except OSError:
        pass

    return subtitled_out, srt_path


def get_slide_start_times(audio_files):
	starts, current = [], 0.0
	for wav in audio_files:
		starts.append(current)
		dur = AudioFileClip(wav).duration
		current += dur
	return starts


# uploaded = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

# Make a 2-column layout just like before


STOPWORDS = {"a","an","the","and","or","but","in","on","at","for","to","with"}
stopwords = STOPWORDS 

col1, col2, col3 = st.columns([3,6,3])

with col1:
	# 1) Language pick
	st.markdown("<h2 style='font-size:24px; font-weight:700'>🎙️ Language</h2>", unsafe_allow_html=True)
	voice_tld = st.selectbox("Choose Language", options=list(VOICE_OPTIONS.keys()))
	selected_lang, selected_tld = VOICE_OPTIONS[voice_tld]


	
	if "last_lang" not in st.session_state:
		st.session_state["last_lang"] = None
	if st.session_state["last_lang"] != selected_lang:
		# For each prefix with slides, regenerate translations
		for key in list(st.session_state.keys()):
			if key.startswith("slides_"):
				prefix = key.split("_", 1)[1]
				eng_slides = st.session_state[f"slides_{prefix}"]
				# Translate each slide
				translated = [
					translator.translate(text, dest=selected_lang).text
					for text in eng_slides
				]
				st.session_state[f"translated_slides_{prefix}"] = translated
		st.session_state["last_lang"] = selected_lang

	# 2) Upload PDF
	st.markdown("<h2  style='font-size:24px; font-weight:700'>📄 Upload PDF</h2>", unsafe_allow_html=True)
	uploaded_files = st.file_uploader("Upload file", type="pdf", accept_multiple_files=True, label_visibility="collapsed")


	# 2a) As soon as the PDF arrives, generate slides once:
	if uploaded_files:
		pdf = uploaded_files[0]
		prefix = os.path.splitext(pdf.name)[0]
		if f"slides_{prefix}" not in st.session_state:
			pages  = extract_pages_from_bytes(pdf)
			summary= summarize(pages, STOPWORDS)
			slides = split_into_slides(summary)
			st.session_state[f"slides_{prefix}"]    = slides
			st.session_state[f"generated_{prefix}"]  = True  # mark that slides exist

	

def rebuild_video_with_edits(slides, prefix):
	st.info("Re-building video with your edits…")
	fetch_image_for_slide.cache_clear()
	out_vid, out_srt = generate_video(slides, prefix, rebuild=True)
	st.session_state[f"video_path_{prefix}_{tag}"] = out_vid
	st.session_state[f"generated_{prefix}_{tag}"]  = True
	st.success("Re-build complete!")


# # hard-coded stopword set; no widget shown
STOPWORDS = {"a","an","the","and","or","but","in","on","at","for","to","with"}
stopwords = STOPWORDS 


if uploaded_files:

	for pdf_file in uploaded_files:
		key    = pdf_file.name
		prefix = key.replace(" ", "_").rsplit(".", 1)[0]

		# 1) generate slides on first upload
		if f"slides_{prefix}" not in st.session_state:
			pages   = extract_pages_from_bytes(pdf_file)
			summary = summarize(pages, STOPWORDS)
			slides  = split_into_slides(summary)
			st.session_state[f"slides_{prefix}"]   = slides
			# mark as "not yet generated"
			st.session_state[f"generated_{prefix}"] = False
			

		
		else:
			slides_to_display = st.session_state[f"slides_{prefix}"]

		# ─── REPLACE MEDIA UI ────────────────────────────────────────────────
		with col1:
			st.markdown("<h2  style='font-size:24px; font-weight:700'	>✏️ Replace Image/Video or Audio</h2>", unsafe_allow_html=True)
			slides = st.session_state[f"slides_{prefix}"]

			chosen     = st.selectbox("Slide # to replace", list(range(len(slides))))
			media_type = st.radio("Replace", ["Image/Video", "Audio"], horizontal=True)



			if media_type == "Image/Video":
				up = st.file_uploader(
					f"Upload new image/video for slide {chosen}",
					type=["png", "jpg", "jpeg", "mp4", "mov"],
					key=f"mediaedit_{prefix}_{chosen}"
				)
				if up:
					ext  = up.name.split(".")[-1]
					dest = os.path.join(IMG_DIR, f"{prefix}_slide_{chosen}.{ext}")
					# delete old, save new, clear cache…
					for old_ext in [".png", ".jpg", ".jpeg", ".mp4", ".mov"]:
						old_file = os.path.join(IMG_DIR, f"{prefix}_slide_{chosen}{old_ext}")
						if os.path.exists(old_file):
							os.remove(old_file)
					with open(dest, "wb") as f:
						f.write(up.read())
					fetch_image_for_slide.cache_clear()
					st.success("✅ Image/Video replaced!")

			else:  # Audio
				up = st.file_uploader(
					f"Upload new audio for slide {chosen}",
					type=["mp3"],
					key=f"audioedit_{prefix}_{chosen}"
				)
				if up:
					tag = f"{selected_lang}" + (f"_{selected_tld}" if selected_tld else "")
					dest = os.path.join(AUDIO_DIR, f"{prefix}_{tag}_scene_{chosen:03d}.mp3")
					with open(dest, "wb") as f:
						f.write(up.read())
					st.success("✅ Audio replaced!")

			# Rebuild button writes to same key as initial generate
			if st.button("Rebuild video with edits", key=f"rebuild_{prefix}"):
				out_vid, out_srt = generate_video(slides, prefix, rebuild=True)
				st.session_state[f"video_path_{prefix}"]   = out_vid
				st.session_state[f"generated_{prefix}"]    = True
				st.success("🎉 Video rebuilt!")

		# ─── EDIT TRANSCRIPT ─────────────────────────────────────────────────
		with col3:
			audio_files = st.session_state.get(f"audio_files_{prefix}", [])
			# if audio_files:
			# 	starts = get_slide_start_times(audio_files)
			# 	start_ts = format_timestamp(starts[chosen])
			# 	st.markdown(f"**Slide {chosen} starts at {start_ts}**")
			st.header("📝 Edit Transcript")
			# orig   = st.session_state[f"slides_{prefix}"]
			source_key = (
				f"translated_slides_{prefix}"
				if f"translated_slides_{prefix}" in st.session_state
				else f"slides_{prefix}"
			)
			orig = st.session_state[source_key]
			edited = st.text_area(f"Slides for {key}", "\n\n".join(orig), height=300)
			if st.button("Update Transcript", key=f"upd_{prefix}"):
				new_slides = [s.strip() for s in edited.split("\n\n") if s.strip()]
				st.session_state[f"slides_{prefix}"]  = new_slides
				st.session_state[f"generated_{prefix}"] = False
				st.success("Transcript updated!")
			

			
			
			

		# ─── VIDEO GENERATION & DISPLAY ────────────────────────────────────
		with col2:
			st.header("🎞️ Generate Video")
			# first-time generate
			if not st.session_state[f"generated_{prefix}"]:
				if st.button("Generate Video", key=f"gen_{prefix}"):
					# clean up old audio
					for fn in os.listdir(AUDIO_DIR):
						if fn.startswith(f"{prefix}_") and fn.endswith(".mp3"):
							try: os.remove(os.path.join(AUDIO_DIR, fn))
							except: pass

					with st.spinner("🔧 Building video…"):
						vid, srt = generate_video(st.session_state[f"slides_{prefix}"], prefix)
						# write to the same key as rebuild
						st.session_state[f"video_path_{prefix}"]   = vid
						st.session_state[f"generated_{prefix}"]    = True
						st.success("✅ Video ready!")

			# and in all cases, if a video_path exists, show it:
			if st.session_state.get(f"video_path_{prefix}"):
				st.video(st.session_state[f"video_path_{prefix}"])








