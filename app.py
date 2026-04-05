"""
MoneyPrinterV2 — Hugging Face Spaces Gradio UI

Generates YouTube Shorts (video only, no upload) using:
  - HF Inference API for LLM text generation
  - Gemini API (Nano Banana 2) for AI image generation
  - KittenTTS for text-to-speech
  - faster-whisper for subtitle generation
  - MoviePy for video assembly
"""

import os
import sys
import json
import tempfile
import shutil
import traceback

# Ensure src/ is importable (same trick as src/main.py)
_root = os.path.dirname(os.path.abspath(__file__))
_src = os.path.join(_root, "src")
if _src not in sys.path:
    sys.path.insert(0, _src)
# Set sys.path[0] so config.ROOT_DIR resolves correctly
if sys.path[0] != _src:
    sys.path.insert(0, _src)

import gradio as gr
from config import assert_folder_structure, ROOT_DIR
from llm_provider import select_model, list_models, generate_text
from status import info, success, error, warning

# Ensure .mp directory exists
assert_folder_structure()

# Available TTS voices (KittenTTS)
TTS_VOICES = ["Jasper", "Bella", "Luna", "Bruno", "Rosie", "Hugo", "Kiki", "Leo"]

LANGUAGES = ["English", "Korean", "Spanish", "French", "German", "Japanese", "Chinese", "Portuguese", "Russian", "Arabic"]

# LLM model choices
LLM_MODELS = list_models()


def generate_short(
    niche: str,
    language: str,
    llm_model: str,
    tts_voice: str,
    sentence_length: int,
    progress=gr.Progress(track_tqdm=True),
):
    """Main generation pipeline — returns (video_path, metadata_json, log_text)."""
    log_lines = []

    def log(msg):
        log_lines.append(msg)

    if not niche.strip():
        return None, {}, "Please enter a niche/topic."

    # Select LLM model
    select_model(llm_model)
    log(f"Using LLM model: {llm_model}")

    # Override TTS voice via env (config reads it)
    os.environ["TTS_VOICE"] = tts_voice
    # Override script sentence length
    os.environ["SCRIPT_SENTENCE_LENGTH"] = str(int(sentence_length))

    try:
        # Import YouTube class (browser-free mode)
        from classes.YouTube import YouTube
        from classes.Tts import TTS

        log("Initializing YouTube pipeline (browser-free)...")
        yt = YouTube(
            account_uuid="gradio-session",
            account_nickname="gradio-user",
            fp_profile_path="",
            niche=niche,
            language=language,
            use_browser=False,
        )

        # Step 1: Generate topic
        log("Generating topic...")
        topic = yt.generate_topic()
        log(f"Topic: {topic}")

        # Step 2: Generate script
        log("Generating script...")
        script = yt.generate_script()
        log(f"Script: {script[:200]}...")

        # Step 3: Generate metadata
        log("Generating metadata (title, description)...")
        metadata = yt.generate_metadata()
        log(f"Title: {metadata['title']}")

        # Step 4: Generate image prompts
        log("Generating image prompts...")
        prompts = yt.generate_prompts()
        log(f"Generated {len(prompts)} image prompts")

        # Step 5: Generate images
        log("Generating images...")
        generated_count = 0
        for i, prompt in enumerate(prompts):
            log(f"  Image {i+1}/{len(prompts)}: {prompt[:80]}...")
            result = yt.generate_image(prompt)
            if result:
                generated_count += 1
        log(f"Generated {generated_count}/{len(prompts)} images")

        if generated_count == 0:
            return None, metadata, "\n".join(log_lines + ["ERROR: No images were generated. Check your GEMINI_API_KEY."])

        # Step 6: TTS
        log("Generating speech (TTS)...")
        tts = TTS()
        yt.generate_script_to_speech(tts)
        log("TTS complete")

        # Step 7: Combine into video
        log("Combining into final video (this may take a few minutes)...")
        video_path = yt.combine()
        log(f"Video generated: {video_path}")

        full_metadata = {
            "title": metadata["title"],
            "description": metadata["description"],
            "topic": topic,
            "script": script,
            "image_prompts": prompts,
            "images_generated": generated_count,
        }

        return video_path, full_metadata, "\n".join(log_lines)

    except Exception as e:
        log_lines.append(f"ERROR: {e}")
        log_lines.append(traceback.format_exc())
        return None, {}, "\n".join(log_lines)


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="MoneyPrinterV2 — YouTube Shorts Generator", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# MoneyPrinterV2 — YouTube Shorts Generator")
    gr.Markdown(
        "Generate YouTube Shorts videos automatically using AI. "
        "The pipeline generates a topic, script, images, speech, subtitles, and assembles them into a video."
    )

    with gr.Row():
        with gr.Column(scale=1):
            niche_input = gr.Textbox(
                label="Niche / Topic",
                placeholder="e.g. 'artificial intelligence', 'cooking tips', 'space exploration'",
                lines=2,
            )
            language_input = gr.Dropdown(
                choices=LANGUAGES,
                value="English",
                label="Language",
            )
            llm_model_input = gr.Dropdown(
                choices=LLM_MODELS,
                value=LLM_MODELS[0] if LLM_MODELS else "",
                label="LLM Model",
            )
            tts_voice_input = gr.Dropdown(
                choices=TTS_VOICES,
                value="Jasper",
                label="TTS Voice",
            )
            sentence_length_input = gr.Slider(
                minimum=2,
                maximum=8,
                value=4,
                step=1,
                label="Script Sentence Count",
            )
            generate_btn = gr.Button("Generate Video", variant="primary", size="lg")

        with gr.Column(scale=2):
            video_output = gr.Video(label="Generated Video")
            metadata_output = gr.JSON(label="Metadata")
            log_output = gr.Textbox(label="Progress Log", lines=15, interactive=False)

    generate_btn.click(
        fn=generate_short,
        inputs=[niche_input, language_input, llm_model_input, tts_voice_input, sentence_length_input],
        outputs=[video_output, metadata_output, log_output],
    )

    gr.Markdown(
        "---\n"
        "**Required HF Space Secrets:** `HF_TOKEN` (for LLM), `GEMINI_API_KEY` (for image generation)\n\n"
        "**Note:** This demo generates videos only. YouTube upload requires browser automation and is not available on HF Spaces."
    )


if __name__ == "__main__":
    demo.launch()
