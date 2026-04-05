import os
import sys
import json
import srt_equalizer

from termcolor import colored

ROOT_DIR = os.path.dirname(sys.path[0])


def is_running_in_spaces() -> bool:
    """Returns True when running inside a Hugging Face Space."""
    return bool(os.environ.get("SPACE_ID"))


def _load_config() -> dict:
    """
    Loads config.json if available; falls back to environment variables
    when running on HF Spaces or when the file is missing.
    """
    config_path = os.path.join(ROOT_DIR, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return json.load(f)

    # Fallback: build minimal config from environment variables
    return {
        "verbose": os.environ.get("VERBOSE", "true").lower() == "true",
        "firefox_profile": "",
        "headless": True,
        "ollama_base_url": os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
        "ollama_model": os.environ.get("OLLAMA_MODEL", ""),
        "twitter_language": os.environ.get("TWITTER_LANGUAGE", "English"),
        "nanobanana2_api_base_url": os.environ.get(
            "NANOBANANA2_API_BASE_URL",
            "https://generativelanguage.googleapis.com/v1beta",
        ),
        "nanobanana2_api_key": os.environ.get("GEMINI_API_KEY", ""),
        "nanobanana2_model": os.environ.get(
            "NANOBANANA2_MODEL", "gemini-3.1-flash-image-preview"
        ),
        "nanobanana2_aspect_ratio": os.environ.get("NANOBANANA2_ASPECT_RATIO", "9:16"),
        "threads": int(os.environ.get("THREADS", "2")),
        "zip_url": os.environ.get("ZIP_URL", ""),
        "is_for_kids": False,
        "stt_provider": os.environ.get("STT_PROVIDER", "local_whisper"),
        "whisper_model": os.environ.get("WHISPER_MODEL", "tiny"),
        "whisper_device": os.environ.get("WHISPER_DEVICE", "cpu"),
        "whisper_compute_type": os.environ.get("WHISPER_COMPUTE_TYPE", "int8"),
        "assembly_ai_api_key": os.environ.get("ASSEMBLYAI_API_KEY", ""),
        "tts_voice": os.environ.get("TTS_VOICE", "Jasper"),
        "font": os.environ.get("FONT", "bold_font.ttf"),
        "imagemagick_path": os.environ.get("IMAGEMAGICK_PATH", "/usr/bin/convert"),
        "script_sentence_length": int(os.environ.get("SCRIPT_SENTENCE_LENGTH", "4")),
        "email": {"smtp_server": "", "smtp_port": 587, "username": "", "password": ""},
        "post_bridge": {
            "enabled": False,
            "api_key": "",
            "platforms": [],
            "account_ids": [],
            "auto_crosspost": False,
        },
    }


def assert_folder_structure() -> None:
    """
    Make sure that the nessecary folder structure is present.

    Returns:
        None
    """
    # Create the .mp folder
    if not os.path.exists(os.path.join(ROOT_DIR, ".mp")):
        if get_verbose():
            print(colored(f"=> Creating .mp folder at {os.path.join(ROOT_DIR, '.mp')}", "green"))
        os.makedirs(os.path.join(ROOT_DIR, ".mp"))

def get_first_time_running() -> bool:
    """
    Checks if the program is running for the first time by checking if .mp folder exists.

    Returns:
        exists (bool): True if the program is running for the first time, False otherwise
    """
    return not os.path.exists(os.path.join(ROOT_DIR, ".mp"))

def get_email_credentials() -> dict:
    return _load_config()["email"]

def get_verbose() -> bool:
    return _load_config()["verbose"]

def get_firefox_profile_path() -> str:
    return _load_config()["firefox_profile"]

def get_headless() -> bool:
    return _load_config()["headless"]

def get_ollama_base_url() -> str:
    return _load_config().get("ollama_base_url", "http://127.0.0.1:11434")

def get_ollama_model() -> str:
    return _load_config().get("ollama_model", "")

def get_twitter_language() -> str:
    return _load_config()["twitter_language"]

def get_nanobanana2_api_base_url() -> str:
    return _load_config().get(
        "nanobanana2_api_base_url",
        "https://generativelanguage.googleapis.com/v1beta",
    )

def get_nanobanana2_api_key() -> str:
    configured = _load_config().get("nanobanana2_api_key", "")
    return configured or os.environ.get("GEMINI_API_KEY", "")

def get_nanobanana2_model() -> str:
    return _load_config().get("nanobanana2_model", "gemini-3.1-flash-image-preview")

def get_nanobanana2_aspect_ratio() -> str:
    return _load_config().get("nanobanana2_aspect_ratio", "9:16")

def get_threads() -> int:
    return _load_config()["threads"]

def get_zip_url() -> str:
    return _load_config()["zip_url"]

def get_is_for_kids() -> bool:
    return _load_config()["is_for_kids"]

def get_google_maps_scraper_zip_url() -> str:
    return _load_config()["google_maps_scraper"]

def get_google_maps_scraper_niche() -> str:
    return _load_config()["google_maps_scraper_niche"]

def get_scraper_timeout() -> int:
    return _load_config()["scraper_timeout"] or 300

def get_outreach_message_subject() -> str:
    return _load_config()["outreach_message_subject"]

def get_outreach_message_body_file() -> str:
    return _load_config()["outreach_message_body_file"]

def get_tts_voice() -> str:
    return _load_config().get("tts_voice", "Jasper")

def get_assemblyai_api_key() -> str:
    return _load_config()["assembly_ai_api_key"]

def get_stt_provider() -> str:
    return _load_config().get("stt_provider", "local_whisper")

def get_whisper_model() -> str:
    return _load_config().get("whisper_model", "base")

def get_whisper_device() -> str:
    return _load_config().get("whisper_device", "auto")

def get_whisper_compute_type() -> str:
    return _load_config().get("whisper_compute_type", "int8")
    
def equalize_subtitles(srt_path: str, max_chars: int = 10) -> None:
    """
    Equalizes the subtitles in a SRT file.

    Args:
        srt_path (str): The path to the SRT file
        max_chars (int): The maximum amount of characters in a subtitle

    Returns:
        None
    """
    srt_equalizer.equalize_srt_file(srt_path, srt_path, max_chars)
    
def get_font() -> str:
    return _load_config()["font"]

def get_fonts_dir() -> str:
    return os.path.join(ROOT_DIR, "fonts")

def get_imagemagick_path() -> str:
    path = _load_config().get("imagemagick_path", "")
    if not path and is_running_in_spaces():
        return "/usr/bin/convert"
    return path

def get_script_sentence_length() -> int:
    val = _load_config().get("script_sentence_length")
    return val if val is not None else 4

def get_post_bridge_config() -> dict:
    defaults = {
        "enabled": False,
        "api_key": "",
        "platforms": ["tiktok", "instagram"],
        "account_ids": [],
        "auto_crosspost": False,
    }
    supported_platforms = {"tiktok", "instagram"}

    config_json = _load_config()

    raw_config = config_json.get("post_bridge", {})
    if not isinstance(raw_config, dict):
        raw_config = {}

    raw_platforms = raw_config.get("platforms")
    normalized_platforms = []
    seen_platforms = set()

    if raw_platforms is None:
        normalized_platforms = defaults["platforms"].copy()
    elif isinstance(raw_platforms, list):
        for platform in raw_platforms:
            normalized_platform = str(platform).strip().lower()
            if (
                normalized_platform in supported_platforms
                and normalized_platform not in seen_platforms
            ):
                normalized_platforms.append(normalized_platform)
                seen_platforms.add(normalized_platform)
    else:
        normalized_platforms = []

    raw_account_ids = raw_config.get("account_ids", defaults["account_ids"])
    normalized_account_ids = []
    if isinstance(raw_account_ids, list):
        for account_id in raw_account_ids:
            try:
                normalized_account_ids.append(int(account_id))
            except (TypeError, ValueError):
                continue

    api_key = str(raw_config.get("api_key", "")).strip()
    if not api_key:
        api_key = os.environ.get("POST_BRIDGE_API_KEY", "").strip()

    return {
        "enabled": bool(raw_config.get("enabled", defaults["enabled"])),
        "api_key": api_key,
        "platforms": normalized_platforms,
        "account_ids": normalized_account_ids,
        "auto_crosspost": bool(
            raw_config.get("auto_crosspost", defaults["auto_crosspost"])
        ),
    }
