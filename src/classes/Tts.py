import os

from config import ROOT_DIR, get_tts_voice, is_running_in_spaces

# Voice mapping: friendly name -> edge-tts voice ID
EDGE_TTS_VOICES = {
    "Jasper": "en-US-GuyNeural",
    "Bella": "en-US-JennyNeural",
    "Luna": "en-GB-SoniaNeural",
    "Bruno": "en-US-ChristopherNeural",
    "Rosie": "en-AU-NatashaNeural",
    "Hugo": "en-GB-RyanNeural",
    "Kiki": "en-US-AriaNeural",
    "Leo": "en-US-DavisNeural",
}


def _use_edge_tts() -> bool:
    """Use edge-tts when KittenTTS is not available (e.g. on HF Spaces)."""
    if is_running_in_spaces():
        return True
    try:
        from kittentts import KittenTTS  # noqa: F401
        return False
    except ImportError:
        return True


class TTS:
    def __init__(self) -> None:
        self._voice = get_tts_voice()
        self._use_edge = _use_edge_tts()

        if not self._use_edge:
            import soundfile  # noqa: F401 — ensure available
            from kittentts import KittenTTS as KittenModel
            self._model = KittenModel("KittenML/kitten-tts-mini-0.8")
            self._sample_rate = 24000
        else:
            self._model = None

    def synthesize(self, text, output_file=os.path.join(ROOT_DIR, ".mp", "audio.wav")):
        if self._use_edge:
            return self._synthesize_edge(text, output_file)
        return self._synthesize_kitten(text, output_file)

    def _synthesize_kitten(self, text, output_file):
        import soundfile as sf
        audio = self._model.generate(text, voice=self._voice)
        sf.write(output_file, audio, self._sample_rate)
        return output_file

    def _synthesize_edge(self, text, output_file):
        import asyncio
        import edge_tts

        voice_id = EDGE_TTS_VOICES.get(self._voice, "en-US-GuyNeural")

        # edge-tts outputs mp3; we write to mp3 then keep as-is
        # MoviePy can handle mp3 audio via ffmpeg
        mp3_path = output_file.rsplit(".", 1)[0] + ".mp3"

        async def _generate():
            communicate = edge_tts.Communicate(text, voice_id)
            await communicate.save(mp3_path)

        asyncio.run(_generate())

        # Convert mp3 to wav for compatibility with the rest of the pipeline
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_mp3(mp3_path)
            audio.export(output_file, format="wav")
            os.remove(mp3_path)
        except ImportError:
            # If pydub not available, just use the mp3 directly
            # Rename mp3 to the expected output path
            if os.path.exists(output_file):
                os.remove(output_file)
            os.rename(mp3_path, output_file)

        return output_file
