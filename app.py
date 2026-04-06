import os
import tempfile
import urllib.request
import fal_client
import fal_client.client
import gradio as gr
from concurrent.futures import ThreadPoolExecutor


def call_fal(model_id, arguments):
    """Call fal API with content-policy error handling."""
    try:
        return fal_client.subscribe(model_id, arguments=arguments)
    except fal_client.client.FalClientHTTPError as e:
        msg = str(e)
        if "content_policy_violation" in msg:
            raise gr.Error(
                "xAI 콘텐츠 정책 위반: 프롬프트가 xAI 서버의 콘텐츠 필터에 의해 거부되었습니다. "
                "이 필터는 xAI에서 강제 적용하며 비활성화할 수 없습니다. 프롬프트를 수정해 주세요."
            )
        raise gr.Error(f"API 오류: {msg}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_path(filepath):
    """Resolve a Gradio file path - handles both str and NamedString/file objects."""
    if filepath is None:
        return None
    if isinstance(filepath, str):
        return filepath
    if hasattr(filepath, 'name'):
        return filepath.name
    if hasattr(filepath, 'path'):
        return filepath.path
    return str(filepath)


def encode_file(filepath):
    """Encode a local file as a data URI for fal API (avoids CDN upload auth)."""
    resolved = _resolve_path(filepath)
    if resolved is None:
        return None
    return fal_client.encode_file(resolved)


def encode_files(filepaths):
    """Encode multiple files in parallel, return list of data URIs."""
    if not filepaths:
        return []
    with ThreadPoolExecutor(max_workers=min(len(filepaths), 7)) as pool:
        return list(pool.map(encode_file, filepaths))


def download_video(url):
    """Download video from URL to a temp file and return the path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    urllib.request.urlretrieve(url, tmp.name)
    return tmp.name


def download_image(url):
    """Download image from URL to a temp file and return the path."""
    ext = ".png" if "png" in url else ".webp" if "webp" in url else ".jpg"
    tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    urllib.request.urlretrieve(url, tmp.name)
    return tmp.name


# ---------------------------------------------------------------------------
# Tab 1: Text to Image
# ---------------------------------------------------------------------------

def text_to_image(prompt, num_images, aspect_ratio, resolution, output_format):
    if not prompt:
        raise gr.Error("프롬프트를 입력해 주세요")

    result = call_fal("xai/grok-imagine-image", {
        "prompt": prompt,
        "num_images": int(num_images),
        "aspect_ratio": aspect_ratio,
        "resolution": resolution,
        "output_format": output_format,
    })

    output_images = []
    for img in result.get("images", []):
        path = download_image(img["url"])
        output_images.append(path)

    revised = result.get("revised_prompt", "")
    return output_images, revised


# ---------------------------------------------------------------------------
# Tab 2: Image Edit
# ---------------------------------------------------------------------------

def image_edit(prompt, images, num_images, resolution, output_format):
    if not prompt:
        raise gr.Error("프롬프트를 입력해 주세요")
    if not images or len(images) == 0:
        raise gr.Error("최소 1개의 이미지를 업로드해 주세요")

    image_urls = encode_files(images)

    result = call_fal("xai/grok-imagine-image/edit", {
        "prompt": prompt,
        "image_urls": image_urls,
        "num_images": int(num_images),
        "resolution": resolution,
        "output_format": output_format,
    })

    output_images = []
    for img in result.get("images", []):
        path = download_image(img["url"])
        output_images.append(path)

    revised = result.get("revised_prompt", "")
    return output_images, revised


# ---------------------------------------------------------------------------
# Tab 3: Text to Video
# ---------------------------------------------------------------------------

def text_to_video(prompt, duration, aspect_ratio, resolution):
    if not prompt:
        raise gr.Error("프롬프트를 입력해 주세요")

    result = call_fal("xai/grok-imagine-video/text-to-video", {
        "prompt": prompt,
        "duration": int(duration),
        "aspect_ratio": aspect_ratio,
        "resolution": resolution,
    })

    return download_video(result["video"]["url"])


# ---------------------------------------------------------------------------
# Tab 4: Reference to Video
# ---------------------------------------------------------------------------

def reference_to_video(prompt, ref_images, duration, aspect_ratio, resolution):
    if not prompt:
        raise gr.Error("프롬프트를 입력해 주세요")
    if not ref_images or len(ref_images) == 0:
        raise gr.Error("최소 1개의 참조 이미지를 업로드해 주세요")

    ref_urls = encode_files(ref_images)

    result = call_fal("xai/grok-imagine-video/reference-to-video", {
        "prompt": prompt,
        "reference_image_urls": ref_urls,
        "duration": int(duration),
        "aspect_ratio": aspect_ratio,
        "resolution": resolution,
    })

    return download_video(result["video"]["url"])


# ---------------------------------------------------------------------------
# Tab 5: Image to Video
# ---------------------------------------------------------------------------

def image_to_video(prompt, image, duration, aspect_ratio, resolution):
    if not prompt:
        raise gr.Error("프롬프트를 입력해 주세요")
    if image is None:
        raise gr.Error("이미지를 업로드해 주세요")

    image_url = encode_file(image)

    result = call_fal("xai/grok-imagine-video/image-to-video", {
        "prompt": prompt,
        "image_url": image_url,
        "duration": int(duration),
        "aspect_ratio": aspect_ratio,
        "resolution": resolution,
    })

    return download_video(result["video"]["url"])


# ---------------------------------------------------------------------------
# Tab 6: Edit Video
# ---------------------------------------------------------------------------

def edit_video(prompt, video, resolution):
    if not prompt:
        raise gr.Error("프롬프트를 입력해 주세요")
    if video is None:
        raise gr.Error("비디오를 업로드해 주세요")

    video_url = encode_file(video)

    result = call_fal("xai/grok-imagine-video/edit-video", {
        "prompt": prompt,
        "video_url": video_url,
        "resolution": resolution,
    })

    return download_video(result["video"]["url"])


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

ASPECT_RATIOS = ["16:9", "4:3", "3:2", "1:1", "2:3", "3:4", "9:16"]
IMAGE_ASPECT_RATIOS = ["1:1", "2:1", "20:9", "19.5:9", "16:9", "4:3", "3:2", "2:3", "3:4", "9:16", "9:19.5", "9:20", "1:2"]

with gr.Blocks(title="Grok Imagine Studio") as demo:
    gr.Markdown("# Grok Imagine Studio\nxAI Grok Imagine API 기반 이미지 & 비디오 생성 스튜디오 (fal.ai)")

    with gr.Tabs():
        # ---- Tab 1: Text to Image ----
        with gr.Tab("텍스트 to 이미지"):
            gr.Markdown(
                "### 텍스트로 이미지 생성\n"
                "텍스트 프롬프트를 입력하면 AI가 이미지를 생성합니다. "
                "최대 4장까지 동시에 생성할 수 있으며, 다양한 비율과 해상도를 지원합니다."
            )
            with gr.Row():
                with gr.Column():
                    t2i_prompt = gr.Textbox(label="프롬프트", lines=3, max_lines=10, placeholder="생성하고 싶은 이미지를 설명해 주세요...")
                    with gr.Row():
                        t2i_num = gr.Slider(minimum=1, maximum=4, step=1, value=1, label="생성 이미지 수")
                        t2i_ar = gr.Dropdown(choices=IMAGE_ASPECT_RATIOS, value="1:1", label="비율")
                    with gr.Row():
                        t2i_res = gr.Dropdown(choices=["1k", "2k"], value="1k", label="해상도")
                        t2i_fmt = gr.Dropdown(choices=["jpeg", "png", "webp"], value="png", label="출력 형식")
                    t2i_btn = gr.Button("생성하기", variant="primary")
                with gr.Column():
                    t2i_gallery = gr.Gallery(label="생성 결과", columns=2)
                    t2i_revised = gr.Textbox(label="수정된 프롬프트", interactive=False)

            t2i_btn.click(text_to_image, inputs=[t2i_prompt, t2i_num, t2i_ar, t2i_res, t2i_fmt], outputs=[t2i_gallery, t2i_revised])

        # ---- Tab 2: Image Edit ----
        with gr.Tab("이미지 편집"):
            gr.Markdown(
                "### AI 이미지 편집\n"
                "기존 이미지를 업로드하고 텍스트로 편집 내용을 설명하면 AI가 이미지를 수정합니다. "
                "최대 3장의 이미지를 동시에 입력할 수 있습니다."
            )
            with gr.Row():
                with gr.Column():
                    ie_prompt = gr.Textbox(label="프롬프트", lines=3, max_lines=10, placeholder="원하는 편집 내용을 설명해 주세요...")
                    ie_images = gr.File(label="입력 이미지 (최대 3장)", file_count="multiple", file_types=["image"])
                    with gr.Row():
                        ie_num = gr.Slider(minimum=1, maximum=4, step=1, value=1, label="출력 이미지 수")
                        ie_res = gr.Dropdown(choices=["1k", "2k"], value="1k", label="해상도")
                        ie_fmt = gr.Dropdown(choices=["jpeg", "png", "webp"], value="jpeg", label="출력 형식")
                    ie_btn = gr.Button("편집하기", variant="primary")
                with gr.Column():
                    ie_gallery = gr.Gallery(label="편집 결과", columns=2)
                    ie_revised = gr.Textbox(label="수정된 프롬프트", interactive=False)

            ie_btn.click(image_edit, inputs=[ie_prompt, ie_images, ie_num, ie_res, ie_fmt], outputs=[ie_gallery, ie_revised])

        # ---- Tab 3: Text to Video ----
        with gr.Tab("텍스트 to 비디오"):
            gr.Markdown(
                "### 텍스트로 비디오 생성\n"
                "텍스트 프롬프트만으로 최대 15초 길이의 비디오를 생성합니다. "
                "480p 또는 720p 해상도를 선택할 수 있습니다."
            )
            with gr.Row():
                with gr.Column():
                    t2v_prompt = gr.Textbox(label="프롬프트", lines=3, max_lines=10, placeholder="생성하고 싶은 비디오를 설명해 주세요...")
                    with gr.Row():
                        t2v_dur = gr.Slider(minimum=1, maximum=15, step=1, value=6, label="길이 (초)")
                        t2v_ar = gr.Dropdown(choices=ASPECT_RATIOS, value="16:9", label="비율")
                        t2v_res = gr.Dropdown(choices=["480p", "720p"], value="720p", label="해상도")
                    t2v_btn = gr.Button("생성하기", variant="primary")
                with gr.Column():
                    t2v_video = gr.Video(label="생성 결과")

            t2v_btn.click(text_to_video, inputs=[t2v_prompt, t2v_dur, t2v_ar, t2v_res], outputs=[t2v_video])

        # ---- Tab 4: Reference to Video ----
        with gr.Tab("참조 이미지 to 비디오"):
            gr.Markdown(
                "### 참조 이미지 기반 비디오 생성\n"
                "참조 이미지를 업로드하고 프롬프트에서 @Image1, @Image2 등으로 참조하여 비디오를 생성합니다. "
                "최대 7장의 참조 이미지를 사용할 수 있습니다."
            )
            with gr.Row():
                with gr.Column():
                    r2v_prompt = gr.Textbox(label="프롬프트 (@Image1, @Image2... 로 이미지 참조)", lines=3, max_lines=10,
                                           placeholder="예: @Image1의 캐릭터가 @Image2 배경에서 걷고 있는 장면")
                    r2v_images = gr.File(label="참조 이미지 (최대 7장)", file_count="multiple", file_types=["image"])
                    with gr.Row():
                        r2v_dur = gr.Slider(minimum=1, maximum=10, step=1, value=8, label="길이 (초)")
                        r2v_ar = gr.Dropdown(choices=ASPECT_RATIOS, value="16:9", label="비율")
                        r2v_res = gr.Dropdown(choices=["480p", "720p"], value="480p", label="해상도")
                    r2v_btn = gr.Button("생성하기", variant="primary")
                with gr.Column():
                    r2v_video = gr.Video(label="생성 결과")

            r2v_btn.click(reference_to_video, inputs=[r2v_prompt, r2v_images, r2v_dur, r2v_ar, r2v_res], outputs=[r2v_video])

        # ---- Tab 5: Image to Video ----
        with gr.Tab("이미지 to 비디오"):
            gr.Markdown(
                "### 이미지를 비디오로 변환\n"
                "정지 이미지를 업로드하고 원하는 움직임을 설명하면 AI가 이미지를 애니메이션 비디오로 변환합니다. "
                "비율을 'auto'로 설정하면 원본 이미지 비율을 유지합니다."
            )
            with gr.Row():
                with gr.Column():
                    i2v_prompt = gr.Textbox(label="프롬프트", lines=3, max_lines=10, placeholder="원하는 움직임이나 변화를 설명해 주세요...")
                    i2v_image = gr.Image(label="입력 이미지", type="filepath")
                    with gr.Row():
                        i2v_dur = gr.Slider(minimum=1, maximum=15, step=1, value=6, label="길이 (초)")
                        i2v_ar = gr.Dropdown(choices=["auto"] + ASPECT_RATIOS, value="auto", label="비율")
                        i2v_res = gr.Dropdown(choices=["480p", "720p"], value="720p", label="해상도")
                    i2v_btn = gr.Button("생성하기", variant="primary")
                with gr.Column():
                    i2v_video = gr.Video(label="생성 결과")

            i2v_btn.click(image_to_video, inputs=[i2v_prompt, i2v_image, i2v_dur, i2v_ar, i2v_res], outputs=[i2v_video])

        # ---- Tab 6: Edit Video ----
        with gr.Tab("비디오 편집"):
            gr.Markdown(
                "### AI 비디오 편집\n"
                "기존 비디오를 업로드하고 텍스트로 편집 내용을 설명하면 AI가 비디오를 수정합니다. "
                "입력 비디오는 최대 854x480 크기, 8초 길이로 자동 조정됩니다."
            )
            with gr.Row():
                with gr.Column():
                    edv_prompt = gr.Textbox(label="프롬프트", lines=3, max_lines=10, placeholder="원하는 편집 내용을 설명해 주세요...")
                    edv_video = gr.Video(label="입력 비디오")
                    edv_res = gr.Dropdown(choices=["auto", "480p", "720p"], value="auto", label="해상도")
                    edv_btn = gr.Button("편집하기", variant="primary")
                with gr.Column():
                    edv_output = gr.Video(label="편집 결과")

            edv_btn.click(edit_video, inputs=[edv_prompt, edv_video, edv_res], outputs=[edv_output])

demo.queue(default_concurrency_limit=4).launch(theme=gr.themes.Soft(), ssr_mode=False)
