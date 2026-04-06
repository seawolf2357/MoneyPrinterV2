import os
import tempfile
import fal_client
import gradio as gr
from concurrent.futures import ThreadPoolExecutor

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def upload_file(filepath):
    """Upload a local file to fal CDN and return the URL."""
    if filepath is None:
        return None
    return fal_client.upload_file(filepath)


def upload_files(filepaths):
    """Upload multiple files in parallel, return list of URLs."""
    if not filepaths:
        return []
    with ThreadPoolExecutor(max_workers=min(len(filepaths), 7)) as pool:
        return list(pool.map(upload_file, filepaths))


def download_video(url):
    """Download video from URL to a temp file and return the path."""
    import urllib.request
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    urllib.request.urlretrieve(url, tmp.name)
    return tmp.name


def download_image(url):
    """Download image from URL to a temp file and return the path."""
    import urllib.request
    ext = ".jpg"
    if "png" in url:
        ext = ".png"
    elif "webp" in url:
        ext = ".webp"
    tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    urllib.request.urlretrieve(url, tmp.name)
    return tmp.name


# ---------------------------------------------------------------------------
# Tab 1: Image Edit
# ---------------------------------------------------------------------------

def image_edit(prompt, images, num_images, resolution, output_format):
    if not prompt:
        raise gr.Error("Prompt is required")
    if not images or len(images) == 0:
        raise gr.Error("At least one image is required")

    image_urls = upload_files(images)

    result = fal_client.subscribe("xai/grok-imagine-image/edit", arguments={
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
# Tab 2: Text to Video
# ---------------------------------------------------------------------------

def text_to_video(prompt, duration, aspect_ratio, resolution):
    if not prompt:
        raise gr.Error("Prompt is required")

    result = fal_client.subscribe("xai/grok-imagine-video/text-to-video", arguments={
        "prompt": prompt,
        "duration": int(duration),
        "aspect_ratio": aspect_ratio,
        "resolution": resolution,
    })

    video_url = result["video"]["url"]
    return download_video(video_url)


# ---------------------------------------------------------------------------
# Tab 3: Reference to Video
# ---------------------------------------------------------------------------

def reference_to_video(prompt, ref_images, duration, aspect_ratio, resolution):
    if not prompt:
        raise gr.Error("Prompt is required")
    if not ref_images or len(ref_images) == 0:
        raise gr.Error("At least one reference image is required")

    ref_urls = upload_files(ref_images)

    result = fal_client.subscribe("xai/grok-imagine-video/reference-to-video", arguments={
        "prompt": prompt,
        "reference_image_urls": ref_urls,
        "duration": int(duration),
        "aspect_ratio": aspect_ratio,
        "resolution": resolution,
    })

    return download_video(result["video"]["url"])


# ---------------------------------------------------------------------------
# Tab 4: Image to Video
# ---------------------------------------------------------------------------

def image_to_video(prompt, image, duration, aspect_ratio, resolution):
    if not prompt:
        raise gr.Error("Prompt is required")
    if image is None:
        raise gr.Error("Image is required")

    image_url = upload_file(image)

    result = fal_client.subscribe("xai/grok-imagine-video/image-to-video", arguments={
        "prompt": prompt,
        "image_url": image_url,
        "duration": int(duration),
        "aspect_ratio": aspect_ratio,
        "resolution": resolution,
    })

    return download_video(result["video"]["url"])


# ---------------------------------------------------------------------------
# Tab 5: Extend Video
# ---------------------------------------------------------------------------

def extend_video(prompt, video, duration):
    if not prompt:
        raise gr.Error("Prompt is required")
    if video is None:
        raise gr.Error("Video is required")

    video_url = upload_file(video)

    result = fal_client.subscribe("xai/grok-imagine-video/extend-video", arguments={
        "prompt": prompt,
        "video_url": video_url,
        "duration": int(duration),
    })

    return download_video(result["video"]["url"])


# ---------------------------------------------------------------------------
# Tab 6: Edit Video
# ---------------------------------------------------------------------------

def edit_video(prompt, video, resolution):
    if not prompt:
        raise gr.Error("Prompt is required")
    if video is None:
        raise gr.Error("Video is required")

    video_url = upload_file(video)

    result = fal_client.subscribe("xai/grok-imagine-video/edit-video", arguments={
        "prompt": prompt,
        "video_url": video_url,
        "resolution": resolution,
    })

    return download_video(result["video"]["url"])


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

ASPECT_RATIOS = ["16:9", "4:3", "3:2", "1:1", "2:3", "3:4", "9:16"]

with gr.Blocks(title="Grok Imagine Studio", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Grok Imagine Studio\nImage & Video generation powered by xAI Grok Imagine via fal.ai")

    with gr.Tabs():
        # ---- Tab 1: Image Edit ----
        with gr.Tab("Image Edit"):
            with gr.Row():
                with gr.Column():
                    ie_prompt = gr.Textbox(label="Prompt", lines=3, max_lines=10, placeholder="Describe the edit you want...")
                    ie_images = gr.File(label="Input Images (max 3)", file_count="multiple", file_types=["image"])
                    with gr.Row():
                        ie_num = gr.Slider(minimum=1, maximum=4, step=1, value=1, label="Number of outputs")
                        ie_res = gr.Dropdown(choices=["1k", "2k"], value="1k", label="Resolution")
                        ie_fmt = gr.Dropdown(choices=["jpeg", "png", "webp"], value="jpeg", label="Format")
                    ie_btn = gr.Button("Generate", variant="primary")
                with gr.Column():
                    ie_gallery = gr.Gallery(label="Results", columns=2)
                    ie_revised = gr.Textbox(label="Revised Prompt", interactive=False)

            ie_btn.click(image_edit, inputs=[ie_prompt, ie_images, ie_num, ie_res, ie_fmt], outputs=[ie_gallery, ie_revised])

        # ---- Tab 2: Text to Video ----
        with gr.Tab("Text to Video"):
            with gr.Row():
                with gr.Column():
                    t2v_prompt = gr.Textbox(label="Prompt", lines=3, max_lines=10, placeholder="Describe the video...")
                    with gr.Row():
                        t2v_dur = gr.Slider(minimum=1, maximum=15, step=1, value=6, label="Duration (sec)")
                        t2v_ar = gr.Dropdown(choices=ASPECT_RATIOS, value="16:9", label="Aspect Ratio")
                        t2v_res = gr.Dropdown(choices=["480p", "720p"], value="720p", label="Resolution")
                    t2v_btn = gr.Button("Generate", variant="primary")
                with gr.Column():
                    t2v_video = gr.Video(label="Result")

            t2v_btn.click(text_to_video, inputs=[t2v_prompt, t2v_dur, t2v_ar, t2v_res], outputs=[t2v_video])

        # ---- Tab 3: Reference to Video ----
        with gr.Tab("Reference to Video"):
            with gr.Row():
                with gr.Column():
                    r2v_prompt = gr.Textbox(label="Prompt (use @Image1, @Image2... to reference images)", lines=3, max_lines=10)
                    r2v_images = gr.File(label="Reference Images (max 7)", file_count="multiple", file_types=["image"])
                    with gr.Row():
                        r2v_dur = gr.Slider(minimum=1, maximum=10, step=1, value=8, label="Duration (sec)")
                        r2v_ar = gr.Dropdown(choices=ASPECT_RATIOS, value="16:9", label="Aspect Ratio")
                        r2v_res = gr.Dropdown(choices=["480p", "720p"], value="480p", label="Resolution")
                    r2v_btn = gr.Button("Generate", variant="primary")
                with gr.Column():
                    r2v_video = gr.Video(label="Result")

            r2v_btn.click(reference_to_video, inputs=[r2v_prompt, r2v_images, r2v_dur, r2v_ar, r2v_res], outputs=[r2v_video])

        # ---- Tab 4: Image to Video ----
        with gr.Tab("Image to Video"):
            with gr.Row():
                with gr.Column():
                    i2v_prompt = gr.Textbox(label="Prompt", lines=3, max_lines=10, placeholder="Describe the motion...")
                    i2v_image = gr.Image(label="Input Image", type="filepath")
                    with gr.Row():
                        i2v_dur = gr.Slider(minimum=1, maximum=15, step=1, value=6, label="Duration (sec)")
                        i2v_ar = gr.Dropdown(choices=["auto"] + ASPECT_RATIOS, value="auto", label="Aspect Ratio")
                        i2v_res = gr.Dropdown(choices=["480p", "720p"], value="720p", label="Resolution")
                    i2v_btn = gr.Button("Generate", variant="primary")
                with gr.Column():
                    i2v_video = gr.Video(label="Result")

            i2v_btn.click(image_to_video, inputs=[i2v_prompt, i2v_image, i2v_dur, i2v_ar, i2v_res], outputs=[i2v_video])

        # ---- Tab 5: Extend Video ----
        with gr.Tab("Extend Video"):
            with gr.Row():
                with gr.Column():
                    ev_prompt = gr.Textbox(label="Prompt", lines=3, max_lines=10, placeholder="What should happen next...")
                    ev_video = gr.Video(label="Input Video (MP4, 2-15 sec)")
                    ev_dur = gr.Slider(minimum=2, maximum=10, step=1, value=6, label="Extension Duration (sec)")
                    ev_btn = gr.Button("Generate", variant="primary")
                with gr.Column():
                    ev_output = gr.Video(label="Extended Video")

            ev_btn.click(extend_video, inputs=[ev_prompt, ev_video, ev_dur], outputs=[ev_output])

        # ---- Tab 6: Edit Video ----
        with gr.Tab("Edit Video"):
            with gr.Row():
                with gr.Column():
                    edv_prompt = gr.Textbox(label="Prompt", lines=3, max_lines=10, placeholder="Describe the edit...")
                    edv_video = gr.Video(label="Input Video")
                    edv_res = gr.Dropdown(choices=["auto", "480p", "720p"], value="auto", label="Resolution")
                    edv_btn = gr.Button("Generate", variant="primary")
                with gr.Column():
                    edv_output = gr.Video(label="Edited Video")

            edv_btn.click(edit_video, inputs=[edv_prompt, edv_video, edv_res], outputs=[edv_output])

demo.queue(default_concurrency_limit=4).launch()
