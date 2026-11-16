import os, sys, json, time, uuid, shutil, subprocess, requests, math
import nbformat
from pathlib import Path

nb = nbformat.v4.new_notebook()
cells = []

cells.append(nbformat.v4.new_markdown_cell("""# VideoClone AI Studio v0.3 — Geração direta com Wan 2.2 Animate

**Esta versão já contém o fluxo de geração**, não apenas a preparação do ZIP.

### O que você faz
1. Executa as células de instalação e download dos modelos.
2. Abre a interface Gradio.
3. Envia o vídeo e a imagem de referência.
4. Ajusta o ponto positivo sobre a pessoa no primeiro quadro (padrão: centro).
5. Clica em **Gerar vídeo** e aguarda o resultado.

### Pipeline
Gradio → ComfyUI API → Wan 2.2 Animate (Mix) → FFmpeg → vídeo MP4 final.

### Observações importantes
- O primeiro setup baixa aproximadamente 25–30 GB de modelos e pode demorar bastante.
- O notebook foi preparado para GPU NVIDIA com bastante VRAM (A100 80 GB recomendada).
- O primeiro teste usa 352×640, adequado ao vídeo vertical de exemplo. As dimensões devem ser múltiplas de 16.
- O workflow oficial gera o primeiro segmento de 77 frames (~4,8 s a 16 FPS). O pós-processamento ajusta a duração ao vídeo original e converte para 30 FPS.
- O modo **Mix** troca a pessoa e procura preservar movimento, iluminação e cenário original. Troca completa de cenário e refinamento com FLUX Fill/VACE ficam para uma etapa posterior.
- O ponto positivo deve ficar sobre a pessoa que será substituída no primeiro quadro. Se o ponto estiver errado, a segmentação pode selecionar o objeto errado.
- Use apenas vídeos e imagens que você tenha autorização para editar.

Fontes do workflow e dos modelos: documentação oficial do [Wan 2.2 Animate](https://docs.comfy.org/tutorials/video/wan/wan2-2-animate) e workflow [ComfyUI](https://comfy.org/hub/workflows/wan2-2-animate-character-animation-and-replacement-c2c4b45b7df5/). """))

cells.append(nbformat.v4.new_markdown_cell("""## 1. Verificar GPU e preparar pastas

Selecione uma GPU em **Ambiente de execução → Alterar tipo de ambiente de execução → GPU** antes de executar."""))

cells.append(nbformat.v4.new_code_cell("""import os, sys, json, time, uuid, shutil, subprocess, requests, math
from pathlib import Path

BASE = Path("/content/VideoCloneAI")
COMFY = Path("/content/ComfyUI")
MODELS = COMFY / "models"

for folder in [
    BASE / "inputs", BASE / "outputs",
    COMFY / "input", COMFY / "output",
    MODELS / "diffusion_models", MODELS / "text_encoders",
    MODELS / "clip_vision", MODELS / "vae", MODELS / "loras",
    MODELS / "sam2", MODELS / "detection"
]:
    folder.mkdir(parents=True, exist_ok=True)

import torch
if not torch.cuda.is_available():
    raise RuntimeError("GPU CUDA não encontrada. Ative uma GPU no Colab e execute novamente.")

gpu = torch.cuda.get_device_name(0)
vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
print(f"GPU detectada: {gpu} | VRAM: {vram_gb:.1f} GB")
if vram_gb < 48:
    print("AVISO: o workflow Wan Animate 14B pode não caber nesta GPU. A100 80 GB é recomendada.")
"""))

cells.append(nbformat.v4.new_markdown_cell("""## 2. Instalar ComfyUI e os nós necessários

Inclui KJNodes, ControlNet Aux e Segment Anything 2, usados pelo workflow oficial."""))

cells.append(nbformat.v4.new_code_cell("""if not COMFY.exists():
    subprocess.run(["git", "clone", "https://github.com/comfyanonymous/ComfyUI.git", str(COMFY)], check=True)
else:
    print("ComfyUI já existe; mantendo a instalação atual.")

subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(COMFY / "requirements.txt")], check=True)

CUSTOM = COMFY / "custom_nodes"
CUSTOM.mkdir(exist_ok=True)
NODE_REPOS = {
    "ComfyUI-KJNodes": "https://github.com/kijai/ComfyUI-KJNodes.git",
    "comfyui_controlnet_aux": "https://github.com/Fannovel16/comfyui_controlnet_aux.git",
    "ComfyUI-segment-anything-2": "https://github.com/kijai/ComfyUI-segment-anything-2.git",
}
for folder, repo in NODE_REPOS.items():
    target = CUSTOM / folder
    if not target.exists():
        subprocess.run(["git", "clone", repo, str(target)], check=True)
    req = target / "requirements.txt"
    if req.exists():
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(req)], check=True)

print("ComfyUI e extensões instalados.")
"""))

cells.append(nbformat.v4.new_markdown_cell("""## 3. Baixar os modelos automaticamente

O download é feito uma única vez por sessão do Colab. Se a sessão reiniciar, os arquivos em `/content` serão perdidos e precisarão ser baixados novamente."""))

cells.append(nbformat.v4.new_code_cell("""subprocess.run([sys.executable, "-m", "pip", "install", "-U", "huggingface_hub"], check=True)
from huggingface_hub import hf_hub_download

def fetch_model(repo_id, filename, destination_dir, final_name=None):
    destination_dir = Path(destination_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    final_path = destination_dir / (final_name or Path(filename).name)
    if final_path.exists() and final_path.stat().st_size > 1_000_000:
        print("Já existe:", final_path.name)
        return final_path
    print(f"Baixando {Path(filename).name} de {repo_id} ...")
    downloaded = Path(hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(destination_dir)
    ))
    if downloaded != final_path:
        final_path.parent.mkdir(parents=True, exist_ok=True)
        if downloaded.exists():
            shutil.move(str(downloaded), str(final_path))
    if not final_path.exists():
        raise FileNotFoundError(f"Download não localizado: {final_path}")
    print(f"OK: {final_path.name} ({final_path.stat().st_size/1024**3:.2f} GB)")
    return final_path

# Modelo principal FP8
fetch_model(
    "Kijai/WanVideo_comfy_fp8_scaled",
    "Wan22Animate/Wan2_2-Animate-14B_fp8_e4m3fn_scaled_KJ.safetensors",
    MODELS / "diffusion_models"
)

# Encoder de texto e CLIP Vision
fetch_model(
    "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
    "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
    MODELS / "text_encoders"
)
fetch_model(
    "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
    "split_files/clip_vision/clip_vision_h.safetensors",
    MODELS / "clip_vision"
)

# VAE
fetch_model(
    "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
    "split_files/vae/wan_2.1_vae.safetensors",
    MODELS / "vae"
)

# LoRAs usadas pelo workflow oficial
fetch_model(
    "Kijai/WanVideo_comfy",
    "Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors",
    MODELS / "loras"
)
fetch_model(
    "Kijai/WanVideo_comfy",
    "LoRAs/Wan22_relight/WanAnimate_relight_lora_fp16.safetensors",
    MODELS / "loras"
)

# Modelo SAM2 e detectores de pose
fetch_model(
    "Kijai/sam2-safetensors",
    "sam2.1_hiera_base_plus.safetensors",
    MODELS / "sam2"
)
fetch_model(
    "Wan-AI/Wan2.2-Animate-14B",
    "process_checkpoint/det/yolov10m.onnx",
    MODELS / "detection"
)
fetch_model(
    "JunkyByte/easy_ViTPose",
    "onnx/wholebody/vitpose-l-wholebody.onnx",
    MODELS / "detection"
)

print("Download dos modelos concluído.")
"""))

cells.append(nbformat.v4.new_markdown_cell("""## 4. Baixar o workflow oficial e iniciar o ComfyUI

O workflow é o JSON API publicado pelo ComfyUI. O notebook ajusta automaticamente os arquivos de entrada, prompt, resolução e ponto de segmentação."""))

cells.append(nbformat.v4.new_code_cell("""COMFY_URL = "http://127.0.0.1:8188"
WORKFLOW_URL = "https://comfy.org/catalog-media/bdcf5025-d570-47e8-8842-12c958679fa2/a0901e31-f75a-42cb-b0bc-09fb6b3ec18a"

wf_response = requests.get(WORKFLOW_URL, timeout=60)
wf_response.raise_for_status()
BASE_WORKFLOW = wf_response.json()
print("Workflow API carregado:", len(BASE_WORKFLOW), "nós")

# Confere os nós críticos antes de iniciar.
required_nodes = ["145", "10", "21", "159", "160", "229", "232:62", "232:15", "19"]
missing = [node for node in required_nodes if node not in BASE_WORKFLOW]
if missing:
    raise RuntimeError(f"Workflow oficial mudou; nós ausentes: {missing}")

# Inicia ComfyUI uma única vez nesta sessão.
if "comfy_proc" not in globals() or comfy_proc.poll() is not None:
    comfy_log = open("/content/comfyui.log", "w")
    comfy_proc = subprocess.Popen(
        [sys.executable, "main.py", "--listen", "127.0.0.1", "--port", "8188"],
        cwd=str(COMFY),
        stdout=comfy_log,
        stderr=subprocess.STDOUT
    )

for attempt in range(180):
    try:
        ping = requests.get(COMFY_URL + "/system_stats", timeout=3)
        if ping.ok:
            print("ComfyUI online.")
            break
    except Exception:
        time.sleep(2)
else:
    raise RuntimeError("ComfyUI não iniciou. Veja /content/comfyui.log")

# Verifica os tipos de nós requeridos.
object_info = requests.get(COMFY_URL + "/object_info", timeout=60).json()
needed_types = {
    "UNETLoader", "CLIPLoader", "CLIPVisionLoader", "VAELoader",
    "LoadImage", "LoadVideo", "GetVideoComponents", "DWPreprocessor",
    "DownloadAndLoadSAM2Model", "Sam2Segmentation", "PointsEditor",
    "WanAnimateToVideo", "SaveVideo"
}
missing_types = sorted(t for t in needed_types if t not in object_info)
if missing_types:
    print("Nós ainda não reconhecidos pelo ComfyUI:", missing_types)
    print("Se instalou agora, reinicie o runtime e execute novamente desde a célula 1.")
else:
    print("Todos os tipos de nós principais foram reconhecidos.")
"""))

cells.append(nbformat.v4.new_markdown_cell("""## 5. Criar a interface de geração

O ponto de segmentação é definido em porcentagem da imagem processada. Coloque-o sobre o tronco da pessoa original no primeiro quadro; o padrão central funciona apenas se a pessoa estiver no centro."""))

cells.append(nbformat.v4.new_code_cell("""import gradio as gr
from fractions import Fraction

def ffprobe_json(path):
    p = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        capture_output=True, text=True
    )
    if p.returncode:
        raise RuntimeError(p.stderr[-2000:])
    return json.loads(p.stdout)

def video_duration(path):
    data = ffprobe_json(path)
    return float(data.get("format", {}).get("duration", 0))

def get_comfy_output(history_item, node_id="19"):
    outputs = history_item.get("outputs", {})
    node_out = outputs.get(node_id, {})
    files = node_out.get("videos") or node_out.get("gifs") or node_out.get("images") or []
    if not files:
        raise RuntimeError("O nó Save Video não retornou arquivo. Consulte o log do ComfyUI.")
    return files[0]

def run_wan_generation(
    video_path, character_path, instruction, wardrobe, environment, visual_style,
    width, height, point_x_pct, point_y_pct, progress=gr.Progress()
):
    if not video_path or not character_path:
        raise gr.Error("Envie o vídeo e a imagem de referência.")
    video_path = Path(video_path)
    character_path = Path(character_path)
    if not video_path.exists() or not character_path.exists():
        raise gr.Error("Um dos arquivos enviados não foi encontrado.")

    width, height = int(width), int(height)
    if width % 16 or height % 16:
        raise gr.Error("Largura e altura precisam ser múltiplos de 16.")

    duration = video_duration(video_path)
    if duration <= 0:
        raise gr.Error("Não foi possível ler a duração do vídeo.")
    if duration > 5.2:
        raise gr.Error("Este primeiro workflow está configurado para um segmento de até aproximadamente 5 segundos. Use um trecho curto para o primeiro teste.")

    job_id = uuid.uuid4().hex[:10]
    safe_video = f"vc_{job_id}{video_path.suffix.lower() or '.mp4'}"
    safe_image = f"vc_{job_id}{character_path.suffix.lower() or '.png'}"
    shutil.copy2(video_path, COMFY / "input" / safe_video)
    shutil.copy2(character_path, COMFY / "input" / safe_image)

    # Prompt do modo Mix: identidade vem da imagem; movimento e cenário vêm do vídeo.
    prompt_parts = [
        instruction.strip() if instruction else "Replace the person in the source video with the person from the reference image.",
        f"Clothing: {wardrobe.strip()}." if wardrobe and wardrobe.strip() else "Keep the original clothing design unless the reference image clearly specifies otherwise.",
        f"Environment: {environment.strip()}." if environment and environment.strip() else "Preserve the original environment and background.",
        visual_style.strip() if visual_style and visual_style.strip() else "Photorealistic, natural lighting, coherent anatomy.",
        "Preserve the original person's body motion, pose, timing, facial expression and camera movement. Keep the reference character identity consistent across frames."
    ]
    positive_prompt = " ".join(prompt_parts)

    workflow = json.loads(json.dumps(BASE_WORKFLOW))
    workflow["145"]["inputs"]["file"] = safe_video
    workflow["10"]["inputs"]["image"] = safe_image
    workflow["21"]["inputs"]["text"] = positive_prompt
    workflow["159"]["inputs"]["value"] = width
    workflow["160"]["inputs"]["value"] = height

    # O workflow usa um ponto positivo para orientar SAM2 sobre a pessoa a substituir.
    px = max(0, min(width - 1, round(width * float(point_x_pct) / 100)))
    py = max(0, min(height - 1, round(height * float(point_y_pct) / 100)))
    point = {"x": px, "y": py}
    workflow["229"]["inputs"]["width"] = width
    workflow["229"]["inputs"]["height"] = height
    workflow["229"]["inputs"]["coordinates"] = json.dumps([point])
    workflow["229"]["inputs"]["neg_coordinates"] = json.dumps([{"x": 0, "y": 0}])
    workflow["229"]["inputs"]["points_store"] = json.dumps(
        {"positive": [point], "negative": [{"x": 0, "y": 0}]}
    )

    # Usar apenas o primeiro bloco de 77 frames neste teste curto; evita executar o bloco de extensão.
    workflow.pop("243", None)
    workflow["19"]["inputs"]["filename_prefix"] = f"videoclone/{job_id}"

    progress(0.08, desc="Enviando workflow ao ComfyUI...")
    submit = requests.post(COMFY_URL + "/prompt", json={"prompt": workflow}, timeout=120)
    if not submit.ok:
        raise RuntimeError(f"ComfyUI recusou o workflow: {submit.text[-4000:]}")
    prompt_id = submit.json().get("prompt_id")
    if not prompt_id:
        raise RuntimeError(f"Resposta sem prompt_id: {submit.text[:1000]}")

    progress(0.15, desc="Wan 2.2 Animate processando...")
    deadline = time.time() + 3600
    history = None
    while time.time() < deadline:
        time.sleep(5)
        try:
            h = requests.get(f"{COMFY_URL}/history/{prompt_id}", timeout=30).json()
            if prompt_id in h:
                history = h[prompt_id]
                status = history.get("status", {})
                if status.get("completed"):
                    break
                if status.get("status_str") == "error":
                    raise RuntimeError(json.dumps(status, ensure_ascii=False)[:5000])
        except requests.RequestException:
            pass
        progress(0.45, desc="Gerando personagem e movimento...")

    if history is None or not history.get("status", {}).get("completed"):
        raise RuntimeError("Tempo limite aguardando o ComfyUI. Consulte /content/comfyui.log e a fila do ComfyUI.")

    progress(0.82, desc="Baixando vídeo gerado...")
    out_info = get_comfy_output(history, "19")
    view = requests.get(
        COMFY_URL + "/view",
        params={
            "filename": out_info["filename"],
            "subfolder": out_info.get("subfolder", ""),
            "type": out_info.get("type", "output")
        },
        timeout=300
    )
    view.raise_for_status()

    raw_video = BASE / "outputs" / f"wan_raw_{job_id}.mp4"
    raw_video.write_bytes(view.content)

    # Ajusta levemente a duração ao vídeo original e converte para 30 FPS.
    raw_duration = video_duration(raw_video)
    if raw_duration <= 0:
        raise RuntimeError("O arquivo gerado não contém duração válida.")
    speed_factor = duration / raw_duration
    final_video = BASE / "outputs" / f"VideoClone_{job_id}.mp4"
    ffmpeg_cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(raw_video), "-i", str(video_path),
        "-filter:v", f"setpts=PTS*{speed_factor:.8f},fps=30",
        "-map", "0:v:0", "-map", "1:a?",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
        "-shortest", "-movflags", "+faststart", str(final_video)
    ]
    proc = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError("Falha na finalização FFmpeg: " + proc.stderr[-3000:])

    progress(1.0, desc="Concluído.")
    report = (
        f"Geração concluída: {job_id}\\n"
        f"Duração original: {duration:.3f} s\\n"
        f"Duração do render Wan: {raw_duration:.3f} s\\n"
        f"Resolução de geração: {width}x{height}\\n"
        f"FPS final: 30\\n"
        f"Arquivo: {final_video}"
    )
    return str(final_video), report

with gr.Blocks(title="VideoClone AI Studio — Wan 2.2") as demo:
    gr.Markdown("# VideoClone AI Studio")
    gr.Markdown("### Substituição de personagem com Wan 2.2 Animate — modo Mix")
    with gr.Row():
        source_video = gr.Video(label="Vídeo de referência", sources=["upload"])
        reference_image = gr.Image(label="Imagem da personagem", type="filepath")
    instruction = gr.Textbox(
        label="Instrução principal",
        value="Replace the person in the source video with the person from the reference image.",
        lines=2
    )
    with gr.Accordion("Roupa, ambiente e estilo", open=True):
        wardrobe = gr.Textbox(label="Roupa", placeholder="Vazio = manter a roupa original.")
        environment = gr.Textbox(label="Ambiente", placeholder="Vazio = preservar o cenário original.")
        visual_style = gr.Textbox(label="Estilo visual", value="Photorealistic, natural lighting, realistic textures, cinematic video.")
    with gr.Accordion("Resolução e seleção da pessoa", open=True):
        with gr.Row():
            out_width = gr.Slider(256, 832, value=352, step=16, label="Largura")
            out_height = gr.Slider(256, 832, value=640, step=16, label="Altura")
        with gr.Row():
            point_x = gr.Slider(0, 100, value=50, step=1, label="Ponto X da pessoa (%)")
            point_y = gr.Slider(0, 100, value=50, step=1, label="Ponto Y da pessoa (%)")
        gr.Markdown("O ponto X/Y deve cair sobre o corpo da pessoa original no primeiro quadro. O workflow redimensiona o vídeo para a resolução escolhida.")
    generate = gr.Button("Gerar vídeo com Wan 2.2 Animate", variant="primary")
    result_video = gr.Video(label="Vídeo gerado")
    result_report = gr.Textbox(label="Relatório", lines=6)
    generate.click(
        fn=run_wan_generation,
        inputs=[
            source_video, reference_image, instruction, wardrobe, environment,
            visual_style, out_width, out_height, point_x, point_y
        ],
        outputs=[result_video, result_report]
    )

demo.queue(max_size=2)
demo.launch(server_name="0.0.0.0", server_port=7860, share=False, debug=False, prevent_thread_lock=True)

from google.colab import output
output.serve_kernel_port_as_window(7860)
"""))

cells.append(nbformat.v4.new_markdown_cell("""## Se ocorrer erro

- **CUDA out of memory:** reduza resolução para 352×512 ou 320×576 e confirme que o runtime é A100 80 GB.
- **Missing node:** o ComfyUI pode precisar ser reiniciado após a instalação dos custom nodes. Reinicie o runtime e execute novamente desde a célula 1.
- **Model not found:** confira se a célula de download terminou sem erro.
- **Máscara seleciona o fundo:** ajuste os pontos X/Y para o centro do tronco da pessoa no primeiro quadro.
- **Vídeo cortado ou incompatível:** este workflow inicial foi limitado a um segmento curto de 77 frames. Depois que a primeira geração funcionar, integramos a extensão temporal para vídeos maiores.
- Logs do servidor: `/content/comfyui.log`.

O notebook não promete resultado perfeito em toda referência: qualidade depende do enquadramento, visibilidade da pessoa, pose e compatibilidade dos modelos."""))

nb["cells"] = cells
nb["metadata"] = {
    "colab": {"name": "VideoClone_AI_Studio_v0_3_Wan_Geracao_Direta.ipynb", "provenance": []},
    "kernelspec": {"display_name": "Python 3", "name": "python3"},
    "language_info": {"name": "python"}
}

out = "/mnt/data/VideoClone_AI_Studio_v0_3_Wan_Geracao_Direta.ipynb"
nbformat.write(nb, out)
print("Notebook salvo com sucesso:", out, "Tamanho:", os.path.getsize(out))