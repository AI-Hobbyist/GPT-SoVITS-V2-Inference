import gradio as gr
from gradio.processing_utils import PUBLIC_HOSTNAME_WHITELIST
from json import loads, dumps
from requests import post, get
from base64 import b64encode
from pathlib import Path
from startup.functions import start
from startup.libs.agreement import webui_agreement

import argparse
import subprocess
import atexit
import time

#===================启动参数===================#
parser = argparse.ArgumentParser()
parser.add_argument("-s","--server_name",type=str,default="127.0.0.1",help="WebUI地址")
parser.add_argument("-p","--port",type=int,default=8080,help="WebUI端口")
parser.add_argument("-ak","--api_key",type=str,default="",help="API密钥")
parser.add_argument("-sr","--share",action="store_true",help="共享WebUI")
parser.add_argument("-d","--device",type=str,default="cuda",help="推理设备(cuda/cpu)")
args = parser.parse_args()
app_key = args.api_key
api_root = f"http://{args.server_name}:{args.port + 1}"
PUBLIC_HOSTNAME_WHITELIST.append("127.0.0.1")

start()
#===================组件样式===================#
css = """
.add-conv {
    height: 80px;
}
"""

#===================启动后端===================#
#启动后端
def start_server():
    subprocess.Popen(["runtime/python.exe", "gsvi_api.py", "-p", str(args.port + 1), "-s", "0.0.0.0", "-d", args.device])

#关闭webui后关闭后端
def close_server():
    backend_process = subprocess.Popen(["runtime/python.exe", "gsvi_api.py", "-p", str(args.port + 1), "-s", "0.0.0.0", "-d", args.device])
    backend_process_pid = backend_process.pid
    subprocess.Popen(["taskkill", "/f", "/pid", str(backend_process_pid)])

# 等待API启动成功
def wait_for_api():
    while True:
        try:
            response = get(f"{api_root}/models")
            if response.status_code == 200:
                break
        except:
            pass
        time.sleep(1)

#===================公共函数===================#
# 启动服务
start_server()
wait_for_api()
atexit.register(close_server)

# 获取模型列表
def get_models():
    data = get(f"{api_root}/models").json()
    return data

model_list = get_models()

#获取说话人列表
def get_characters(model):
    headers = {"Content-Type": "application/json"}
    content = {"model": model}
    data = post(f"{api_root}/spks", headers=headers, data=dumps(content),timeout=86400)
    res = loads(data.text)
    return res["speakers"]

#更新说话人列表
def update_characters(model):
    spk_list = get_characters(model)
    speakers = list(spk_list.keys())
    emo_lang = list(set([emo_lang for emo_lang in spk_list.values() for emo_lang in emo_lang.keys()]))
    emotions = list(set([emotion for emo_lang in spk_list.values() for emotions in emo_lang.values() for emotion in emotions]))
    return speakers, emo_lang, emotions

def update_settings(model):
    speakers, emo_lang, emotions = update_characters(model)
    return gr.update(choices=speakers, value=speakers[0] if speakers else None), gr.update(choices=emo_lang, value=emo_lang[0] if emo_lang else None), gr.update(choices=emotions, value=emotions[0] if emotions else None)

#根据模型和角色获取情感列表
def update_emotion_lang(model, character):
    spk_list = get_characters(model)
    emo_lang = list(spk_list[character].keys())
    emotion = list(set([emotion for emotions in spk_list[character].values() for emotion in emotions]))
    return gr.update(choices=emo_lang, value=emo_lang[0] if emo_lang else None), gr.update(choices=emotion, value=emotion[0] if emotion else None)

#根据模型、角色、参考语言获取情感列表
def update_emotion(model, character, emotion_lang):
    spk_list = get_characters(model)
    emotion = spk_list[character][emotion_lang]
    return gr.update(choices=emotion, value=emotion[0] if emotion else None)

#构建对话
def build_conversation(modelname, character, text_lang, prompt_lang, emotion, speed, text):
    conv_text = f"{modelname}|{character}|{text_lang}|{prompt_lang}|{emotion}|{float(speed)}|#{text}‖"
    return conv_text

def add_conversation(old_text, new_text):
    append_text = f"{old_text}\n{new_text}"
    return append_text

def add_and_build_conversation(model, character, text_lang, emotion_lang, emotion, speed, text, old_text):
    new_conv = build_conversation(model, character, text_lang, emotion_lang, emotion, speed, text)
    return add_conversation(old_text, new_conv)

#将音频编码为base64
def encode_audio(audio):
    audio_data = Path(audio).read_bytes()
    audio_base64 = b64encode(audio_data).decode("utf-8")
    return audio_base64

#===================推理函数===================#
#单人推理
def infer_single(model, character, emotion_lang, emotion, text, text_lang, top_k, top_p, temperature, text_split_method, batch_size, batch_threshold, split_bucket, speed_facter, fragment_interval, media_type, parallel_infer, repetition_penalty, seed):
    headers = {"Content-Type": "application/json"}
    content = {"app_key": app_key,"audio_dl_url": api_root, "model_name": model, "speaker_name": character, "prompt_text_lang": emotion_lang, "emotion": emotion, "text": text, "text_lang": text_lang, "top_k": top_k, "top_p": top_p, "temperature": temperature, "text_split_method": text_split_method, "batch_size": batch_size, "batch_threshold": batch_threshold, "split_bucket": split_bucket, "speed_facter": speed_facter, "fragment_interval": fragment_interval, "media_type": media_type, "parallel_infer": parallel_infer, "repetition_penalty": repetition_penalty, "seed": seed}
    data = post(f"{api_root}/infer_single", headers=headers, data=dumps(content),timeout=86400)
    res = loads(data.text)
    gr.Info(res["msg"])
    return res["audio_url"]

#多人对话
def infer_multi(content, top_k, top_p, temperature, text_split_method, batch_size, batch_threshold, split_bucket, fragment_interval, media_type, parallel_infer, repetition_penalty, seed):
    headers = {"Content-Type": "application/json"}
    content = content.replace("\n", "")
    data = {"app_key": app_key, "audio_dl_url": api_root, "content": content, "top_k": top_k, "top_p": top_p, "temperature": temperature, "text_split_method": text_split_method, "batch_size": batch_size, "batch_threshold": batch_threshold, "split_bucket": split_bucket, "fragment_interval": fragment_interval, "media_type": media_type, "parallel_infer": parallel_infer, "repetition_penalty": repetition_penalty, "seed": seed}
    data = post(f"{api_root}/infer_multi", headers=headers, data=dumps(data),timeout=86400)
    res = loads(data.text)
    gr.Info(res["msg"])
    return res["archive_url"]

#自定义参考音频
def infer_custom(model, ref_audio, text, text_lang, prompt_text, prompt_text_lang, top_k, top_p, temperature, text_split_method, batch_size, batch_threshold, split_bucket, speed_facter, fragment_interval, media_type, parallel_infer, repetition_penalty, seed):
    b64_audio = encode_audio(ref_audio)
    headers = {"Content-Type": "application/json"}
    content = {"app_key": app_key, "audio_dl_url": api_root, "model_name": model, "ref_audio_b64": b64_audio, "text": text, "text_lang": text_lang, "prompt_text": prompt_text, "prompt_text_lang": prompt_text_lang, "top_k": top_k, "top_p": top_p, "temperature": temperature, "text_split_method": text_split_method, "batch_size": batch_size, "batch_threshold": batch_threshold, "split_bucket": split_bucket, "speed_facter": speed_facter, "fragment_interval": fragment_interval, "media_type": media_type, "parallel_infer": parallel_infer, "repetition_penalty": repetition_penalty, "seed": seed}
    data = post(f"{api_root}/infer_ref", headers=headers, data=dumps(content),timeout=86400)
    res = loads(data.text)
    gr.Info(res["msg"])
    return res["audio_url"]

with gr.Blocks(title="GPT-Sovits Inference WebUI", css=css) as app:
    gr.Markdown("## <center>[GPT-Sovits](https://github.com/RVC-Boss/GPT-SoVITS) 语音合成</center>")
    gr.Markdown(f"{webui_agreement()}")
    with gr.Tabs(selected="single"):
        with gr.Tab("单人推理", id="single"):
            with gr.Row():
                with gr.Column(scale=5):
                    with gr.Tab("要合成的文本"):
                        text = gr.Textbox(lines=28, label="输入要合成的文本", placeholder="请输入要合成的文本")
                        output_single = gr.Audio(label="合成音频", interactive=False)
                with gr.Column(scale=1):
                    with gr.Tab("合成设置"):
                        model_single = gr.Dropdown(label="选择模型", choices=model_list, value="请选择模型", interactive=True, allow_custom_value=True)
                        character_single = gr.Dropdown(label="选择角色", choices=[], value=None, interactive=True)
                        emotion_lang_single = gr.Dropdown(label="参考语言", choices=[], value=None, interactive=True)
                        emotion_single = gr.Dropdown(label="参考情感", choices=[], value=None, interactive=True)
                        text_language_single = gr.Dropdown(label="文本语言", choices=["中文", "英语", "日语", "粤语", "韩语", "中英混合", "日英混合", "粤英混合", "韩英混合", "多语种混合", "多语种混合(粤语)"], value="中文", interactive=True)
                        cut_method_single = gr.Dropdown(label="切分方法", choices=["不切", "凑四句一切", "凑50字一切", "按中文句号。切", "按英文句号.切", "按标点符号切"], value="按标点符号切", interactive=True)
                        seed_single = gr.Number(label="种子码", minimum=-1, maximum=10000000, value=-1, interactive=True)
                        speed_single = gr.Slider(minimum=0.01, maximum=2.0, label="语速", value=1.0, step=0.01, interactive=True)
                        btn_single = gr.Button("一键合成", interactive=True, variant="primary")
                        
            model_single.change(fn=update_settings, inputs=model_single, outputs=[character_single, emotion_lang_single, emotion_single])
            character_single.change(fn=update_emotion_lang, inputs=[model_single, character_single], outputs=[emotion_lang_single, emotion_single])
            emotion_lang_single.change(fn=update_emotion, inputs=[model_single, character_single, emotion_lang_single], outputs=emotion_single)
                        
        with gr.Tab("多人对话", id="multi"):
            with gr.Row():
                with gr.Column(scale=5):
                    with gr.Tab("要合成的文本"):
                        with gr.Column():
                            with gr.Row():
                                model_multi = gr.Dropdown(label="选择模型", choices=model_list, value="请选择模型", interactive=True, allow_custom_value=True)
                                character_multi = gr.Dropdown(label="选择角色", choices=[], value=None, interactive=True)
                                emotion_lang_multi = gr.Dropdown(label="参考语言", choices=[], value=None, interactive=True)
                                emotion_multi = gr.Dropdown(label="参考情感", choices=[], value=None, interactive=True)
                                text_lang_multi = gr.Dropdown(label="文本语言", choices=["中文", "英语", "日语", "粤语", "韩语", "中英混合", "日英混合", "粤英混合", "韩英混合", "多语种混合", "多语种混合(粤语)"], value="中文", interactive=True)
                                speed_multi = gr.Number(minimum=0.01, maximum=2.0, label="语速", value=1.0, step=0.01, interactive=True)
                            with gr.Row():
                                with gr.Column(scale=5):
                                    text_conversation = gr.Textbox(lines=2, label="对话文本", placeholder="请输入对话文本", show_label=False)
                                with gr.Column(scale=1):
                                    with gr.Row():
                                        btn_conv = gr.Button("添加并构建对话", interactive=True, variant="primary", elem_classes=["add-conv"])
                        text_multi = gr.Textbox(lines=15, label="如需添加后修改内容，请参考：模型名|角色|合成语言|参考语言|参考情感|语速(0.01~2.0)|#要合成的内容‖", placeholder="请先构建对话文本")
                        output_multi = gr.File(label="压缩包下载", type="filepath", interactive=False)
                        btn_conv.click(fn=add_and_build_conversation, inputs=[model_multi, character_multi, text_lang_multi, emotion_lang_multi, emotion_multi, speed_multi, text_conversation, text_multi], outputs=text_multi)
                with gr.Column(scale=1):
                    with gr.Tab("合成设置"):
                        cut_method_multi = gr.Dropdown(label="切分方法", choices=["不切", "凑四句一切", "凑50字一切", "按中文句号。切", "按英文句号.切", "按标点符号切"], value="按标点符号切", interactive=True)
                        seed_multi = gr.Number(label="种子码", minimum=-1, maximum=10000000, value=-1, interactive=True)
                        btn_multi = gr.Button("一键合成", interactive=True, variant="primary")
            model_multi.change(fn=update_settings, inputs=model_multi, outputs=[character_multi, emotion_lang_multi, emotion_multi])
            character_multi.change(fn=update_emotion_lang, inputs=[model_multi, character_multi], outputs=[emotion_lang_multi, emotion_multi])
            emotion_lang_multi.change(fn=update_emotion, inputs=[model_multi, character_multi, emotion_lang_multi], outputs=emotion_multi)
            
        with gr.Tab("自定义参考音频", id="custom"):
            with gr.Row():
                with gr.Column(scale=5):
                    with gr.Tab("要合成的文本"):
                        ref_text = gr.Textbox(lines=1, label="参考文本", placeholder="参考文本", show_label=False)
                        ref_audio = gr.Audio(label="参考音频", type="filepath", interactive=True)
                        text_custom = gr.Textbox(lines=28, label="输入要合成的文本", placeholder="请输入要合成的文本")
                        output_custom = gr.Audio(label="合成音频", type="filepath", interactive=False)
                with gr.Column(scale=1):
                    with gr.Tab("合成设置"):
                        model_custom = gr.Dropdown(label="选择模型", choices=model_list, value="请选择模型", interactive=True, allow_custom_value=True)
                        ref_text_language_custom = gr.Dropdown(label="参考文本语言", choices=["中文", "英语", "日语", "粤语", "韩语", "中英混合", "日英混合", "粤英混合", "韩英混合", "多语种混合", "多语种混合(粤语)"], value="中文", interactive=True)
                        text_language_custom = gr.Dropdown(label="合成文本语言", choices=["中文", "英语", "日语", "粤语", "韩语", "中英混合", "日英混合", "粤英混合", "韩英混合", "多语种混合", "多语种混合(粤语)"], value="中文", interactive=True)
                        cut_method_custom = gr.Dropdown(label="切分方法", choices=["不切", "凑四句一切", "凑50字一切", "按中文句号。切", "按英文句号.切", "按标点符号切"], value="按标点符号切", interactive=True)
                        seed_custom = gr.Number(label="种子码", minimum=-1, maximum=10000000, value=-1, interactive=True)
                        speed_custom = gr.Slider(minimum=0.01, maximum=2.0, label="语速", value=1.0, step=0.01, interactive=True)
                        btn_custom = gr.Button("一键合成", interactive=True, variant="primary")
    
        with gr.Tab("全局设置", id="global"):
            with gr.Column():
                with gr.TabItem("基本设置"):
                    with gr.Row():
                        with gr.Column(scale=2):
                            media_type = gr.Radio(label="音频格式", choices=["wav", "ogg", "aac"], value="wav", interactive=True)
                        with gr.Column(scale=2):
                            fragment_interval = gr.Slider(label="分段间隔(秒)", minimum=0.01, maximum=1.0, step=0.01, value=0.3, interactive=True)
            with gr.Column():
                with gr.TabItem("并行推理"):
                    with gr.Row():
                        with gr.Column(scale=2):
                            parallel_infer = gr.Checkbox(label="启用并行推理", value=True, interactive=True, show_label=True)
                        with gr.Column(scale=2):
                            split_bucket = gr.Checkbox(label="启用数据分桶(并行推理时会降低一点计算量)", value=True, interactive=True, show_label=True)
                    with gr.Row():
                        with gr.Column(scale=2):
                            batch_size = gr.Slider(minimum=1, maximum=200, step=1, label="批量大小", value=10, interactive=True)
                        with gr.Column(scale=2):
                            batch_threshold = gr.Slider(minimum=0, maximum=1, step=0.01, label="批处理阈值", value=0.75, interactive=True)
            with gr.Column():
                with gr.TabItem("推理参数"):
                    with gr.Row():
                        with gr.Column(scale=2):
                            top_k = gr.Slider(label="前k个采样（Top-k）", minimum=1, maximum=100, step=1, value=10, interactive=True)
                        with gr.Column(scale=2):
                            top_p = gr.Slider(label="累计概率采样 (Top-p)", minimum=0.01, maximum=1.0, step=0.01, value=1.0, interactive=True)
                    with gr.Row():
                        with gr.Column(scale=2):
                            temperature = gr.Slider(label="温度系数 (Temperature)", minimum=0.01, maximum=1, step=0.01, value=1.0, interactive=True)
                        with gr.Column(scale=2):
                            repetition_penalty = gr.Slider(minimum=0, maximum=2, step=0.05, label="重复惩罚", value=1.35, interactive=True)
    btn_single.click(infer_single, inputs=[model_single, character_single, emotion_lang_single, emotion_single, text, text_language_single, top_k, top_p, temperature, cut_method_single, batch_size, batch_threshold, split_bucket, speed_single, fragment_interval, media_type, parallel_infer, repetition_penalty, seed_single], outputs=output_single)
    btn_multi.click(infer_multi, inputs=[text_multi, top_k, top_p, temperature, cut_method_multi, batch_size, batch_threshold, split_bucket, fragment_interval, media_type, parallel_infer, repetition_penalty, seed_multi], outputs=output_multi)
    btn_custom.click(infer_custom, inputs=[model_custom, ref_audio, text_custom, text_language_custom, ref_text, ref_text_language_custom, top_k, top_p, temperature, cut_method_custom, batch_size, batch_threshold, split_bucket, speed_custom, fragment_interval, media_type, parallel_infer, repetition_penalty, seed_custom], outputs=output_custom)
            
app.queue(default_concurrency_limit=1)

def main():
    app.launch(show_api=False,server_name=args.server_name,server_port=args.port,share=args.share)

if __name__ == "__main__":
    main()
            