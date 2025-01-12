import os
import sys

now_dir = os.getcwd()
sys.path.append(now_dir)
sys.path.append("%s/GPT_SoVITS" % (now_dir))

import subprocess
import numpy as np
import soundfile as sf
import torch
import gc
from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config
from glob import glob
from pathlib import Path
from re import split
from io import BytesIO
from pydub import AudioSegment
from random import choice, randint
from hashlib import md5
from base64 import b64decode
from tqdm import tqdm
from time import time
from datetime import datetime

#===============推理预备================
def pre_infer(config_path):
    global tts_config, tts_pipeline
    if config_path in [None, ""]:
        config_path = "GPT-SoVITS/configs/tts_infer.yaml"
    Path("outputs").mkdir(parents=True, exist_ok=True)
    Path("custom_refs").mkdir(parents=True, exist_ok=True)
    tts_config = TTS_Config(config_path)
    print(tts_config)
    tts_pipeline = TTS(tts_config)
    
def load_weights(gpt, sovits):
    if gpt != "":
        tts_pipeline.init_t2s_weights(gpt)
    if sovits != "":
        tts_pipeline.init_vits_weights(sovits)
    
#===============推理函数================
def pack_ogg(io_buffer:BytesIO, data:np.ndarray, rate:int):
    with sf.SoundFile(io_buffer, mode='w', samplerate=rate, channels=1, format='ogg') as audio_file:
        audio_file.write(data)
    return io_buffer


def pack_raw(io_buffer:BytesIO, data:np.ndarray, rate:int):
    io_buffer.write(data.tobytes())
    return io_buffer


def pack_wav(io_buffer:BytesIO, data:np.ndarray, rate:int):
    io_buffer = BytesIO()
    sf.write(io_buffer, data, rate, format='wav')
    return io_buffer

def pack_aac(io_buffer:BytesIO, data:np.ndarray, rate:int):
    process = subprocess.Popen([
        'ffmpeg',
        '-f', 's16le',  # 输入16位有符号小端整数PCM
        '-ar', str(rate),  # 设置采样率
        '-ac', '1',  # 单声道
        '-i', 'pipe:0',  # 从管道读取输入
        '-c:a', 'aac',  # 音频编码器为AAC
        '-b:a', '192k',  # 比特率
        '-vn',  # 不包含视频
        '-f', 'adts',  # 输出AAC数据流格式
        'pipe:1'  # 将输出写入管道
    ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, _ = process.communicate(input=data.tobytes())
    io_buffer.write(out)
    return io_buffer

def pack_audio(io_buffer:BytesIO, data:np.ndarray, rate:int, media_type:str):
    if media_type == "ogg":
        io_buffer = pack_ogg(io_buffer, data, rate)
    elif media_type == "aac":
        io_buffer = pack_aac(io_buffer, data, rate)
    elif media_type == "wav":
        io_buffer = pack_wav(io_buffer, data, rate)
    else:
        io_buffer = pack_raw(io_buffer, data, rate)
    io_buffer.seek(0)
    return io_buffer

def tts_infer(text, text_lang, ref_audio_path, prompt_text, prompt_lang, top_k, top_p, temperature, text_split_method, batch_size, batch_threshold, split_bucket, speed_facter, fragment_interval, seed, media_type, parallel_infer, repetition_penalty):
    t_lang = ["all_zh","en","all_ja","all_yue","all_ko","zh","ja","yue","ko","auto","auto_yue"][["中文","英语","日语","粤语","韩语","中英混合","日英混合","粤英混合","韩英混合","多语种混合","多语种混合(粤语)"].index(text_lang)]
    p_lang = ["all_zh","en","all_ja","all_yue","all_ko","zh","ja","yue","ko","auto","auto_yue"][["中文","英语","日语","粤语","韩语","中英混合","日英混合","粤英混合","韩英混合","多语种混合","多语种混合(粤语)"].index(prompt_lang)]
    cut_method = ["cut0","cut1","cut2","cut3","cut4","cut5"][["不切","凑四句一切","凑50字一切","按中文句号。切","按英文句号.切","按标点符号切"].index(text_split_method)]
    infer_dict = {
        "text": text,
        "text_lang": t_lang,
        "ref_audio_path": ref_audio_path,
        "prompt_text": prompt_text,
        "prompt_lang": p_lang,
        "top_k": top_k,
        "top_p": top_p,
        "temperature": temperature,
        "text_split_method": cut_method,
        "batch_size": batch_size,
        "batch_threshold": batch_threshold,
        "split_bucket": split_bucket,
        "speed_factor": speed_facter,
        "fragment_interval": fragment_interval,
        "seed": seed,
        "media_type": media_type,
        "streaming_mode": False,
        "parallel_infer": parallel_infer,
        "repetition_penalty": repetition_penalty
    }
    with torch.no_grad():
        tts_gen = tts_pipeline.run(infer_dict)
        sr, audio = next(tts_gen)
        torch.cuda.empty_cache()
        gc.collect()
    
    audio = pack_audio(BytesIO(), audio, sr, media_type).getvalue()
    
    return audio

#===============通用函数================
# 将base64编码音频转换为音频数据，并写出到文件，文件名为md5值
def base64_to_audio(base64_str):
    audio_data = b64decode(base64_str)
    audio_md5 = md5(audio_data).hexdigest()
    audio_path = f"custom_refs/{audio_md5}.wav"
    Path(audio_path).write_bytes(audio_data)
    return audio_path

# 随机种子码
def random_seed():
    seed = randint(0, 4294967295)
    return seed

# 根据情感参考音频文件名分离情感名称和参考文本
def get_emotion_text(file_name):
    emotion = split("【|】", file_name)[1]
    emo_text = split("【|】", file_name)[2]
    return emotion, emo_text

# 获取说话人列表
def get_speakers(modelname):
    speakers = glob(f"models/{modelname}/reference_audios/emotions/*")
    speaker_list = []
    for speaker in speakers:
        speaker_name = Path(speaker).name
        speaker_list.append(speaker_name)
    return speaker_list

# 获取说话人支持的参考音频语言
def get_ref_audio_langs(modelname, speaker):
    langs = []
    lang_dir = glob(f"models/{modelname}/reference_audios/emotions/{speaker}/*")
    for lang in lang_dir:
        lang_name = Path(lang).name
        langs.append(lang_name)
    return langs

# 根据语言获取参考情感列表
def get_ref_audios(modelname, speaker, lang):
    audios = glob(f"models/{modelname}/reference_audios/emotions/{speaker}/{lang}/*.wav")
    audio_list = []
    for audio in audios:
        audio_name = Path(audio).name
        emotion, emo_text = get_emotion_text(audio_name)
        audio_list.append(emotion)
    if Path(f"models/{modelname}/reference_audios/randoms/{speaker}/{lang}").exists():
        audio_list.append("随机")
    return audio_list

# 获取指定情感的完整参考音频文件名
def get_ref_audio(modelname, speaker, lang, emotion):
    audios = glob(f"models/{modelname}/reference_audios/emotions/{speaker}/{lang}/*.wav")
    for audio in audios:
        audio_name = str(Path(audio).name).replace(".wav", "")
        if f"【{emotion}】" in audio_name:
            emo, emo_text = get_emotion_text(audio_name)
    return emo, emo_text

# 随机选择参考音频
def random_ref_audio(modelname, speaker, lang):
    if Path(f"models/{modelname}/reference_audios/randoms/{speaker}/{lang}").exists():
        audios = glob(f"models/{modelname}/reference_audios/randoms/{speaker}/{lang}/*.wav")
        audio = choice(audios)
        lab_name = audio.replace(".wav", ".lab")
        lab_content = Path(lab_name).read_text(encoding="utf-8")
    else:
        audio = ""
        lab_content = ""
    return audio, lab_content

#判断参考音频长度是否符合要求
def check_audio_length(audio_path):
    audio = AudioSegment.from_file(audio_path)
    audio_length = len(audio) / 1000
    if audio_length < 3 or audio_length > 10:
        return False
    else:
        return True
    
#获取模型路径
def get_model_path(model_name):
    gpt_models = glob(f"models/{model_name}/*.ckpt")
    sovits_models = glob(f"models/{model_name}/*.pth")
    gpt_model = gpt_models[0] if len(gpt_models) > 0 else ""
    sovits_model = sovits_models[0] if len(sovits_models) > 0 else ""
    return gpt_model, sovits_model

#加载模型
def load_model(model_name):
    gpt_model, sovits_model = get_model_path(model_name)
    load_weights(gpt_model, sovits_model)
    return gpt_model, sovits_model

#===============接口函数================
# 获取模型列表
def get_models():
    models = glob(f"models/*")
    model_list = []
    for mod in models:
        model_name = Path(mod).name
        model_list.append(model_name)
    return model_list

# 获取多人对话参考单人模板（不支持自定义参考音频）
def get_multi_ref_template(modelname):
    speakers = glob(f"models/{modelname}/reference_audios/emotions/*")
    template_list = []
    if len(speakers) == 0:
        msg = "该模型不存在或未设置参考音频"
    for speaker in speakers:
        speaker_name = Path(speaker).name
        multi_template = f"{modelname}|{speaker_name}|合成语言|参考语言|情感|语速|#内容请自由发挥‖"
        template_list.append(multi_template)
        msg = "获取成功"
    return template_list, msg

# 创建说话人列表
def create_speaker_list(modelname):
    speakers = get_speakers(modelname)
    spk_list = {}
    if len(speakers) == 0:
        msg = "该模型不存在!"
    else:
        for speaker in speakers:
            langs = get_ref_audio_langs(modelname, speaker)
            spk_list[speaker] = {}
            for lang in langs:
                audios = get_ref_audios(modelname, speaker, lang)
                spk_list[speaker][lang] = audios
        msg = "获取成功"
    return spk_list, msg

# 根据自定义参考音频以及文本合成语音（仅支持单人合成）
def custom_ref(modelname, refaudio_b64, text, text_lang, prompt_text, prompt_lang, top_k, top_p, temperature, text_split_method, batch_size, batch_threshold, split_bucket, speed_facter, fragment_interval, media_type, parallel_infer, repetition_penalty, seed):
    ref_audio_path = base64_to_audio(refaudio_b64)
    if check_audio_length(ref_audio_path) == False:
        msg = "参考音频长度不符合要求"
        audio_path = ""
    else:
        load_model(modelname)
        if seed == -1:
            seed = random_seed()
        audio = tts_infer(text, text_lang, ref_audio_path, prompt_text, prompt_lang, top_k, top_p, temperature, text_split_method, batch_size, batch_threshold, split_bucket, speed_facter, fragment_interval, seed, media_type, parallel_infer, repetition_penalty)
        audio_md5 = md5(audio).hexdigest()
        audio_path = f"outputs/{audio_md5}.wav"
        Path(audio_path).write_bytes(audio)
        msg = "合成成功"
    return audio_path, msg
    
# 根据说话人和情感合成语音（单人合成）
def single_infer(modelname, speaker, prompt_lang, emotion, text, text_lang, top_k, top_p, temperature, text_split_method, batch_size, batch_threshold, split_bucket, speed_facter, fragment_interval, media_type, parallel_infer, repetition_penalty, seed):
    if emotion == "随机":
        ref_audio, lab_content = random_ref_audio(modelname, speaker, prompt_lang)
        prompt_text = lab_content
    else:
        emo, prompt_text = get_ref_audio(modelname, speaker, prompt_lang, emotion)
        ref_audio = f"models/{modelname}/reference_audios/emotions/{speaker}/{prompt_lang}/【{emo}】{prompt_text}.wav"
        
    if ref_audio == "":
        msg = "参考音频不存在"
        audio_path = ""
    else:
        load_model(modelname)
        if seed == -1:
            seed = random_seed()
        audio = tts_infer(text, text_lang, ref_audio, prompt_text, prompt_lang, top_k, top_p, temperature, text_split_method, batch_size, batch_threshold, split_bucket, speed_facter, fragment_interval, seed, media_type, parallel_infer, repetition_penalty)
        audio_md5 = md5(audio).hexdigest()
        audio_path = f"outputs/{audio_md5}.wav"
        Path(audio_path).write_bytes(audio)
        msg = "合成成功"
    return audio_path, msg

# 根据说话人和情感合成语音（多人合成）
def multi_infer(content, top_k, top_p, temperature, text_split_method, batch_size, batch_threshold, split_bucket, fragment_interval, media_type, parallel_infer, repetition_penalty, seed):
    log_list = []
    try:
        content_list = content.split("‖")
        filtered_list = list(filter(str.strip, content_list))
        content_md5 = f"{md5(content.encode()).hexdigest()}_{int(time())}"
        content_md5 = md5(content_md5.encode()).hexdigest()
        Path(f"outputs/conv_{content_md5}").mkdir(parents=True, exist_ok=True)
        for i, single_content in enumerate(filtered_list):
            try:
                single_content_list = single_content.split("|")
                model_name = single_content_list[0]
                speaker = single_content_list[1]
                text_lang = single_content_list[2]
                prompt_lang = single_content_list[3]
                emotion = single_content_list[4]
                speed_facter = float(single_content_list[5])
                text = single_content_list[6]
                text = text.replace("#", "")
                if emotion == "随机":
                    ref_audio, lab_content = random_ref_audio(model_name, speaker, prompt_lang)
                    prompt_text = lab_content
                else:
                    emo, prompt_text = get_ref_audio(model_name, speaker, prompt_lang, emotion)
                    ref_audio = f"models/{model_name}/reference_audios/emotions/{speaker}/{prompt_lang}/【{emo}】{prompt_text}.wav"
            except:
                log_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                log_list.append(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] 第 {i+1} 段对话格式错误或参数有误，已跳过！")
                continue
            log_list.append(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] 正在合成第 {i+1} 段对话，模型：{model_name}，说话人：{speaker}，情感：{emotion}")
            load_model(model_name)
            if seed == -1:
                seed = random_seed()
            audio = tts_infer(text, text_lang, ref_audio, prompt_text, prompt_lang, top_k, top_p, temperature, text_split_method, batch_size, batch_threshold, split_bucket, speed_facter, fragment_interval, seed, media_type, parallel_infer, repetition_penalty)
            Path(f"outputs/conv_{content_md5}/{i+1}_{speaker}.wav").write_bytes(audio)
            Path(f"outputs/conv_{content_md5}/{i+1}_{speaker}.txt").write_text(text, encoding="utf-8")
            log_list.append(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] 第 {i+1} 段对话合成成功！")
        Path(f"outputs/conv_{content_md5}/log.txt").write_text("\n".join(log_list), encoding="utf-8")
        if os.name == "nt":
            subprocess.run(f"./7-Zip/7za.exe a -t7z outputs/conv_{content_md5}.7z outputs/conv_{content_md5}",stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        else:
            subprocess.run(f"7za a -t7z outputs/conv_{content_md5}.7z outputs/conv_{content_md5}", shell=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        archive_path = f"outputs/conv_{content_md5}.7z"
        msg = "合成成功"
    except:
        msg = "合成失败，参数错误！"
        archive_path = ""
    return archive_path, msg