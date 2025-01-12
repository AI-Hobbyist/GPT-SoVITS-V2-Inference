from tools.my_infer import get_models, get_multi_ref_template, create_speaker_list, custom_ref, single_infer, multi_infer, pre_infer
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
import argparse

APP = FastAPI()

# 定义请求参数模型
class requestModel(BaseModel):
    model: str
    
class inferWithCustomRefAaudio(BaseModel):
    app_key: str = ""
    audio_dl_url: str = ""
    model_name: str = ""
    ref_audio_b64: str = ""
    text: str = ""
    text_lang: str = ""
    prompt_text: str = ""
    prompt_text_lang: str = ""
    top_k: int = 10
    top_p: float = 1.0
    temperature: float = 1.0
    text_split_method: str = "按标点符号切"
    batch_size: int = 1
    batch_threshold: float = 0.75
    split_bucket: bool = True
    speed_facter: float = 1.0
    fragment_interval: float = 0.3
    media_type: str = "wav"
    parallel_infer: bool = True
    repetition_penalty: float = 1.35
    seed: int = -1
    
class inferWithEmotions(BaseModel):
    app_key: str = ""
    audio_dl_url: str = ""
    model_name: str = ""
    speaker_name: str = ""
    prompt_text_lang: str = ""
    emotion: str = ""
    text: str = ""
    text_lang: str = ""
    top_k: int = 10
    top_p: float = 1.0
    temperature: float = 1.0
    text_split_method: str = "按标点符号切"
    batch_size: int = 1
    batch_threshold: float = 0.75
    split_bucket: bool = True
    speed_facter: float = 1.0
    fragment_interval: float = 0.3
    media_type: str = "wav"
    parallel_infer: bool = True
    repetition_penalty: float = 1.35
    seed: int = -1
    
class inferWithMulti(BaseModel):
    app_key: str = ""
    archive_dl_url: str = ""
    content : str = ""
    top_k: int = 10
    top_p: float = 1.0
    temperature: float = 1.0
    text_split_method: str = "按标点符号切"
    batch_size: int = 1
    batch_threshold: float = 0.75
    split_bucket: bool = True
    fragment_interval: float = 0.3
    media_type: str = "wav"
    parallel_infer: bool = True
    repetition_penalty: float = 1.35
    seed: int = -1
    
    
    
# 初始化
@APP.get("/")
async def root():
    return {"message": "This is a TTS inference API. If you see this page, it means the server is running."}

# 获取模型列表
@APP.get("/models")
async def models():
    return get_models()

# 获取多人对话模板
@APP.post("/template")
async def template(model: requestModel):
    templates, msg = get_multi_ref_template(model.model)
    return {"msg": msg, "templates": templates}

# 获取说话人列表
@APP.post("/spks")
async def speaker_list(model: requestModel):
    spk_list, msg = create_speaker_list(model.model)
    return {"msg": msg, "speakers": spk_list}

# 根据自定义参考音频进行推理
@APP.post("/infer_ref")
async def infer_ref(model: inferWithCustomRefAaudio):
    try:
        if model.app_key != infer_key and infer_key != "":
            msg = "app_key错误"
            audio_url = ""
        else:
            audio_path, msg = custom_ref(model.model_name, model.ref_audio_b64, model.text, model.text_lang, model.prompt_text, model.prompt_text_lang, model.top_k, model.top_p, model.temperature, model.text_split_method, model.batch_size, model.batch_threshold, model.split_bucket, model.speed_facter, model.fragment_interval, model.media_type, model.parallel_infer, model.repetition_penalty, model.seed)
            if model.audio_dl_url == "":
                audio_url = f"http://{host}:{port}/{audio_path}"
            else:
                audio_url = f"{model.audio_dl_url}/{audio_path}"
    except:
        msg = "参数错误"
        audio_url = ""
    return {"msg": msg, "audio_url": audio_url}

# 根据情感进行推理
@APP.post("/infer_emotion")
async def infer_emotion(model: inferWithEmotions):
    try:
        if model.app_key != infer_key and infer_key != "":
            msg = "app_key错误"
            audio_url = ""
        else:
            audio_path, msg = single_infer(model.model_name, model.speaker_name, model.prompt_text_lang, model.emotion, model.text, model.text_lang, model.top_k, model.top_p, model.temperature, model.text_split_method, model.batch_size, model.batch_threshold, model.split_bucket, model.speed_facter, model.fragment_interval, model.media_type, model.parallel_infer, model.repetition_penalty, model.seed)
            if model.audio_dl_url == "":
                audio_url = f"http://{host}:{port}/{audio_path}"
            else:
                audio_url = f"{model.audio_dl_url}/{audio_path}"
    except:
        msg = "参数错误"
        audio_url = ""
    return {"msg": msg, "audio_url": audio_url}

# 根据多人对话模板进行推理
@APP.post("/infer_multi")
async def infer_multi(model: inferWithMulti):
    try:
        if model.app_key != infer_key and infer_key != "":
            msg = "app_key错误"
            archive_url = ""
        else:
            archive_path, msg = multi_infer(model.content, model.top_k, model.top_p, model.temperature, model.text_split_method, model.batch_size, model.batch_threshold, model.split_bucket, model.fragment_interval, model.media_type, model.parallel_infer, model.repetition_penalty, model.seed)
            if model.archive_dl_url == "":
                archive_url = f"http://{host}:{port}/{archive_path}"
            else:
                archive_url = f"{model.archive_dl_url}/{archive_path}"
    except:
        msg = "参数错误"
        archive_url = ""
    return {"msg": msg, "archive_url": archive_url}

# 下载生成结果
from fastapi.staticfiles import StaticFiles

APP.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")
@APP.get("/outputs/{result_path}")
async def download(result_path: str):
    return FileResponse(f"outputs/{result_path}")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="TTS Inference API")
    parser.add_argument("-s","--host", type=str, default="127.0.0.1", help="主机地址")
    parser.add_argument("-p","--port", type=int, default=8000, help="端口")
    parser.add_argument("-k","--key", type=str, default="", help="推理密钥")
    parser.add_argument("-c","--config", type=str, default="GPT_SoVITS/configs/tts_infer.yaml", help="默认配置文件")
    args = parser.parse_args()
    
    infer_key = args.key
    host = args.host
    port = args.port
    
    pre_infer(args.config)
    
    import uvicorn
    uvicorn.run(app=APP, host=host, port=port)