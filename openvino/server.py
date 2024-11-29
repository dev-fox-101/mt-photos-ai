from dotenv import load_dotenv
import os
import sys
import threading
from fastapi import Depends, FastAPI, File, UploadFile, HTTPException, Header
from fastapi.responses import HTMLResponse
import uvicorn
import numpy as np
import cv2
import asyncio
from pydantic import BaseModel
from rapidocr_openvino import RapidOCR
from PIL import Image
from io import BytesIO
import insightface
from insightface.utils import storage
from insightface.app import FaceAnalysis
import logging
import utils.clip as clip

logging.basicConfig(level=logging.WARNING)


# import onnxruntime as ort
# device = ort.get_device()
# print(f"Using device: {device}")

on_linux = sys.platform.startswith('linux')

load_dotenv()
app = FastAPI()
api_auth_key = os.getenv("API_AUTH_KEY", "mt_photos_ai_extra")
http_port = int(os.getenv("HTTP_PORT", "8060"))
server_restart_time = int(os.getenv("SERVER_RESTART_TIME", "300"))
env_auto_load_txt_modal = os.getenv("AUTO_LOAD_TXT_MODAL", "off") == "on" # 是否自动加载CLIP文本模型，开启可以优化第一次搜索时的响应速度,文本模型占用700多m内存

restart_timer = None
rapid_ocr = None
clip_img_model = None
clip_txt_model = None
face_model = None


detector_backend = os.getenv("DETECTOR_BACKEND", "insightface")
recognition_model = os.getenv("RECOGNITION_MODEL", "buffalo_l")
detection_thresh = float(os.getenv("DETECTION_THRESH", "0.65"))
# 设置下载模型URL
storage.BASE_REPO_URL = 'https://github.com/kqstone/mt-photos-insightface-unofficial/releases/download/models'
on_win = sys.platform.startswith('win')
model_folder_path = '~/.insightface'
if on_win :
    current_folder = os.path.dirname(os.path.abspath(__file__))
    model_folder_path = os.path.join(current_folder, "_insightface_root")


class ClipTxtRequest(BaseModel):
    text: str

def load_ocr_model():
    global rapid_ocr
    if rapid_ocr is None:
        rapid_ocr = RapidOCR()

def load_clip_img_model():
    global clip_img_model
    if clip_img_model is None:
        clip_img_model = clip.load_img_model()

def load_clip_txt_model():
    global clip_txt_model
    if clip_txt_model is None:
        clip_txt_model = clip.load_txt_model()

def load_face_model():
    global face_model
    if face_model is None:
        faceAnalysis = FaceAnalysis(providers= ["OpenVINOExecutionProvider","CPUExecutionProvider"],root=model_folder_path, allowed_modules=['detection', 'recognition'], name=recognition_model)
        faceAnalysis.prepare(ctx_id=0, det_thresh=detection_thresh, det_size=(640, 640))
        face_model = faceAnalysis


@app.on_event("startup")
async def startup_event():
    if env_auto_load_txt_modal:
        load_clip_txt_model()


@app.middleware("http")
async def check_activity(request, call_next):
    global restart_timer

    if restart_timer:
        restart_timer.cancel()

    restart_timer = threading.Timer(server_restart_time, restart_program)
    restart_timer.start()

    response = await call_next(request)
    return response

async def verify_header(api_key: str = Header(...)):
    # 在这里编写验证逻辑，例如检查 api_key 是否有效
    if api_key != api_auth_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return api_key


def to_fixed(num):
    return str(round(num, 2))


def trans_result(result):
    texts = []
    scores = []
    boxes = []
    if result is None:
        return {'texts': texts, 'scores': scores, 'boxes': boxes}
    for res_i in result:
        dt_box = res_i[0]
        box = {
            'x': to_fixed(dt_box[0][0]),
            'y': to_fixed(dt_box[0][1]),
            'width': to_fixed(dt_box[1][0] - dt_box[0][0]),
            'height': to_fixed(dt_box[2][1] - dt_box[0][1])
        }
        boxes.append(box)
        texts.append(res_i[1])
        scores.append(f"{res_i[2]:.2f}")
    return {'texts': texts, 'scores': scores, 'boxes': boxes}


@app.get("/", response_class=HTMLResponse)
async def top_info():
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>MT Photos AI Server</title>
    <style>p{text-align: center;}</style>
</head>
<body>
<p style="font-weight: 600;">MT Photos智能识别服务</p>
<p>服务状态： 运行中</p>
<p>使用方法： <a href="https://mtmt.tech/docs/advanced/ocr_api">https://mtmt.tech/docs/advanced/ocr_api</a></p>
</body>
</html>"""
    return html_content


@app.post("/check")
async def check_req(api_key: str = Depends(verify_header)):
    return {
        'result': 'pass',
        "title": "mt-photos-ai服务",
        "help": "https://mtmt.tech/docs/advanced/ocr_api",
        "detector_backend": detector_backend,
        "recognition_model": recognition_model,
        "facial_min_score": detection_thresh, # 推荐的人脸最低置信度阈值
        "facial_max_distance": 0.5, # 推荐的人脸差异值
    }


@app.post("/restart")
async def check_req(api_key: str = Depends(verify_header)):
    # 客户端可调用，触发重启进程来释放内存
    # restart_program()
    return {'result': 'pass'}

@app.post("/restart_v2")
async def check_req(api_key: str = Depends(verify_header)):
    # 预留触发服务重启接口-自动释放内存
    restart_program()
    return {'result': 'pass'}

@app.post("/ocr")
async def process_image(file: UploadFile = File(...), api_key: str = Depends(verify_header)):
    load_ocr_model()
    image_bytes = await file.read()
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        height, width, _ = img.shape
        if width > 10000 or height > 10000:
            return {'result': [], 'msg': 'height or width out of range'}
        _result = rapid_ocr(img)
        result = trans_result(_result[0])
        del img
        del _result
        return {'result': result}
    except Exception as e:
        print(e)
        return {'result': [], 'msg': str(e)}

@app.post("/clip/img")
async def clip_process_image(file: UploadFile = File(...), api_key: str = Depends(verify_header)):
    load_clip_img_model()
    image_bytes = await file.read()
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        result = await predict(clip.process_image, img, clip_img_model)
        return {'result': ["{:.16f}".format(vec) for vec in result]}
    except Exception as e:
        print(e)
        return {'result': [], 'msg': str(e)}

@app.post("/clip/txt")
async def clip_process_txt(request:ClipTxtRequest, api_key: str = Depends(verify_header)):
    load_clip_txt_model()
    text = request.text
    result = await predict(clip.process_txt, text, clip_txt_model)
    return {'result': ["{:.16f}".format(vec) for vec in result]}



@app.post("/represent")
async def process_image(file: UploadFile = File(...), api_key: str = Depends(verify_header)):
    load_face_model()
    content_type = file.content_type
    image_bytes = await file.read()
    try:
        img = None
        if content_type == 'image/gif':
            # Use Pillow to read the first frame of the GIF file
            with Image.open(BytesIO(image_bytes)) as img:
                if img.is_animated:
                    img.seek(0)  # Seek to the first frame of the GIF
                frame = img.convert('RGB')  # Convert to RGB mode
                np_arr = np.array(frame)  # Convert to NumPy array
                img = cv2.cvtColor(np_arr, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV
        if img is None:
            # Use OpenCV for other image types
            np_arr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            err = f"The uploaded file {file.filename} is not a valid image format or is corrupted."
            print(err)
            return {'result': [], 'msg': str(err)}

        height, width, _ = img.shape
        if width > 10000 or height > 10000:
            return {'result': [], 'msg': 'height or width out of range'}

        data = {"detector_backend": detector_backend, "recognition_model": recognition_model}

        # embedding_objs = await predict(_represent, img)
        embedding_objs = _represent(img)  # DmlExecutionProvider使用异步并发时会导致程序退出
        del img
        data["result"] = embedding_objs
        # logging.info("detector_backend: %s", detector_backend)
        # logging.info("recognition_model: %s", recognition_model)
        logging.info("detected_img: %s", file.filename)
        logging.info("img_type: %s", content_type)
        logging.info("detected_persons: %d", len(embedding_objs))
        for embedding_obj in embedding_objs:
            logging.info("facial_area: %s", str(embedding_obj["facial_area"]))
            logging.info("face_confidence: %f", embedding_obj["face_confidence"])
        return data
    except Exception as e:
        if 'set enforce_detection' in str(e):
            return {'result': []}
        print(e)
        return {'result': [], 'msg': str(e)}

def _represent(img):
    faces = face_model.get(img)
    results = []
    for face in faces:
        resp_obj = {}
        embedding = face.normed_embedding.astype(float)
        resp_obj["embedding"] = embedding.tolist()
        # print(len(resp_obj["embedding"]))
        box = face.bbox
        resp_obj["facial_area"] = {"x" : int(box[0]), "y" : int(box[1]), "w" : int(box[2] - box[0]), "h" : int(box[3] - box[1])}
        resp_obj["face_confidence"] = face.det_score.astype(float)
        results.append(resp_obj)
    return results


async def predict(predict_func, inputs,model):
    return await asyncio.get_running_loop().run_in_executor(None, predict_func, inputs,model)

def restart_program():
    python = sys.executable
    os.execl(python, python, *sys.argv)


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=http_port)
