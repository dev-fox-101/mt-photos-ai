FROM openvino/ubuntu22_runtime:latest
USER root
# 镜像加速
COPY ./sources.list /etc/apt/sources.list
RUN apt update && \
    apt install -y python3.10-dev libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libfontconfig1 && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/log/*

WORKDIR /app

COPY requirements.txt .

# 安装依赖包
RUN pip3 install --no-cache-dir -r requirements.txt --index-url=https://pypi.tuna.tsinghua.edu.cn/simple/

RUN apt update && \
    apt install -y build-essential && \
    pip3 install --no-cache-dir insightface==0.7.3 --index-url=https://pypi.tuna.tsinghua.edu.cn/simple/ && \
    apt remove -y build-essential && \
    apt purge -y build-essential && \
    apt autoremove -y && apt autoclean -y && rm -rf /var/lib/apt/lists/* /tmp/* /var/log/*

COPY ./utils/vit-b-16.img.fp32.onnx ./utils/vit-b-16.img.fp32.onnx
COPY ./utils/vit-b-16.txt.fp32.onnx ./utils/vit-b-16.txt.fp32.onnx
COPY ./utils/bert_tokenizer.py ./utils/bert_tokenizer.py
COPY ./utils/vocab.txt ./utils/vocab.txt
COPY ./utils/clip.py ./utils/clip.py
COPY ./_insightface_root /root/.insightface

COPY server.py .

ENV NO_ALBUMENTATIONS_UPDATE=1
ENV API_AUTH_KEY=mt_photos_ai_extra
EXPOSE 8060

CMD [ "python3", "/app/server.py" ]
