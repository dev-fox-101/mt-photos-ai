# MT Photos AI识别相关任务独立部署项目

合并了 https://github.com/MT-Photos/mt-photos-ai 和 https://github.com/kqstone/mt-photos-insightface-unofficial 部分的代码

运行一个程序就可以支持以上2个程序的功能

## 目录说明

- openvino：基于onnxruntime_openvino库，进行识别任务。**适用于Intel Xeon、Core CPU运行**
- onnx：基于onnxruntime库，进行识别任务。**适用于所有CPU运行**
  
> 在Intel cpu上运行时OpenVINO版本会快很多；

>RapidOCR更多配置可参考官方仓库 https://github.com/RapidAI/RapidOCR 

## 镜像说明

DockerHub镜像仓库地址：
https://hub.docker.com/r/devfox101/mt-photos-ai

镜像Tags说明：

- latest：基于openvino文件夹打包生成，推荐**Intel CPU**机型安装这个镜像
- onnx:基于onnx文件夹打包生产，推荐**AMD CPU**机型安装这个镜像


### 打包docker镜像

```bash
cd onnx
docker build  . -t mt-photos-ai:onnx-latest
```
`onnx`为文件夹，可根据需要替换为`cuda`、`openvino`

### 运行docker容器

```bash
docker run -i -p 8000:8000 -e API_AUTH_KEY=mt_photos_ai_extra --name mt-photos-ai --restart="unless-stopped" mt-photos-ai:onnx-latest
```
`onnx-latest`可以替换为`latest`、`onnx-latest`


### 下载源码本地运行

- 安装python **3.10版本**
- 根据硬件环境选择onnx或openvino文件夹
- 在选择文件夹下执行`pip install -r requirements.txt`
- 复制`.env.example`生成`.env`文件，然后修改`.env`文件内的API_AUTH_KEY
- 执行 `python server.py` ，启动服务

> API_AUTH_KEY为MT Photos填写api_key需要输入的值

> ./onnx/utils 和 ./openvino/utils 目录下需要手动添加 vit-b-16.img.fp32.onnx、vit-b-16.txt.fp32.onnx 2个模型文件
>
> 可以从release中下载附件：https://github.com/MT-Photos/mt-photos-ai/releases/tag/v1.1.0
>
> 手动转换ONNX模型方法见：
>
> https://github.com/OFA-Sys/Chinese-CLIP/blob/master/deployment.md#%E8%BD%AC%E6%8D%A2%E5%92%8C%E8%BF%90%E8%A1%8Connx%E6%A8%A1%E5%9E%8B


看到以下日志，则说明服务已经启动成功
```bash
INFO:     Started server process [3024]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8060 (Press CTRL+C to quit)
```


## API

### /check

检测服务是否可用，及api-key是否正确

```bash
curl --location --request POST 'http://127.0.0.1:8000/check' \
--header 'api-key: api_key'
```

**response:**

```json
{
  "result": "pass"
}
```

### /ocr

```bash
curl --location --request POST 'http://127.0.0.1:8000/ocr' \
--header 'api-key: api_key' \
--form 'file=@"/path_to_file/test.jpg"'
```

**response:**

- result.texts : 识别到的文本列表
- result.scores : 为识别到的文本对应的置信度分数，1为100%
- result.boxes : 识别到的文本位置，x,y为左上角坐标，width,height为框的宽高

```json
{
  "result": {
    "texts": [
      "识别到的文本1",
      "识别到的文本2"
    ],
    "scores": [
      "0.98",
      "0.97"
    ],
    "boxes": [
      {
        "x": "4.0",
        "y": "7.0",
        "width": "283.0",
        "height": "21.0"
      },
      {
        "x": "7.0",
        "y": "34.0",
        "width": "157.0",
        "height": "23.0"
      }
    ]
  }
}
```


### /clip/img

```bash
curl --location --request POST 'http://127.0.0.1:8000/clip/img' \
--header 'api-key: api_key' \
--form 'file=@"/path_to_file/test.jpg"'
```

**response:**

- results : 图片的特征向量

```json
{
  "results": [
    "0.3305919170379639",
    "-0.4954293668270111",
    "0.0217289477586746",
    ...
  ]
}
```

### /clip/txt

```bash
curl --location --request POST 'http://127.0.0.1:8000/clip/txt' \
--header "Content-Type: application/json" \
--header 'api-key: api_key' \
--data '{"text":"飞机"}'
```

**response:**

- results : 文字的特征向量

```json
{
  "results": [
    "0.3305919170379639",
    "-0.4954293668270111",
    "0.0217289477586746",
    ...
  ]
}
```

### /restart_v2

通过重启进程来释放内存

```bash
curl --location --request POST 'http://127.0.0.1:8000/restart_v2' \
--header 'api-key: api_key'
```

**response:**

请求中断,没有返回，因为服务重启了
