# 运行要求

onnxruntime-openvino 需要 python >= 3.10

Ubuntu20只能安装python 3.8 ，因此将基础镜像升级到3.10了


由于容器内install insightface 需要 gcc，因此放在一个单独的命令中build
```
pip install -r requirements.txt --index-url=https://pypi.tuna.tsinghua.edu.cn/simple/
pip install insightface==0.7.3 --index-url=https://pypi.tuna.tsinghua.edu.cn/simple/
```

### Windows下人脸识别无法调用openvino加速

会提示下面的错误，因此windows exe版本先不打包 openvino加速版本：
`EP Error C:\Users\Administrator\Desktop\rel-1.19_final\onnxruntime\onnxruntime\core\session\provider_bridge_ort.cc:1637 onnxruntime::ProviderLibrary::Get [ONNXRuntimeError] : 1 : FAIL : LoadLibrary failed with error 127 "" when trying to load "C:\Python310\lib\site-packages\onnxruntime\capi\onnxruntime_providers_openvino.dll"`
