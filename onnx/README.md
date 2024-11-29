# dml运行注意点

python 3.8

卸载onnxruntime，单独安装 onnxruntime-directml
```
pip uninstall onnxruntime
pip install onnxruntime-directml -i https://pypi.tuna.tsinghua.edu.cn/simple/
```

server.py 修改
```
    embedding_objs = await predict(_represent, img)
    # embedding_objs = _represent(img)  # DmlExecutionProvider使用异步并发时会导致程序退出
```


## 运行中错误

Windows server 2016 不支持dml版本的onnxruntime

用纯cpu版本版本即可
