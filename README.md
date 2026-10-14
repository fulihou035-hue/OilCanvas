# OilCanvas 照片油画工作室

将照片与参考画作组合成油画风格图片的本地应用。现支持“明亮柔和油画”（Stable Diffusion 1.5 + IP-Adapter + Canny ControlNet）和原有 StyleID 两种模式，通过 Gradio 提供中文互动界面。当前版本只处理静态图片，不训练模型，不调用付费图像 API。

## 2026年10月13日升级

根据“上一版像模糊低像素图”的反馈，增加“明显油画笔触”：IP-Adapter Plus细节参考与厚涂提示，默认使用768像素/30步。原有柔和模式仍可选择。新增细节适配器约98MB；缓存就绪后的768校园样板约4.8秒，仍可能重绘天空、建筑与人物。768为实际模型生成尺寸，不是放大512结果。

- 模型结果直接保存 PNG，取消最终结果的 JPEG 中转。
- 完整构图使用等比缩放、补边推理和去除补边，修复比例拉伸。
- 曝光、阴影、鲜艳度即时调节；每次从原始模型结果调色，避免累计损失，下载参数包含调色设置。
- 新模式可分别调整重绘程度、参考画风强度和轮廓保留；夜景选择“保留原图光照”。
- 新模式在应用进程内常驻模型，缓存最多8个参考图特征；切换旧模式时释放常驻模型。重启后需要重新加载。

同一张512像素校园图，新模式首次任务10.06秒，重复任务1.43秒；这只是本机小样记录。六类照片已完成新旧对照，色彩与绘画感有所改善，五官、手部、建筑和小文字仍可能改变，不保证身份一致或真实细节完整。

## 这台电脑如何启动

双击 `启动APP.bat`，随后在浏览器中使用本地界面。也可以访问 http://127.0.0.1:7860 。关闭启动窗口即可结束本次服务。

本提交包不包含本机配置、环境和权重。首次使用请按下方安装说明准备环境与模型；在本机原项目目录中已配置完成。

## 如何使用

1. 上传 JPG、PNG 或 WebP 照片，或点击公开示例按钮。
2. 选择梵高《柏树麦田》或《戴草帽的自画像》，也可以上传自己的参考画作。
3. 新模式分别调整重绘、参考画风和轮廓保留；旧模式的人像可先用0.95并勾选减少原始色调变化。
4. 先用 512 像素和 20 步快速预览。需要细节时选择 768 像素和 30 步。
5. 点击生成，查看原图和结果，下载 PNG 与参数文件。

默认保留完整构图，图片等比缩放后补边到64的倍数，输出移除补边；最终图片尺寸不要求为64的倍数。正方形模式会居中裁剪。结果保存在 `results`。旧模式保存进程日志，新模式保存参数与缓存命中记录。

## 在另一台电脑上安装

推荐 Python 3.10 或 3.11、支持 CUDA 的 NVIDIA 显卡和至少 8GB 显存。先依据 [PyTorch 官方安装说明](https://pytorch.org/get-started/locally/) 安装适合显卡的 CUDA 版本，RTX 50 系列需要支持该硬件的 PyTorch/CUDA 组合。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe prepare_model.py
.\.venv\Scripts\python.exe prepare_artistic.py
.\.venv\Scripts\python.exe app.py --browser
```

首次准备模型需要联网，之后推理可离线。新模式新增组件约3.4GB。`prepare_artistic.py` 支持断点续传。也可通过 `OILCANVAS_MODEL_DIR` 指向已有 SD1.5 Diffusers 半精度组件目录，通过 `OILCANVAS_ARTISTIC_DIR` 指向新的组件目录。没有准备新组件时，界面默认使用旧模式。

## 源码结构

```text
app.py                    中文交互界面
engine.py                 图片预处理、参数校验、任务调度与导出
artistic_backend.py       新绘画后端、模型常驻和参考图特征缓存
color_tools.py            独立明暗与色彩调节
launch.py / 启动APP.bat    本地启动器
prepare_model.py           模型准备
prepare_artistic.py        新绘画组件准备
assets/styles/            公有领域画作与来源记录
vendor/styleid/           未修改的 StyleID 官方代码与许可
vendor/styleid_runtime/   已实测的显存适配版本
tests/test_engine.py      图片预处理与参数校验检查
```

## 与上游代码的关系

算法核心复用 [StyleID](https://github.com/jiwoogit/StyleID) 官方 Diffusers 实现，MIT 许可保留在两个 vendor 目录。改动包括：半精度组件加载、SDPA 注意力计算、CPU 特征缓存、只解码末步、修正浮点参数解析，以及本地模型路径与离线配置。`vendor/styleid/sources.json` 记录原始文件 URL 与 SHA-256。未宣称提出新的风格迁移算法或复现原论文指标。

界面流程参考 MIT 许可的 [coopercox315/neural-style-transfer](https://github.com/coopercox315/neural-style-transfer)，没有直接复制该项目的应用代码。交互组件使用 [Gradio](https://github.com/gradio-app/gradio)。

## 实测与局限

RTX 5070 Laptop 8GB、约 32GB 内存上，已有 512×512 与 768×768 样板成功运行。具体记录见汇报材料。人像的鼻子、嘴唇、眼镜和表情仍可能变化。当前没有独立的身份保持模块，内容保留滑块不能保证五官完全还原。每次任务独立启动推理进程，模型加载开销仍存在。

原有StyleID模式仍按任务启动独立推理进程；新模式使用模型常驻与参考特征缓存。这里的显存指标是 PyTorch 分配峰值，不是系统总显存占用。耗时来自小规模单次运行，不是稳定性能保证。

## 开源素材

两幅画作来自 The Metropolitan Museum of Art，并由馆藏 API 标注为 Public Domain。具体标题、年代、下载地址和许可记录在 `assets/styles/sources.json`。公开示例照片来自 scikit-image 的 NASA 宇航员图片。

Stable Diffusion 1.5 权重采用 CreativeML Open RAIL-M 许可。模型许可与代码 MIT 许可分别适用。未经授权的图片、个人照片、结果、模型权重和机器配置不应放进公开仓库。

## 提交 GitHub 前

保留所有来源与许可证。`.gitignore` 已排除模型、环境、结果和本机配置。将本项目源码作为自己的工程作品提交，并在介绍中说明开源算法、AI 协助和自己需要理解的实现部分。

