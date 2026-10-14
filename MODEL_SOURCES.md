# 模型与实现来源

- SD1.5：https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5 ，CreativeML Open RAIL-M。基础权重复用本机已准备组件。
- IP-Adapter：https://huggingface.co/h94/IP-Adapter ，模型卡标注 Apache-2.0；代码来源：https://github.com/tencent-ailab/IP-Adapter ，Apache-2.0。使用公开SD1.5适配器及models/image_encoder。
- Canny ControlNet：https://huggingface.co/lllyasviel/control_v11p_sd15_canny ，模型卡标注OpenRAIL / CreativeML OpenRAIL-M。使用公开Diffusers转换的fp16组件。
- 组合实现参考：https://huggingface.co/docs/diffusers/v0.35.1/using-diffusers/ip_adapter 。本项目通过已安装Diffusers API组合模型，未复制InstantStyle-Plus代码。

模型准备脚本保留对应模型README。交付源码包不包含权重；模型许可和本项目代码许可分别适用。参考图像编码器来源也需遵循IP-Adapter模型卡列出的上游说明。研究样板没有训练、微调或更新这些权重。
