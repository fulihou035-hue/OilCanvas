"""Local Gradio UI, inspired by existing open-source neural-style-transfer apps."""
import os
os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'
os.environ['NO_PROXY'] = 'localhost,127.0.0.1,' + os.environ.get('NO_PROXY', '')
from pathlib import Path
import json
import argparse
import uuid
import gradio as gr
from PIL import Image
from engine import ROOT, stylize,stylize_artistic
from color_tools import adjust_colors
from artistic_backend import artistic_directory

STYLES = ROOT / 'assets' / 'styles'
PRESETS = {'梵高 · 柏树麦田': STYLES / 'vangogh.jpg', '梵高 · 戴草帽的自画像': STYLES / 'vangoghportrait.jpg'}

def pick_style(name):
    path = PRESETS.get(name)
    return Image.open(path).convert('RGB') if path and path.exists() else None

def run(photo, style, gamma, size, steps, tones, framing, backend, strength, style_scale, structure_scale, lighting, painting_effect, progress=gr.Progress()):
    try:
        values = (stylize_artistic(photo,style,int(size),int(steps),framing,strength,style_scale,structure_scale,lighting,progress,texture='厚涂' if painting_effect=='明显油画笔触' else '柔和')
                  if backend == '明亮柔和油画（新方案）' else stylize(photo, style, gamma, int(size), int(steps), tones, framing, progress))
        return (*values, values[2])
    except ValueError as error:
        raise gr.Error(str(error))

def recolor(path, exposure, shadows, saturation):
    if not path:
        raise gr.Error('请先生成一张油画。')
    output = Path(path).parent/'adjusted_painting.png'
    image = adjust_colors(Image.open(path),exposure,shadows,saturation)
    temporary = output.with_name('adjusted-'+uuid.uuid4().hex+'.png')
    image.save(temporary)
    os.replace(temporary,output)
    parameters = json.loads((output.parent/'parameters.json').read_text(encoding='utf-8'))
    parameters['color_adjustments'] = {'exposure':exposure,'shadows':shadows,'saturation':saturation}
    export = output.parent/'export_parameters.json'
    temporary = export.with_name('parameters-'+uuid.uuid4().hex+'.json')
    temporary.write_text(json.dumps(parameters,ensure_ascii=False,indent=2),encoding='utf-8')
    os.replace(temporary,export)
    return image,str(output),str(export)

CSS = '''
.gradio-container {max-width:1180px !important; background:#faf7f0;}
#heading h1 {font-size:38px; color:#25382f; margin-bottom:10px;}
#heading p {color:#657369; font-size:16px;}
#generate {background:#315646; color:white; border-radius:12px; min-height:48px;}
footer {display:none !important;}
'''

def create_app():
    history = sorted((ROOT / 'results').glob('*/oil_painting.png'), reverse=True)
    last = history[0].parent if history else None
    theme = gr.themes.Soft(primary_hue='green', neutral_hue='stone', font=['Microsoft YaHei', 'sans-serif'])
    with gr.Blocks(title='OilCanvas 照片油画工作室', theme=theme, css=CSS) as demo:
        gr.Markdown('# OilCanvas 照片油画工作室\n把你的照片，转成一幅有笔触的油画。', elem_id='heading')
        with gr.Row():
            with gr.Column(scale=1):
                photo = gr.Image(type='pil', label='1 上传照片', sources=['upload'], height=300)
                gr.Button('试用公开示例照片').click(lambda: Image.open(ROOT / 'assets' / 'demo_photo.png'), outputs=photo)
            with gr.Column(scale=1):
                preset = gr.Dropdown(list(PRESETS) + ['自选参考画作'], value='梵高 · 柏树麦田', label='2 选择油画风格')
                style = gr.Image(type='pil', value=pick_style('梵高 · 柏树麦田'), label='参考画作（也可自行上传）', sources=['upload'], height=258)
                preset.change(pick_style, inputs=preset, outputs=style)
        ready = (artistic_directory(ROOT)/'controlnet'/'diffusion_pytorch_model.fp16.safetensors').exists() and (artistic_directory(ROOT)/'ip_adapter'/'models'/'image_encoder'/'model.safetensors').exists()
        detail_ready = ready and (artistic_directory(ROOT)/'ip_adapter'/'models'/'ip-adapter-plus_sd15.safetensors').exists()
        backend = gr.Dropdown(['明亮柔和油画（新方案）','原有油画模式'],value='明亮柔和油画（新方案）' if ready else '原有油画模式',label='绘画模式')
        with gr.Group(visible=not ready) as old_controls:
            with gr.Row():
                gamma = gr.Slider(0.5, 0.99, value=0.95, step=0.01, label='五官与结构保留', info='数值越高，越倾向保留原图结构；仍可能改变五官。')
                tones = gr.Checkbox(value=True, label='减少原始色调变化')
        with gr.Accordion('绘画细节设置',open=False,visible=ready) as new_controls:
            painting_effect = gr.Radio(['柔和','明显油画笔触'],value='明显油画笔触' if detail_ready else '柔和',label='笔触表现',info='明显笔触建议768像素、30步。')
            strength = gr.Slider(0.2,0.8,value=0.6 if detail_ready else 0.55,step=0.05,label='重绘程度',info='越高越接近重新绘画，也更容易改变细节。')
            style_scale = gr.Slider(0,1,value=0.65 if detail_ready else 0.45,step=0.05,label='参考画风强度')
            structure_scale = gr.Slider(0,1,value=0.65,step=0.05,label='轮廓保留程度')
            lighting = gr.Radio(['明亮日光','保留原图光照'],value='明亮日光',label='光照风格',info='夜景建议保留原图光照。')
        backend.change(lambda name:(gr.update(visible=name=='原有油画模式'),gr.update(visible=name!='原有油画模式')),inputs=backend,outputs=[old_controls,new_controls])
        with gr.Accordion('画面与速度设置', open=False):
            with gr.Row():
                size = gr.Radio(['512', '768'], value='768' if detail_ready else '512', label='输出长边（像素）')
                steps = gr.Radio(['20', '30'], value='30' if detail_ready else '20', label='处理步数')
                framing = gr.Radio(['保持完整构图', '中心裁剪正方形'], value='保持完整构图', label='构图方式')
        generate = gr.Button('生成我的油画', variant='primary', elem_id='generate')
        status = gr.Textbox(label='生成状态', value='下方显示最近一次生成结果。' if last else None,
            interactive=False, lines=2, max_lines=3, placeholder='上传照片后，点击生成。首次加载会稍慢一些。')
        with gr.Row():
            original = gr.Image(value=str(last / ('display_photo.png' if (last/'display_photo.png').exists() else 'photo.png')) if last else None, label='处理后的原图', interactive=False, height=400)
            result = gr.Image(value=str(last / 'oil_painting.png') if last else None, label='油画结果', interactive=False, height=400)
        with gr.Row():
            download = gr.File(value=str(last / 'oil_painting.png') if last else None, label='下载 PNG 图片', interactive=False)
            parameters = gr.File(value=str(last / 'parameters.json') if last else None, label='下载本次参数', interactive=False)
        raw_result = gr.State(value=str(last/'oil_painting.png') if last else None)
        with gr.Accordion('明暗与色彩（无需重新生成）',open=True):
            exposure = gr.Slider(-1.5,1.5,value=0,step=0.05,label='曝光',elem_id='exposure-control')
            shadows = gr.Slider(0,1,value=0,step=0.05,label='提亮阴影')
            saturation = gr.Slider(0,2,value=1,step=0.05,label='色彩鲜艳度')
            reset = gr.Button('恢复模型原始色彩')
        for control in [exposure,shadows,saturation]:
            control.change(recolor,inputs=[raw_result,exposure,shadows,saturation],outputs=[result,download,parameters],queue=False)
        reset.click(lambda:(0,0,1),outputs=[exposure,shadows,saturation]).then(recolor,inputs=[raw_result,exposure,shadows,saturation],outputs=[result,download,parameters],queue=False)
        generate.click(run, inputs=[photo, style, gamma, size, steps, tones, framing,backend,strength,style_scale,structure_scale,lighting,painting_effect],
            outputs=[original, result, download, parameters, status,raw_result], api_name='stylize').then(recolor,inputs=[raw_result,exposure,shadows,saturation],outputs=[result,download,parameters])
        gr.Markdown('人像提示：油画重绘可能改变鼻子、嘴唇和表情。当前版本适合艺术化尝试。\n\n'
            '图片在本机处理，结果保存在 results 文件夹。参考画作来自博物馆开放馆藏。')
    return demo

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=7860)
    parser.add_argument('--browser', action='store_true')
    args = parser.parse_args()
    create_app().queue(default_concurrency_limit=1, max_size=4).launch(
        server_name='127.0.0.1', server_port=args.port, inbrowser=args.browser,
        share=False, max_file_size='20mb', allowed_paths=[str(ROOT / 'results')])

