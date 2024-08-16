import gradio as gr
from PIL import Image
from chatbot_new import chat_gen
from draw_syz import l2s_chain,text2img_chain
# 全局respond，传给图像
# glob_respond = "start"
glob_response = None



def text_echo(message, history):
    # 根据message, history得到回复
    # respond = fn(message, history)
    global glob_response


    glob_response = chat_gen(message,history,)
   # print(type(glob_response))
    respond= ''.join(list(glob_response)[-1])
    return respond

def image_echo():
    # 根据glob_respond得到生成图片
    # picture = fn(glob_respond)
    global glob_response
    concise_prompt = l2s_chain.invoke(glob_response)
    picture = text2img_chain.invoke(concise_prompt)
    return picture

# 主题，JS，CSS设置
theme = gr.themes.Glass(
    primary_hue="amber",
    neutral_hue="amber",
    text_size="lg",
)
js = """
function createGradioAnimation() {
    var container = document.createElement('div');
    container.id = 'gradio-animation';
    container.style.fontSize = '4em';
    container.style.fontWeight = 'bold';
    container.style.textAlign = 'center';
    container.style.marginBottom = '15px';
    container.style.color = 'black';

    var text = 'Welcome to the Three-Body World!';
    for (var i = 0; i < text.length; i++) {
        (function(i){
            setTimeout(function(){
                var letter = document.createElement('span');
                letter.style.opacity = '0';
                letter.style.transition = 'opacity 0.2s';
                letter.innerText = text[i];

                container.appendChild(letter);

                setTimeout(function() {
                    letter.style.opacity = '1';
                }, 50);
            }, i * 250);
        })(i);
    }

    var gradioContainer = document.querySelector('.gradio-container');
    gradioContainer.insertBefore(container, gradioContainer.firstChild);

    return 'Animation created';
}
"""
css = """
.gradio-container {background: url('https://images.pexels.com/photos/3986695/pexels-photo-3986695.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2')}

"""
# 预设组件
chatbot = gr.Chatbot(height=500,bubble_full_width=False)
text_submit = gr.Button(icon="pic/paper_plane.png",value='submit')
retry = gr.Button(icon="pic/update.png",value='retry')
undo = gr.Button(icon="pic/undooutline.png",value='undo')
clear = gr.Button(icon="pic/biggarbagebin.png",value='clear')
# gradio主界面
with gr.Blocks(theme=theme,js=js,css=css) as demo:
    # 主页面
    with gr.Tab('Home Page'):
        with gr.Row():
            with gr.Group():
                gr.Image('pic/647.jpg',show_label=False,show_download_button=False)
                gr.Markdown("<center>记忆是一条早已干涸的河流，<br />只在毫无生气的河床中剩下零落的砾石</center>")
            with gr.Group():
                gr.Image('pic/646.jpg',show_label=False,show_download_button=False)
                gr.Markdown("<center>是对规律的渴望，<br />还是对混沌的屈服</center>")
            with gr.Group():
                gr.Image('pic/643.jpg',show_label=False,show_download_button=False)
                gr.Markdown("<center>弱小和无知不是生存的障碍，<br />傲慢才是</center>")
            with gr.Group():
                gr.Image('pic/644.jpg',show_label=False,show_download_button=False)
                gr.Markdown("<center>给时光以生命，<br />给岁月以文明。</center>")
            with gr.Group():
                gr.Image('pic/645.jpg',show_label=False,show_download_button=False)
                gr.Markdown("<center>我们都是阴沟里的虫子,<br />但总还是得有人仰望星空</center>")
    # 聊天机器人页面
    with gr.Tab('Chatbot'):
        # 展示画
        with gr.Row():
            with gr.Group():
                gr.Image('pic/1.png', show_label=False, show_download_button=False)
            with gr.Group():
                gr.Image('pic/2.png', show_label=False, show_download_button=False)
            with gr.Group():
                gr.Image('pic/3.png', show_label=False, show_download_button=False)
            with gr.Group():
                gr.Image('pic/7.png', show_label=False, show_download_button=False)

        # 参数设置
        # with gr.Accordion(label="Params Details", open=False):
        #     prompt = gr.Textbox("You are helpful AI.", label="System Prompt",info='What kind of chatbot do you want?')
        #     temp = gr.Slider(0, 1, step=0.1, label="Temperature", info='The degree of freedom in which the bot answers')
        with gr.Row():
            # 输入组件
            with gr.Column(scale=7):
                # 聊天机器人
                with gr.Group():
                    chat = gr.ChatInterface(
                        fn=text_echo,
                        chatbot=chatbot,
                        # additional_inputs=[
                        #     temp,
                        #     prompt,
                        # ],
                        retry_btn=retry,
                        undo_btn=undo,
                        clear_btn=clear,
                        submit_btn=text_submit,
                    )

            # 图片输出组件
            with gr.Column(scale=4):
                image = gr.Image(value='pic/5.png',interactive=False, type='pil', label='image', show_download_button=True, show_share_button=True,height=530,width=530)
                pic_button = gr.Button("Generate Image")
                pic_button.click(image_echo,[],[image])
        # 样例
        examples = gr.Examples(
            examples=[
                "Can you share a description of the appearance and demeanor of Ye Wenjie, one of the main character in the book?",
                "How does the book depict the potential impact of alien concat on humen society and civilization?",

            ],
            inputs=[chat.textbox],
            label="Text Input Example"
        )


try:
    demo.launch(debug=True, share=True, show_api=False, server_port=5000, server_name="0.0.0.0")
    demo.close()
except Exception as e:
    demo.close()
    print(e)
    raise e

# demo.queue()
# demo.launch()


