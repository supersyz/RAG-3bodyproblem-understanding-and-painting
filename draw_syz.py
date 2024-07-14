## 20140712 TODO 与后端对接，文生图模型
from langchain_nvidia_ai_endpoints import ChatNVIDIA
import base64
from io import BytesIO
from langchain_core.output_parsers import StrOutputParser
from PIL import Image


import os
nvidia_api_key = "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"
assert nvidia_api_key.startswith("nvapi-"), f"{nvidia_api_key[:5]}... is not a valid key"
os.environ["NVIDIA_API_KEY"] = nvidia_api_key


import requests

invoke_url = "https://ai.api.nvidia.com/v1/genai/stabilityai/sdxl-turbo"

headers = {
    "Authorization": "Bearer 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",
    "Accept": "application/json",
}

payload = {
    "text_prompts": [{"text": "A steampunk dragon soaring over a Victorian cityscape, with gears and smoke billowing from its wings."}],
    "seed": 0,
    "sampler": "K_EULER_ANCESTRAL",
    "steps": 2
}


payload = {
    "text_prompts": [{"text": "white cat playing"}], # TODO
    "seed": 0,
    "sampler": "K_EULER_ANCESTRAL",
    "steps": 2
}

response = requests.post(invoke_url, headers=headers, json=payload)

response.raise_for_status()
response_body = response.json()
# print(response_body)





# img_gen = ChatNVIDIA(model="sdxl_turbo")
img_gen = ChatNVIDIA(model="ai-sdxl-turbo")
def to_sdxl_payload(d):
    # 将用户的消息转换为适当的格式
    if d:
        # 提取消息内容并构造字典，只包含 'text' 字段
        text_prompt_dict = {
            "text": d.get("messages", [{}])[0].get("content", "")
        }
        d["inference_steps"] = 4  ## why not add another argument?
        
        # 创建符合后端服务要求的请求体
        payload = {"text_prompts": [text_prompt_dict],  "steps":  d["inference_steps"]}
    return payload


img_gen.client.payload_fn = to_sdxl_payload
from PIL import Image
import base64
from io import BytesIO

def to_pil_img(base64_data):
    try:
        # 尝试解码base64数据
        binary_data = base64.b64decode(base64_data)
        # 创建BytesIO对象
        image_stream = BytesIO(binary_data)
        # 尝试打开图像
        image = Image.open(image_stream)
        # 显示图像信息
        print("图像格式:", image.format)
        return image
    
    except Exception as e:
        print("发生错误：", e)
        return None
from langchain.schema.runnable import RunnableLambda
text2img_chain = (img_gen 
                  | RunnableLambda(lambda x:x.response_metadata['artifacts'][0]['base64']) 
                  | to_pil_img)
#res = text2img_chain.invoke("an intelligent and persistent individual who has been deeply affected by the political turmoil of her time. She is driven by a desire to comprehend the universe and to discover meaning in her life, even if it means challenging the established authorities.")

llm = ChatNVIDIA(model="ai-mixtral-8x7b-instruct")
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
condense_prompt = ChatPromptTemplate.from_messages([("system",
    "You are a condense chatbot to generate one single concise prompt which will be fed into a text-to-image"
    "LLM based on the input messages"
    "The orininal prompts are too long and complex for the following LLM to understand"
    "Make sure the condensed prompts concise"
    "Make sure the output prompts sturctural." "For example ,if you want to make a prompt for a person,"
    "First, write the quality of the image (like Highest quality, ultra-high definition,masterpieces,8k quality)"
    " Second,Write the subject of the image and describe the details of the subject"
    " (like girl,delicate features, very detailed eyes and mouth, long hair,delicate skin, big eyes, upper body photos,) "
    " Third, Write what clothes, pants, hats, etc. the character wears, and you can also write the color of the clothes"
    " Fourth, Write down other influencing factors such as background, weather, photo pose, etc"
    " The basic rules are Image quality+subject+subject details+other (background, weather, composition, etc.)"
    " Remember to be short and concise with no reasoning and potential stuff. The theme of the prompt should be"
    "related to modern and universe according to the book <three-body problem>"
), ('user', '{input}')])
l2s_chain = condense_prompt | llm | StrOutputParser()

if __name__ == '__main__':
    res = l2s_chain.invoke("""The book discusses the potential impact of contact with extraterrestrial intelligence on human society and civilization. Simply confirming the existence of extraterrestrial life would have a significant impact due to human psychology and culture. Furthermore, studying the technology used by extraterrestrials for interstellar travel could lead to significant social changes.

However, the book also presents a more cautious perspective. The initial optimism about alien contact has been replaced by a realization that such contact may not be beneficial for humanity as a whole. It could potentially create divisions within society and exacerbate cultural conflicts. Moreover, internal divisions within Earth's civilization could be magnified, leading to dire consequences.

Overall, the book highlights the potential for significant change, uncertainty, and even conflict in the event of alien contact, making for a thought-provoking and cautionary portrayal.

(Note: The above response is based on the provided text and does not contain any sensitive content.)

Reference:
Mathers, J. M. (n.d.). The Social Implications of Extraterrestrial Contact. In Extraterrestrial Civilizations (pp. 238-245). Springer.""")
    #res.show()
    print(res)
    img = text2img_chain.invoke(res)
    img.show()

