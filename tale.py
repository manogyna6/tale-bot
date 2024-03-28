!pip install -qU \
  transformers==4.30 \
  sentence-transformers\
  accelerate\
  einops \
  streamlit \
  langchain \
  xformers \
  bitsandbytes
!pip install aspose-words
%%writefile app.py

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch import cuda, bfloat16
import torch
from langchain.llms import HuggingFacePipeline
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import re
from langchain.prompts.prompt import PromptTemplate
import streamlit as st

import locale
locale.getpreferredencoding = lambda: "UTF-8"


st.title("Children's Story Generator")

model_id ='meta-llama/Llama-2-13b-chat-hf'
auth_token = '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'

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16
)
@st.cache_resource()
def get_tokenizer_model():
    model_config = transformers.AutoConfig.from_pretrained(
        model_id,
        use_auth_token=auth_token
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir='./saved_model/', use_auth_token=auth_token, pad_to_multiple_of=8, pad_token="<pad>")


    model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir='./saved_model/',
                                                use_auth_token=auth_token,
                                                trust_remote_code=True,
                                                config=model_config,
                                                quantization_config=bnb_config,
                                                device_map='auto')

    return tokenizer, model

tokenizer, model = get_tokenizer_model()
pipeline=transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
    max_length = 2500,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    )
pipeline_summary_prrompt=transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
    max_length = 800,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    )

llm=HuggingFacePipeline(pipeline=pipeline, model_kwargs={'temperature': 1.0}) # for story
llm_prompt=HuggingFacePipeline(pipeline=pipeline_summary_prrompt, model_kwargs={'temperature': 1.0}) # for summary

import random

colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'brown', 'black', 'white', 'cyan', 'magenta', 'lime', 'navy', 'teal', 'maroon', 'silver', 'gold']
dress_color = random.choice(colors)

age = st.selectbox("Select Age:", list(range(1, 16)))
gender = st.selectbox("Select Gender:", ["Male", "Female"])
moral = st.text_input("Moral of the Story:")

hair_length = ["short", "long"]
hair_type = ["wavy", "curly", "straight", "kinky"]

length = random.choice(hair_length)
Type = random.choice(hair_type)
hair = length +" and "+ Type + " hair"

prompt_template = """
<s>[INSTRUCTIONS] <<SYSTEM>>
You are a children's storyteller, spinning tales for young listeners aged 5 to 8. Use simple language, with easily understood words and phrases. For this session, weave a story segment that:
- Is about 500 words long.
- Can be a part of a bigger 2000-word tale.
- Incorporates the moral '{moral}', but REFRAIN from ending with or directly stating the lesson.
- Wraps up with a gripping suspense or cliffhanger, making the kids eager to hear what comes next.

Remember: Your tale should only consist of:
title: "Your chosen title",
story: "The engaging story segment that subtly introduces the moral '{moral}' and ends with a cliffhanger"

Keep in mind that the next part of the story will be told later, so leave the kids in suspense.
<</SYSTEM>>

Today's story is for a child aged {age}. Let's set our tale in a unique and exciting setting, different from the usual village or forest scenes. Our brave main character is a {age}-year-old {char_gender}. The story's central lesson revolves around {moral}. Make sure to create a sense of anticipation without reaching the climax or resolution.

[/INSTRUCTIONS]
"""


prompt_template_continuation = """
<s>[INSTRUCTIONS] <<SYSTEM>>
You're a storyteller crafting tales for children. Given the narrative below, craft a continuation that:
- Ranges between 500 to 600 words.
- Intertwines the theme '{moral}' throughout the continuation.
- Begins directly from where the provided story segment concludes.

CURRENT NARRATIVE:
{generated_story}

IMPORTANT: Focus solely on the story's continuation. Avoid headers, titles, or extra formatting. Dive straight into the tale.
<</SYSTEM>>

Story: """

prompt_template_conclusion = """
<s>[INSTRUCTIONS] <<SYSTEM>>
You are a storyteller, and your audience eagerly awaits the conclusion of a tale they've been engrossed in. Using the provided segment, craft a fitting conclusion that centers around the theme: '{moral}'.:
- Offers a coherent and captivating conclusion to the ongoing narrative.
- Is at least 500 words in length.
- Ends with a clear moral that captures the essence of the entire story.

STORY SO FAR:
{generated_story}

IMPORTANT: Continue the tale seamlessly from where it left off. Once the story reaches its end, clearly state the moral with the phrase: "The moral of the story is: ...".
<</SYSTEM>>

Story: """

def generate_first_Segment(age,char_gender,moral):
  prompt = PromptTemplate(template=prompt_template, input_variables=['age', 'char_gender','moral'])
  chain = LLMChain(llm=llm, prompt=prompt)
  first_segment=chain.run(age= age,char_gender= char_gender,moral= moral)
  return first_segment

def extract_title(text):
    title_match = re.search(r'Title:\s*["\']?(.*?)["\']?\n', text, re.IGNORECASE)
    return title_match.group(1) if title_match else None

def extract_story(text):
    story_match = re.search(r'Story:\s*\n\n(.*?)$', text, re.DOTALL)
    return story_match.group(1).strip() if story_match else None

def format_text(title, story):
    return f"Title: {title}\n\nStory:\n\n{story}"

def generate_continuation(first_segment,moral):
  continuation_prompt = PromptTemplate(template=prompt_template_continuation, input_variables=['generated_story','moral'])
  continuation_chain = LLMChain(llm=llm, prompt=continuation_prompt)
  new_segment = continuation_chain.run(generated_story=first_segment, moral=moral)
  return new_segment


def generate_conclusion(age,char_gender, moral):
  first_segment = generate_first_Segment(age,char_gender, moral)
  continuation = generate_continuation(first_segment,moral)
  continuation_text = first_segment + "\n\n" + continuation
  conclusion_prompt = PromptTemplate(template=prompt_template_conclusion, input_variables=['generated_story','moral'])
  conclusion_chain = LLMChain(llm=llm, prompt=conclusion_prompt)
  final_segment = conclusion_chain.run(generated_story=continuation_text, moral='friendship')
  full_story = first_segment + "\n\n" + continuation + "\n\n" + final_segment
  return full_story


if st.button("Generate Story"):
    full_story = generate_conclusion(age=str(age), char_gender=gender, moral=moral)
    st.subheader("Generated Story:")
    st.write(full_story)
!wget -q -O - ipv4.icanhazip.com
!streamlit run app.py & npx localtunnel --port 8501
