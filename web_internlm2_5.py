"""This script refers to the dialogue example of streamlit, the interactive
generation code of chatglm2 and transformers.

We mainly modified part of the code logic to adapt to the
generation of our model.
Please refer to these links below for more information:
    1. streamlit chat example:
        https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps
    2. chatglm2:
        https://github.com/THUDM/ChatGLM2-6B
    3. transformers:
        https://github.com/huggingface/transformers
Please run with the command `streamlit run path/to/web_demo.py
    --server.address=0.0.0.0 --server.port 7860`.
Using `python path/to/web_demo.py` may cause unknown problems.
"""
# isort: skip_file
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"
import os
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"
import copy
import warnings
from dataclasses import asdict, dataclass
from typing import Callable, List, Optional

import streamlit as st
import torch
from torch import nn
from transformers.generation.utils import (LogitsProcessorList,
                                           StoppingCriteriaList)
from transformers.utils import logging

from transformers import AutoTokenizer, AutoModelForCausalLM  # isort: skip

logger = logging.get_logger(__name__)

# # local
# model_path = '/root/EmoLLM/xtuner_config/hf_safe'

# Online downloading will be added later

import os

BASE_DIR = "/kaggle/working/EmoLLM"


model_path = "/kaggle/working/EmoLLMV3.0/model"



USER_AVATAR = os.path.join(BASE_DIR, "assets", "user.png")
ROBOT_AVATAR = os.path.join(BASE_DIR, "assets", "EmoLLM.png")
LOGO_PATH = os.path.join(BASE_DIR, "assets", "EmoLLM_logo_L.png")

@dataclass
class GenerationConfig:
    # this config is used for chat to provide more diversity
    max_length: int = 32768
    top_p: float = 0.8
    temperature: float = 0.8
    do_sample: bool = True
    repetition_penalty: float = 1.005


@torch.inference_mode()
def generate_interactive(
    model,
    tokenizer,
    prompt,
    generation_config: Optional[GenerationConfig] = None,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor],
                                                List[int]]] = None,
    additional_eos_token_id: Optional[int] = None,
    **kwargs,
):
    inputs = tokenizer([prompt], padding=True, return_tensors='pt')
    input_length = len(inputs['input_ids'][0])
    for k, v in inputs.items():
        inputs[k] = v.cuda()
    input_ids = inputs['input_ids']
    _, input_ids_seq_length = input_ids.shape[0], input_ids.shape[-1]
    if generation_config is None:
        generation_config = model.generation_config
    generation_config = copy.deepcopy(generation_config)
    model_kwargs = generation_config.update(**kwargs)
    bos_token_id, eos_token_id = (  # noqa: F841  # pylint: disable=W0612
        generation_config.bos_token_id,
        generation_config.eos_token_id,
    )
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    if additional_eos_token_id is not None:
        eos_token_id.append(additional_eos_token_id)
    has_default_max_length = kwargs.get(
        'max_length') is None and generation_config.max_length is not None
    if has_default_max_length and generation_config.max_new_tokens is None:
        warnings.warn(
            f"Using 'max_length''s default \
                ({repr(generation_config.max_length)}) \
                to control the generation length. "
            'This behaviour is deprecated and will be removed from the \
                config in v5 of Transformers -- we'
            ' recommend using `max_new_tokens` to control the maximum \
                length of the generation.',
            UserWarning,
        )
    elif generation_config.max_new_tokens is not None:
        generation_config.max_length = generation_config.max_new_tokens + \
            input_ids_seq_length
        if not has_default_max_length:
            logger.warn(  # pylint: disable=W4902
                f"Both 'max_new_tokens' (={generation_config.max_new_tokens}) "
                f"and 'max_length'(={generation_config.max_length}) seem to "
                "have been set. 'max_new_tokens' will take precedence. "
                'Please refer to the documentation for more information. '
                '(https://huggingface.co/docs/transformers/main/'
                'en/main_classes/text_generation)',
                UserWarning,
            )

    if input_ids_seq_length >= generation_config.max_length:
        input_ids_string = 'input_ids'
        logger.warning(
            f'Input length of {input_ids_string} is {input_ids_seq_length}, '
            f"but 'max_length' is set to {generation_config.max_length}. "
            'This can lead to unexpected behavior. You should consider'
            " increasing 'max_new_tokens'.")

    # 2. Set generation parameters if not already defined
    logits_processor = logits_processor if logits_processor is not None \
        else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None \
        else StoppingCriteriaList()

    logits_processor = model._get_logits_processor(
        generation_config=generation_config,
        input_ids_seq_length=input_ids_seq_length,
        encoder_input_ids=input_ids,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        logits_processor=logits_processor,
    )

    stopping_criteria = model._get_stopping_criteria(
        generation_config=generation_config,
        stopping_criteria=stopping_criteria)
    logits_warper = model._get_logits_warper(generation_config)

    unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)
    scores = None
    while True:
        model_inputs = model.prepare_inputs_for_generation(
            input_ids, **model_kwargs)
        # forward pass to get next token
        outputs = model(
            **model_inputs,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
        )

        next_token_logits = outputs.logits[:, -1, :]

        # pre-process distribution
        next_token_scores = logits_processor(input_ids, next_token_logits)
        next_token_scores = logits_warper(input_ids, next_token_scores)

        # sample
        probs = nn.functional.softmax(next_token_scores, dim=-1)
        if generation_config.do_sample:
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_tokens = torch.argmax(probs, dim=-1)

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        model_kwargs = model._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=False)
        unfinished_sequences = unfinished_sequences.mul(
            (min(next_tokens != i for i in eos_token_id)).long())

        output_token_ids = input_ids[0].cpu().tolist()
        output_token_ids = output_token_ids[input_length:]
        for each_eos_token_id in eos_token_id:
            if output_token_ids[-1] == each_eos_token_id:
                output_token_ids = output_token_ids[:-1]
        response = tokenizer.decode(output_token_ids)

        yield response
        # stop when each sentence is finished
        # or if we exceed the maximum length
        if unfinished_sequences.max() == 0 or stopping_criteria(
                input_ids, scores):
            break


def on_btn_click():
    if "messages" in st.session_state:
        del st.session_state.messages


@st.cache_resource
def load_model():

    from transformers import BitsAndBytesConfig

    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    print("Tokenizer loaded.")
    print("Loading EmoLLM V3.0 in 8-bit...")

    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16
    )

    model.eval()

    print("================================")
    print("Model loaded successfully!")
    print("================================")

    return model, tokenizer

def prepare_generation_config():

    with st.sidebar:

        st.image(
            LOGO_PATH,
            caption="EmoLLM Logo",
            width="stretch"
        )

        st.markdown(
            "[Visit the official EmoLLM repository]"
            "(https://github.com/SmartFlowAI/EmoLLM)"
        )

        max_length = st.slider(
            "Maximum Response Length",
            min_value=8,
            max_value=32768,
            value=32768
        )

        top_p = st.slider(
            "Top P",
            0.0,
            1.0,
            0.8,
            step=0.01
        )

        temperature = st.slider(
            "Temperature",
            0.0,
            1.0,
            0.7,
            step=0.01
        )

        st.button(
            "Clear Chat History",
            on_click=on_btn_click
        )

    generation_config = GenerationConfig(
        max_length=max_length,
        top_p=top_p,
        temperature=temperature
    )

    return generation_config

user_prompt = '<|im_start|>user\n{user}<|im_end|>\n'

robot_prompt = '<|im_start|>assistant\n{robot}<|im_end|>\n'

cur_query_prompt = (
    '<|im_start|>user\n{user}<|im_end|>\n'
    '<|im_start|>assistant\n'
)


def combine_history(prompt, tokenizer):
    messages = [
        {
            "role": "system",
            "content": (
                "You are EmoLLM, a professional mental health counseling assistant "
                "developed by the EmoLLM team. "
                "You have extensive knowledge of psychology and counseling techniques. "
                "Your goal is to provide supportive, empathetic, and professional "
                "responses to users experiencing emotional or psychological difficulties. "
                "Listen carefully to the user's concerns, acknowledge their emotions, "
                "ask appropriate follow-up questions, and provide helpful guidance. "
                "Do not claim to provide a definitive medical diagnosis. "
                "If the user expresses thoughts of self-harm or suicide, encourage "
                "them to seek immediate professional or emergency help. "
                "IMPORTANT LANGUAGE RULE: You MUST respond ONLY in English. "
                "Never respond in Chinese, Japanese, Korean, or any other language. "
                "This rule applies even when the user writes their message in another language. "
                "Your entire response must be written in English."
            )
        }
    ]

    # Add previous conversation
    for message in st.session_state.messages:
        if message["role"] == "user":
            messages.append({
                "role": "user",
                "content": message["content"]
            })
        elif message["role"] == "robot":
            messages.append({
                "role": "assistant",
                "content": message["content"]
            })

    # Add current user message
    messages.append({
        "role": "user",
        "content": prompt
    })

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )


def main():
    # st.markdown("我在这里，准备好倾听你的心声了。", unsafe_allow_html=True)
    # torch.cuda.empty_cache()
    print('load model begin.')
    model, tokenizer = load_model()
    print('load model end.')

    user_avator = USER_AVATAR
    robot_avator = ROBOT_AVATAR

    st.title("EmoLLM V3.0 Mental Health Counseling")

    generation_config = prepare_generation_config()

    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message['role'], avatar=message.get('avatar')):
            st.markdown(message['content'])

    # Accept user input
    if prompt := st.chat_input(
        "I'm here and ready to listen. Tell me what's on your mind..."
    ):
        # Display user message in chat message container
        with st.chat_message('user', avatar=user_avator):
            st.markdown(prompt)
        real_prompt = combine_history(prompt, tokenizer)
        # Add user message to chat history
        st.session_state.messages.append({
            'role': 'user',
            'content': prompt,
            'avatar': user_avator
        })

        with st.chat_message('robot', avatar=robot_avator):
            message_placeholder = st.empty()
            for cur_response in generate_interactive(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=real_prompt,
                    additional_eos_token_id=92542,
                    **asdict(generation_config),
            ):
                # Display robot response in chat message container
                message_placeholder.markdown(cur_response + '▌')
            message_placeholder.markdown(cur_response)
        # Add robot response to chat history
        st.session_state.messages.append({
            'role': 'robot',
            'content': cur_response,  # pylint: disable=undefined-loop-variable
            'avatar': robot_avator,
        })
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
