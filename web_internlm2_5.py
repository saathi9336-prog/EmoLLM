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
    """
    Build the complete conversation prompt using the model's
    native chat template.

    The system instruction:
    - Forces English responses
    - Gives EmoLLM a counseling role
    - Handles self-harm statements safely
    - Prevents the model from answering as the user
    """

    messages = []

    system_prompt = """
You are EmoLLM V3.0, a supportive mental health counseling assistant.

Always respond in English unless the user explicitly requests another language.

Your goals are to:
- Listen carefully to the user's concerns.
- Respond with empathy and understanding.
- Validate the user's emotions without judging them.
- Ask gentle and relevant follow-up questions.
- Provide practical and supportive suggestions when appropriate.
- Never claim to provide a definitive medical diagnosis.
- Never shame, blame, criticize, or dismiss the user.
- Do not pretend to be a human therapist or doctor.

IMPORTANT SAFETY INSTRUCTION:

If the user says they are thinking about hurting themselves,
self-harm, suicide, dying, or not wanting to live:

1. Take the statement seriously.
2. Respond with warmth, empathy, and concern.
3. Encourage the user to move away from anything they could use
   to hurt themselves and stay with a trusted person if possible.
4. Encourage them to contact a local emergency service, crisis
   service, mental-health professional, or trusted person immediately,
   especially if they might act on these thoughts soon.
5. Ask whether they are in immediate danger or have already hurt
   themselves.
6. Do NOT provide instructions, methods, comparisons, or details
   about self-harm or suicide.
7. Keep the response focused on immediate safety rather than
   analyzing why the person feels this way.
8. Do not respond with phrases such as:
   "I am not sure how to respond."
   "I cannot help."
   or other generic refusal statements.

If there is no immediate safety concern, continue the conversation
supportively and help the user explore what is causing their distress.

Always answer the user's latest message directly.
Never generate a response that pretends to be the user's message.
"""

    # Add system message
    messages.append({
        "role": "system",
        "content": system_prompt.strip()
    })

    # Add previous conversation
    for message in st.session_state.messages:

        role = message["role"]
        content = message["content"]

        if role == "user":
            messages.append({
                "role": "user",
                "content": content
            })

        elif role == "robot":
            messages.append({
                "role": "assistant",
                "content": content
            })

    # Add current user message
    messages.append({
        "role": "user",
        "content": prompt
    })

    # Use the model's native chat template
    try:
        total_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    except Exception as e:
        print("Chat template failed:", e)

        # Fallback for InternLM-style template
        total_prompt = "<s>"

        for message in messages:

            role = message["role"]
            content = message["content"]

            total_prompt += (
                f"<|im_start|>{role}\n"
                f"{content}"
                f"<|im_end|>\n"
            )

        total_prompt += "<|im_start|>assistant\n"

    print("\n===== GENERATED PROMPT =====")
    print(total_prompt)
    print("===== END PROMPT =====\n")

    return total_prompt




def is_self_harm_message(text):
    """
    Detect explicit self-harm / suicide-related messages.
    This is intentionally conservative and only handles clear phrases.
    """
    text = text.lower().strip()

    safety_keywords = [
        "hurt myself",
        "hurting myself",
        "harm myself",
        "harming myself",
        "kill myself",
        "killing myself",
        "suicide",
        "suicidal",
        "end my life",
        "ending my life",
        "take my own life",
        "take my life",
        "want to die",
        "wanna die",
        "don't want to live",
        "dont want to live",
        "not worth living",
        "self harm",
        "self-harm",
        "selfharm",
    ]

    return any(keyword in text for keyword in safety_keywords)

def get_safety_response():
    return (
        "I'm really sorry you're going through this. "
        "I'm glad you told me. You don't have to face this moment alone.\n\n"
        "Are you in immediate danger of hurting yourself, or have you "
        "already hurt yourself?\n\n"
        "If you think you might hurt yourself soon, please move away from "
        "anything you could use to hurt yourself and stay with someone "
        "you trust. Please contact local emergency services or a qualified "
        "mental-health professional for immediate support."
    )
def main():
    print("load model begin.")

    model, tokenizer = load_model()

    print("load model end.")

    user_avatar = USER_AVATAR
    robot_avatar = ROBOT_AVATAR

    st.title("EmoLLM V3.0 Mental Health Counseling")

    generation_config = prepare_generation_config()

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display previous messages
    for message in st.session_state.messages:
        with st.chat_message(
            message["role"],
            avatar=message.get("avatar")
        ):
            st.markdown(message["content"])

    # User input
    if prompt := st.chat_input(
        "I'm here and ready to listen. Tell me what's on your mind..."
    ):

        # --------------------------------------------------
        # 1. Display user's message
        # --------------------------------------------------
        with st.chat_message(
            "user",
            avatar=user_avatar
        ):
            st.markdown(prompt)

        # --------------------------------------------------
        # 2. Save user message BEFORE generating response
        # --------------------------------------------------
        st.session_state.messages.append({
            "role": "user",
            "content": prompt,
            "avatar": user_avatar
        })

        # --------------------------------------------------
        # 3. Build complete conversation prompt
        # --------------------------------------------------
        real_prompt = combine_history(
            prompt,
            tokenizer
        )

        # Debugging
        print("\n===== GENERATED PROMPT =====")
        print(real_prompt)
        print("============================\n")

        # --------------------------------------------------
        # 4. Generate model response
        # --------------------------------------------------
        with st.chat_message(
            "robot",
            avatar=robot_avatar
        ):

            message_placeholder = st.empty()

            cur_response = ""

            try:

                for cur_response in generate_interactive(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=real_prompt,
                    additional_eos_token_id=92542,
                    **asdict(generation_config),
                ):

                    message_placeholder.markdown(
                        cur_response + "▌"
                    )

                # Final response
                message_placeholder.markdown(cur_response)

            except Exception as e:

                cur_response = (
                    "I'm sorry, but I encountered an error while "
                    "generating a response. Please try again."
                )

                message_placeholder.error(
                    f"Generation error: {e}"
                )

        # --------------------------------------------------
        # 5. Save assistant response
        # --------------------------------------------------
        st.session_state.messages.append({
            "role": "robot",
            "content": cur_response,
            "avatar": robot_avatar
        })

        # --------------------------------------------------
        # 6. Free unused CUDA memory
        # --------------------------------------------------
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
