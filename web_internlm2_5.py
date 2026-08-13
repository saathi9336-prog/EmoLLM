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
    prefix_allowed_tokens_fn: Optional[
        Callable[[int, torch.Tensor], List[int]]
    ] = None,
    additional_eos_token_id: Optional[int] = None,
    **kwargs,
):

    # --------------------------------------------------
    # Tokenize prompt
    # --------------------------------------------------
    inputs = tokenizer(
        [prompt],
        padding=True,
        return_tensors="pt"
    )

    input_length = len(inputs["input_ids"][0])

    # Move tensors to GPU
    for k, v in inputs.items():
        inputs[k] = v.cuda()

    input_ids = inputs["input_ids"]

    _, input_ids_seq_length = input_ids.shape

    # --------------------------------------------------
    # Generation configuration
    # --------------------------------------------------
    if generation_config is None:
        generation_config = model.generation_config

    generation_config = copy.deepcopy(generation_config)

    model_kwargs = generation_config.update(**kwargs)

    bos_token_id = generation_config.bos_token_id
    eos_token_id = generation_config.eos_token_id

    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]

    if additional_eos_token_id is not None:
        eos_token_id.append(additional_eos_token_id)

    # --------------------------------------------------
    # Length configuration
    # --------------------------------------------------
    has_default_max_length = (
        kwargs.get("max_length") is None
        and generation_config.max_length is not None
    )

    if (
        has_default_max_length
        and generation_config.max_new_tokens is None
    ):
        warnings.warn(
            f"Using 'max_length' default "
            f"({repr(generation_config.max_length)}) "
            "to control the generation length. "
            "Consider using 'max_new_tokens'.",
            UserWarning,
        )

    elif generation_config.max_new_tokens is not None:

        generation_config.max_length = (
            generation_config.max_new_tokens
            + input_ids_seq_length
        )

    # --------------------------------------------------
    # Check input length
    # --------------------------------------------------
    if input_ids_seq_length >= generation_config.max_length:

        logger.warning(
            f"Input length is {input_ids_seq_length}, "
            f"but max_length is {generation_config.max_length}. "
            "Consider increasing max_new_tokens."
        )

    # --------------------------------------------------
    # Logits processor
    # --------------------------------------------------
    logits_processor = (
        logits_processor
        if logits_processor is not None
        else LogitsProcessorList()
    )

    stopping_criteria = (
        stopping_criteria
        if stopping_criteria is not None
        else StoppingCriteriaList()
    )

    logits_processor = model._get_logits_processor(
        generation_config=generation_config,
        input_ids_seq_length=input_ids_seq_length,
        encoder_input_ids=input_ids,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        logits_processor=logits_processor,
    )

    stopping_criteria = model._get_stopping_criteria(
        generation_config=generation_config,
        stopping_criteria=stopping_criteria,
    )

    logits_warper = model._get_logits_warper(
        generation_config
    )

    # --------------------------------------------------
    # Generation state
    # --------------------------------------------------
    unfinished_sequences = (
        input_ids.new(input_ids.shape[0]).fill_(1)
    )

    scores = None

    # --------------------------------------------------
    # Generation loop
    # --------------------------------------------------
    while True:

        model_inputs = model.prepare_inputs_for_generation(
            input_ids,
            **model_kwargs
        )

        # Forward pass
        outputs = model(
            **model_inputs,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
        )

        next_token_logits = outputs.logits[:, -1, :]

        # Process logits
        next_token_scores = logits_processor(
            input_ids,
            next_token_logits
        )

        next_token_scores = logits_warper(
            input_ids,
            next_token_scores
        )

        # --------------------------------------------------
        # Sample next token
        # --------------------------------------------------
        probs = nn.functional.softmax(
            next_token_scores,
            dim=-1
        )

        if generation_config.do_sample:

            next_tokens = torch.multinomial(
                probs,
                num_samples=1
            ).squeeze(1)

        else:

            next_tokens = torch.argmax(
                probs,
                dim=-1
            )

        # --------------------------------------------------
        # Append token
        # --------------------------------------------------
        input_ids = torch.cat(
            [
                input_ids,
                next_tokens[:, None]
            ],
            dim=-1
        )

        model_kwargs = model._update_model_kwargs_for_generation(
            outputs,
            model_kwargs,
            is_encoder_decoder=False,
        )

        # --------------------------------------------------
        # Check EOS
        # --------------------------------------------------
        for eos_id in eos_token_id:

            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.ne(eos_id).long()
            )

        # --------------------------------------------------
        # Decode ONLY newly generated tokens
        # --------------------------------------------------
        output_token_ids = input_ids[0].cpu().tolist()

        output_token_ids = output_token_ids[input_length:]

        # Remove EOS tokens
        while (
            output_token_ids
            and output_token_ids[-1] in eos_token_id
        ):
            output_token_ids.pop()

        response = tokenizer.decode(
            output_token_ids,
            skip_special_tokens=True
        )

        # --------------------------------------------------
        # Yield response to Streamlit
        # --------------------------------------------------
        yield response

        # --------------------------------------------------
        # Stop generation
        # --------------------------------------------------
        if unfinished_sequences.max() == 0:
            break

        if stopping_criteria(
            input_ids,
            scores
        ):
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

    # --------------------------------------------------
    # System instruction
    # --------------------------------------------------
    system_prompt = """
You are EmoLLM, a professional and supportive mental health
counseling assistant.

Always respond in English.

Your role is to:

- listen carefully to the user's concerns
- acknowledge and validate their emotions
- respond with empathy and warmth
- ask relevant follow-up questions when appropriate
- provide practical and supportive suggestions
- never judge, shame, or criticize the user
- never claim to provide a definitive medical diagnosis

IMPORTANT SAFETY RULE:

If the user expresses thoughts about suicide, self-harm,
hurting themselves, wanting to die, or not wanting to live:

- Take the statement seriously.
- Respond with empathy and concern.
- Do not ask the user to justify why they feel this way.
- Do not provide instructions or methods for self-harm.
- Encourage the person to move away from anything they could
  use to hurt themselves.
- Encourage them to stay with a trusted person.
- Encourage them to contact a mental-health professional
  or emergency service if they may act on these thoughts.
- Ask whether they are in immediate danger or have already
  hurt themselves.

For ordinary emotional concerns, provide supportive,
empathetic and practical responses.

Never repeat the user's message as your response.

Always respond naturally as the assistant.
"""

    # --------------------------------------------------
    # Conversation messages
    # --------------------------------------------------
    messages = []

    messages.append({
        "role": "system",
        "content": system_prompt.strip()
    })

    # --------------------------------------------------
    # Previous conversation
    # --------------------------------------------------
    if "messages" in st.session_state:

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

    # --------------------------------------------------
    # Current user message
    # --------------------------------------------------
    messages.append({
        "role": "user",
        "content": prompt
    })

    # --------------------------------------------------
    # Use InternLM's chat template
    # --------------------------------------------------
    try:

        total_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    except Exception as e:

        print(
            "WARNING: tokenizer chat template failed:"
        )

        print(e)

        # --------------------------------------------------
        # Fallback InternLM format
        # --------------------------------------------------
        total_prompt = (
            "<s>"
            "<|im_start|>system\n"
            + system_prompt.strip()
            + "<|im_end|>\n"
        )

        for message in messages[1:]:

            if message["role"] == "user":

                total_prompt += (
                    "<|im_start|>user\n"
                    + message["content"]
                    + "<|im_end|>\n"
                )

            elif message["role"] == "assistant":

                total_prompt += (
                    "<|im_start|>assistant\n"
                    + message["content"]
                    + "<|im_end|>\n"
                )

        total_prompt += (
            "<|im_start|>assistant\n"
        )

    # --------------------------------------------------
    # Debug output
    # --------------------------------------------------
    print()
    print("===== GENERATED PROMPT =====")
    print(total_prompt)
    print("============================")
    print()

    return total_prompt
def is_self_harm_message(text):
    text = text.lower()

    self_harm_keywords = [
        "hurt myself",
        "hurting myself",
        "harm myself",
        "harming myself",
        "kill myself",
        "killing myself",
        "suicide",
        "suicidal",
        "end my life",
        "take my own life",
        "want to die",
        "don't want to live",
        "dont want to live"
    ]

    return any(keyword in text for keyword in self_harm_keywords)

def get_safety_response():
    return (
        "I'm really sorry that you're going through this. "
        "I'm glad you told me. Your safety is important right now.\n\n"
        "If you feel you might hurt yourself, please move away from "
        "anything you could use to hurt yourself and stay with someone "
        "you trust. Please contact a mental-health professional or "
        "emergency service in your area if you might act on these thoughts.\n\n"
        "If possible, tell someone you trust exactly what you're "
        "experiencing so they can stay with you.\n\n"
        "Are you in immediate danger of hurting yourself right now, "
        "or have you already hurt yourself?"
    )
def main():

    print("load model begin.")

    # Load model and tokenizer
    model, tokenizer = load_model()

    print("load model end.")

    # -----------------------------------------
    # Avatar paths
    # -----------------------------------------
    user_avator = USER_AVATAR
    robot_avator = ROBOT_AVATAR

    # -----------------------------------------
    # Page title
    # -----------------------------------------
    st.title("EmoLLM V3.0 Mental Health Counseling")

    # -----------------------------------------
    # Generation configuration / sidebar
    # -----------------------------------------
    generation_config = prepare_generation_config()

    # -----------------------------------------
    # Initialize chat history
    # -----------------------------------------
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # -----------------------------------------
    # Display previous conversation
    # -----------------------------------------
    for message in st.session_state.messages:

        with st.chat_message(
            message["role"],
            avatar=message.get("avatar")
        ):
            st.markdown(message["content"])

    # -----------------------------------------
    # User input
    # -----------------------------------------
    if prompt := st.chat_input(
        "I'm here and ready to listen. Tell me what's on your mind..."
    ):

        # -------------------------------------
        # Display user message
        # -------------------------------------
        with st.chat_message(
            "user",
            avatar=user_avator
        ):
            st.markdown(prompt)

        # -------------------------------------
        # Save user message
        # -------------------------------------
        st.session_state.messages.append({
            "role": "user",
            "content": prompt,
            "avatar": user_avator
        })

        # -------------------------------------
        # SELF-HARM SAFETY CHECK
        # -------------------------------------
        if is_self_harm_message(prompt):

            safety_response = get_safety_response()

            # Display safety response
            with st.chat_message(
                "robot",
                avatar=robot_avator
            ):
                st.markdown(safety_response)

            # Save safety response
            st.session_state.messages.append({
                "role": "robot",
                "content": safety_response,
                "avatar": robot_avator
            })

        # -------------------------------------
        # NORMAL MODEL GENERATION
        # -------------------------------------
        else:

            # Build complete conversation prompt
            real_prompt = combine_history(
                prompt,
                tokenizer
            )

            print("\n===== GENERATED PROMPT =====")
            print(real_prompt)
            print("============================\n")

            # ---------------------------------
            # Generate model response
            # ---------------------------------
            with st.chat_message(
                "robot",
                avatar=robot_avator
            ):

                message_placeholder = st.empty()

                cur_response = ""

                for response in generate_interactive(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=real_prompt,
                    additional_eos_token_id=92542,
                    **asdict(generation_config),
                ):

                    cur_response = response

                    message_placeholder.markdown(
                        cur_response + "▌"
                    )

                # ---------------------------------
                # Final response
                # ---------------------------------
                if cur_response.strip():

                    message_placeholder.markdown(
                        cur_response
                    )

                else:

                    cur_response = (
                        "I'm sorry, I wasn't able to generate "
                        "a response. Could you please try again?"
                    )

                    message_placeholder.markdown(
                        cur_response
                    )

            # ---------------------------------
            # Save model response
            # ---------------------------------
            st.session_state.messages.append({
                "role": "robot",
                "content": cur_response,
                "avatar": robot_avator
            })

        # -------------------------------------
        # Clear unused CUDA memory
        # -------------------------------------
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
