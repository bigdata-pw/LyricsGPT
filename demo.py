import gradio as gr
from transformers import GPT2TokenizerFast, GPT2LMHeadModel
from typing import List, Optional, Union

tokenizer: GPT2TokenizerFast = GPT2TokenizerFast.from_pretrained(
    "bigdata-pw/lyrics-gpt"
)
model: GPT2LMHeadModel = GPT2LMHeadModel.from_pretrained("bigdata-pw/lyrics-gpt").cuda()


def generate(
    artist: Union[str, List[str]],
    lines: Optional[List[str]] = None,
    min_length: int = 50,
    max_length: int = 150,
    repetition_penalty: float = 1.2,
    temperature: float = 0.4,
    top_p: float = 0.95,
    top_k: int = 50,
):
    prompt = "<|artist|>"
    if isinstance(artist, str):
        artist = [artist]
    prompt += "<|artist|>".join(artist)
    prompt += "<|lyrics|>"
    if lines:
        prompt += "\n".join(lines)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.cuda()
    attention_mask = inputs.attention_mask.cuda()
    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        min_length=min_length,
        max_length=max_length,
        repetition_penalty=repetition_penalty,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        do_sample=True,
    )
    output = output[:, input_ids.shape[1] :]
    text = tokenizer.batch_decode(
        output, clean_up_tokenization_spaces=True, skip_special_tokens=True
    )[0]
    return text


with gr.Blocks() as demo:
    gr.Markdown("# Lyrics-GPT\nGenerate lyrics!\n[Known artists](https://huggingface.co/datasets/bigdata-pw/lyrics-gpt-info) and frequency in training dataset")

    with gr.Row():
        with gr.Column(scale=1):
            artist_input = gr.Textbox(
                lines=1, label="Artist (comma-separated if multiple)"
            )
            lines_input = gr.Textbox(
                lines=10, placeholder="Enter initial lines here", label="Initial Lines"
            )

        with gr.Column(scale=1):
            min_length_slider = gr.Slider(10, 300, value=50, step=5, label="Min Length")
            max_length_slider = gr.Slider(
                20, 500, value=150, step=5, label="Max Length"
            )
            repetition_penalty_slider = gr.Slider(
                1.0, 2.0, value=1.2, step=0.1, label="Repetition Penalty"
            )
            temperature_slider = gr.Slider(
                0.1, 1.0, value=0.4, step=0.1, label="Temperature"
            )
            top_p_slider = gr.Slider(0.1, 1.0, value=0.95, step=0.05, label="Top P")
            top_k_slider = gr.Slider(10, 100, value=50, step=10, label="Top K")

    generate_button = gr.Button("Generate Lyrics")
    output_box = gr.Textbox(label="Lyrics", lines=10)

    def gradio_interface(
        artist,
        lines,
        min_length,
        max_length,
        repetition_penalty,
        temperature,
        top_p,
        top_k,
    ):
        if artist == "":
            raise gr.Error("Artist required!")
        lines = lines.split("\n") if lines else None
        artist = [text.strip() for text in artist.split(",")]
        return generate(
            artist,
            lines,
            min_length,
            max_length,
            repetition_penalty,
            temperature,
            top_p,
            top_k,
        )

    generate_button.click(
        gradio_interface,
        inputs=[
            artist_input,
            lines_input,
            min_length_slider,
            max_length_slider,
            repetition_penalty_slider,
            temperature_slider,
            top_p_slider,
            top_k_slider,
        ],
        outputs=output_box,
    )


if __name__ == "__main__":
    demo.launch()
