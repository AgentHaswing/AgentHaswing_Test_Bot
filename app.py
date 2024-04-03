import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained("ai21labs/Jamba-v0.1", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("ai21labs/Jamba-v0.1")

def generate_text(prompt, max_new_tokens=216):
    # Tokenize the prompt
    input_ids = tokenizer(prompt, return_tensors='pt').to(model.device)["input_ids"]

    # Generate text
    outputs = model.generate(input_ids, max_new_tokens=max_new_tokens)

    # Decode the generated text
    generated_text = tokenizer.batch_decode(outputs)

    return generated_text

# Example usage
prompt = "In the recent Super Bowl LVIII,"
generated_text = generate_text(prompt)
print(generated_text)

def greet(name):
    return "Hello " + name + "!!"

iface = gr.Interface(fn=greet, inputs="text", outputs="text")
iface.launch()
