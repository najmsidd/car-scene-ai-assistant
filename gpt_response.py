from gpt4all import GPT4All

model = GPT4All("mistral-7b-instruct-v0.1.Q4_0.gguf")  # Just the filename, not full path

def build_prompt(scene_dict):
    prompt_list = []

    for key,value in scene_dict.items():
        if isinstance(value, int):
            label = key if value == 1 else key + "s"
            prompt_list.append(f"{value} {label}")
        elif isinstance(value, str):
            prompt_list.append(f"a {value} {key}")

    prompt = "There are " + ", ".join(prompt_list) + ". Describe the scene and what the driver should do."
    return prompt

def gpt_response(prompt):
    with model.chat_session():
        return model.generate(prompt)

