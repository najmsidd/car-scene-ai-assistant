from gpt4all import GPT4All

model = GPT4All("mistral-7b-instruct-v0.1.Q4_0.gguf")  # Just the filename, not full path

with model.chat_session():
    prompt = "There are 3 cars, 1 person, and a red traffic light. Describe the scene and what the driver should do."
    response = model.generate(prompt)
    print("\nScene description:\n")
    print(response)

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

sample_scene = {
    "car": 3,
    "person": 1,
    "traffic light": "red"
}

sample_prompt = build_prompt(sample_scene)
print(sample_prompt)
