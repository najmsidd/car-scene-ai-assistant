from detect import detect_image
from gpt_response import build_prompt, gpt_response


IMG_PATH = "images/scene1.jpg"

detected_image = detect_image(IMG_PATH)
print("\nDetected images:\n")
print(f"{detected_image}")

builded_prompt = build_prompt(detected_image)
print("\nGenerated Prompt\n")
print(f"{builded_prompt}")

gpted_response = gpt_response(builded_prompt)
print("\nGPT Generated response:\n")
print(f"{gpted_response}")
