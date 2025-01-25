# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import ast
import torch
import torch
from utils import draw_point, load_model, load_processor, prepare_inputs
from config import MIN_PIXELS, MAX_PIXELS, MODEL_NAME, SYSTEM_PROMT, IMG_URL


torch.cuda.empty_cache()
torch.cuda.memory_summary()


model = load_model(MODEL_NAME)
processor = load_processor(MODEL_NAME, MIN_PIXELS, MAX_PIXELS)

query = "Play button"

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": SYSTEM_PROMT},
            {"type": "image", "image": IMG_URL, "min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS},
            {"type": "text", "text": query}
        ],
    }
]

inputs = prepare_inputs(messages, processor)

generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0]

click_xy = ast.literal_eval(output_text)

draw_point(IMG_URL, click_xy, 10)
