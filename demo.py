import os
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates

import numpy as np
import copy
import warnings
from decord import VideoReader, cpu

warnings.filterwarnings("ignore")
# Load the OneVision model
pretrained = "lmms-lab/llava-onevision-qwen2-7b-ov"
model_name = "llava_qwen"
device = "cuda"
device_map = "auto"
llava_model_args = {
    "multimodal": True,
}
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map, attn_implementation="sdpa", **llava_model_args)
model_wrapper = os.environ.get("WRAPPER")
if model_wrapper in ["visionzip", "holitom", "fastvid", "unicomp"]:
    wrapper_module = __import__(model_wrapper)
    wrapper_class = getattr(wrapper_module, model_wrapper)
    model = wrapper_class(model)
elif model_wrapper is not None:
    raise ValueError(f"Invalid model wrapper: {model_wrapper}")

# model = model.to(model.device)

model.eval()




# Function to extract frames from video
def load_video(video_path, max_frames_num):
    if type(video_path) == str:
        vr = VideoReader(video_path, ctx=cpu(0))
    else:
        vr = VideoReader(video_path[0], ctx=cpu(0))
    total_frame_num = len(vr)
    uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
    frame_idx = uniform_sampled_frames.tolist()
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    return spare_frames  # (frames, height, width, channels)


# Load and process video
video_path = str(os.environ.get("video_path", ""))
video_frames = load_video(video_path, 32)


image_tensors = []
frames = image_processor.preprocess(video_frames, return_tensors="pt")["pixel_values"].half().cuda()
image_tensors.append(frames)

# Prepare conversation input
conv_template = "qwen_1_5"
Q = str(os.environ.get("Q", ""))
question = f"{DEFAULT_IMAGE_TOKEN}\n"+Q
# question = f"{DEFAULT_IMAGE_TOKEN}\nDescribe the video."

conv = copy.deepcopy(conv_templates[conv_template])
conv.append_message(conv.roles[0], question)
conv.append_message(conv.roles[1], None)
prompt_question = conv.get_prompt()

input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
image_sizes = [frame.size for frame in video_frames]

# Generate response
import time
t_start = time.time()
cont = model.generate(
    input_ids,
    images=image_tensors,
    image_sizes=image_sizes,
    do_sample=False,
    temperature=0,
    max_new_tokens=4096,
    modalities=["video"],
)
print("Time", time.time()-t_start)
text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
print("+++++++++++answer+++++++++++")
print(text_outputs[0])
print("++++++++++++++++++++++++++++")
