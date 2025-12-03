# 64 frames, retain ratio 0.1
WRAPPER=holitom RETAIN_RATIO=0.1 T=0.80 HOLITOM_k=18 HOLITOM_r=0.5 \
accelerate launch --num_processes=8 --main_process_port=25000 \
-m lmms_eval \
--model llava_vid \
--model_args pretrained=lmms-lab/LLaVA-Video-7B-Qwen2,conv_template=qwen_1_5,mm_spatial_pool_mode=average,max_frames_num=64 \
--tasks longvideobench_val_v,mlvu \
--batch_size 1 \
--log_samples \
--log_samples_suffix llava_vid \
--output_path ./logs/vid-7b-holitom/0.10 2>&1 | tee ./logs/vid-7b-holitom/0.10/vid-7b-holitom-0.10.log

# 64 frames, retain ratio 0.25
WRAPPER=holitom RETAIN_RATIO=0.25 T=0.80 HOLITOM_k=18 HOLITOM_r=0.5 \
accelerate launch --num_processes=8 --main_process_port=25000 \
-m lmms_eval \
--model llava_vid \
--model_args pretrained=lmms-lab/LLaVA-Video-7B-Qwen2,conv_template=qwen_1_5,mm_spatial_pool_mode=average,max_frames_num=64 \
--tasks longvideobench_val_v,mlvu \
--batch_size 1 \
--log_samples \
--log_samples_suffix llava_vid \
--output_path ./logs/vid-7b-holitom/0.25 2>&1 | tee ./logs/vid-7b-holitom/0.25/vid-7b-holitom-0.25.log


#################################################
# Long video settings!!!
# 320 frames, compress to 64 frames (64*169 tokens)
WRAPPER=holitom RETAIN_RATIO=0.2 T=0.80 HOLITOM_k=18 HOLITOM_r=0.5 \
accelerate launch --num_processes=8 --main_process_port=25000 \
-m lmms_eval \
--model llava_vid \
--model_args pretrained=lmms-lab/LLaVA-Video-7B-Qwen2,conv_template=qwen_1_5,mm_spatial_pool_mode=average,max_frames_num=320 \
--tasks longvideobench_val_v,mlvu \
--batch_size 1 \
--log_samples \
--log_samples_suffix llava_vid \
--output_path ./logs/vid-7b-holitom/320fto64f 2>&1 | tee ./logs/vid-7b-holitom/320fto64f/vid-7b-holitom-320fto64f.log

# 256 frames, compress to 64 frames (64*169 tokens)
WRAPPER=holitom RETAIN_RATIO=0.25 T=0.80 HOLITOM_k=18 HOLITOM_r=0.5 \
accelerate launch --num_processes=8 --main_process_port=25000 \
-m lmms_eval \
--model llava_vid \
--model_args pretrained=lmms-lab/LLaVA-Video-7B-Qwen2,conv_template=qwen_1_5,mm_spatial_pool_mode=average,max_frames_num=256 \
--tasks longvideobench_val_v,mlvu \
--batch_size 1 \
--log_samples \
--log_samples_suffix llava_vid \
--output_path ./logs/vid-7b-holitom/256fto64f 2>&1 | tee ./logs/vid-7b-holitom/256fto64f/vid-7b-holitom-256fto64f.log