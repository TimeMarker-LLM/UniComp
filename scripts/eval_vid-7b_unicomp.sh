# 64 frames, retain ratio 0.1
WRAPPER=unicomp RETAIN_RATIO=0.1 Uf=0.005 Uc=0.2 \
accelerate launch --num_processes=8 --main_process_port=25000 \
-m lmms_eval \
--model llava_vid \
--model_args pretrained=lmms-lab/LLaVA-Video-7B-Qwen2,conv_template=qwen_1_5,mm_spatial_pool_mode=average,max_frames_num=64 \
--tasks mlvu,longvideobench_val_v \
--batch_size 1 \
--log_samples \
--log_samples_suffix llava_vid \
--output_path ./logs/vid-7b-unicomp/0.10 2>&1 | tee ./logs/vid-7b-unicomp/0.10/vid-7b-unicomp-0.10.log

# 64 frames, retain ratio 0.25
WRAPPER=unicomp RETAIN_RATIO=0.25 Uf=0.005 Uc=0.2 \
accelerate launch --num_processes=8 --main_process_port=25000 \
-m lmms_eval \
--model llava_vid \
--model_args pretrained=lmms-lab/LLaVA-Video-7B-Qwen2,conv_template=qwen_1_5,mm_spatial_pool_mode=average,max_frames_num=64 \
--tasks mlvu,longvideobench_val_v \
--batch_size 1 \
--log_samples \
--log_samples_suffix llava_vid \
--output_path ./logs/vid-7b-unicomp/0.25 2>&1 | tee ./logs/vid-7b-unicomp/0.25/vid-7b-unicomp-0.25.log



#################################################
# Long video settings!!!
# 320 frames, compress to 64 frames (64*169 tokens)
WRAPPER=unicomp RETAIN_RATIO=0.1 Uf=0.005 Uc=0.2 \
accelerate launch --num_processes=8 --main_process_port=25000 \
-m lmms_eval \
--model llava_vid \
--model_args pretrained=lmms-lab/LLaVA-Video-7B-Qwen2,conv_template=qwen_1_5,mm_spatial_pool_mode=average,max_frames_num=320 \
--tasks mlvu,longvideobench_val_v \
--batch_size 1 \
--log_samples \
--log_samples_suffix llava_vid \
--output_path ./logs/vid-7b-ours/320fto64f 2>&1 | tee ./logs/vid-7b-ours/320fto64f/vid-7b-ours-320fto64f.log

# 256 frames, compress to 64 frames (64*169 tokens)
WRAPPER=unicomp RETAIN_RATIO=0.25 Uf=0.005 Uc=0.2 \
accelerate launch --num_processes=8 --main_process_port=25000 \
-m lmms_eval \
--model llava_vid \
--model_args pretrained=lmms-lab/LLaVA-Video-7B-Qwen2,conv_template=qwen_1_5,mm_spatial_pool_mode=average,max_frames_num=256 \
--tasks mlvu,longvideobench_val_v \
--batch_size 1 \
--log_samples \
--log_samples_suffix llava_vid \
--output_path ./logs/vid-7b-ours/256fto64f 2>&1 | tee ./logs/vid-7b-ours/256fto64f/vid-7b-ours-256fto64f.log