# 32 frames, retain ratio 0.1
WRAPPER=unicomp RETAIN_RATIO=0.1 Uf=0.005 Uc=0.2 \
accelerate launch --num_processes=8 --main_process_port=25000 \
-m lmms_eval \
--model llava_onevision \
--model_args pretrained=lmms-lab/llava-onevision-qwen2-7b-ov,conv_template=qwen_1_5,model_name=llava_qwen,max_frames_num=32 \
--tasks egoschema,mlvu,videomme,longvideobench_val_v \
--batch_size 1 \
--log_samples \
--log_samples_suffix llava_onevision \
--output_path ./logs/ov-7b-unicomp/0.10 2>&1 | tee ./logs/ov-7b-unicomp/0.10/ov-7b-unicomp-0.10.log

# 32 frames, retain ratio 0.15
WRAPPER=unicomp RETAIN_RATIO=0.15 Uf=0.005 Uc=0.2 \
accelerate launch --num_processes=8 --main_process_port=25000 \
-m lmms_eval \
--model llava_onevision \
--model_args pretrained=lmms-lab/llava-onevision-qwen2-7b-ov,conv_template=qwen_1_5,model_name=llava_qwen,max_frames_num=32 \
--tasks egoschema,mlvu,videomme,longvideobench_val_v \
--batch_size 1 \
--log_samples \
--log_samples_suffix llava_onevision \
--output_path ./logs/ov-7b-unicomp/0.15 2>&1 | tee ./logs/ov-7b-unicomp/0.15/ov-7b-unicomp-0.15.log

# 32 frames, retain ratio 0.2
WRAPPER=unicomp RETAIN_RATIO=0.2 Uf=0.005 Uc=0.2 \
accelerate launch --num_processes=8 --main_process_port=25000 \
-m lmms_eval \
--model llava_onevision \
--model_args pretrained=lmms-lab/llava-onevision-qwen2-7b-ov,conv_template=qwen_1_5,model_name=llava_qwen,max_frames_num=32 \
--tasks egoschema,mlvu,videomme,longvideobench_val_v \
--batch_size 1 \
--log_samples \
--log_samples_suffix llava_onevision \
--output_path ./logs/ov-7b-unicomp/0.20 2>&1 | tee ./logs/ov-7b-unicomp/0.20/ov-7b-unicomp-0.20.log

# 32 frames, retain ratio 0.25
WRAPPER=unicomp RETAIN_RATIO=0.25 Uf=0.005 Uc=0.2 \
accelerate launch --num_processes=8 --main_process_port=25000 \
-m lmms_eval \
--model llava_onevision \
--model_args pretrained=lmms-lab/llava-onevision-qwen2-7b-ov,conv_template=qwen_1_5,model_name=llava_qwen,max_frames_num=32 \
--tasks egoschema,mlvu,videomme,longvideobench_val_v \
--batch_size 1 \
--log_samples \
--log_samples_suffix llava_onevision \
--output_path ./logs/ov-7b-unicomp/0.25 2>&1 | tee ./logs/ov-7b-unicomp/0.25/ov-7b-unicomp-0.25.log

#################################################
# Long video settings!!!
# 320 frames, compress to 32 frames (32*196 tokens)
WRAPPER=unicomp RETAIN_RATIO=0.1 Uf=0.005 Uc=0.2 \
accelerate launch --num_processes=8 --main_process_port=25000 \
-m lmms_eval \
--model llava_onevision \
--model_args pretrained=lmms-lab/llava-onevision-qwen2-7b-ov,conv_template=qwen_1_5,model_name=llava_qwen,max_frames_num=320 \
--tasks egoschema,mlvu,videomme,longvideobench_val_v \
--batch_size 1 \
--log_samples \
--log_samples_suffix llava_onevision \
--output_path ./logs/ov-7b-unicomp/320fto32f 2>&1 | tee ./logs/ov-7b-unicomp/320fto32f/ov-7b-unicomp-320fto32f.log

# 256 frames, compress to 32 frames (32*196 tokens)
WRAPPER=unicomp RETAIN_RATIO=0.125 Uf=0.005 Uc=0.2 \
accelerate launch --num_processes=8 --main_process_port=25000 \
-m lmms_eval \
--model llava_onevision \
--model_args pretrained=lmms-lab/llava-onevision-qwen2-7b-ov,conv_template=qwen_1_5,model_name=llava_qwen,max_frames_num=256 \
--tasks egoschema,mlvu,videomme,longvideobench_val_v \
--batch_size 1 \
--log_samples \
--log_samples_suffix llava_onevision \
--output_path ./logs/ov-7b-unicomp/256fto32f 2>&1 | tee ./logs/ov-7b-unicomp/256fto32f/ov-7b-unicomp-256fto32f.log

# 128 frames, compress to 32 frames (32*196 tokens)
WRAPPER=unicomp RETAIN_RATIO=0.25 Uf=0.005 Uc=0.2 \
accelerate launch --num_processes=8 --main_process_port=25000 \
-m lmms_eval \
--model llava_onevision \
--model_args pretrained=lmms-lab/llava-onevision-qwen2-7b-ov,conv_template=qwen_1_5,model_name=llava_qwen,max_frames_num=128 \
--tasks egoschema,mlvu,videomme,longvideobench_val_v \
--batch_size 1 \
--log_samples \
--log_samples_suffix llava_onevision \
--output_path ./logs/ov-7b-unicomp/128fto32f 2>&1 | tee ./logs/ov-7b-unicomp/128fto32f/ov-7b-unicomp-128fto32f.log
