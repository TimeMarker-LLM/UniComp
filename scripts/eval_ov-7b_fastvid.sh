# FastVid
WRAPPER=fastvid RETAIN_RATIO=0.1 fastvid_DySeg_c=8 fastvid_DySeg_tau=0.9 fastvid_STPrune_d=0.4 fastvid_DTM_p=4 fastvid_DTM_alpha=0.6 \
accelerate launch --num_processes=2 --main_process_port=25000 \
-m lmms_eval \
--model llava_onevision \
--model_args pretrained=lmms-lab/llava-onevision-qwen2-7b-ov,conv_template=qwen_1_5,model_name=llava_qwen,max_frames_num=32 \
--tasks egoschema,mlvu,videomme,longvideobench_val_v \
--batch_size 1 \
--log_samples \
--log_samples_suffix llava_onevision \
--output_path ./logs/ov-7b-fastvid/0.10 2>&1 | tee ./logs/ov-7b-fastvid/0.10/ov-7b-fastvid-0.10.log
