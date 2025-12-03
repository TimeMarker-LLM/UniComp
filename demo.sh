RR=0.1

video_path=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-basecv-hl/hadoop-basecv/yuanchao/benchmarks/MVBench/video/perception/videos/video_593.mp4
Q='Describe the video.'
Q="描述这个视频。"

# Base
# Q=$Q video_path=$video_path CUDA_VISIBLE_DEVICES=0 python demo.py

# UniComp
# WRAPPER=unicomp Q=$Q video_path=$video_path RETAIN_RATIO=$RR Uf=0.005 Uc=0.2 CUDA_VISIBLE_DEVICES=0 python demo.py

WRAPPER=unicomp Q=$Q video_path=$video_path RETAIN_RATIO=$RR AUTO=True Uf=0.005 Uc=0.2 CUDA_VISIBLE_DEVICES=0 python demo.py


# Holitom
# WRAPPER=holitom Q=$Q video_path=$video_path RETAIN_RATIO=$RR T=0.8 HOLITOM_k=18 HOLITOM_r=0.5 CUDA_VISIBLE_DEVICES=0 python demo.py

# VisionZip
# Tokens=$(echo "$RR * 196" | bc | awk '{print int($1)}')
# WRAPPER=visionzip Q=$Q video_path=$video_path CUDA_VISIBLE_DEVICES=0 SPATIAL_TOKENS=$Tokens python demo.py

# RETAIN_RATIO=$RR Q=$Q video_path=$video_path CUDA_VISIBLE_DEVICES=0 python /mnt/dolphinfs/ssd_pool/docker/user/hadoop-basecv-hl/hadoop-basecv/yuanchao/FastVID/demo.py

# FastVid
# WRAPPER=fastvid RETAIN_RATIO=$RR fastvid_DySeg_c=8 fastvid_DySeg_tau=0.9 fastvid_STPrune_d=0.4 fastvid_DTM_p=4 fastvid_DTM_alpha=0.6 Q=$Q video_path=$video_path CUDA_VISIBLE_DEVICES=0 python demo.py