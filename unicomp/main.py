import os
from .utils import apply_info
from .llava_arch import LlavaMetaForCausalLM_ours

def unicomp(model):
    
    retain_ratio = float(os.environ.get("RETAIN_RATIO", 1))
    Uf = float(os.environ.get("Uf", 0.005))
    Uc = float(os.environ.get("Uc", 0.2))
    print("################################")
    print("########### unicomp ############")
    print(f"retain_ratio: {retain_ratio}")
    print(f"frame_uniqueness (Uf): {Uf}")
    print(f"spatial_uniqueness (Uc): {Uc}")
    print("################################")

    
    apply_info(model.model.vision_tower.vision_tower)
    
    from llava.model.llava_arch import LlavaMetaForCausalLM
    LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal = LlavaMetaForCausalLM_ours.prepare_inputs_labels_for_multimodal
    LlavaMetaForCausalLM.encode_images = LlavaMetaForCausalLM_ours.encode_images
    LlavaMetaForCausalLM.encode_images_multi = LlavaMetaForCausalLM_ours.encode_images_multi
    LlavaMetaForCausalLM.token_allocation = LlavaMetaForCausalLM_ours.token_allocation
    LlavaMetaForCausalLM.spatial_dynamic_compression = LlavaMetaForCausalLM_ours.spatial_dynamic_compression
    LlavaMetaForCausalLM.frame_group_fusion = LlavaMetaForCausalLM_ours.frame_group_fusion
    LlavaMetaForCausalLM.add_newline_token = LlavaMetaForCausalLM_ours.add_newline_token

    
    return model
