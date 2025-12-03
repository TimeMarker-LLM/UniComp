from abc import ABC, abstractmethod

import math
import re
import torch
import torch.nn as nn
import os

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from llava.mm_utils import get_anyres_image_grid_shape
from llava.utils import rank0_print, rank_print
import random

class LlavaMetaForCausalLM_ours(ABC):

    def encode_images(self, images):
        image_features, _ = self.get_model().get_vision_tower()(images)
        # image_features = self.get_model().vision_resampler(image_features, images=images)
        image_features = self.get_model().mm_projector(image_features)
        return image_features
    
    def encode_images_multi(self, images):
        image_features, attn_weights, metric, images_dtype = self.get_model().get_vision_tower()(images)
        frame_global_features=None
        frame_global_features = self.get_model().get_vision_abstract()(image_features.float())
        image_features = self.get_model().mm_projector(image_features)
        return image_features, None, metric, images_dtype, frame_global_features
    

    def frame_group_fusion(self, feat, uniqueness = 0.005, metric=None, min_interval = 10000):
        dtype = feat.dtype
        feat = feat.to(torch.float32)
        metric = feat.mean(1)
        metric = nn.functional.normalize(metric, dim=-1)
        u = 1-torch.cosine_similarity(metric[:, None, :], metric[None, :, :], dim=-1)
        new_feat = []
        num= 1
        selected_ids = [0]
        for i in range(1, feat.shape[0]):
            if u[i,selected_ids[-1]] <= uniqueness and num<=min_interval:
                num+=1
            else:
                new_feat.append(feat[i-num:i].sum(0)/num)
                num = 1
                selected_ids.append(i)

        new_feat.append(feat[feat.shape[0]-num:feat.shape[0]].sum(0)/num)
        feat = torch.stack(new_feat).to(dtype)
        selected_ids = torch.tensor(selected_ids)

        return feat, selected_ids
    
    def token_allocation(self, feat, max_tokens=10000, max_token_per_item=256, global_metric=None):
        if global_metric is not None:
            num_frames = global_metric.shape[0]
            global_metric = torch.nn.functional.normalize(global_metric, p=2, dim=-1)
            sims = torch.cosine_similarity(global_metric[:, None, :], global_metric[None, :, :], dim=-1)
        else:
            feat = feat.clone().detach().to(torch.float32)
            num_frames = feat.shape[0]
            global_feat = feat.mean(dim=-1) # b, dim
            global_feat = torch.nn.functional.normalize(global_feat, p=2, dim=-1)
            sims = torch.cosine_similarity(global_feat[:, None, :], global_feat[None, :, :], dim=-1)
            del feat


        sims = sims.mean(dim=-1)
        uniques = sims.mean()-sims
        uniques = uniques * (num_frames ** 0.5)

        max_token_per_item = min(max_token_per_item, max(1, int((max_tokens // num_frames) * 2))) 

        tokens_per_frame = torch.tensor([max_token_per_item] * len(uniques)).to(uniques.device)
        if len(uniques) > max_tokens // max_token_per_item:
            normalized_similarities = torch.softmax(uniques, dim=-1)
            tokens_per_frame = torch.floor(normalized_similarities * max_tokens).int()
            mask = tokens_per_frame>max_token_per_item
            tokens_per_frame[mask] = max_token_per_item
            extra_tokens = max_tokens - tokens_per_frame.sum()
            extra_tokens = max(0, extra_tokens)
            pad_num = (~mask).sum()
            avg_pad_tokens = extra_tokens // pad_num
            while mask.sum() > 0 and avg_pad_tokens>0:
                tokens_per_frame[~mask] += avg_pad_tokens
                mask = tokens_per_frame>max_token_per_item
                tokens_per_frame[mask] = max_token_per_item
                extra_tokens = max_tokens - tokens_per_frame.sum()
                extra_tokens = max(0, extra_tokens)
                pad_num = (~mask).sum()
                avg_pad_tokens = extra_tokens // pad_num
        return tokens_per_frame
    

    def spatial_dynamic_compression(self, feats, metric=None, tokens_per_frame=None, uniqueness = 0.2, nextline_token=torch.tensor([198])):        
        all_selected_ids = []
        new_feats = []
        num_frames, num_tokens, feat_dim = feats.shape
        
        feats_copy = feats.clone().to(torch.float32)

        if metric is None:
            metric = feats
        feats_norm = torch.nn.functional.normalize(metric.detach(), p=2, dim=-1)
        sim_all =  torch.matmul(feats_norm, feats_norm.transpose(1, 2)) # 64,144,144
      
        all_sim_avg = sim_all.mean(dim=-1)
        col_mask_all = 1 - sim_all < uniqueness
        
        # Sort with incresing order, uniqueness higher -> first to be selected
        unique_sorted_indices = torch.argsort(all_sim_avg, dim=-1, descending=False).to('cpu')

        for num in range(num_frames):
            sorted_indices = unique_sorted_indices[num]
            col_mask = col_mask_all[num]

            device = sorted_indices.device
            N = feats[num].shape[0]

            sorted_indices = sorted_indices.to(torch.long)
            col_mask_tensor = col_mask[sorted_indices][:, sorted_indices]
            col_mask_tensor = col_mask_tensor.to(torch.long)
            all_indices = torch.arange(N, device=device)

            causal_mask = (col_mask_tensor.bool() & torch.triu(torch.ones_like(col_mask_tensor), diagonal=1).bool())
            causal_mask_bool = causal_mask>0
            mask = torch.ones_like(all_indices, dtype=torch.bool)

            # simple logic but slow implementation of SDC with loops
            # # visited_indices = torch.zeros_like(sorted_indices)
            # # select_indices = []
            # # for idx in sorted_indices:
            # #     if visited_indices[idx]:
            # #         continue
            # #     select_indices.append(idx.item())
            # #     # fuse
            # #     feats[num][idx] = (feats[num][idx]+feats_copy[num][col_mask[idx]].mean(0))/2
            # #     # feats[num][idx] = feats_copy[num][col_mask[idx]].mean(0)
            # #     visited_indices[col_mask[idx]] = True

            # fast inplementation of SDC (result is the same with the original code)
            keep_indices = torch.arange(N, device=device)
            while keep_indices.shape[0] != 0:
                mask[keep_indices] = True
                remove_indices = torch.where(causal_mask_bool[keep_indices].any(dim=0))[0]
                mask[remove_indices] = False
                rest_indices = all_indices[mask]
                rest_indices = torch.where(causal_mask_bool[rest_indices].any(dim=0))[0]
                remove_indices = torch.where(causal_mask_bool[remove_indices].any(dim=0))[0]
                keep_indices = remove_indices[~torch.isin(remove_indices, rest_indices)]
    
            all_indices = all_indices[mask]
            select_indices = sorted_indices[all_indices]

            mask_valid = col_mask == 1
            gathered = feats_copy[num]
            sum_feat = (gathered * mask_valid.unsqueeze(-1)).sum(1)
            counts = mask_valid.sum(1).clamp(min=1).unsqueeze(-1)
            fused_all = (sum_feat / counts).to(torch.float32)
            feats[num][select_indices] = ((fused_all[select_indices] + feats_copy[num][select_indices])/2).to(feats.dtype)
            select_indices = select_indices.tolist()

            all_selected_ids.append(select_indices)
        selected_nums = torch.tensor([len(x) for x in all_selected_ids]).to(tokens_per_frame.device)

        if tokens_per_frame is not None:
            tokens_per_frame_tmp = tokens_per_frame.clone()
            if sum(selected_nums) > sum(tokens_per_frame):
                mask = selected_nums < tokens_per_frame_tmp
                remain_tokens = sum(tokens_per_frame) - sum(tokens_per_frame_tmp)
                remain_tokens += sum(tokens_per_frame_tmp[mask]) - sum(selected_nums[mask])
                if remain_tokens > 0:
                    avg_tokens = remain_tokens//(~mask).sum()
                    tokens_per_frame_tmp[~mask] += avg_tokens
            tokens_per_frame = tokens_per_frame_tmp

        for i in range(len(selected_nums)):
            select_indices = all_selected_ids[i]
            select_indices = select_indices[:tokens_per_frame[i]]
            select_indices.sort()
            new_feats.append(feats[i][select_indices])
            new_feats.append(nextline_token.to(feats.device))

        # new_feats = torch.cat(new_feats, dim=0)
        del feats_copy
        return new_feats
    
    def add_newline_token(self, feat, pos, grid_size, newline_token):
        expanded_feat_list = []
        nextline_token = torch.tensor([198]).to(self.get_model().device)
        nextline_token = self.get_model().embed_tokens(nextline_token)
        for cur_feat, cur_pos in zip(feat, pos):
            cur_row_pos = torch.tensor(cur_pos) // grid_size
            expanded_feat = []
            for row in range(grid_size):
                find_row_feat = cur_feat[cur_row_pos == row]
                if len(find_row_feat) > 0:
                    expanded_feat.append(torch.cat((find_row_feat, newline_token), dim=0))
                # else:
                #     expanded_feat.append(find_row_feat)
            
            expanded_feat.append(nextline_token)
            batch_feat = torch.cat(expanded_feat, dim=0)
            expanded_feat_list.append(batch_feat)
            
        image_feat = torch.cat(expanded_feat_list, dim=0)
        return image_feat
 
    def prepare_inputs_labels_for_multimodal(self, input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities=["image"], image_sizes=None):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        if isinstance(modalities, str):
            modalities = [modalities]

        if type(images) is list or images.ndim == 5:
            mm_patch_merge_type = getattr(self.config, "mm_patch_merge_type", "flat")
            image_aspect_ratio = getattr(self.config, "image_aspect_ratio", "square")
            mm_newline_position = getattr(self.config, "mm_newline_position", "one_token")
            
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]

            video_idx_in_batch = []
            for _ in range(len(modalities)):
                if modalities[_] == "video":
                    video_idx_in_batch.append(_)

            images_list = []
            for image in images:
                if image.ndim == 4:
                    images_list.append(image)
                else:
                    images_list.append(image.unsqueeze(0))

            concat_images = torch.cat([image for image in images_list], dim=0)
            split_sizes = [image.shape[0] for image in images_list]
            encoded_image_features, attn_weights, key, images_dtype, frame_global_features = self.encode_images_multi(concat_images)       # vit implementation      

            fix_position = bool(os.environ.get("fix_position", False))
            
            if mm_newline_position != "grid":
                nextline_token = torch.tensor([198]).to(encoded_image_features.device)
                nextline_token = self.get_model().embed_tokens(nextline_token)
            else:
                nextline_token = self.model.image_newline[None].to(encoded_image_features.device)

            encoded_image_features = torch.split(encoded_image_features, split_sizes)
            image_features = []
            video_position_ids = []
            frame_uniqueness = float(os.environ.get("Uf", 0.005))
            retain_ratio = float(os.environ.get("RETAIN_RATIO", 1))
            spatial_uniqueness = float(os.environ.get("Uc", 0.2))
            auto_compress = bool(os.environ.get("AUTO", False))
            min_interval = int(1 / retain_ratio) if not auto_compress else 1000000

            for idx, image_feat in enumerate(encoded_image_features):
                if idx in video_idx_in_batch:
                    pooled_image_feat = self.get_2dPool(image_feat) # (batch_size, seq_len', embed_dim)     
                    key = self.get_2dPool(key) 
                    frames = pooled_image_feat.shape[0]
                    frame_tokens = pooled_image_feat.shape[1]
         
                    selected_ids = list(range(frames))
                    pooled_image_feat, selected_ids = self.frame_group_fusion(pooled_image_feat, uniqueness = frame_uniqueness, min_interval=min_interval)
                    if frame_global_features!=None:
                        frame_global_features = frame_global_features[selected_ids]
                    key = key[selected_ids]
    
                    max_tokens=int(retain_ratio*frames*frame_tokens)
                    tokens_per_frame = torch.tensor([pooled_image_feat.shape[1]] * pooled_image_feat.shape[0]).to(pooled_image_feat.device)
                    
                    if auto_compress:
                        tokens_per_frame = torch.tensor([max_tokens//pooled_image_feat.shape[0]] * pooled_image_feat.shape[0]).to(pooled_image_feat.device)
                        pooled_image_feat = self.spatial_dynamic_compression(pooled_image_feat, key, tokens_per_frame, uniqueness=spatial_uniqueness,nextline_token=nextline_token)
                    else:
                        if pooled_image_feat.shape[0]*frame_tokens>max_tokens:
                            tokens_per_frame = self.token_allocation(pooled_image_feat, max_tokens=max_tokens, max_token_per_item=frame_tokens, global_metric=frame_global_features)   
                            pooled_image_feat = self.spatial_dynamic_compression(pooled_image_feat, key, tokens_per_frame, uniqueness=spatial_uniqueness,nextline_token=nextline_token)
                        else:
                            pooled_image_feat = [pooled_image_feat.flatten(0, 1)]


                    image_features.append(pooled_image_feat)

                else:
                    image_features.append(image_feat)

            
            if mm_patch_merge_type == "flat":
                image_features = [x.flatten(0, 1) for x in image_features]

            elif mm_patch_merge_type.startswith("spatial"):
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    # FIXME: now assume the image is square, and split to 2x2 patches
                    # num_patches = h * w, where h = w = sqrt(num_patches)
                    # currently image_feature is a tensor of shape (4, num_patches, hidden_size)
                    # we want to first unflatten it to (2, 2, h, w, hidden_size)
                    # rank0_print("At least we are reaching here")
                    # import pdb; pdb.set_trace()
                    if image_idx in video_idx_in_batch:  # video operations
                        # rank0_print(mm_newline_position)
                        if mm_newline_position == "grid":
                            # grid_size = int(math.sqrt(frame_tokens))
                            # newline_token = self.model.image_newline[None].to(image_feature[0].device)
                            # image_feature = self.add_newline_token(image_feature, all_selected_ids, grid_size, newline_token)
                            
                        
                            image_feature = torch.cat(image_feature, dim=0)
                            new_image_features.append(image_feature)
                        elif mm_newline_position == "frame":
                            # Frame-wise
                            # image_feature = self.add_token_per_frame(image_feature)
                            # new_image_features.append(image_feature.flatten(0, 1))

                            image_feature = torch.cat(image_feature, dim=0)
                            new_image_features.append(image_feature)

                            
                        elif mm_newline_position == "one_token":
                            image_feature = torch.cat(image_feature, dim=0)
                            # one-token
                            # image_feature = image_feature.flatten(0, 1)
                            if 'unpad' in mm_patch_merge_type:
                                image_feature = torch.cat((
                                    image_feature,
                                    self.model.image_newline[None].to(image_feature.device)
                                ), dim=0)
                            new_image_features.append(image_feature)      
                        elif mm_newline_position == "no_token":
                            new_image_features.append(image_feature.flatten(0, 1))
                        else:
                            raise ValueError(f"Unexpected mm_newline_position: {mm_newline_position}")
                    elif image_feature.shape[0] > 1:  # multi patches and multi images operations
                        # rank0_print("Single-images")
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]
                        height = width = self.get_vision_tower().num_patches_per_side
                        assert height * width == base_image_feature.shape[0]

                        if "anyres_max" in image_aspect_ratio:
                            matched_anyres_max_num_patches = re.match(r"anyres_max_(\d+)", image_aspect_ratio)
                            if matched_anyres_max_num_patches:
                                max_num_patches = int(matched_anyres_max_num_patches.group(1))

                        if image_aspect_ratio == "anyres" or "anyres_max" in image_aspect_ratio:
                            if hasattr(self.get_vision_tower(), "image_size"):
                                vision_tower_image_size = self.get_vision_tower().image_size
                            else:
                                raise ValueError("vision_tower_image_size is not found in the vision tower.")
                            try:
                                num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, vision_tower_image_size)
                            except Exception as e:
                                rank0_print(f"Error: {e}")
                                num_patch_width, num_patch_height = 2, 2
                            image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                        else:
                            image_feature = image_feature.view(2, 2, height, width, -1)

                        if "maxpool2x2" in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = nn.functional.max_pool2d(image_feature, 2)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        elif "unpad" in mm_patch_merge_type and "anyres_max" in image_aspect_ratio and matched_anyres_max_num_patches:
                            unit = image_feature.shape[2]
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            c, h, w = image_feature.shape
                            times = math.sqrt(h * w / (max_num_patches * unit**2))
                            if times > 1.1:
                                image_feature = image_feature[None]
                                image_feature = nn.functional.interpolate(image_feature, [int(h // times), int(w // times)], mode="bilinear")[0]
                            image_feature = torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        elif "unpad" in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            image_feature = torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        else:
                            image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                            image_feature = image_feature.flatten(0, 3)
                        if "nobase" in mm_patch_merge_type:
                            pass
                        else:
                            image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                        new_image_features.append(image_feature)
                    else:  # single image operations
                        image_feature = image_feature[0]
                        if "unpad" in mm_patch_merge_type:
                            image_feature = torch.cat((image_feature, self.model.image_newline[None]), dim=0)

                        new_image_features.append(image_feature)
                image_features = new_image_features
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        else:
            image_features = self.encode_images(images)



        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(self.config, "mm_use_im_start_end", False):
            raise NotImplementedError
        # rank_print(f"Total images : {len(image_features)}")

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        fix_position = bool(os.environ.get("fix_position", False))


        new_input_embeds = []
        new_labels = []
        new_position_ids = []
        cur_image_idx = 0
        # rank_print("Inserting Images embedding")
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
     
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                
                new_position_ids.append(torch.arange(cur_input_embeds.shape[0]))

                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1 : image_token_indices[i + 1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i] + 1 : image_token_indices[i + 1]])
    
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            cur_position_ids = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if fix_position:
                    start_id = torch.ceil(cur_position_ids[-1][-1]).long() + 1 if len(cur_position_ids) > 0 else 0
                    cur_position_ids.append(torch.arange(cur_input_embeds_no_im[i].shape[0]) + start_id)
            
                if i < num_images:
                    try:
                        cur_image_features = image_features[cur_image_idx]
                    except IndexError:
                        cur_image_features = image_features[cur_image_idx - 1]
                    
                    if fix_position:
                        start_id = torch.ceil(cur_position_ids[-1][-1]).long() + 1 if len(cur_position_ids) > 0 else 0
                        cur_position_ids.append(video_position_ids[cur_image_idx] + start_id)
   
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            # import pdb; pdb.set_trace()
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

            if fix_position:
                new_position_ids.append(torch.cat(cur_position_ids))


        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, "tokenizer_model_max_length", None)
        # rank_print("Finishing Inserting")

        new_input_embeds = [x[:tokenizer_model_max_length] for x, modality in zip(new_input_embeds, modalities)]
        new_labels = [x[:tokenizer_model_max_length] for x, modality in zip(new_labels, modalities)]
        # TODO: Hard code for control loss spike
        # if tokenizer_model_max_length is not None:
        #     new_input_embeds = [x[:4096] if modality != "video" else x[:tokenizer_model_max_length] for x, modality in zip(new_input_embeds, modalities)]
        #     new_labels = [x[:4096] if modality != "video" else x[:tokenizer_model_max_length] for x, modality in zip(new_labels, modalities)]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)
        # rank0_print("Prepare pos id")


        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, "tokenizer_padding_side", "right") == "left":
                new_input_embeds_padded.append(torch.cat((torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device), cur_new_embed), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    if fix_position:
                        position_ids[i, -cur_len:] = new_position_ids[i][-cur_len:]
                    else:
                        position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

            else:
                new_input_embeds_padded.append(torch.cat((cur_new_embed, torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    if fix_position:
                        position_ids[i, :cur_len] = new_position_ids[i][:cur_len]
                    else:
                        position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)
        # rank0_print("tokenizer padding")

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if fix_position:
            position_ids = position_ids.to(new_input_embeds.device)
        elif _position_ids is None:
            position_ids = None
        # print(position_ids)
        assert position_ids==None or position_ids.shape[1] == new_input_embeds.shape[1]
        
        if getattr(self.config, "use_pos_skipping", False) and self.training:
            position_ids = torch.arange(new_input_embeds.size(1), device=new_input_embeds.device).unsqueeze(0).to(new_input_embeds.device)
            split_position = random.randint(0, new_input_embeds.size(1))
            left_add = random.randint(0, self.config.pos_skipping_range)
            right_add = random.randint(left_add, self.config.pos_skipping_range)
            position_ids[:, :split_position] += left_add
            position_ids[:, split_position:] += right_add

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels
