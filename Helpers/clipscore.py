import torch
from torch import nn
import numpy as np
import math
from opt_einsum import contract
import tqdm
import itertools
from torch.nn.utils.rnn import pad_sequence
from PIL import Image, ImageFilter
from transformers import AutoTokenizer, AutoProcessor, AutoModel,CLIPImageProcessor, CLIPVisionConfig, CLIPConfig, CLIPTextConfig
from transformers.modeling_outputs import BaseModelOutputWithPooling
from typing import Tuple, Optional, Union, List
from transformers.models.clip.modeling_clip import CLIPVisionEmbeddings, CLIPEncoder,CLIPVisionTransformer, CLIPTextTransformer, CLIPModel, CLIPVisionModel
import matplotlib.pyplot as plt
import gc

def text_collate_fn(batch):
    """
    Collates a batch of data with variable textual input lengths and fixed-sized image data.
    """
    new_batch = {}
    pad_token = batch[0]["padding_value"]

    for k in batch[0].keys():
        if k != "padding_value":
            # Handle variable-length text inputs
            if k == "input_ids":
                # Pad input_ids along sequence length dimension
                new_batch[k] = pad_sequence(
                    [item[k] for item in batch], padding_value=pad_token, batch_first=True
                ).squeeze(-1)

            elif k in ["attention_mask", "token_type_ids"]:
                # Pad attention mask and token type ids similarly
                new_batch[k] = pad_sequence(
                    [item[k] for item in batch], padding_value=0, batch_first=True
                ).squeeze(-1)

            # Handle image data (pixel_values, pixel_mask) by stacking along batch dimension
            elif k in ["pixel_values", "pixel_mask"]:
                new_batch[k] = torch.stack([item[k] for item in batch], dim=0)

            else:
                # If there are other fields, add them as-is
                new_batch[k] = list(map(lambda dict: dict[k], batch))

    return new_batch

class CLIPCapDataset(torch.utils.data.Dataset):
    """
    Textual Dataset. 
        - Adds a prefix string to the textual input and encodes the text sequence with an textual processor
    """
    def __init__(self, data, text_processor,use_open_clip = False):
        self.data = data
        self.use_open_clip = use_open_clip
        self.text_processor = text_processor

    def __getitem__(self, idx):
        c_data = self.data[idx]
        if self.use_open_clip:
            c_data = self.text_processor(c_data)
            c_data = {"input_ids": c_data.squeeze(0)}
            c_data["padding_value"] = 0
        else:
            c_data = self.text_processor(c_data, truncation = True, return_tensors="pt")
            c_data = {k: v.squeeze(0) for k,v in c_data.items()}
            c_data["padding_value"] = c_data["input_ids"][-1]
        
        return c_data

    def __len__(self):
        return len(self.data)

class CLIPSampleDataset(torch.utils.data.Dataset):
    """
    Textual Dataset. 
        - Adds a prefix string to the textual input and encodes the text sequence with an textual processor
    """
    def __init__(self, data, text_processor,tokenized_data,pos_tagging,valid_pos):
        self.data = data
        self.tokenized_data = tokenized_data
        self.pos_tagging = pos_tagging
        self.text_processor = text_processor
        self.valid_pos = valid_pos
        if valid_pos == None:
            self.valid_pos = [valid_pos]
            
    
    def string_mapping(self,words,pos_tag):
        positions = {}
        start_index = 0

        for i,(word,tag) in enumerate(zip(words,pos_tag)):
            end_index = start_index + len(word)
            if (self.valid_pos == [None]) | (tag in self.valid_pos):
                positions[i] = {"string_pos":(start_index, end_index),"word":word}
            start_index = end_index + 1


        return positions

    def __getitem__(self, idx):
        c_data = self.data[idx]
        tokenized_data = self.tokenized_data[idx]
        pos_tagging = self.pos_tagging[idx]
        position_map = self.string_mapping(words=tokenized_data,pos_tag=pos_tagging)
        c_data = self.text_processor(c_data, truncation = True, return_tensors="pt",return_offsets_mapping=True)
        for i in position_map.keys():
            tokens = []
            string_pos = position_map[i]["string_pos"]
            for j,o in enumerate(c_data["offset_mapping"][0]):
                if j == 0:
                    continue
                if (string_pos[0] <= o[0]) & (string_pos[1] >= o[1]):
                    tokens.append(j)
                elif (string_pos[1] < o[0]):
                    break
                elif (string_pos[0] >= o[1]):
                    continue
            position_map[i]["tokens_in_word"] = tokens
        c_data = {k: v.squeeze(0) for k,v in c_data.items() if k != "offset_mapping"}
        c_data["padding_value"] = c_data["input_ids"][-1]
        c_data["word_info"] = position_map
        return c_data

    def __len__(self):
        return len(self.data)

class CLIPImageDataset(torch.utils.data.Dataset):
    """
    Visual Dataset. 
        - Encodes the input image with an image processor. When image_processor is None, 
        a generic transformation is applied based on the original clip-vit-base-patch32 model
    """
    def __init__(self, data,image_processor, use_open_clip = False):
        self.data = data
        # only 224x224 ViT-B/32 supported for now
        self.preprocess = image_processor
        
        self.use_open_clip = use_open_clip

    def __getitem__(self, idx):
        c_data = self.data[idx]
        image = Image.open(c_data)
        image = image.convert("RGB")
        if self.use_open_clip:
            image = self.preprocess(image)
        else:
            image = self.preprocess(image,return_tensors="pt")
        if type(image) is torch.Tensor:
            return {'pixel_values':image}
        else:
            return {k: v.squeeze(0) for k,v in image.items()}

    def __len__(self):
        return len(self.data)


class SampleWrapper(nn.Module):
    def __init__(self,model,sampling_ratio,n_samples,device = "cuda"):
        super(SampleWrapper,self).__init__()

        self.model = model
        self.sampling_ratio = sampling_ratio
        self.n_samples = n_samples
        self.device = device
    
    def torch_delete(self,tensor, indices):
        """
        Deleats torch values based on indices
        """
        mask = torch.ones(tensor.numel(), dtype=torch.bool)
        mask[indices] = False
        return tensor[mask]
    
    def extract_all_k(self,word_info,k = "tokens_in_word"):
        return  [item[k] for item in word_info.values()]
    
    def tensor_sampling(self, input_ids: Optional[torch.Tensor] = None,attention_mask: Optional[torch.Tensor] = None, token_type_ids: Optional[torch.Tensor] = None,word_info: Optional[List] = None,all_tokens: Optional[List] = None):
        """
        Samples tokens from input_ids and attention_mask tensors
        """
        assert attention_mask is not None
        max_length = len(word_info)
        k = math.ceil(self.sampling_ratio*max_length)
        ids = list(np.arange(max_length))
        removed_valid_words = np.random.choice(ids,size=k,replace=False)
        possible_words = list(word_info.keys())
        removed_words = [possible_words[rm] for rm in removed_valid_words]
        removed_ids = [all_tokens[rm] for rm in removed_valid_words]
        removed_ids = list(itertools.chain.from_iterable(removed_ids))
        new_I = input_ids.clone()
        if token_type_ids != None:
            new_T = token_type_ids.clone()
        new_A = attention_mask.clone()
        new_A[removed_ids] = 0
        if token_type_ids != None:
            return removed_words,{"input_ids": new_I.to(torch.int), "attention_mask": new_A.to(torch.int), "token_type_ids": new_T.to(torch.int)}
        else:
            return removed_words,{"input_ids": new_I.to(torch.int), "attention_mask": new_A.to(torch.int)}

    def get_text_features(self, input_ids: Optional[torch.Tensor] = None,attention_mask: Optional[torch.Tensor] = None,token_type_ids: Optional[torch.Tensor] = None,word_info: Optional[List] = None):
        input_ids = input_ids.squeeze(0)
        attention_mask = attention_mask.squeeze(0)
        if token_type_ids != None:
            token_type_ids = token_type_ids.squeeze(0)
        word_info = word_info[0]
        new_input_ids = [input_ids]
        new_input_att = [attention_mask]
        if token_type_ids != None:
            new_input_tok = [token_type_ids]

        removed_ids_list = []
        all_tokens = self.extract_all_k(word_info)
        for i in range(self.n_samples):
            removed_ids,aux = self.tensor_sampling(input_ids=input_ids,attention_mask=attention_mask,token_type_ids = token_type_ids,word_info=word_info,all_tokens = all_tokens)
            new_input_ids.append(aux["input_ids"])
            new_input_att.append(aux["attention_mask"])
            if token_type_ids != None:
                new_input_tok.append(aux["token_type_ids"])
            removed_ids_list.append(removed_ids)

        if token_type_ids != None:
            new_input = {"input_ids": torch.stack(new_input_ids),"attention_mask":torch.stack(new_input_att),"token_type_ids":torch.stack(new_input_tok)}
        else:
            new_input = {"input_ids": torch.stack(new_input_ids),"attention_mask":torch.stack(new_input_att)}
        new_input = {k: v.to(self.device) for k,v in new_input.items()}
        with torch.no_grad():
            text_output = self.model.get_text_features(**new_input)

        return removed_ids_list, text_output

class CLIPVisionModelMask(CLIPVisionModel):

    def __init__(self, config: CLIPVisionConfig):
        super().__init__(config)
        self.vision_model = CLIPVisionTransformerWithMask(config)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.vision_model.embeddings.patch_embedding

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, CLIPVisionModel

        >>> model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        >>> processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled CLS states
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        with torch.no_grad():
            return self.vision_model(
                pixel_values=pixel_values,
                attention_mask= attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

class CLIPModelWithMask(CLIPModel):
    config_class = CLIPConfig
    def __init__(self, config: CLIPConfig):
        super().__init__(config)

        if not isinstance(config.text_config, CLIPTextConfig):
            raise TypeError(
                "config.text_config is expected to be of type CLIPTextConfig but is of type"
                f" {type(config.text_config)}."
            )

        if not isinstance(config.vision_config, CLIPVisionConfig):
            raise TypeError(
                "config.vision_config is expected to be of type CLIPVisionConfig but is of type"
                f" {type(config.vision_config)}."
            )

        self.config = config
        text_config = config.text_config
        vision_config = config.vision_config

        self.projection_dim = config.projection_dim
        self.text_embed_dim = text_config.hidden_size
        self.vision_embed_dim = vision_config.hidden_size

        self.text_model = CLIPTextTransformer(text_config)
        self.vision_model = CLIPVisionModelMask(vision_config)

        self.visual_projection = nn.Linear(self.vision_embed_dim, self.projection_dim, bias=False)
        self.text_projection = nn.Linear(self.text_embed_dim, self.projection_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(self.config.logit_scale_init_value))

        # Initialize weights and apply final processing
        self.post_init()

    def get_image_features(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:
    
        # Use CLIP model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        
        with torch.no_grad():
            vision_outputs = self.vision_model(
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            pooled_output = vision_outputs[1]  # pooled_output
            image_features = self.visual_projection(pooled_output)

        return image_features


def prepare_attention_mask(attention_mask):
        inverted_mask = 1.0 - attention_mask
        return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(torch.float32).min)

class CLIPVisionTransformerWithMask(CLIPVisionTransformer):
    def __init__(self, config: CLIPVisionConfig):
        super().__init__(config)
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = CLIPVisionEmbeddings(config)
        self.pre_layrnorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.encoder = CLIPEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")
        
        with torch.no_grad():
            hidden_states = self.embeddings(pixel_values)
            hidden_states_2 = self.pre_layrnorm(hidden_states)
            
            
            attention_mask_2 = prepare_attention_mask(attention_mask)

            encoder_outputs = self.encoder(
                inputs_embeds=hidden_states_2,
                attention_mask=attention_mask_2,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            last_hidden_state = encoder_outputs[0]
            pooled_output = last_hidden_state[:, 0, :]
            pooled_output = self.post_layernorm(pooled_output)

            if not return_dict:
                return (last_hidden_state, pooled_output) + encoder_outputs[1:]
            hidden_states_o = encoder_outputs.hidden_states
            attentions_o = encoder_outputs.attentions
            
            return BaseModelOutputWithPooling(
                last_hidden_state=last_hidden_state,
                pooler_output=pooled_output,
                hidden_states=hidden_states_o,
                attentions=attentions_o,
            )


# Modified CLIP Vision Transformer with Attention Mask
class VisionSampleWrapper(nn.Module):
    def __init__(self, model, sampling_ratio, n_samples, device="cuda"):
        super(VisionSampleWrapper, self).__init__()

        self.model = model
        self.config = model.config.vision_config
        self.sampling_ratio = sampling_ratio
        self.n_samples = n_samples
        self.device = device
    
    

    def tensor_sampling(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Randomly masks a fraction of image patches, simulating noise, by modifying the attention mask.
        """
        # Assuming pixel_values is a 4D tensor: (batch_size, num_patches, patch_size, channels)
        _, num_patches, _ = attention_mask.shape
        flatten_patch_matrix_size = num_patches*num_patches
        k = math.ceil(self.sampling_ratio * flatten_patch_matrix_size)  # Number of patches to sample
        ids = np.arange(1,flatten_patch_matrix_size)

        removed_patches = np.random.choice(ids, size=k, replace=False)
        new_A = attention_mask[0].clone()
        new_A = new_A.flatten()  # Clone current attention mask
        new_A[removed_patches] = 0  # Set chosen patches as masked
        new_A = new_A.view((num_patches,num_patches))
        
        removed_patches = torch.nonzero(new_A == 0, as_tuple=False)

        return removed_patches.to(torch.int),new_A.unsqueeze(0).to(torch.int)

    def get_image_features(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")
        if attention_mask is None:
            batch_size, num_channels, height, width = pixel_values.shape
            patch_size = self.config.patch_size  # Adjust if necessary
            num_patches = (height // patch_size) * (width // patch_size)
            attention_mask = torch.ones(batch_size, 1, num_patches +1, num_patches+1, dtype=torch.float32)
        attention_mask = attention_mask.squeeze(0)
        new_attention_masks = [attention_mask]
        
        removed_patches_list = []
        for _ in range(self.n_samples):
            removed_patches, new_attention_mask = self.tensor_sampling(attention_mask)
            new_attention_masks.append(new_attention_mask)
            removed_patches_list.append(removed_patches)
        
        # Stack sampled versions of input pixel values and attention masks
        new_input = {
            "pixel_values": pixel_values.repeat(self.n_samples + 1,1,1,1),
            "attention_mask": torch.stack(new_attention_masks)
        }
        
        new_input = {k: v.to(self.device) for k, v in new_input.items()}
        
        with torch.no_grad():
            vision_output = self.model.get_image_features(**new_input)
        
        return removed_patches_list, vision_output

def extract_all_texts(text,model,tokenized_words=None,pos_tagging=None,text_processor=None,valid_pos = None, device = "cuda",batch_size=64, num_workers=0,use_open_clip=False,use_attention_mask_sampling = False):
    """
    Text extraction function. 
        - gets the encoding vectors of a list of strings
    """
    if use_attention_mask_sampling:
        batch_size = 1
    if use_attention_mask_sampling:
        data = torch.utils.data.DataLoader(
            CLIPSampleDataset(text,text_processor=text_processor,tokenized_data=tokenized_words,pos_tagging=pos_tagging,valid_pos=valid_pos),
            collate_fn=text_collate_fn,
            batch_size=batch_size, num_workers=num_workers, shuffle=False)
    else:
        data = torch.utils.data.DataLoader(
            CLIPCapDataset(text,text_processor=text_processor,use_open_clip=use_open_clip),
            collate_fn=text_collate_fn,
            batch_size=batch_size, num_workers=num_workers, shuffle=False)
    all_text_features = []
    all_remove_ids = []
    with torch.no_grad():
        for i,b in enumerate(tqdm.tqdm(data)):
            b = {k: v.to(device) if k != "word_info" else v for k, v in b.items()}
            if use_attention_mask_sampling:
                removed_ids ,text_model_output = model.get_text_features(**b)
                all_text_features.append(text_model_output.detach().cpu().unsqueeze(0))
                all_remove_ids.append(removed_ids)
            else:
                all_text_features.append(model.get_text_features(**b).detach().cpu())
            del b, text_model_output
            if i % 1000 == 0:
                gc.collect()
            torch.cuda.empty_cache()
    all_text_features = torch.concat(all_text_features)
    if use_attention_mask_sampling:
        return all_remove_ids, all_text_features
    return all_text_features


def extract_all_images(images, model,image_processor, device, batch_size=64, num_workers=8, use_open_clip = False, use_attention_mask_sampling= False):
    """
    Visual extraction function. 
        - gets the encoding vectors of a list of images. 
        The input variable "images" corresponds to the list of image paths.
    """
    if use_attention_mask_sampling:
        batch_size = 1
    data = torch.utils.data.DataLoader(
        CLIPImageDataset(images,image_processor=image_processor,use_open_clip=use_open_clip),
        batch_size=batch_size, num_workers=num_workers, shuffle=False)
    all_image_features = []
    all_remove_patches = []
    with torch.no_grad():
        for i,b in enumerate(tqdm.tqdm(data)):
            b = {k: v.to(device) for k,v in b.items()}
            if device == 'cuda':
                b = {k: v.to(torch.float) for k,v in b.items()}
            if use_attention_mask_sampling:
                removed_patch ,image_model_output = model.get_image_features(**b)
                all_image_features.append(image_model_output.detach().cpu().unsqueeze(0))
                all_remove_patches.append(removed_patch)
                del image_model_output
            else:
                all_image_features.append(model.get_image_features(**b).detach().cpu())
            
            del b
            if i % 1000 == 0:
                gc.collect()
            torch.cuda.empty_cache()
    all_image_features = torch.concat(all_image_features)
    if use_attention_mask_sampling:
        return all_remove_patches, all_image_features
    return all_image_features

def predict(text,images,tokenized_words,pos_tagging,valid_pos, text_model, image_model,text_processor,image_processor,use_attention_mask_sampling_text,use_attention_mask_sampling_image, clipscale):
    print("... Extract Images")
    image_feats = extract_all_images(images, image_model,image_processor=image_processor, device="cuda", batch_size=64, num_workers=1,use_attention_mask_sampling=use_attention_mask_sampling_image)
    
    print("... Extract Candidates")
    candidate_feats = extract_all_texts(text=text,tokenized_words=tokenized_words,pos_tagging=pos_tagging,valid_pos=valid_pos, model=text_model,text_processor=text_processor, batch_size=64, num_workers=1, use_attention_mask_sampling=use_attention_mask_sampling_text)
    
    if use_attention_mask_sampling_image:
        removed_patches, image_feats = image_feats
    else:
        image_feats = image_feats.unsqueeze(1)
    removed_ids, candidate_feats = candidate_feats

    print("==== Scores ====")
    clip_scores_image_text = get_clip_score(image_feats, candidate_feats, clipscale)
    if use_attention_mask_sampling_text | use_attention_mask_sampling_image:
        clipscore_samples = clip_scores_image_text[:,1:,:]
        clipscore = clip_scores_image_text[:,0,:]
        if use_attention_mask_sampling_image:
            return clipscore, clipscore_samples, removed_ids, removed_patches
        else:
            return clipscore, clipscore_samples, removed_ids
    else:
        return clip_scores_image_text


def agregate_scores_into_predictions(tokenized_words,removed_ids,masked_clipscores,original_clipscore):
    sampled_scores = []
    for word_tokens,r_ids, cs_m,cs_o in zip(tokenized_words,removed_ids,masked_clipscores,original_clipscore):
        cum_sum = [ [] for i in range(len(word_tokens))]
        for r, s in zip(r_ids,cs_m):
            for idx in r:
                cum_sum[idx].append(torch.mean((s-cs_o)/len(r)))

        for j,v in enumerate(word_tokens):
            if len(cum_sum[j]) >0:
                cum_sum[j] = np.mean(cum_sum[j])
            else:
                cum_sum[j] = -np.inf
                
        sampled_scores.append(torch.tensor(cum_sum))
    return sampled_scores

def get_valid_1hot_predictions_and_targets(text,images,labels,tokenized_words,pos_tagging,valid_pos,text_model,image_model,text_processor,image_processor, sampling_method,clipscale):
    clipscore, clipscore_samples, removed_ids = predict(text=text,images=images,tokenized_words=tokenized_words,pos_tagging=pos_tagging,valid_pos=valid_pos,use_attention_mask_sampling_text=True, use_attention_mask_sampling_image= sampling_method == "both",text_model = text_model,image_model=image_model,text_processor=text_processor,image_processor=image_processor,clipscale=clipscale)
    S = agregate_scores_into_predictions(tokenized_words,removed_ids,clipscore_samples,clipscore)
    remove_invalid_words = [np.array(item != -np.inf) for item in S]
    f_s = [nn.functional.sigmoid(x) for x in S]
    valid_f_s = []
    valid_labels = []
    
    for i,(s,L,m) in enumerate(zip(f_s,labels,remove_invalid_words)):
        valid_f_s.append(s[m])
        valid_labels.append(L[m])

    return f_s, valid_f_s, valid_labels

def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))

def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t())
    return (caption_loss + image_loss) / 2.0

def get_indices(element, lst):
    return [i for i, x in enumerate(lst) if x == element]

def get_clip_score(images_feats, candidates_feats, logit_scale):
    '''
    get standard image-text clipscore.
    images can either be:
    - a list of strings specifying filepaths for images
    - a precomputed, ordered matrix of image features
    '''
    #images_feats = torch.rand_like(images_feats)
    #candidates_feats = torch.rand_like(candidates_feats)
    images_feats2 = images_feats / images_feats.norm(p=2, dim=-1, keepdim=True)
    candidates_feats2 = candidates_feats / candidates_feats.norm(p=2, dim=-1, keepdim=True)
    all_results = contract("kab,kcb -> kac", candidates_feats2,images_feats2)
    all_results = torch.maximum(all_results*logit_scale,torch.tensor(0))
    return all_results

def scale_to_range(arr, a=0.5, b=1.5):
    # Find the min and max of the original array
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    
    # Apply the min-max scaling formula
    scaled_arr = a + ((arr - arr_min) * (b - a)) / (arr_max - arr_min)
    
    return scaled_arr

def non_linear_function(arr, a=1,p=1):
    return a*arr + arr**p

def load_model(model_name, config_sampler):
    if "calpt" in model_name:
        trust_remote_code=True
    else: 
        trust_remote_code = False
    # ==== Load Model & Processor ====
    print("==== Load Model & Processor ====")
    print("... Load Model {}".format(model_name))
    model = AutoModel.from_pretrained(model_name, trust_remote_code = trust_remote_code)  
    logit_scale = model.logit_scale.cpu().detach()
    print("... Load Processor")
    if "calpt" in model_name:
        if "ViT-B" in model_name:
            image_processor = CLIPImageProcessor.from_pretrained("laion/CLIP-ViT-B-32-laion2B-s34B-b79K")
            tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
        elif "ViT-H" in model_name:
            image_processor = CLIPImageProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
            tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")
    else:
        processor = AutoProcessor.from_pretrained(model_name)
        image_processor = processor.image_processor
        tokenizer = processor.tokenizer

    if config_sampler["use_attention_mask_sampling_text"]:
        model = model
        text_model = SampleWrapper(model=model,sampling_ratio=config_sampler["t_sampling_ratio"],n_samples=config_sampler["t_number_of_samples"],device=config_sampler["device"])
        text_model.eval()
        text_model.to(config_sampler["device"])
        if config_sampler["use_attention_mask_sampling_image"]:
            model = CLIPModelWithMask.from_pretrained(model_name)
            image_model = VisionSampleWrapper(model=model,sampling_ratio=config_sampler["i_sampling_ratio"],n_samples=config_sampler["i_number_of_samples"],device=config_sampler["device"])
            image_model.eval()
            image_model.to(config_sampler["device"])
        else:
            image_model = model
            image_model.eval()
            image_model.to(config_sampler["device"])

    return {"text_model":text_model,"image_model":image_model,"text_processor":tokenizer,"image_processor":image_processor,"logit_scale":logit_scale}

def plot_and_save_histograms(calib_data, test_data, valid_pos_tags, data_path):
    # Calculate lengths of candidates for calibration and test datasets
    calib_candidate_lengths = [len(candidate.split()) for candidate in calib_data["candidates"].to_list()]
    test_candidate_lengths = [len(candidate.split()) for candidate in test_data["candidates"].to_list()]

    # Calculate lengths of valid POS tags for calibration and test datasets
    calib_valid_pos_lengths = [
        sum(1 for pos in pos_tags if pos in valid_pos_tags) for pos_tags in calib_data["pos_tagging"].to_list()
    ]
    test_valid_pos_lengths = [
        sum(1 for pos in pos_tags if pos in valid_pos_tags) for pos_tags in test_data["pos_tagging"].to_list()
    ]

    # Plot and save histogram for calibration candidate lengths
    plt.figure(figsize=(10, 5))
    plt.hist(calib_candidate_lengths, bins=20, alpha=0.7, color='blue', label='Calibration Candidates')
    plt.title('Calibration Candidate Lengths Distribution')
    plt.xlabel('Length')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig(f"{data_path}/calib_candidate_lengths_histogram.png")
    plt.close()

    # Plot and save histogram for test candidate lengths
    plt.figure(figsize=(10, 5))
    plt.hist(test_candidate_lengths, bins=20, alpha=0.7, color='green', label='Test Candidates')
    plt.title('Test Candidate Lengths Distribution')
    plt.xlabel('Length')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig(f"{data_path}/test_candidate_lengths_histogram.png")
    plt.close()

    # Plot and save histogram for calibration valid POS lengths
    plt.figure(figsize=(10, 5))
    plt.hist(calib_valid_pos_lengths, bins=20, alpha=0.7, color='red', label='Calibration Valid POS')
    plt.title('Calibration Valid POS Lengths Distribution')
    plt.xlabel('Length')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig(f"{data_path}/calib_valid_pos_lengths_histogram.png")
    plt.close()

    # Plot and save histogram for test valid POS lengths
    plt.figure(figsize=(10, 5))
    plt.hist(test_valid_pos_lengths, bins=20, alpha=0.7, color='orange', label='Test Valid POS')
    plt.title('Test Valid POS Lengths Distribution')
    plt.xlabel('Length')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig(f"{data_path}/test_valid_pos_lengths_histogram.png")
    plt.close()

    print(f"Separate histograms saved in {data_path}")