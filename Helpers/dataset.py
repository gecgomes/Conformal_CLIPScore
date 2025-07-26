import joblib
import stanza
import json
import pandas as pd
from tqdm import tqdm 
import re
import os
import requests
from PIL import Image
import pickle
import torch
import numpy as np
from datasets import load_dataset
import random
def configure_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

configure_seed(888)



def load_flickr8k_extentions(input_json):
    data = {}
    with open(input_json) as f:
        data.update(json.load(f))
    print('Loaded {} images'.format(len(data)))

    new_pd = pd.DataFrame()
    images = []
    refs = []
    candidates = []
    human_scores = []
    ids = []
    for k, v in list(data.items()):
        for human_judgement in v['human_judgement']:
            if np.isnan(human_judgement['rating']):
                print('NaN')
                continue
            images.append("/"+v['image_path'])
            refs.append([' '.join(gt.split()) for gt in v['ground_truth']])
            candidates.append(' '.join(human_judgement['caption'].split()))
            human_scores.append(human_judgement['rating'])
            ids.append(k)
    new_pd["image_ids"] = ids
    new_pd["image_path"] = images
    new_pd["refs"] = refs
    new_pd["candidates"] = candidates
    new_pd["human_scores"] = human_scores
    return new_pd

def prepare_foilit_dataset(foil_path):
    print("... Prepare Foil ...")
    pos_tagger = stanza.Pipeline(processors='tokenize,mwt,pos', lang='en', tokenize_pretokenized=True)
    for name in ["test","train"]:
        if name == "train":
            coco_url = "http://images.cocodataset.org/train2014/"
        else:
            coco_url = "http://images.cocodataset.org/val2014/"
        with open(foil_path + "/foilv1.0_{}_2017.json".format(name), "r") as fl:
            data = json.load(fl)
        
        df_images = pd.DataFrame(data["images"])
        df_annotations = pd.DataFrame(data["annotations"])
        target_word = []
        label_1hot_list = []
        df_annotations.sort_values(["id","foil"], inplace=True)
        df_annotations["caption"] = df_annotations["caption"].apply(lambda caption: re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', '', caption.lower())).strip())
        remove_ids_from_typos = []
        for index in tqdm(df_annotations["id"].unique()):
            true_sentence = df_annotations[df_annotations["id"] == index]["caption"].iloc[0].split(" ")
            foil_sentence = df_annotations[df_annotations["id"] == index]["caption"].iloc[1].split(" ")
            if len(true_sentence) != len(foil_sentence):
                remove_ids_from_typos.append(index)
        print("REMOVE {} CAPTIONS FROM TYPOS".format(len(remove_ids_from_typos)))
        df_annotations = df_annotations[df_annotations["id"].apply(lambda x: x not in remove_ids_from_typos)]
        for index in tqdm(df_annotations["id"].unique()):
            true_sentence = df_annotations[df_annotations["id"] == index]["caption"].iloc[0].split(" ")
            foil_sentence = df_annotations[df_annotations["id"] == index]["caption"].iloc[1].split(" ")
            label_1hot = torch.zeros(len(true_sentence))
            label_1hot_list.append(torch.zeros(len(true_sentence)))
            for i,(t,f) in enumerate(zip(true_sentence,foil_sentence)):
                if t !=f:
                    label_1hot[i] = 1
                    target_word.append(t)
                    target_word.append(f)
                    break
            label_1hot_list.append(label_1hot)
        df_annotations["label_1hot"] = label_1hot_list
        df_annotations["target_word"] = target_word
        df_annotations.set_index("image_id",inplace = True)
        df_images.set_index("id",inplace=True)
        df_annotations["image_ids"] = df_images["file_name"]
        df_annotations["url"] = df_images["flickr_url"]
        df_annotations.reset_index(inplace=True,drop=True)
        df_annotations = df_annotations[["id","url","image_ids","caption","target_word","foil","label_1hot"]]
        df_annotations.columns = ["pair_ids","url","image_ids","candidates","target_word","foil","label_1hot"]
        df_annotations = df_annotations[df_annotations["target_word"].apply(lambda x: len(x) > 0)]
        df_annotations.sort_values(["pair_ids","foil"], inplace=True)
        candidates = df_annotations["candidates"].to_list()
        target_word = df_annotations["target_word"].to_list()
        token_pos = [ [(word.text,word.upos) for item in pos_tagger([sentence.split()]).sentences for word in item.words] for sentence in tqdm(candidates)]
        target_token_pos = [ [(word.text,word.upos) for item in pos_tagger([sentence.split()]).sentences for word in item.words] for sentence in tqdm(target_word)]
        
        tokenized_words = [ [x[0] for x in sentence_items] for sentence_items in token_pos]
        pos_tagging = [ [x[1] for x in sentence_items] for sentence_items in token_pos]
        df_annotations["tokenized_words"] = tokenized_words
        df_annotations["pos_tagging"] = pos_tagging

        tokenized_words = [ [x[0] for x in sentence_items] for sentence_items in target_token_pos]
        pos_tagging = [ [x[1] for x in sentence_items] for sentence_items in target_token_pos]
        df_annotations["target_tokenized_words"] = tokenized_words
        df_annotations["target_pos_tagging"] = pos_tagging
    
        df_annotations["image_path"] = foil_path +"/images/" + df_annotations["image_ids"]
        if not os.path.exists(foil_path + "/images"):
            os.makedirs(foil_path+ "/images")
        count = 0
        for coco_id in tqdm(df_annotations["image_ids"].unique()):
            url = coco_url + coco_id
            if not os.path.exists(foil_path+ "/images/" + coco_id):
                img_object = requests.get(url, stream=True).raw
                img = Image.open(img_object)
                img = img.convert("RGB")
                img.save(foil_path+ "/images/" + coco_id, 'JPEG')
            count += 1
        
        df_annotations["candidates"] = df_annotations["candidates"].str.lower()
        df_annotations = df_annotations.sample(frac=1)
        with open(foil_path + "/foil_{}.pkl".format(name),"wb") as f:
            pickle.dump(df_annotations,f)


def prepare_foilnocaps_dataset(foil_path):
    def generate_1hot(correct_sentence, foil_sentence, replacement):
        """
        Generates a one-hot encoded vector marking positions in the foil sentence
        where the correct word(s) were replaced by the wrong word(s).

        Args:
            correct_sentence (str): The original correct sentence.
            foil_sentence (str): The foil sentence with incorrect words.
            replacement (list): A list with two strings: [correct_word, wrong_word].

        Returns:
            np.ndarray: One-hot encoded vector.
        """
        correct_word, wrong_word = replacement  # Unpack replacement pair
        correct_words = correct_sentence.split(" ")
        foil_words = foil_sentence.split(" ")

        correct_tokens = correct_word.lower().split(" ")  # Handles multi-word replacements
        wrong_tokens = wrong_word.lower().split(" ")

        one_hot = torch.zeros(len(foil_words))  # Initialize one-hot vector

        i = 0
        while i <= len(foil_words) - len(wrong_tokens):
            # Check if a sequence in the foil matches the wrong word(s)
            if foil_words[i:i + len(wrong_tokens)] == wrong_tokens:
                # Check if the corresponding sequence in the correct sentence matches the correct word(s)
                if correct_words[i:i + len(correct_tokens)] == correct_tokens:
                    one_hot[i:i + len(wrong_tokens)] = 1  # Mark all wrong word positions
                    i += len(wrong_tokens)  # Skip the matched tokens
                    continue
            i += 1  # Move to the next word

        return one_hot
    print("... Prepare Foil ...")
    pos_tagger = stanza.Pipeline(processors='tokenize,mwt,pos', lang='en', tokenize_pretokenized=True)
    coco_url = "https://s3.amazonaws.com/nocaps/val/"
    nocaps_path = "/nocaps-val-foil.json"
    #['pair_ids', 'url', 'image_ids', 'candidates', 'target_word','foil_word', 'foil','label_1hot', 'tokenized_words', 'pos_tagging','foil_tokenized_word', 'foil_pos_tagging', 'image_path']
    new_data = pd.DataFrame()
    data = pd.read_json(foil_path + nocaps_path)
    
    data.columns = ["image_ids","refs","baseline","foil_sentence","replacement","domain"]
    data["baseline"] = data["baseline"].apply(lambda caption: re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', '', caption.lower())).strip())
    data["foil_sentence"] = data["foil_sentence"].apply(lambda caption: re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', '', caption.lower())).strip())
    foil_1hot = data.apply(lambda x: generate_1hot(x["baseline"],x["foil_sentence"],x["replacement"]), axis=1).to_list()
    correct_1hot = data.apply(lambda x: torch.zeros(len(x["baseline"].split(" "))),axis = 1).to_list()
    new_data["pair_ids"] = list(data.index) + list(data.index)
    new_data["image_ids"] = data["image_ids"].to_list() + data["image_ids"].to_list()
    new_data["candidates"] = data["baseline"].to_list() + data["foil_sentence"].to_list()
    new_data["label_1hot"] = correct_1hot + foil_1hot
    new_data["target_word"] =  data["replacement"].apply(lambda x: x[0]).to_list() + data["replacement"].apply(lambda x: x[1]).to_list()
    new_data["foil"] = [False]*len(correct_1hot) + [True]*len(foil_1hot)
    new_data["domain"] = data["domain"].to_list() + data["domain"].to_list()
    candidates = new_data["candidates"].to_list()
    target_word = new_data["target_word"].to_list()
    
    token_pos = [ [(word.text,word.upos) for item in pos_tagger([sentence.split()]).sentences for word in item.words] for sentence in tqdm(candidates)]
    target_token_pos = [ [(word.text,word.upos) for item in pos_tagger([sentence.split()]).sentences for word in item.words] for sentence in tqdm(target_word)]
    
    tokenized_words = [ [x[0] for x in sentence_items] for sentence_items in token_pos]
    pos_tagging = [ [x[1] for x in sentence_items] for sentence_items in token_pos]
    new_data["tokenized_words"] = tokenized_words
    new_data["pos_tagging"] = pos_tagging

    tokenized_words = [ [x[0] for x in sentence_items] for sentence_items in target_token_pos]
    pos_tagging = [ [x[1] for x in sentence_items] for sentence_items in target_token_pos]
    new_data["target_tokenized_words"] = tokenized_words
    new_data["target_pos_tagging"] = pos_tagging
    new_data["image_path"] = foil_path +"/images/" + new_data["image_ids"]
    if not os.path.exists(foil_path + "/images"):
        os.makedirs(foil_path+ "/images")
    count = 0
    for coco_id in tqdm(new_data["image_ids"].unique()):
        url = coco_url + coco_id
        if not os.path.exists(foil_path+ "/images/" + coco_id):
            img_object = requests.get(url, stream=True).raw
            img = Image.open(img_object)
            img = img.convert("RGB")
            img.save(foil_path+ "/images/" + coco_id, 'JPEG')
        count += 1
    
    new_data = new_data.sample(frac=1)
    with open(foil_path + "/foilnocaps.pkl","wb") as f:
        pickle.dump(new_data,f)

def prepare_general_dataset(path):
    def remove_punctuation_and_extra_spaces(text):
        # Regular expression pattern to match specified punctuation marks
        pattern = r'([.,!?\.\.\.])'
        # Substitute matched punctuation with spaces around it
        result = re.sub(pattern, r' ', text)
        # Removing extra spaces created by multiple matches (if any)
        result = re.sub(r'\s{2,}', ' ', result)
        result.strip()
        return result
    
    pos_tagger = stanza.Pipeline(processors='tokenize,mwt,pos', lang='en', tokenize_pretokenized=True)
    if "flickr" in path: 
        data = load_flickr8k_extentions(input_json=path)
        data = data.groupby(
            ['image_ids', 'candidates'], 
            as_index=False
        ).agg({
            'human_scores': list,  # Collect human_scores into a list
            'image_path': 'first',  # Keep the first occurrence of image_path
            'refs': 'first',  # Keep the first occurrence of refs
        })
        data.rename(columns={'human_scores': 'all_human_scores'}, inplace=True)
        data["mean_human_scores"] = data["all_human_scores"].apply(lambda x: np.mean(x))
    elif "Vicr" in path:
        data = joblib.load(path)
        data.rename(columns={'human_scores': 'all_human_scores'}, inplace=True)
        data = data.sample(frac=1)
    else:
        if "Composite" in path:
            data = joblib.load(path)
            data["image_ids"] = data["image_path"].apply(lambda x: x.split("/")[-1])
            data = data.groupby(
                ['image_ids', 'candidates'], 
                as_index=False
            ).agg({
                'human_scores': list,  # Collect human_scores into a list
                'image_path': 'first',  # Keep the first occurrence of image_path
                'refs': 'first',  # Keep the first occurrence of refs
            })
        elif "Polaris" in path:
            data = joblib.load(path)
            
        data.rename(columns={'human_scores': 'all_human_scores'}, inplace=True)
        data["mean_human_scores"] = data["all_human_scores"].apply(lambda x: np.mean(x))
    
    data["candidates"] = data["candidates"].apply(lambda x: remove_punctuation_and_extra_spaces(str(x)))
    data = data[data["candidates"].apply(lambda x: len(x) >= 1)] #remove empty strings
    candidates = data["candidates"].to_list()
    token_pos = [ [(word.text,word.upos) for item in pos_tagger(sentence).sentences for word in item.words] for sentence in tqdm(candidates)]
    tokenized_words = [ [x[0] for x in sentence_items] for sentence_items in token_pos]
    pos_tagging = [ [x[1] for x in sentence_items] for sentence_items in token_pos]
    data["tokenized_words"] = tokenized_words
    data["pos_tagging"] = pos_tagging

    with open((path.split(".")[0] +"_tokenized.pkl"),"wb") as f:
        pickle.dump(data,f)

def prepare_richhf_dataset(rich_path):
    def build_lookup_table(pickapic_subset):
        """Build a lookup table for fast UID matching."""
        lookup_table = {}
        for item in tqdm(pickapic_subset):
            lookup_table[item['image_0_uid']] = item
            lookup_table[item['image_1_uid']] = item
        return lookup_table

    def get_caption_and_image(lookup_table, uid):
        """Retrieve caption and image bytes for a given UID."""
        if uid in lookup_table:
            matching_item = lookup_table[uid]
            if matching_item['image_0_uid'] == uid:
                image_data = matching_item.get('jpg_0', '')
            elif matching_item['image_1_uid'] == uid:
                image_data = matching_item.get('jpg_1', '')
            else:
                return None, None

            caption = matching_item.get('caption', '')

            # Ensure image_data is bytes-like
            if isinstance(image_data, str):
                image_data = image_data.encode('utf-8')

            return caption, image_data

        return None, None

    def enrich_and_save_images(df, lookup_tables):
        """Enrich DataFrame with captions and save images locally."""
        images_path = os.path.join(rich_path, "images")
        os.makedirs(images_path, exist_ok=True)

        enriched_data = []
        processed_filenames = set()  # To track and skip duplicates

        for row in tqdm(df.itertuples(), total=len(df)):
            subset, uid = row.filename.split('/')

            if uid in processed_filenames:
                continue

            # Determine the appropriate lookup table
            lookup_table = lookup_tables.get(subset)
            if not lookup_table:
                continue

            caption, image_bytes = get_caption_and_image(lookup_table, uid)
            if caption is not None and image_bytes is not None:
                # Create subdirectory for the subset if it doesn't exist
                subset_path = os.path.join(images_path, subset)
                os.makedirs(subset_path, exist_ok=True)

                # Save the image
                image_path = os.path.join(subset_path, f"{uid}.jpg")
                with open(image_path, "wb") as f:
                    f.write(image_bytes)

                # Enrich the DataFrame row
                enriched_data.append({
                    'filename': f"{subset}/{uid}",
                    'caption': caption,
                    'image_path': image_path
                })

                processed_filenames.add(uid)

        return pd.DataFrame(enriched_data)

    def add_images_captions(rich_path):
        # Load the lightweight pickapic_v1_no_images dataset
        pickapic_no_images = load_dataset('yuvalkirstain/pickapic_v1', num_proc=64)

        # Build separate lookup tables for train, val, and test subsets
        lookup_tables = {
            'train': build_lookup_table(pickapic_no_images['train']),
            'val': build_lookup_table(pickapic_no_images['validation']),
            'test': build_lookup_table(pickapic_no_images['test'])
        }

        # Load the rich_hf DataFrame
        rich_hf = joblib.load(rich_path + "/rich_hf_complete.pkl")
        print(rich_hf)
        print(rich_hf.columns)
        rich_hf["filename"] = rich_hf["filename"].str.replace(".png","")
        # Enrich the DataFrame with data from pickapic_v1 and save images
        enriched_rich_hf = enrich_and_save_images(rich_hf, lookup_tables)

        # Merge enriched data back into the original rich_hf DataFrame
        enriched_rich_hf = enriched_rich_hf.set_index('filename')
        rich_hf = rich_hf.set_index('filename')
        rich_hf = rich_hf.join(enriched_rich_hf, how='left')
        rich_hf.reset_index(inplace=True,drop=True)
        
        with open(rich_path + "/rich_hf_with_images.pkl","wb") as f:
            pickle.dump(rich_hf,f)

    def remove_punctuation_and_extra_spaces(text):
        # Regular expression pattern to match specified punctuation marks
        pattern = r'([.,!?":;\.\.\.])'
        # Substitute matched punctuation with spaces around it
        result = re.sub(pattern, r' ', text)
        # Removing extra spaces created by multiple matches (if any)
        result = re.sub(r'\s{2,}', ' ', result)
        result.strip()
        result.lower()
        return result

    def flip_binary_string(s):
        # Flip binary values in a space-separated string
        binary_list = torch.tensor([int(i) for i in s.split()])
        return 1 - binary_list

    def match_token_label_to_word(token_label, caption):
        # Split caption into tokens based on punctuation and spaces
        delimiters = ',.?!":; '
        pattern = '|'.join(map(re.escape, delimiters))
        tokens = re.split(pattern, caption)
        tokens = [t for t in tokens if t]  # Filter out empty tokens

        if len(tokens) != len(token_label):
            return []
        
        if sum(token_label) == 0:
            return ["<orig>"]
        
        return [tokens[i] for i, lbl in enumerate(token_label) if lbl == 1]

    if not os.path.isfile(rich_path + "/rich_hf_with_images.pkl"):
        add_images_captions(rich_path)

    pos_tagger = stanza.Pipeline(processors='tokenize,mwt,pos', lang='en', tokenize_pretokenized=True)

    rich_hf = joblib.load(rich_path + "/rich_hf_with_images.pkl")
    new_rich_hf = pd.DataFrame()

    # Prepare columns
    new_rich_hf["image_path"] = rich_hf["image_path"].to_list()
    rich_hf["caption"] = rich_hf["caption"].apply(lambda x: remove_punctuation_and_extra_spaces(str(x))).to_list()
    new_rich_hf["candidates"] = rich_hf["caption"].to_list()
    rich_hf["label_1hot"] = rich_hf["token_label"].apply(flip_binary_string).to_list()
    new_rich_hf["label_1hot"] = rich_hf["label_1hot"].to_list()
    new_rich_hf["target_words"] = rich_hf.apply(
        lambda row: match_token_label_to_word(
            token_label=row["label_1hot"], caption=row["caption"]
        ),
        axis=1
    ).to_list()
    new_rich_hf["foil"] = new_rich_hf["label_1hot"].apply(lambda x: any(x == 1)).to_list()
    new_rich_hf["subset"] = rich_hf["subset"].to_list()
    # Remove instances where "target_words" is an empty list
    typo_count = len(new_rich_hf[new_rich_hf["target_words"].apply(len) == 0])
    print(f"REMOVE {typo_count} INSTANCES WITH TYPOS")
    new_rich_hf = new_rich_hf[new_rich_hf["target_words"].apply(len) > 0]

    token_pos = [ [(word.text,word.upos) for item in pos_tagger([sentence.split()]).sentences for word in item.words] for sentence in tqdm(new_rich_hf["candidates"].to_list())]
    target_token_pos = [ [(word.text,word.upos) for item in pos_tagger([sentence]).sentences for word in item.words] for sentence in tqdm(new_rich_hf["target_words"].to_list())]
    
    tokenized_words = [ [x[0] for x in sentence_items] for sentence_items in token_pos]
    pos_tagging = [ [x[1] for x in sentence_items] for sentence_items in token_pos]
    new_rich_hf["tokenized_words"] = tokenized_words
    new_rich_hf["pos_tagging"] = pos_tagging

    tokenized_words = [ [x[0] for x in sentence_items] for sentence_items in target_token_pos]
    pos_tagging = [ [x[1] for x in sentence_items] for sentence_items in target_token_pos]
    new_rich_hf["target_tokenized_words"] = tokenized_words
    new_rich_hf["target_pos_tagging"] = pos_tagging
    # Print results
    with open(rich_path + "/richhf_train.pkl","wb") as f:
        pickle.dump(new_rich_hf[new_rich_hf["subset"] =="train"],f)
    with open(rich_path + "/richhf_calib.pkl","wb") as f:
        pickle.dump(new_rich_hf[new_rich_hf["subset"] =="validation"],f)
    with open(rich_path + "/richhf_test.pkl","wb") as f:
        pickle.dump(new_rich_hf[new_rich_hf["subset"] =="test"],f)

def custom_load_dataset(data_path, config_dataset, return_type = "cal"):
    if ("Foil-it" in data_path):
        if not(os.path.isfile(data_path + "/foil_train.pkl") & os.path.isfile(data_path + "/foil_test.pkl")):
            prepare_foilit_dataset(data_path)
        if return_type == "cal":
            data = joblib.load(data_path  +"/foil_train.pkl")
        else:
            data = joblib.load(data_path  + "/foil_test.pkl")
        #test_data = test_data.iloc[:5000]
        data["image_path"] = data["image_ids"].apply(lambda x: data_path + "/images/" +x)
        data.columns = ['pair_ids', 'url', 'image_ids', 'candidates', 'target_word', 'foil','label_1hot', 'tokenized_words', 'pos_tagging','target_tokenized_word', 'target_pos_tagging', 'image_path']
        valid_pos = config_dataset["valid_pos_tags"]
        data = data[data["pos_tagging"].apply(lambda x: len(set(x).intersection(set(valid_pos))) > 0)]
        if return_type == "cal":
            data = data.sample(frac=0.1)
            return data, valid_pos
        else:
            return data
    
    elif ("Foil-nocaps" in data_path):
        
        foil_path = data_path.replace("Foil-nocaps","Foil-it")
        
        if not(os.path.isfile(data_path + "/foilnocaps.pkl")):
            prepare_foilnocaps_dataset(data_path)
        if not(os.path.isfile(foil_path + "/foil_train.pkl")):
            prepare_foilit_dataset(foil_path)

        if return_type == "cal":
            data = joblib.load(data_path  +"/foil_train.pkl")
        else:
            data = joblib.load(data_path  +"/foilnocaps.pkl")

        valid_pos = config_dataset["valid_pos_tags"]
        data = data[data["pos_tagging"].apply(lambda x: len(set(x).intersection(set(valid_pos))) > 0)]

        if return_type == "cal":
            data = data.sample(frac=0.1)
            return data, valid_pos
        else:
            return data
    
    elif "Vicr" in data_path:
        if not(os.path.isfile(data_path + "/VICR_tokenized.pkl")):
            prepare_general_dataset(data_path + "/VICR.pkl")
        dataset = joblib.load(data_path + "/VICR_tokenized.pkl")
        image_dir = "/".join(data_path.split("/")[:-1])

        dataset["image_path"] = dataset["image_path"].apply(lambda x: image_dir + "/" + x)
        dataset["mean_human_scores"] = dataset["all_human_scores"].apply(lambda x: np.mean(x))
        dataset["median_human_scores"] = dataset["all_human_scores"].apply(lambda x: np.median(x))
        dataset["var_human_scores"] = dataset["all_human_scores"].apply(lambda x: np.var(x))
        valid_pos = config_dataset["valid_pos_tags"]
        test_data = dataset[dataset["subset"] == "test"]
        val_data = dataset[dataset["subset"] == "val"]
        test_data = test_data[test_data["pos_tagging"].apply(lambda x: len(set(x).intersection(set(valid_pos))) > 0)]
        val_data = val_data[val_data["pos_tagging"].apply(lambda x: len(set(x).intersection(set(valid_pos))) > 0)]
        if return_type == "cal":
            return val_data, valid_pos
        else:
            return test_data
    
    elif "Polaris" in data_path:
        image_dir = "/".join(data_path.split("/")[:-1])
        if not(os.path.isfile(data_path + "/Polaris_tokenized.pkl")):
            prepare_general_dataset(data_path + "/Polaris.pkl")
        data = joblib.load(data_path + "/Polaris_tokenized.pkl")
        print(data)
        print(data.columns)
        test_data = data[data["subset"] == "test"]
        val_data = data[data["subset"] == "val"]
        test_data["image_path"] = test_data["image_path"].apply(lambda x: image_dir + x)
        val_data["image_path"] = val_data["image_path"].apply(lambda x: image_dir + x)
        if return_type == "cal":
            valid_pos = config_dataset["valid_pos_tags"]
            return val_data, valid_pos
        else:
            return test_data

    elif "flickr" in data_path:
        image_dir = "/".join(data_path.split("/")[:-1])
        if not(os.path.isfile(data_path.split(".")[0]+"_tokenized.pkl")):
            prepare_general_dataset(data_path)
        data = joblib.load(data_path.split(".")[0]+"_tokenized.pkl")
        data["image_path"] = data["image_path"].apply(lambda x: image_dir + x)
        return data
        
    elif "Composite" in data_path:
        image_dir = "/cfs/home/u021414/PhD/data/"
        if not(os.path.isfile(data_path.split(".")[0]+"_tokenized.pkl")):
            prepare_general_dataset(data_path)
        data = joblib.load(data_path.split(".")[0]+"_tokenized.pkl")
        data["image_path"] = data["image_path"].apply(lambda x: image_dir + "/".join(x.split("/")[-3:]))
        return data

    elif "Rich-hf" in data_path:
        if not(os.path.isfile(data_path + "/richhf_calib.pkl") & os.path.isfile(data_path + "/richhf_test.pkl")):
            prepare_richhf_dataset(data_path)

        train_data = joblib.load(data_path  +"/richhf_train.pkl")
        val_data = joblib.load(data_path  +"/richhf_calib.pkl")
        test_data = joblib.load(data_path  + "/richhf_test.pkl")

        valid_pos = config_dataset["valid_pos_tags"]
        train_data = train_data[train_data["pos_tagging"].apply(lambda x: len(set(x).intersection(set(valid_pos))) > 0)]
        test_data = test_data[test_data["pos_tagging"].apply(lambda x: len(set(x).intersection(set(valid_pos))) > 0)]
        val_data = val_data[val_data["pos_tagging"].apply(lambda x: len(set(x).intersection(set(valid_pos))) > 0)]
        if return_type == "cal":
            return val_data, valid_pos
        else:
            return test_data