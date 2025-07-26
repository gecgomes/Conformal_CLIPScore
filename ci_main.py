import argparse
import torch
import random
import numpy as np
from Helpers.metrics import Upearson_risk, print_metrics
from Helpers.risk_control import conformal_risk_control_using_LTT
from Helpers.ci import process_truncated_gaussian, process_truncated_gaussian_mean, process_truncated_gaussian_std
from Helpers.dataset import custom_load_dataset
from Helpers.clipscore import load_model, predict
import os
import pickle
import joblib
def configure_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

def argument_parser():
    # Create the argument parser
    parser = argparse.ArgumentParser(
        description="A script that demonstrates the use of command-line arguments."
    )

    # Define command-line arguments
    parser.add_argument(
        '-data_path', '--data_path', type=str, default="data", required=False, help='Select the dataset to be used: "Foil-it" "Foil-nocaps" "Rich-HF"'
    )
    parser.add_argument(
        '-data', '--data', type=str,choices=['Vicr','Ex-8k','Polaris','Composite','Whoops','DistCOCO-train-2014','DistCOCO-val-2014','DistCOCO-train-2017','DistCOCO-val-2017'], default="Vicr", required=False, help='Select the dataset to be used: "Vicr"'
    )
    parser.add_argument(
        '-cal_data', '--cal_data', type=str,choices=['Vicr'], default="Vicr", required=False, help='Select the dataset to be used: "Vicr"'
    )
    parser.add_argument(
        '-ms', '--model_size', type=str,choices=['B', 'H'], default="B", required=False, help='Select the model size: "B" or "H".'
    )
    parser.add_argument(
        '-sm', '--sampling_method', type=str, choices=['text', 'both'], default="both", required=False, help='Select the sampling method: "text" or "both".'
    )
    parser.add_argument(
        '-tsr', '--t_sampling_ratio', type=float, default=0.1, required=False, help='Choose the text encoder sampling ratio. Default is set to 10% (0.1)'
    )
    parser.add_argument(
        '-tns', '--t_number_of_samples', type=int, default=100, required=False, help='Choose the text encoder number of samples. Default is set to 100'
    )
    parser.add_argument(
        '-isr', '--i_sampling_ratio', type=float, default=0.3, required=False, help='Choose the image encoder sampling ratio. Default is set to 10% (0.1)'
    )
    parser.add_argument(
        '-ins', '--i_number_of_samples', type=int, default=100, required=False, help='Choose the image encoder number of samples. Default is set to 100'
    )
    parser.add_argument(
        '-d', '--delta', type=float, default=0.05, required=False, help='Choose the error rate values delta. Default is set to 5% (0.05)'
    )
    parser.add_argument(
        '-vpt', '--valid_pos_tags', type=str,nargs="+", default=["VERB","NOUN","ADJ","ADV","PROPN","NUM"],required=False, help='Threshold to choose the valid pos tags. Default: ["VERB","NOUN","ADJ","ADV","PROPN","NUM"].'
    )
    parser.add_argument(
        '-cal_cache', '--cal_cache', default=False, type=bool,required=False, help='The cache argument is False by default. If True, the folder cache must not be empty.' 
    )
    parser.add_argument(
        '-cache', '--cache', default=False, type=bool,required=False, help='The cache argument is False by default. If True, the folder cache must not be empty.' 
    )
    parser.add_argument(
        '-save', '--save', default=False, type=bool,required=False, help='The save argument is False by default. If True, the results will be saved in cache.' 
    )


    # Parse the arguments
    args = parser.parse_args()

    return args

def main():
    # Parse the arguments
    configure_seed(50)
    device = torch.device("cuda")
    args = argument_parser()

    if not os.path.exists("cache"):
        os.makedirs("cache")

    if args.model_size == "B":
        model_name = "calpt/CLIP-ViT-B-32-xlm-roberta-base-laion5B-s13B-b90k"
    else:
        model_name = "calpt/CLIP-ViT-H-14-frozen-xlm-roberta-large-laion5B-s13B-b90k"
    config_sampler = {"use_attention_mask_sampling_text": True, "use_attention_mask_sampling_image": args.sampling_method == "both","i_sampling_ratio": args.i_sampling_ratio,"i_number_of_samples": args.i_number_of_samples,"t_sampling_ratio": args.t_sampling_ratio,"t_number_of_samples": args.t_number_of_samples, "device": device}
    print("=== Loading Model: {}".format(model_name))
    model_loader = load_model(model_name=model_name,config_sampler=config_sampler)
    text_model = model_loader["text_model"]
    text_processor = model_loader["text_processor"]
    image_model = model_loader["image_model"]
    image_processor = model_loader["image_processor"]
    clipscale = model_loader["logit_scale"]
    print("Model Loaded!")

    print("=== Loading Calibration Dataset: {}".format(args.cal_data))
    print("=== Loading Test Dataset: {}".format(args.data))

    cal_path = args.data_path + "/Vicr"
    if not os.path.exists(cal_path):
        os.makedirs(cal_path)
    
    config_dataset = {"valid_pos_tags": args.valid_pos_tags,"sampling_method":args.sampling_method}
    
    if args.data == "Vicr":
        data_dir = args.data_path + "/Vicr"
        data_path = data_dir
    elif args.data == "Ex-8k":
        data_dir = args.data_path + "/flickr8k"
        data_path = data_dir + "/flickr8k.json"
    elif args.data == "Polaris":
        data_dir = args.data_path + "/Polaris"
        data_path = data_dir
    elif args.data == "Composite":
        data_dir = args.data_path + "/Composite"
        data_path = data_dir + "/Composite.pkl"
    else:
        raise ValueError(f"Unsupported dataset '{args.data}'")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    test_data = custom_load_dataset(data_path=data_path,config_dataset = config_dataset,return_type ="test")

    cal_data, valid_pos = custom_load_dataset(cal_path, config_dataset)
    
    #Calib Set
    cal_image_path = cal_data["image_path"].to_list()
    cal_candidates = cal_data["candidates"].to_list()
    cal_tokenized_words = cal_data["tokenized_words"].to_list()
    cal_pos_tagging = cal_data["pos_tagging"].to_list()

    #Test Set
    test_image_path = test_data["image_path"].to_list()
    test_candidates = test_data["candidates"].to_list()
    test_tokenized_words = test_data["tokenized_words"].to_list()
    test_pos_tagging = test_data["pos_tagging"].to_list()
    

    print("Number of instances:", len(test_data))
    print("Number of unique images:", len(test_data["image_path"].unique()))

    #Calib predict
    if args.cal_cache:
        cal_data = joblib.load("cache/{}_calib_cache_data_{}_{}_{}.pkl".format(args.cal_data,args.sampling_method, model_name.split("/")[-1],valid_pos)) 
    else:
        clipscore, clipscore_samples, removed_ids, _ = predict(text=cal_candidates,images=cal_image_path,tokenized_words=cal_tokenized_words,pos_tagging=cal_pos_tagging,valid_pos=valid_pos,use_attention_mask_sampling_text=True, use_attention_mask_sampling_image= args.sampling_method == "both",text_model = text_model,image_model=image_model,text_processor=text_processor,image_processor=image_processor,clipscale=clipscale)
        clipscore_samples = clipscore_samples.numpy()
        cal_data["clipscore"] = [np.array(row) for row in clipscore_samples]
        cal_data["model_mean_values"] = [np.mean(row) for row in clipscore_samples]
        cal_data["model_var_values"] = [np.var(row) for row in clipscore_samples]
        cal_data["model_std_values"] = [np.std(row) for row in clipscore_samples]
        if args.save:
            with open("cache/{}_calib_cache_data_{}_{}_{}.pkl".format(args.cal_data,args.sampling_method, model_name.split("/")[-1],valid_pos),"wb") as f:
                pickle.dump(cal_data,f)
    error = 1.8

    lower_bound = cal_data["model_mean_values"].min()/(error) #0
    upper_bound = cal_data["model_mean_values"].max()*(error) 

    
    alpha = Upearson_risk(prediction_mean=process_truncated_gaussian_mean(cal_data,1,lower_bound=lower_bound,upper_bound=upper_bound),prediction_std=process_truncated_gaussian_std(cal_data,1,lower_bound=lower_bound,upper_bound=upper_bound),Y=cal_data["mean_human_scores"])

    #Risk Control
    risks, lambdas_set, pvalues = conformal_risk_control_using_LTT(prediction_data=cal_data,Y=cal_data["mean_human_scores"],risk_fn=Upearson_risk,lower_bound=lower_bound,upper_bound=upper_bound, alpha=alpha, delta=args.delta,n=len(cal_data),N=50, start_lambda=1,stop_lambda=50)
    #Select lambda with the lowest p-value
    l_star = lambdas_set[np.argmin(risks)]
    pv_star = pvalues[np.argmin(risks)]
    print("CALIB")
    for L,pv in zip([1,l_star],[np.nan,pv_star]):
        aux_df = process_truncated_gaussian(cal_data,L,lower_bound=lower_bound,upper_bound=upper_bound)
        print("====")
        print("Lambda {}: P-value: {} ".format(L,pv))
        print_metrics(human_scores=cal_data["mean_human_scores"].to_numpy(), system_scores=aux_df["mean"].to_numpy(),system_std=aux_df["std"].to_numpy())
        print("====")
   
    else:
        _, clipscore_samples, _, _ = predict(text=test_candidates,images=test_image_path,tokenized_words=test_tokenized_words,pos_tagging=test_pos_tagging,valid_pos=valid_pos,use_attention_mask_sampling_text=True, use_attention_mask_sampling_image= args.sampling_method == "both",text_model = text_model,image_model=image_model,text_processor=text_processor,image_processor=image_processor,clipscale=clipscale)

        clipscore_samples = clipscore_samples.numpy()
        test_data["clipscore"] = [np.array(row) for row in clipscore_samples]
        test_data["model_mean_values"] = [np.mean(row) for row in clipscore_samples]
        test_data["model_var_values"] = [np.var(row) for row in clipscore_samples]
        test_data["model_std_values"] = [np.std(row) for row in clipscore_samples]
        aux_df = process_truncated_gaussian(test_data,l_star,lower_bound=lower_bound,upper_bound=upper_bound)
        if args.save:
            with open("cache/{}_test_cache_data_{}_{}_{}.pkl".format(args.data,args.sampling_method, model_name.split("/")[-1],valid_pos),"wb") as f:
                pickle.dump(test_data,f)
            with open("cache/{}_test_scores_data_{}_{}_{}.pkl".format(args.data,args.sampling_method, model_name.split("/")[-1],valid_pos),"wb") as f:
                pickle.dump(clipscore_samples,f)
    
    print("TEST")
    for L,pv in zip([1,l_star],[np.nan,pv_star]):
        print("====")
        print("Lambda {}: P-value: {} ".format(L,pv))
        print_metrics(human_scores=test_data["mean_human_scores"].to_numpy(), system_scores=aux_df["mean"].to_numpy(),system_std=aux_df["std"].to_numpy())
        print("====")

if __name__ == "__main__":
    main()