import argparse
import torch
import random
import numpy as np
from Helpers.dataset import custom_load_dataset
from Helpers.clipscore import load_model, get_valid_1hot_predictions_and_targets
from Helpers.metrics import false_discovery_rate, false_positive_rate, false_negative_rate,foil_accuracy,location_accuracy,location_precision,location_recall,location_f1 ,location_accuracy_foil_only, top_location_accuracy_foil_only
from Helpers.risk_control import conformal_risk_control_using_concentration_inequalities
from sklearn.metrics import recall_score, precision_score, f1_score, average_precision_score
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
        '-data_path', '--data_path', type=str, default="/cfs/home/u021414/PhD/ConformalFoil/test_data", required=False, help='Select the dataset to be used: "Foil-it" "Foil-nocaps" "Rich-HF"'
    )
    parser.add_argument(
        '-data', '--data', type=str,choices=['Foil-it','Foil-nocaps', 'Rich-HF'], default="Foil-it", required=False, help='Select the dataset to be used: "Foil-it" "Foil-nocaps" "Rich-HF"'
    )
    parser.add_argument(
        '-cal_data', '--cal_data', type=str,choices=['Foil-it', 'Rich-HF'], default="Foil-it", required=False, help='Select the dataset to be used: "Foil-it" "Foil-nocaps" "Rich-HF"'
    )
    parser.add_argument(
        '-ms', '--model_size', type=str,choices=['B', 'H'], default="B", required=False, help='Select the model size: "B" or "H".'
    )
    parser.add_argument(
        '-sm', '--sampling_method', type=str, choices=['text', 'both'], default="text", required=False, help='Select the sampling method: "text" or "both".'
    )
    parser.add_argument(
        '-r', '--risk_function', type=str, choices=['FDR', 'FPR'], default="FDR", required=False, help='Select the risk function: False Discovery Rate ("FDR") or False Positive Rate ("FPR").'
    )
    parser.add_argument(
        '-tsr', '--t_sampling_ratio', type=float, default=0.1, required=False, help='Choose the text encoder sampling ratio. Default is set to 10% (0.1)'
    )
    parser.add_argument(
        '-tns', '--t_number_of_samples', type=int, default=100, required=False, help='Choose the text encoder number of samples. Default is set to 100'
    )
    parser.add_argument(
        '-isr', '--i_sampling_ratio', type=float, default=0.05, required=False, help='Choose the image encoder sampling ratio. Default is set to 10% (0.1)'
    )
    parser.add_argument(
        '-ins', '--i_number_of_samples', type=int, default=100, required=False, help='Choose the image encoder number of samples. Default is set to 100'
    )
    parser.add_argument(
        '-d', '--delta', type=float, default=0.05, required=False, help='Choose the error rate values delta. Default is set to 5% (0.05)'
    )
    parser.add_argument(
        '-a', '--alpha', type=float, nargs='+', default=[0.2], required=False, help='Choose the risk tolerance values alpha. Default is set to 15% [0.15]. Note: this argument is a list of floats.'
    )
    parser.add_argument(
        '-c','--calibration', action='store_true', help='The calibration argument is True by default. If False, the lambda argument value is required.' 
    )
    parser.add_argument(
        '-l', '--lambda', type=float, required=False, help='Lambda is the threshold probability value. Required if calibration is set to False.'
    )
    parser.add_argument(
        '-vpt', '--valid_pos_tags', type=str,nargs="+", default=["VERB","NOUN","ADJ","ADV","PROPN","NUM"],required=False, help='Threshold to choose the valid pos tags. Default: ["VERB","NOUN","ADJ","ADV","PROPN","NUM"].'
    )
    parser.add_argument(
        '-cache', '--cache', default=False, type=bool,required=False, help='The cache argument is False by default. If True, the folder cache must not be empty.' 
    )
    parser.add_argument(
        '-cal_cache', '--cal_cache', default=False, type=bool,required=False, help='The cache argument is False by default. If True, the folder cache must not be empty.' 
    )
    parser.add_argument(
        '-save', '--save', default=False, type=bool,required=False, help='The save argument is False by default. If True, the results will be saved in cache.' 
    )


    # Parse the arguments
    args = parser.parse_args()

    return args

def main():
    # Parse the arguments
    #configure_seed(888)
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

    print("=== Loading Dataset: {}".format(args.data))
    if args.cal_data == "Foil-it":
        cal_data_path = args.data_path + "/Foil-it"
    elif args.cal_data == "Rich-HF":
        cal_data_path = args.data_path + "/Rich-hf"

    if not os.path.exists(cal_data_path):
        os.makedirs(cal_data_path)


    if args.data == "Foil-it":
        data_path = args.data_path + "/Foil-it"
    elif args.data == "Foil-nocaps":
        data_path = args.data_path + "/Foil-nocaps"
    elif args.data == "Rich-HF":
        data_path = args.data_path + "/Rich-hf"

    if not os.path.exists(data_path):
        os.makedirs(data_path)
    config_dataset = {"valid_pos_tags": args.valid_pos_tags}
    
    test_data = custom_load_dataset(data_path, config_dataset,return_type="test")
    calib_data, valid_pos = custom_load_dataset(cal_data_path, config_dataset)
    

    #Calibration Set
    calib_candidates = calib_data["candidates"].to_list()
    calib_image_path = calib_data["image_path"].to_list()
    calib_tokenized_words = calib_data["tokenized_words"].to_list()
    calib_pos_tagging = calib_data["pos_tagging"].to_list()
    calib_labels = calib_data["label_1hot"].to_list()
    #Test Set 
    test_candidates = test_data["candidates"].to_list()
    test_image_path = test_data["image_path"].to_list()
    test_tokenized_words = test_data["tokenized_words"].to_list()
    test_pos_tagging = test_data["pos_tagging"].to_list()
    test_labels = test_data["label_1hot"].to_list()

    if args.data == "Foil-nocaps":
        test_domain = test_data["domain"]
        indom_index = test_domain.apply(lambda x: x == "in-domain").to_list()
        indom_index = [i for i, val in enumerate(indom_index) if val]
        neardom_index = test_domain.apply(lambda x: x == "near-domain").to_list()
        neardom_index = [i for i, val in enumerate(neardom_index) if val]
        outdom_index = test_domain.apply(lambda x: x == "out-domain").to_list()
        outdom_index = [i for i, val in enumerate(outdom_index) if val]
    
    if args.risk_function == "FPR":
        risk_fn = false_positive_rate
    else:  
        risk_fn = false_discovery_rate
    
    #Calibration

    if args.cal_cache:
        calib_cache_data = joblib.load("cache/{}_calib_cache_data_{}_{}_{}.pkl".format(args.cal_data,args.sampling_method, model_name.split("/")[-1],valid_pos)) 
    else:
        calib_cache_data = get_valid_1hot_predictions_and_targets(text=calib_candidates,images=calib_image_path,labels = calib_labels,tokenized_words=calib_tokenized_words,pos_tagging=calib_pos_tagging,valid_pos=valid_pos,sampling_method = args.sampling_method,text_model=text_model,image_model=image_model,text_processor=text_processor, image_processor=image_processor,clipscale = clipscale)
        if args.save:
            with open("cache/{}_calib_cache_data_{}_{}_{}.pkl".format(args.cal_data,args.sampling_method, model_name.split("/")[-1],valid_pos),"wb") as f:
                pickle.dump(calib_cache_data,f)
    calib_f_s, calib_valid_f_s, calib_valid_1hot_labels = calib_cache_data


    lam_list = []
    calib_n = len(calib_valid_f_s) 
    for tolerance in args.alpha:
        print("Calibrate to risk tolerance: {} with an error rate of: {}".format(tolerance,args.delta))
        lamhat = conformal_risk_control_using_concentration_inequalities(prediction_set=calib_valid_f_s,Y=calib_valid_1hot_labels,risk_fn=risk_fn,alpha=tolerance,delta=args.delta, n=calib_n)
        lam_list.append(lamhat)
    
    if args.cache:
        test_cache_data = joblib.load("cache/{}_test_cache_data_{}_{}_{}.pkl".format(args.data,args.sampling_method, model_name.split("/")[-1],valid_pos))
    else:
        test_cache_data = get_valid_1hot_predictions_and_targets(text=test_candidates,images=test_image_path,labels = test_labels,tokenized_words=test_tokenized_words,pos_tagging=test_pos_tagging,valid_pos=valid_pos,sampling_method = args.sampling_method,text_model=text_model,image_model=image_model,text_processor=text_processor, image_processor=image_processor,clipscale = clipscale)
        if args.save:
            with open("cache/{}_test_cache_data_{}_{}_{}.pkl".format(args.data,args.sampling_method, model_name.split("/")[-1],valid_pos),"wb") as f:
                pickle.dump(test_cache_data,f)
            
    test_f_s, test_valid_f_s, test_valid_1hot_labels = test_cache_data
    for risk,lamhat in zip(args.alpha,lam_list):
        print("CALIB: Percentage of Foils: {}, Percentage of Corrects: {}".format((sum([(i == 1).any() for i in calib_labels])/len(calib_labels)),(sum([(i == 0).all() for i in calib_labels])/len(calib_labels))))
        print("TEST: Percentage of Foils: {}, Percentage of Corrects: {}".format((sum([(i == 1).any() for i in test_labels])/len(test_labels)),(sum([(i == 0).all() for i in test_labels])/len(test_labels))))
        print("==== Results for RISK < {} ====".format(risk))
        print("LAMHAT: ", lamhat)
        
        fdr_list = []
        fpr_list = []
        acc_list = []
        ap_list = []
        prec_list = []
        rec_list = []
        f1_list = []
        
        for experiment_type,f_s,valid_fs, labels,valid_labels in zip(["calib","test"],[calib_f_s,test_f_s],[calib_valid_f_s,test_valid_f_s],[calib_labels,test_labels],[calib_valid_1hot_labels,test_valid_1hot_labels]):
            predictions = [item > lamhat for item in f_s]
            new_predictions = [item > lamhat for item in valid_fs]

            #Risk Metrics
            fdr = false_discovery_rate(new_predictions, valid_labels)
            fpr = false_positive_rate(new_predictions,valid_labels)
            fnr = false_negative_rate(new_predictions,valid_labels)

            #Instance Level
            acc, foil, correct = foil_accuracy(predictions, labels)
            ap = average_precision_score(
                y_true=np.array([(instance == 1).any() for instance in labels]),
                y_score=np.array([(instance == 1).any() for instance in predictions]), average="macro"
            )
            prec = precision_score(
                y_true=np.array([(instance == 1).any() for instance in labels]),
                y_pred=np.array([(instance == 1).any() for instance in predictions]), average="macro"
            )
            rec = recall_score(
                y_true=np.array([(instance == 1).any() for instance in labels]),
                y_pred=np.array([(instance == 1).any() for instance in predictions]), average="macro"
            )
            f1 = f1_score(
                y_true=np.array([(instance == 1).any() for instance in labels]),
                y_pred=np.array([(instance == 1).any() for instance in predictions]), average="macro"
            )

            #All Instances
            if "Foil" in args.data:
                la_set = location_accuracy(predictions,labels, k=1)
            else:
                la_set = location_accuracy(predictions,labels)
            lp_set = location_precision(predictions,labels)
            lr_set = location_recall(predictions,labels)
            lf1_set = location_f1(predictions,labels)

            print("=== {} ===".format(experiment_type.upper()))
            print("...Risk Metrics...")
            print("FDR: ",fdr)
            print("FPR: ",fpr)
            print("FNR: ",fnr)

            print("...Instance Level...")
            print("Instance ACC: {}, Correct: {}, Foil: {}".format(acc, correct, foil))
            print("Instance AP: {}, Prec: {}, Recall: {}, F1: {}".format(ap,prec, rec,f1))

            print("...All Instances...")
            print("LA-set: ", la_set)
            print("LP-set: ", lp_set)
            print("LR-set: ", lr_set)
            print("LF1-set: ", lf1_set)
            fdr_list.append(fdr)
            fpr_list.append(fpr)
            acc_list.append(acc)
            ap_list.append(ap)
            prec_list.append(prec)
            rec_list.append(rec)
            f1_list.append(f1)
            if "Foil" in args.data:
                #Foil Only
                la_set_only = location_accuracy_foil_only(predictions, labels, k=1)
                la_1object_only = top_location_accuracy_foil_only(f_s=f_s, labels=labels, lamhat=lamhat, k=1)
                print("...Foil Instance Only...")
                print("LA-set: ",la_set_only)
                print("LA-1object: ",la_1object_only)
            
            
        if (args.data == "Foil-nocaps"):
            print("Domain Results")
            for domain_type, domain_index in zip(["In Domain","Near Domain","Out Domain"],[indom_index,neardom_index,outdom_index]):
                print("... {}".format(domain_type))
                domain_test_f_s = [test_f_s[i] for i in domain_index]
                domain_test_valid_1hot_labels = [test_valid_1hot_labels[i] for i in domain_index]
                domain_predictions = [predictions[i] for i in domain_index]
                domain_new_predictions = [new_predictions[i] for i in domain_index]
                domain_test_labels = [test_labels[i] for i in domain_index]

                fdr = false_discovery_rate(domain_new_predictions, domain_test_valid_1hot_labels)
                fpr = false_positive_rate(domain_new_predictions,domain_test_valid_1hot_labels)
                fnr = false_negative_rate(domain_new_predictions,domain_test_valid_1hot_labels)
                acc, foil, correct = foil_accuracy(domain_predictions, domain_test_labels)
                ap = precision_score(
                    y_true=np.array([(instance == 1).any() for instance in domain_test_labels]),
                    y_pred=np.array([(instance == 1).any() for instance in domain_predictions])
                )
                la_set = location_accuracy_foil_only(domain_predictions, domain_test_labels, k=1)
                la_1object = top_location_accuracy_foil_only(f_s=domain_test_f_s, labels=domain_test_labels, lamhat=lamhat, k=1)
                
                print("FDR: ",fdr)
                print("FPR: ",fpr)
                print("FNR: ",fnr)
                print("Foil ACC: {}, Correct: {}, Foil: {}".format(acc, correct, foil))
                print("AP: ",ap)
                print("LA-set: ",la_set)
                print("LA-1object: ",la_1object)
        if "Foil" in args.data:
            print(f"{fdr_list[0]},{acc_list[0]},{ap_list[0]},{prec_list[0]},{rec_list[0]},{f1_list[0]},{fdr_list[1]},{acc_list[1]},{ap_list[1]},{prec_list[1]},{rec_list[1]},{f1_list[1]},{la_set_only},{la_1object_only}")
        else:
            print(f"{fpr_list[0]},{acc_list[0]},{ap_list[0]},{prec_list[0]},{rec_list[0]},{f1_list[0]},{fpr_list[1]},{acc_list[1]},{ap_list[1]},{prec_list[1]},{rec_list[1]},{f1_list[1]},{lp_set},{lr_set},{lf1_set}")
        # Perform qualitative analysis
        random_indices = random.sample(range(len(test_candidates)), 25)  # Select 10 random indices
        print(f"Selected Indices: {random_indices}\n")
        
        for idx in random_indices:
            image = test_image_path[idx]
            caption = test_candidates[idx]
            true_target_words = [word for word, label in zip(test_tokenized_words[idx], test_labels[idx]) if label == 1]
            predicted_words = [word for word, pred in zip(test_tokenized_words[idx], predictions[idx]) if pred == 1]
            
            print(f"Instance {idx + 1}:")
            print(f"  Image: {image}")
            print(f"  Caption: {caption}")
            print(f"  True Target Words: {true_target_words}")
            print(f"  Predicted Words: {predicted_words}\n")
if __name__ == "__main__":
    main()