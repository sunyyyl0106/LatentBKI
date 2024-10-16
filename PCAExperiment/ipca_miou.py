import os
import yaml
from ipca_utils import get_miou, get_accuracy
    
def main():
    config_file = os.path.join(os.getcwd(), "ipca.yaml")
    with open(config_file, "r") as stream:
        try:
            config = yaml.safe_load(stream)
            target_dimension = config["target_dimension"]
            val_seqs = config["split"]["val"] 
            labels = config["labels"]
            data_dir = config["mp3d_data_dir"]
            save_dir = config["save_dir"]
        except yaml.YAMLError as exc:
            print(exc)
    num_classes = len(labels)
    result_file_path = os.path.join(os.getcwd(), "predictions", target_dimension, 'result.txt')
    mean_iou = get_miou(num_classes, val_seqs, data_dir, save_dir, target_dimension)
    accuracy = get_accuracy(val_seqs, data_dir, save_dir, target_dimension)
    print(f"mean iou: {mean_iou}\n")
    print(f"accuracy: {accuracy}\n")
    with open(result_file_path, mode='a') as file:
        file.write(f"mean iou: {mean_iou}\n")
        file.write(f"accuracy: {accuracy}\n")
    
    

    
            


    
       


if __name__ == "__main__":
    main()
