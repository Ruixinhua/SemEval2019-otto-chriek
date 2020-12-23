import argparse

import torch
import yaml
import os
from hparams import HParams
from models.title_body_head_att import TitleBodyHeadAtt
from utils.data_loader import convert_text_data, read_text

if __name__ == "__main__":
    yaml_file = "configs/title_body_head_att.yaml"
    config = yaml.full_load(open(yaml_file))
    parser = argparse.ArgumentParser()
    parser.add_argument("-A", default=None, type=str, required=True, help="Article XML file")
    parser.add_argument("-O", default=None, type=str, required=True, help="Output path")

    args = parser.parse_args()
    hparams = HParams(**config)
    model_name = config["model_name"]
    model_path = "checkpoint/test_model.ckpt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    content_dir = {"article": [], "title": [], "id": []}
    if os.path.isdir(args.A):
        for file in os.scandir(args.A):
            print(f"The {file.path} is processing")
            content_dir.update(read_text(file.path))
    else:
        print(f"The {args.A} is processing")
        content_dir.update(read_text(args.A))
    x_data, article_ids = convert_text_data(content_dir, config["article_size"], config["title_size"])

    x_data = torch.tensor(x_data, dtype=torch.long).to(device)
    model = TitleBodyHeadAtt.load_from_checkpoint(model_path, hparam=hparams).to(device)
    model.freeze()

    probs, predicted = model.inference(x_data)
    with open(args.O, "w") as w:
        for i, prob, pred in zip(article_ids, probs, predicted):
            w.write(f"{i} {pred} {prob[pred]} \n")
