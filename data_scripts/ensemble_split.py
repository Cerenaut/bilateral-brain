import glob
import random
import argparse
from pathlib import Path
random.seed(42)

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dir", type=str, 
                    default="/home/chandramouli/Documents/kaggle/CIFAR-100")
parser.add_argument("-m", "--mode", default="train", 
                    choices=["train", "val", "test"])

def main(args):
    src_dir = Path(args.dir)
    file_path = glob.glob(str(src_dir / args.mode / "*" / "*"))
    random.shuffle(file_path)
    total_len = len(file_path)
    split_len = int(total_len / 5)
    dest_dir = f"/home/chandramouli/kaggle/cerenaut/analysis/{args.mode}"
    with open(dest_dir + "_ensemble5_split1.txt", "w") as f:
        f.writelines("\n".join(file_path[split_len:]))
    with open(dest_dir + "_ensemble5_split2.txt", "w") as f:
        f.writelines("\n".join(file_path[:split_len] + file_path[2*split_len:]))
    with open(dest_dir + "_ensemble5_split3.txt", "w") as f:
        f.writelines("\n".join(file_path[:2*split_len] + file_path[3*split_len:]))
    with open(dest_dir + "_ensemble5_split4.txt", "w") as f:
        f.writelines("\n".join(file_path[:3*split_len] + file_path[4*split_len:]))
    with open(dest_dir + "_ensemble5_split5.txt", "w") as f:
        f.writelines("\n".join(file_path[:4*split_len]))

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)