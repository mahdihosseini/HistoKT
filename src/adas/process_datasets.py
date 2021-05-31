from preprocessing.transforms import ProcessImages, ProcessDatasets
from datasets import MHIST, BACH

def main():
    root = "../../.adas-data"
    dataset_list = [MHIST(root=root, split="Train")]
    dataset_processor = ProcessDatasets(dataset_list=dataset_list,)


if __name__ == "__main__":
    main()
