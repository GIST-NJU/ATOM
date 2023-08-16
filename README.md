# ATOM
## <div id="0" >MICS to be tested</div>

| MICS | Dataset |
| :---: | :---: |
| [MSRN](https://github.com/chehao2628/MSRN) | VOC, COCO |
| [ML-GCN](https://github.com/megvii-research/ML-GCN) | VOC, COCO |
| [DSDL](https://github.com/ZFT-CQU/DSDL) | VOC, COCO |
| [ASL](https://github.com/Alibaba-MIIL/ASL) | VOC, COCO |
| [MCAR](https://github.com/gaobb/MCAR) | VOC |
| [MLDecoder](https://github.com/alibaba-miil/ml_decoder) | COCO |


## File Structure
```plain
.
├─data  # Data and results directory
│  ├─heat_map
│  ├─imgs
│  │  ├─default  # Default dataset from VOC2007 & COCO2014
│  │  │  ├─coco
│  │  │  └─voc
│  │  ├─google  # Images from google
│  │  │  ├─coco1
│  │  │  │  ├─airplane
│  │  │  │  │  ├─airplane-1.png
│  │  │  │  │  ├─airplane-2.png
│  │  │  │  │  └─airplane-3.png
│  │  │  │  ├─apple
│  │  │  │  ├─backpack
│  │  │  │  ├─......
│  │  │  │  └─zebra
│  │  │  ├─coco2
│  │  │  │  ├─airplane-apple
│  │  │  │  │  └─airplane-apple-1.png
│  │  │  │  ├─airplane-backpack
│  │  │  │  ├─......
│  │  │  │  └─zebra-wine glass
│  │  │  ├─voc1
│  │  │  └─voc2
│  │  └─ILSVRC2012_img_test_v10102019  # ImageNet2012 testset
│  │      └─test
│  │        ├─ILSVRC2012_test_00000001.JPEG
│  │        ├─......
│  │        └─ILSVRC2012_test_00100000.JPEG
│  ├─info  # Other information
│  │  ├─all_info
│  │  ├─annotation  # Annotation information by all people
│  │  ├─trainval_info  # Information related to VOC and COCO training and validation sets
│  │  └─vote  # Voting results of annotation information
│  └─results  # Results of the model
├─SearchCode  # Code for obtaining images from google
│  ├─coco_synonyms.txt  # COCO labels' synonyms
│  ├─......
│  └─voc_synonyms.txt  # VOC labels' synonyms
├─rqs  # Code for calculating the information required for RQ1-RQ3
└─tests  # Code for inputting images into the model and saving the test results
    ├─checkpoints  # Checkpoints released by various models
    │  ├─asl
    │  ├─......
    │  └─msrn
    ├─dataloaders
    ├─files
    │  ├─coco
    │  └─voc
    ├─models  # Model code
    │  ├─ASL
    │  │  └......
    │  ├─DSDL
    │  ├─......
    │  └─MSRN
    └─pretrained  # Pretrained model of MSRN
```

## Requirements

```bash
pip install -r requirements.txt
pip install -U pip setuptools wheel
pip install -U spacy
python -m spacy download en_core_web_lg
```

## Obtain Images from Google

```bash
python search.py --k=1 --dataset=voc --store_path=testimages/voc1 --synonym_file=voc_synonym.txt  --img_num=5 --check_img_num=20

python search.py --k=2 --dataset=voc --store_path=testimages/voc2 --synonym_file=voc_synonym.txt --img_num=5 --check_img_num=20

python search.py --k=1 --dataset=coco --store_path=testimages/coco1 --synonym_file=coco_synonym.txt --img_num=5 --check_img_num=20

python search.py --k=2 --dataset=coco --store_path=testimages/coco2 --synonym_file=coco_synonym.txt --img_num=5 --check_img_num=20
```

The parameter `--k` means the label combination dimension, `--dataset` means the dataset (label) to be searched, `--store_path` means the path to save the images, `--synonym_file` means the synonym file of the dataset label, `--img_num` means the number of images saved for each keyword, and `--check_img_num` means the number of images checked for each search.

## Test Process

1. Put the images obtained through google search into `data/imgs/google/$dataset`, where `$dataset` represents one of `voc1`, `voc2`, `coco1`, `coco2`, and then name the subfolder as a label combination (multiple labels are separated by `-`), and the image file is named `label-i.png` (refer to the sample in the file directory).
2. Put the checkpoint of each model into `test/checkpoints/$model`, and put the pretrained model of MSRN into `test/pretrained`. Please download the checkpoint from the repository of each model, see [MICS to be tested](#0).
3. Use the images obtained through google search to test the MICS: `cd tests & python google_main.py`, the test results will be saved in `data/results`.
4. Use the images obtained through google search and processed by MR to test the MICS: `cd tests & python google_main.py --followup`, the test results will be saved in `data/results` (must be executed after step 3).
5. Use the images randomly selected from ImageNet with the same number of images as in step 3 (tested 5 times separately) to test the MICS: `cd tests & python imagenet_main.py`, the test results will be saved in `data/results`.
6. Use the images selected from ImageNet and processed by MR to test the MICS: `cd tests & python imagenet_main.py --followup`, the test results will be saved in `data/results` (must be executed after step 5).

## Result Analysis

1. Calculate the information related to the training and validation sets of VOC and COCO required for RQ1-RQ3 and save it in `data/info/trainval_info`: `cd rqs & python gen_trainval_info.py`.
2. Integrate the test results with the information in `data/info` to obtain the data required for RQ1-RQ3, and save the results in `data/info/all_info`: `cd rqs & python integrate_data.py`.
3. Generate tables required for RQ1-RQ3 and save them in `data`: `cd rqs & python rqs.py`. Or use `python rqs.py rq1`, `python rqs.py rq2`, `python rqs.py rq3` to generate the required information separately.
4. Calculate the Cohen’s Kappa of all people's annotations: `cd rqs & python kappa.py`.

## Data download

You can download all the data related to our work from [here](https://drive.google.com/drive/folders/1Tf6B5g0uUi1kdyuES6UxXWIljLkbGrrk), which includes the following parts:

- [`Source test images`](./data/imgs/google/): Images obtained through google search.
- [`annotations`](./data/info/annotation/): Annotation information by all people.
- [`prediction results`](./data/results/): Results of the model.
