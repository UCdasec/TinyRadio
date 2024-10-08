# RadioNet

** The dataset and code are for research purpose only**

## Reference
When reporting results that use the dataset or code in this repository, please cite the paper below:

Haipeng Li, Kaustubh Gupta, Chenggang Wang, Nirnimesh Ghose, Boyang Wang, "RadioNet: Robust Deep-Learning Based Radio Fingerprinting" by IEEE Conference on Communications and Network Security (CNS 2022), Austin, TX, USA, 3-5 October 2022

## Contacts
Haipeng Li, Ph.D. candidate, li2hp@mail.uc.edu, University of Cincinnati 

Boyang Wang, Assistant Professor, boyang.wang@uc.edu, University of Cincinnati

Nirnimesh Ghose, Assistant Professor, nghose@unl.edu, University of Nebraska Lincoln  


## Dataset (last modified: Feb 2024)

https://mailuc-my.sharepoint.com/:f:/g/personal/wang2ba_ucmail_uc_edu/EszcvBI-h49JiBACymmxkigBwa8la4nppRxjMwtB2U5Wsw?e=Ovy8Um

Note: the above link need to be updated every 6 months due to certain settings of OneDrive. If you find the links are expired and you cannot access the data, please feel free to email us (boyang.wang@uc.edu). We will be update the links as soon as we can. Thanks!

## Same-Day and Cross-Day Evaluation

### Step 1

**Model Training** 

`model_training.py` for the same-day scenario and cross-day scenario evaluation. 

run `python model_training.py` to get the performance of radio classification in the same-day and cross-day scenario without the help of transfer learning

### Step 2

**Transfer learning**

After you train your same-day model using `model_training.py`, you can select one of the three followsing transfer learning methods to improve the performance in a cross-day scenario.

**ADA** contains code for Adversary Domain Adptation. `af_classifier.py` is the main function for ADA.

**Triplet_network** contains code for Triplet Network. `triplet_training.py` is the main function for triplet network.

**Fine-tuning** `finetune.py` for fine-tuning pre-trained models. 

## Defination of parameters

We use the same definations of parameters for all model training and transfer training functions. Currently, you can modify those parameters at the entrance of each function (such as "testOpts()" or "main()"):

`dataPath`: Path for Training dataset (Day1 dataset).

`testPath`: Path for Test dataset (Day 2 dataset for the cross-day scenario).

`location`: Specify the location of collected dataset, such as "before_fft", "after_fft" and "symbols".

`slice_len`: length for each slice, such as 288.

`dataType`: Data representation. "IQ" for I/Q data rapresentation; "spectrogram" for spectrogram data representation.

`num_slice`: Number of slices to extract from each device.

`start_idx`: Start index to extract slices.

`stride`: Stride when cutting slices, such as 144 or 288.

`modelType`: Model architecture, such as "homegrown" and "DF".

## Usage 
After you modify the parameters, run `python *.py` (replace `*` with `model_training`, `triplet_training` or `finetune`) to train and test your models. 

