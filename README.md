# Feature Fusion for Image Retrieval with Adaptive Bitrate Allocation and Hard Negative Mining
This is a toolbox that implements the training and testing of the approach described in our papers:
**Feature Fusion for Image Retrieval with Adaptive Bitrate Allocation and Hard Negative Mining**  

## 1. Deep features extract

The code of deep features extraction is modified based on this
 [repo](https://github.com/filipradenovic/cnnimageretrieval-pytorch), and off-the-shelf models are shown below.


### Prerequisites

In order to run this toolbox you will need:

1. Python3 (tested with Python 3.7.0 on Debian 8.1)
1. PyTorch deep learning framework (tested with version 1.0.0)

 
### Usage
 1. **Dataset Downloading**
    
    First of all, navigate (cd) to the root of the toolbox, and make test and train fold. 
    
           mkdir train
           mkdir test
           
    And then you can download the dataset auxiliary file here (
    [train](https://drive.google.com/file/d/1b8WXds46TlhHvDWW7Oga_M37pWfIIf-y/view?usp=sharing), 
    [test](https://drive.google.com/open?id=1NDrr2gRIBhc9GO8sBfB7lgMK-UOZPQuh)). After download the zip file of 
     train and test, please decompression them into the corresponding folder.
     
    Note:
    please download the sfm, oxford, paris CDVS dataset advance, and modify the path of the dataset auxiliary file, 
    don't forget to do that.
 1. **CNNs' Models Training**
 
    Example training script is located in ```rootdir/cirtorch/examples/train.py```
    ```
    python3 -um cirtorch.examples.train [-h] [--training-dataset DATASET] [--no-val]
                  [--test-datasets DATASETS]
                  [--test-freq N] [--arch ARCH] [--pool POOL]
                  [--local-whitening] [--regional] [--whitening]
                  [--not-pretrained] [--loss LOSS] [--loss-margin LM]
                  [--image-size N] [--neg-num N] [--query-size N]
                  [--pool-size N] [--gpu-id N] [--workers N] [--epochs N]
                  [--batch-size N] [--optimizer OPTIMIZER] [--lr LR]
                  [--momentum M] [--weight-decay W] [--print-freq N]
                  [--resume FILENAME] [--using-cdvs CDVS_PARA]
                  EXPORT_DIR
     ```
     The main modify is the parameter of --using-cdvs. While specify the CDVS_PARA, you can control
      the parameter of CDVS weight (_β_) in hard-mining stage. And while the CDVS_PARA equals 0, represent not do the
       hard-mining mentioned in our paper.
       
     For example, to train our best network described in our paper test on oxford5k, run the following command.
    ```
    python3 -um cirtorch.examples.train YOUR_EXPORT_DIR --gpu-id '0' --training-dataset 'retrieval-SfM-120k' 
              --test-datasets 'oxford5k' --arch 'resnet101' --pool 'gem' --loss 'contrastive' 
              --loss-margin 0.85 --optimizer 'adam' --lr 5e-7 --neg-num 5 --query-size=2000 
              --pool-size=2000 --batch-size 5 --image-size 362 --using-cdvs 0.2
    ```
 1. **Testing our pretrained networks**
 
    You can download our pretrained network for oxford5k [here](https://drive.google.com/file/d/1OZkrkLVkfSwm9dJnGB99tcBhbNwnzBk9/view?usp=sharing).
    
    To evaluate them run:
    ```
    python3 -m cirtorch.examples.test_extract2 --gpu-id '0' --network-path YOUR_DOWNLOAD_PATH
                  --datasets 'oxford5k'  --multiscale '[1, 1/2**(1/2), 1/2]'
                  [--using-cdvs 0]  [--ir-remove 0]  [--ir-adaption 0]
    ```
    --using-cdvs is equivalent to the parameter _`λr`_, 
    --ir-remove is equivalent to _`α`_, and --ir-adaption is equivalent to _`T`_ in our paper.
    The script will generate a rank.txt file to record the result.
    
    Note: Before re-ranking the retrieval result, please run the match_ir.exe script first.
 ---
## 2. CDVS features extract
### Prerequisites

In order to run this toolbox you will need:

1. Python3 (tested with Python 3.7.0 on Debian 8.1)
1. PyTorch deep learning framework (tested with version 1.0.0)
1. Windows to run exe file(tested with Windows10)


### Usage
 1. **Global_dis.py**
 
    It's the python script to compute the RHD of a picture in our paper.
    The lambda_RHD in line 9 is the parameter _`λRHD`_, lambda_R in line 10 is the parameter _`λR`_, 
    and bitave in line 11 is the average bitrate we used in the bitrate allocation process. Don't forget 
    to modify your own deep feature path in line 17.
    
    After run the script, you will get a result global txt file in which each line represent 
    the bitrate for each image in the database.
    ```
    python3 Global_dis.py
    ```
1. **parameters_ukbench.txt**
    
    The parameters file we used for extract CDVS descriptor. the common mode we used is 8, which
    bitrate is 1024 bytes.
1. **extract_global_resize.exe**
    
    To extract the CDVS global descriptor of a image list run:
    ```
    extract_global_resize.exe test_database.txt 8 parameters_ukbench.txt
    ```
 1. **extract_resize.exe**
    
    To extract the CDVS descriptor of a image list run:
    ```
    extract_resize.exe test_database.txt 8 parameters_ukbench.txt
    ```
 1. **txt_defined_resize.exe**
    
    To extract the CDVS descriptor of a image list based on the bitrate allocation run:
    ```
    txt_defined_resize.exe test_database.txt result_global.txt 8 parameters_ukbench.txt
    ```
    The result_global.txt is generated by the Global_dis.py above.

 1. **retrieve.exe**
    
    To do image retrieval using the CDVS descriptor of a image retrieval list run:
    ```
    retrieve.exe DB test_retrieval.txt 8 parameters_ukbench.txt 
    ```
 1. **ti-mk_index.exe**
    
    To build a database of a image list run:
    ```
    mkdir dataindex
    ti-mk_index.exe test_database.txt DB 8 parameters_ukbench.txt
    ```
 1. **match_ir.exe**
    
    To re-ranking the image retrieval ranks file generated by the test_extract.py file run :
    ```
    match_ir.exe test_query.txt test_database.txt cdvs_test_retrieval_ranks.txt 8 parameters_ukbench.txt 100
    ```
    Then it will generate a new rank list file. Put it into the origin rank file fold to do the next
    re-ranking steps in the test_extract2.py.
