
-----------------------------------------------------------------------------------------------------------------------------
**VERY IMPORTANT!!!!**


Before reading this file, please read the ORIGINAL_README.md file. It was written by the authors of DITTO, and it is crucial to understand how the original DITTO is deployed, before understanding the modifications that I propose.

Moreover, please read my report, which is uploaded in this repo and stored in its main directory as **REPORT.pdf**.

-----------------------------------------------------------------------------------------------------------------------------

This repo is a fork of the Original DITTO repository (https://github.com/megagonlabs/ditto). It aims to modify DITTO's original architecture regarding the Matcher and propose four alternative model structures. These models are deployed in four newly created .py files that are stored in ditto_light directory (feel free to compare them to the original ditto.py file). These files are 
**ditto_cls_sep_gru.py, ditto_gru.py , ditto_lstm.py and ditto_cls_sep.py.**. ditto_original.py is almost the same file as ditto.py in the original repo, with some minor modifications. 

Suppose you want to execute the new DITTO with my proposed modifications; you can execute either **multiple_ditto_light_colab.ipynb** on Google Colab, or **multiple_ditto_light_kaggle.ipynb** on Kaggle, depending on whether you want to enable the FP16 optimization or not.


If executed from scratch, these two notebooks will produce the .xlsx files stored in the results directory of this forked repo (FINAL_F1_SCORES.xlsx and FINAL_F1_SCORES_FP16.xlsx). Of course, when the notebooks are fully executed, you will have to download these files from either Colab's or Kaggle's working directories. 

Please always have a CUDA enabled when executing these notebooks, and do not infuse GENERAL DK as the DK argument when training, because of the reasons I explain in my report.

To select one of the five architectures available for training DITTO, please provide the appropriate input for **neural argument**, when executing both train_ditto.py and matcher.py. The available input options for that argument are one of : [linear,cls_sep,gru,cls_sep_gru,lstm] (linear stands for the original proposal of the authors and the other four are the architectures that I introduced).**Moreover, you always have to provide the same arguments' inputs both for train_ditto.py and matcher.py.**

Matcher.py is modified to print the result of the produced model (by train_ditto.py) on the respective testset of the Dataset provided. I have forked all the datasets the DITTO authors offered, so there is a wide range of choices to experiment with. 

All the results from the matcher are saved in an .xlsx file in this repo's main directory. In this case, it is either F1_SCORES_FP16.xlsx or F1_SCORES.xlsx, depending on whether you have enabled FP16 optimization. The name of the .xlsx file is an input argument for matcher.py, and it must be a file stored in the same directory as the one where matcher.py is stored (in our case this is the main directory of the repo).

As mentioned before, the results from my experiments are stored in the results directory. There is also a .ipynb file saved in that directory, which serves as a mini report for the results earned from my experiments.  

So feel free to experiment and do not hesitate to ask me anything you want regarding my modifications on **cv13805@di.uoa.gr**. 

**(P.S.: A big thank you to the authors of DITTO for their robust work and their thorough explanations is more than necessary)**



