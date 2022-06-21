
-----------------------------------------------------------------------------------------------------------------------------
**VERY IMPORTANT!!!!**


Before reading this file, please read the ORIGINAL_README.md file. It was written by the authors of DITTO, and it is crucial to understand how the original DITTO is deployed, before understanding the modifications that I propose.

Moreover, please read my report, which is uploaded in this repo and stored in its main directory as **REPORT.pdf**.

-----------------------------------------------------------------------------------------------------------------------------

This repo is a fork of the Original DITTO repository (https://github.com/megagonlabs/ditto). Its aim is to modify DITTO's original architecture regarding the Matcher and propose 4 alternative model structures. These models are deployed in four, newly created .py files that are stored in ditto_light directory. These files are 
**ditto_cls_sep_gru.py, ditto_gru.py , ditto_lstm.py and ditto_cls_sep.py.**. ditto_original.py is the same file as ditto.py in the original repo, with some minor modifications. 

If you want to execute the new DITTO with my proposed modifications, you can execute either **multiple_ditto_light_colab.ipynb** on Google Colab, or **multiple_ditto_light_kaggle.ipynb** on Kaggle, depending on whether you want to enable the FP16 optimization or not.

These 2 notebooks, if executed from scratch, will produce the the .xlsx files that are stored in results directory of this forked repo (FINAL_F1_SCORES.xlsx and FINAL_F1_SCORES_FP16.xlsx). Of course, when the notebooks are done and fully executed, you will have to download these files from either Colab's or Kaggle's working directories. 

Please have always a CUDA enabled when executing these notebooks, and do not infuse GENERAL DK as DK argument when training, because of the reasons that I explain in my report.

In order to select one of the 5 architectures available for training DITTO, please provide the appropriate input for **neural argument**, when executing both train_ditto.py and matcher.py. **Moreover, you always have to provide the same arguments' inputs both for train_ditto.py and matcher.py.**


