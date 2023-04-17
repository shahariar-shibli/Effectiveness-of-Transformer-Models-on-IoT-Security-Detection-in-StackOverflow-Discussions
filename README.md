# Effectiveness of Transformer Models on IoT Security Detection in StackOverflow Discussions
 
This repository contains the dataset of the following paper:

arXiv link: https://arxiv.org/abs/2207.14542

Researchgate link: https://www.researchgate.net/publication/362325733_Effectiveness_of_Transformer_Models_on_IoT_Security_Detection_in_StackOverflow_Discussions 

In this paper, we present the "IoT Security Dataset", a domain-specific dataset of 7,147 samples focused solely on IoT security discussions. We further employed multiple transformer models to automatically detect security discussions. Through rigorous investigations, we found that IoT security discussions are different and more complex than traditional security discussions.

To download the csv file:
------------------------

(1) Download the whole repository 

or 

(2) Click on the csv file -> click "download" -> right click on mouse and select "save as"

Dataset description:
---------------------

You will mainly need the following four columns:

(1) PostId: Id of the post in SO from where the sentence is taken

(2) Sentence: Extracted iot related sentence

(3) Security: Contains value 0 (Not security related) or 1 (security related)

(4) Cleaned Sentence: Sentence after text pre-processing.