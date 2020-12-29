# Custom Named Entity Recognition
According to Wikipedia, Named Entity Recognition (NER) is a subtask of information extraction that seeks to locate and classify named entities mentioned in unstructured text into pre-defined categories such as person names, organizations, locations, medical codes, time expressions, quantities, monetary values, percentages, etc. <br>
NER can be implemented with either statistical or rule-based methods, both of which require a large amount of labeled training data and are typically trained in a fully or semi-supervised manner. <br><br>
**Note:**
This repository houses the code for task 3 (Key Information Extraction from Scanned Receipts) of the ICDAR 2019 Robust Reading Challenge on Scanned Receipts OCR and Information Extraction.

## tl;dr
- Custom trained a blank English language model over reconstructed strings from optical character recognition task data using spaCy.

## Problem Statement
Scanned receipts OCR is a process of recognizing text from scanned structured and semi-structured receipts, and invoices in general. On the other hand, extracting key texts from receipts and invoices and save the texts to structured documents can serve many applications and services, such as efficient archiving, fast indexing and document analytics.<br>

Task Description:<br>
The aim of this task is to extract texts of a number of key fields from given receipts, and save the texts for each receipt image in a json file with format shown in Figure 3. Participants will be asked to submit a zip file containing results for all test invoice images.<br>

Evaluation Protocol:<br>
For each test receipt image, the extracted text is compared to the ground truth. An extract text is marked as correct if both submitted content and category of the extracted text matches the groundtruth; Otherwise, marked as incorrect. The precision is computed over all the extracted texts of all the test receipt images. F1 score is computed based on precision and recall. F1 score will be used for ranking.<br>

 ## Data
The dataset will have 1000 whole scanned receipt images. Each receipt image contains around about four key text fields, such as goods name, unit price and total cost, etc. The text annotated in the dataset mainly consists of digits and English characters.<br>
For receipt OCR task, each image in the dataset is annotated with text bounding boxes (bbox) and the transcript of each text bbox. Locations are annotated as rectangles with four vertices, which are in clockwise order starting from the top. Annotations for an image are stored in a text file with the same file name. The annotation format is similar to that of ICDAR2015 dataset, which is shown below:

x1_1, y1_1,x2_1,y2_1,x3_1,y3_1,x4_1,y4_1, transcript_1

x1_2,y1_2,x2_2,y2_2,x3_2,y3_2,x4_2,y4_2, transcript_2

x1_3,y1_3,x2_3,y2_3,x3_3,y3_3,x4_3,y4_3, transcript_3

…

For the information extraction task, each image in the dataset is annotated with a text file with format shown below:

{<br>"company": "STARBUCKS STORE #10208",<br>"date": "14/03/2015",<br>"address": "11302 EUCLID AVENUE, CLEVELAND, OH (216) 229-0749",<br>"total": "4.95",<br>}

## Technologies Used
- **Python**
- **Spacy**
- **Pandas**

## Solution Approach
- Since the text generated from the reciepts does not follow any grammar rules, I chose to use a blank English Language model from spacy and trained the **ner** component of the model pipeline in accordance with the entity types mentioned in the data description.
- SpaCy is one of the most popular productionized Natural Language Understanding packages. SpaCy uses residual convolutional neural networks (CNN) and incremental parsing with Bloom embeddings for NER. To summarize the algorithm, 1D convolutional filters are applied over the input text to predict how the upcoming words may change the current entity tags. The input sequence is embedded with bloom embeddings, which model the characters, prefix, suffix, and part of speech of each word. Residual blocks are used for the CNNs, andn the filter sizes are chosen with beam search.
- A simple memory tagger is also implemented in order to identify proper nouns such as company names and addresses in the test strings. Lists of company names and addresses are stored in a dictionary format and are searched through if the NER model fails to identify the entity.

## Model Output
Evaluation metric used to measure the model performance is **F1 score**.<br>
F1 score of the above mentioned Custom NER model on the test data set is **0.78**.<br>
[Click here](https://rrc.cvc.uab.es/?ch=13&com=evaluation&view=method_info&task=3&m=82331) to visit the ICDAR SROIE 2019 Challenge solutions page for this model. 

## References
- https://spacy.io/usage/linguistic-features#named-entities
- https://spacy.io/usage/training
- https://aihub.cloud.google.com/u/0/p/products%2F2290fc65-0041-4c87-a898-0289f59aa8ba
















