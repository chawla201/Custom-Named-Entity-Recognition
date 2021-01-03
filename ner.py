import os
import re
import json
import random
import warnings
from pathlib import Path
import pandas as pd
import numpy as np
import spacy
from spacy.util import minibatch, compounding
from spacy.matcher import PhraseMatcher

warnings.filterwarnings("ignore")


class NER:
    def __init__(
        self,
        iter=100,
        dropout=0.5,
        min_batchsize=4.0,
        max_batchsize=32.0,
        compounding_coef=0.01,
        train_text_path=None,
        train_entities_path=None,
        test_text_path=None,
    ):
        self.iter = iter
        self.dropout = dropout
        self.min_batchsize = min_batchsize
        self.max_batchsize = max_batchsize
        self.compounding_coef = compounding_coef
        self.train_text_path = train_text_path
        self.train_entities_path = train_entities_path
        self.test_text_path = test_text_path

    def get_data(self, *args, **kwargs):
        """
        Get data from the text files and transform into a pandas dataframe 
        parameters: None
        returns: dataframe
        """
        set_text = set(os.listdir(self.train_text_path))
        set_ent = set(os.listdir(self.train_entities_path))
        training_set = list(set_text.intersection(set_ent))

        self.data = pd.DataFrame(columns=["filename", "text"])
        self.data["filename"] = training_set
        data_text = []
        for file in self.data["filename"]:
            rec_text = []
            pattern = r"\d+,\d+,\d+,\d+,\d+,\d+,\d+,\d+,(.+)"
            with open(os.path.join(self.train_text_path, file)) as f:
                f.seek(0)
                lines = f.readlines()
                for line in lines:
                    rec_text += re.findall(pattern, line)
            data_text.append(" ".join([x.strip() for x in rec_text]))
        self.data["text"] = data_text
        ent_list = []
        for file in self.data["filename"]:
            with open(f"{self.train_entities_path}/{file}") as f:
                entity_dict = json.load(f)
                ent_list.append(entity_dict)
        self.data["entity_dictionary"] = ent_list

        return self.data

    def transform_data(self, data, *args, **kwargs):
        """
        Transform pandas dataframe to the spaCy compliant training data format 
        parameters: DataFrame
        returns: List of text and entity tuples
        """
        training_data = []
        id_ent = []
        nlp_match = spacy.load("en_core_web_sm")
        matcher = PhraseMatcher(nlp_match.vocab)
        for index, row in self.data.iterrows():
            ent_dic = row["entity_dictionary"]
            ent = []
            phrases = list(ent_dic.values())
            patterns = [nlp_match.make_doc(phrase) for phrase in phrases]
            matcher.add("EntityList", None, *patterns)

            doc = nlp_match(row["text"])
            matches = matcher(doc)
            for match_id, start, end in matches:
                try:
                    span = doc[start:end]
                    if start > 0:
                        sb = doc[0:start]
                        start_index = len(sb.text) + 1
                    else:
                        start_index = 0
                    end_index = start_index + len(span.text)
                except:
                    pass

                for key, value in ent_dic.items():
                    if value == span.text:
                        ent_tup = (start_index, end_index, key)
                        ent.append(ent_tup)
            ent_set = {"company", "date", "total", "address"}
            detected_entities = set([key for start, end, key in ent])
            missed_entities = list(ent_set - detected_entities)
            if "total" in missed_entities:
                value = ent_dic["total"]
                if len(value) > 0:
                    catch_total = re.search(value, str(row["text"]).replace(",", ""))
                    ent_tup = (catch_total.span()[0], catch_total.span()[1], "total")
                    ent.append(ent_tup)
            if "date" in missed_entities:
                value = ent_dic["date"]
                if len(value) > 0:
                    catch_date = re.search(value, str(row["text"]))
                    if catch_date == None:
                        catch_date = re.search(
                            r"\d\d[-/]*\d\d[-/]*\d\d", str(row["text"])
                        )
                    try:
                        ent_tup = (catch_total.span()[0], catch_total.span()[1], "date")
                        ent.append(ent_tup)
                    except:
                        pass
            if "company" in missed_entities:
                value = ent_dic["company"]
                catch_company = re.search(value, str(row["text"]))
                if catch_company != None:
                    ent_tup = (
                        catch_company.span()[0],
                        catch_company.span()[1],
                        "company",
                    )
                    ent.append(ent_tup)
                else:
                    catch_company = re.search(value, str(row["text"]).replace(".", ""))
                    if catch_company != None:
                        ent_tup = (
                            catch_company.span()[0],
                            catch_company.span()[1],
                            "company",
                        )
                        ent.append(ent_tup)
            if "address" in missed_entities:
                try:
                    value = ent_dic["address"]
                    catch_address = re.search(value, str(row["text"]))
                    if catch_address != None:
                        ent_tup = (
                            catch_address.span()[0],
                            catch_address.span()[1],
                            "address",
                        )
                        ent.append(ent_tup)
                except:
                    pass
            id_ent.append(len(ent))
            entity_dictionary = {"entities": ent}
            train_tup = (row["text"], entity_dictionary)
            training_data.append(train_tup)

        return training_data

    def fit(self, train_text_path, train_entities_path, *args, **kwargs):
        """
        Fit a blank English language model from spaCy and save the model in the current directory 
        parameters: None
        returns: None
        """
        self.train_text_path = train_text_path
        self.train_entities_path = train_entities_path
        data = self.get_data(self)
        training_data = self.transform_data(self, data)
        TRAIN_DATA = training_data
        output_dir = os.path.join(os.getcwd(), "model")

        nlp = spacy.blank("en")
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner, last=True)
        for _, annotations in TRAIN_DATA:
            for ent in annotations.get("entities"):
                ner.add_label(ent[2])

        dropout = self.dropout
        min_batchsize = self.min_batchsize
        max_batchsize = self.max_batchsize
        compounding_coef = self.compounding_coef
        n_iter = self.iter

        nlp.begin_training()
        for itn in range(n_iter):
            random.shuffle(TRAIN_DATA)
            losses = {}
            batches = minibatch(
                TRAIN_DATA,
                size=compounding(min_batchsize, max_batchsize, compounding_coef),
            )
            for batch in batches:
                texts, annotations = zip(*batch)
                try:
                    nlp.update(
                        texts, annotations, drop=dropout, losses=losses,
                    )
                except:
                    pass
            print(f"{itn} Losses", losses)

        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.to_disk(output_dir)

    def get_test_data(self, *args, **kwargs):
        """
        Fetch the data from the test directory and transform it into a pandas DataFrame
        returns: DataFrame 
        """
        test_text_files = os.listdir(self.test_text_path)
        test_data = pd.DataFrame(columns=["filename", "text"])
        test_data["filename"] = test_text_files
        data_text = []
        for file in test_data["filename"]:
            rec_text = []
            pattern = r"\d+,\d+,\d+,\d+,\d+,\d+,\d+,\d+,(.+)"
            try:
                with open(os.path.join(self.test_text_path, file)) as f:
                    lines = f.readlines()
                    for line in lines:
                        rec_text += re.findall(pattern, line)
            except:
                pass
            data_text.append(" ".join([x.strip() for x in rec_text]))
        test_data["text"] = data_text
        return test_data

    def predict(self, test_text_path, reciept_text="", is_dir=True, *args, **kwargs):
        """
        Identify entities in a new text string
        """
        cwd = os.getcwd()
        nlp = spacy.load(os.path.join(cwd, "model"))
        train = os.listdir(self.train_entities_path)
        ent_list = []
        for file in train:
            with open(os.path.join(self.train_entities_path, file)) as f:
                entity_dict = json.load(f)
                ent_list.append(entity_dict)
        memory_dictionary = {"company": [], "address": [], "date": [], "total": []}
        for dictionary in ent_list:
            for key, value in dictionary.items():
                memory_dictionary[key].append(value)

        if is_dir:
            self.test_text_path = test_text_path
            test_data = self.get_test_data(self)
            for index, row in test_data.iterrows():
                op_dict = {"company": "", "date": "", "address": "", "total": ""}
                doc = nlp(row["text"])
                for ent in doc.ents:
                    op_dict[ent.label_] = ent.text

                for tag, tag_memory in memory_dictionary.items():
                    for tag_value in tag_memory:
                        if tag == "total":
                            pass
                        elif (re.search(tag_value, row["text"]) != None) and (
                            op_dict[tag] == ""
                        ):
                            op_dict[tag] = tag_value

                print("Entities: ", op_dict)
                op_dir = os.path.join(cwd, "output")
                if not os.path.isdir(op_dir):
                    os.mkdir(path=op_dir)
                json_object = json.dumps(op_dict, indent=4)
                with open(os.path.join(op_dir, f"{row['filename']}"), "w") as op:
                    op.write(json_object)
        else:
            doc = nlp(reciept_text)
            op_dict = {"company": "", "date": "", "address": "", "total": ""}
            for ent in doc.ents:
                op_dict[ent.label_] = ent.text
            for tag, tag_memory in memory_dictionary.items():
                for tag_value in tag_memory:
                    if tag == "total":
                        pass
                    elif (re.search(tag_value, row["text"]) != None) and (
                        op_dict[tag] == ""
                    ):
                        op_dict[tag] = tag_value
            print("Entities: ", op_dict)
