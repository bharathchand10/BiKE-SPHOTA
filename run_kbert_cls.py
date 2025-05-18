# -*- encoding:utf-8 -*-
"""
  This script provides an k-BERT exmaple for classification.
"""
from cmath import nan
import shutil
import sys
import time
import torch
import json
import random
import argparse
import collections
import torch.nn as nn
from uer.utils.vocab import Vocab
from uer.utils.constants import *
from uer.utils.tokenizer import * 
from uer.model_builder import build_model
from uer.utils.optimizers import  BertAdam
from uer.utils.config import load_hyperparam
from uer.utils.seed import set_seed
from uer.model_saver import save_model
from brain import KnowledgeGraph
import brain.config
from multiprocessing import Process, Pool
import numpy as np
import pandas as pd
import re
import traceback
import os
 

from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report

#### AirBnB price id to uri mapping
# FILENAME_TRAIN_DATA_MAPPED_ID_URI = 'datasets/english_datasets/airbnb_train_mapped_id_uri.parquet'
# FILENAME_TEST_DATA_MAPPED_ID_URI = 'datasets/english_datasets/airbnb_test_mapped_id_uri.parquet'


#### AG NEWS id to uri mapping
# FILENAME_TRAIN_DATA_MAPPED_ID_URI = 'datasets/english_datasets/ag_news_train_mapped_id_uri.parquet'
# FILENAME_TEST_DATA_MAPPED_ID_URI = 'datasets/english_datasets/ag_news_dev_test_mapped_id_uri.parquet'

### Empty id to uri mapping
# FILENAME_TEST_DATA_MAPPED_ID_URI_EMPTY = 'datasets/english_datasets/ag_news_test_dev_mapped_id_uri_EMPTY.parquet'
# FILENAME_TEST_DATA_MAPPED_ID_URI = 'datasets/english_datasets/ag_news_test_dev_mapped_id_uri_EMPTY.parquet'

# Open output file for writing results


PHRASES = []
EMBEDDINGS = []


class BertClassifier(nn.Module):
    def __init__(self, args, model):
        super(BertClassifier, self).__init__()
        self.embedding = model.embedding
        self.encoder = model.encoder
        self.labels_num = args.labels_num
        self.pooling = args.pooling
        self.output_layer_1 = nn.Linear(args.hidden_size, args.hidden_size)
        self.output_layer_1a = nn.Linear(args.hidden_size, args.hidden_size) ### added
        self.output_layer_2 = nn.Linear(args.hidden_size, args.labels_num)
        self.dropout = nn.Dropout(args.dropout)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.criterion = nn.NLLLoss()
        self.use_vm = False if args.no_vm else True
        self.proj = nn.Linear(768, 512)
        self.current_sent_tree = None  # Add storage for current sentence tree
        self.current_cleaned_text = None  # Add storage for cleaned text
        
        print("[BertClassifier] use visible_matrix: {}".format(self.use_vm))

    def set_sent_tree(self, sent_tree):
        """Store the current sentence tree"""
        self.current_sent_tree = sent_tree
        
    def set_cleaned_text(self, cleaned_text):
        """Store the current cleaned text"""
        self.current_cleaned_text = cleaned_text

    def forward(self, src, label, mask, pos=None, vm=None):
        """
        Args:
            src: [batch_size x seq_length]
            label: [batch_size]
            mask: [batch_size x seq_length]
        """
        # Embedding.
        emb = self.embedding(src, mask, pos)
        # Encoder.
        if not self.use_vm:
            vm = None
        output = self.encoder(emb, mask, vm)
        # Target.
        if self.pooling == "mean":
            output = torch.mean(output, dim=1)
        elif self.pooling == "max":
            output = torch.max(output, dim=1)[0]
        elif self.pooling == "last":
            output = output[:, -1, :]
        else:
            output = output[:, 0, :]
        pooled_output = torch.tanh(self.output_layer_1(output))
        
        pooled_output_2 = self.proj(pooled_output)
        pooled_output_2 = pooled_output_2.cpu().detach().numpy()
        
        # Print cleaned text, embedding and sentence tree together
        if self.current_cleaned_text is not None:
            print("Cleaned text:")
            print(self.current_cleaned_text, '\n')

        if self.current_sent_tree is not None:
            print("Corresponding sentence tree:")
            print(self.current_sent_tree, '\n')

        print("Embedding:")
        print(pooled_output_2, '\n')

        PHRASES.append(self.current_cleaned_text)
        EMBEDDINGS.append(pooled_output_2.tolist())
        
        print("\n" + "="*80 + "\n")
        
        dropout_output = self.dropout(pooled_output)
        linear_1a_output = torch.relu(self.output_layer_1a(dropout_output))
        logits = self.output_layer_2(linear_1a_output)
        loss = self.criterion(self.softmax(logits.view(-1, self.labels_num)), label.view(-1))
        return loss, logits

def pre_process(text): 
    ### FROM: https://github.com/kavgan/nlp-in-practice/tree/master/tf-idf 
    # lowercase
    text=text.lower() 
    #remove tags
    text=re.sub("</?.*?>"," <> ",text)
    # remove special characters and digits
    text=re.sub("(\\d|\\W)+"," ",text)   
    
    # Swaps line breaks for spaces, also remove apostrophes & single quotes
    text.replace("\n", " ").replace("'"," ").replace("'"," ")
    return text
import ast
def clean_and_sort_entities(df: pd.DataFrame) -> pd.DataFrame:
    df.dropna(axis=0, inplace=True)
    print ('hello')
    print (df['entity_data'].iloc[0])
    
    print (type(df['entity_data'].iloc[0]))
    print (type(df['entity_data'].iloc[0]))
    df["pos"] = df.entity_data.apply(lambda e: e['surface_char_pos'][0])
    #df["pos"] = df.entity_data.apply(lambda e: e['surface_char_pos'][0] if e and 'surface_char_pos' in e and e['surface_char_pos'] else None)
    df = df.sort_values(by=["source_id", "pos"]).drop(columns="pos")
    return df

def add_knowledge_worker(params, filename, max_sentences=0, skip_sentences=0, max_entities = brain.config.MAX_ENTITIES, repeat = 1 ):
    p_id, sentences, columns, kg, vocab, args = params
    sentences_num = len(sentences)
    dataset = []
    df_mapped = pd.read_parquet(filename, engine='pyarrow').dropna(axis=0)
    df_mapped = clean_and_sort_entities(df_mapped)
    
    time_stamp = time.strftime('%Y%m%d-%H%M')
    file = open(args.working_dir + "sent_tree_complete__%s.txt" % time_stamp,'w')

    tot_senteces = 0
    tot_entites = 0
    tot_sentences_with_entities = 0
    tot_triples_added = 0
    tot_sentences_with_triples = 0

    for line_id, line in enumerate(sentences):
        if max_sentences>0 and line_id >= max_sentences + skip_sentences:
            break 
        if line_id < skip_sentences:
            continue
        if line_id % 10000 == 0:
            print("Progress of process {}: {}/{}".format(p_id, line_id, sentences_num))
            sys.stdout.flush()
        line = line.strip().split('\t')
        df_mapped_sel = df_mapped[df_mapped.source_id==line_id]

        tot_senteces += 1
        tot_entites += df_mapped_sel.shape[0]
        tot_sentences_with_entities += df_mapped_sel.shape[0]>0

        try:
            if len(line) == 2:
                label = int(line[columns["label"]])

                print ("TEXT: ",line[columns["text_a"]],'\n')

                

                cleaned_text = pre_process(str(line[columns["text_a"]]))
                print ("CLEANED TEXT: ",cleaned_text,'\n')
                print ("ENTITIES: ")
                for _, row in df_mapped_sel.iterrows():
                    print(f"Entity: {row['entity_data']['surface_form']}, URI: {row['uri']}")
                print('\n')

                tokens, pos, vm, _, sent_tree, num_triples_added_batch = kg.add_knowledge_with_vm([cleaned_text], df_mapped_sel, add_pad=True, max_length=args.seq_length, max_entities = max_entities)
                tokens = tokens[0]
                pos = pos[0]
                vm = vm[0].astype("bool")
                tot_triples_added += sum(num_triples_added_batch[0])
                tot_sentences_with_triples += sum(num_triples_added_batch[0])>0

                file.write(str(sent_tree))
                file.write('\n')

                token_ids = [vocab.get(t) for t in tokens]
                mask = [1 if t != PAD_TOKEN else 0 for t in tokens]
                for r in range(repeat):
                    dataset.append((token_ids, label, mask, pos, vm, sent_tree, cleaned_text))  # Include cleaned_text in dataset
            else:
                pass
        except Exception as e:
            print("Error line: ", line)
            print("Line id:", line_id)
            print("ERROR: ",str(e))
            print(traceback.format_exc())
            pass
    
    print("tot_senteces",tot_senteces)
    print("tot_entites",tot_entites)
    print("tot_sentences_with_entities",tot_sentences_with_entities)
    print("tot_triples_added",tot_triples_added)
    print("tot_sentences_with_triples",tot_sentences_with_triples)
    return dataset

def batch_loader(batch_size, input_ids, label_ids, mask_ids, pos_ids, vms, sent_trees=None, cleaned_texts=None):
    instances_num = input_ids.size()[0]
    for i in range(instances_num // batch_size):
        input_ids_batch = input_ids[i*batch_size: (i+1)*batch_size, :]
        label_ids_batch = label_ids[i*batch_size: (i+1)*batch_size]
        mask_ids_batch = mask_ids[i*batch_size: (i+1)*batch_size, :]
        pos_ids_batch = pos_ids[i*batch_size: (i+1)*batch_size, :]
        vms_batch = vms[i*batch_size: (i+1)*batch_size]
        sent_trees_batch = sent_trees[i*batch_size: (i+1)*batch_size] if sent_trees is not None else None
        cleaned_texts_batch = cleaned_texts[i*batch_size: (i+1)*batch_size] if cleaned_texts is not None else None
        yield input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vms_batch, sent_trees_batch, cleaned_texts_batch
    if instances_num > instances_num // batch_size * batch_size:
        input_ids_batch = input_ids[instances_num//batch_size*batch_size:, :]
        label_ids_batch = label_ids[instances_num//batch_size*batch_size:]
        mask_ids_batch = mask_ids[instances_num//batch_size*batch_size:, :]
        pos_ids_batch = pos_ids[instances_num//batch_size*batch_size:, :]
        vms_batch = vms[instances_num//batch_size*batch_size:]
        sent_trees_batch = sent_trees[instances_num//batch_size*batch_size:] if sent_trees is not None else None
        cleaned_texts_batch = cleaned_texts[instances_num//batch_size*batch_size:] if cleaned_texts is not None else None
        yield input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vms_batch, sent_trees_batch, cleaned_texts_batch

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Path options.
    parser.add_argument("--pretrained_model_path", default=None, type=str,
                        help="Path of the pretrained model.")
    parser.add_argument("--finetuned_model", default=None, type=str,
                        help="Path of the fine tuned model to continue training.")
    parser.add_argument("--output_model_path", default="./models/classifier_model.bin", type=str,
                        help="Path of the output model.")
    parser.add_argument("--vocab_path", default="./models/google_vocab.txt", type=str,
                        help="Path of the vocabulary file.")
    parser.add_argument("--train_path", type=str, required=True,
                        help="Path of the trainset.")
    parser.add_argument("--dev_path", type=str, required=True,
                        help="Path of the devset.") 
    parser.add_argument("--test_path", type=str, required=True,
                        help="Path of the testset.")
    parser.add_argument("--config_path", default="./models/google_config.json", type=str,
                        help="Path of the config file.")
    parser.add_argument("--train_entity_mapping", type=str,
                        help="Path for file containing the mapping from surface form to entity uri in training sentences.")
    parser.add_argument("--test_entity_mapping", type=str,
                        help="Path for file containing the mapping from surface form to entity uri in test sentences.")    
    parser.add_argument("--dev_entity_mapping", type=str,
                        help="Path for file containing the mapping from surface form to entity uri in test sentences.")                            
    
    parser.add_argument("--working_dir", type=str,
                        help="Path for working dir where output files are saved.")                            
    # train, dev, test limitations
    parser.add_argument("--experiment_name", type=str, 
                        help="Name of the experiment for logging outputs")


    parser.add_argument("--train_max_size", type=int, default=0,
                        help="Max number of sentences to take for train.")
    
    parser.add_argument("--dev_max_size", type=int, default=0,
                        help="Max number of sentences to take for dev.")

    parser.add_argument("--test_max_size", type=int, default=0,
                        help="Max number of sentences to take for test.")        

    parser.add_argument("--test_skip_size", type=int, default=0,
                        help="Max number of sentences to skip from the test set.")                                            


    # Model options.
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size.")
    parser.add_argument("--seq_length", type=int, default=256,
                        help="Sequence length.")
    parser.add_argument("--encoder", choices=["bert", "lstm", "gru", \
                                                   "cnn", "gatedcnn", "attn", \
                                                   "rcnn", "crnn", "gpt", "bilstm"], \
                                                   default="bert", help="Encoder type.")
    parser.add_argument("--bidirectional", action="store_true", help="Specific to recurrent model.")
    parser.add_argument("--pooling", choices=["mean", "max", "first", "last"], default="first",
                        help="Pooling type.")

    # Subword options.
    parser.add_argument("--subword_type", choices=["none", "char"], default="none",
                        help="Subword feature type.")
    parser.add_argument("--sub_vocab_path", type=str, default="models/sub_vocab.txt",
                        help="Path of the subword vocabulary file.")
    parser.add_argument("--subencoder", choices=["avg", "lstm", "gru", "cnn"], default="avg",
                        help="Subencoder type.")
    parser.add_argument("--sub_layers_num", type=int, default=2, help="The number of subencoder layers.")

    # Tokenizer options.
    parser.add_argument("--tokenizer", choices=["bert", "char", "word", "space"], default="bert",
                        help="Specify the tokenizer." 
                             "Original Google BERT uses bert tokenizer on Chinese corpus."
                             "Char tokenizer segments sentences into characters."
                             "Word tokenizer supports online word segmentation based on jieba segmentor."
                             "Space tokenizer segments sentences into words according to space."
                             )

    # Optimizer options.
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate.")
    parser.add_argument("--warmup", type=float, default=0.1,
                        help="Warm up value.")

    # Training options.
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="Dropout.")
    parser.add_argument("--epochs_num", type=int, default=5,
                        help="Number of epochs.")
    parser.add_argument("--report_steps", type=int, default=100,
                        help="Specific steps to print prompt.")
    parser.add_argument("--seed", type=int, default=7,
                        help="Random seed.")

    # Evaluation options.
    parser.add_argument("--mean_reciprocal_rank", action="store_true", help="Evaluation metrics for DBQA dataset.")

    # kg
    parser.add_argument("--kg_name", required=True, help="KG name or path")
    parser.add_argument("--workers_num", type=int, default=1, help="number of process for loading dataset")
    parser.add_argument("--no_vm", action="store_true", help="Disable the visible_matrix")
    parser.add_argument("--skip_train", action="store_true", help="Disable train phase", default=False)
    parser.add_argument("--repeat_validation", action="store_true", help="Repeat validation on an existing fine tuned model.", default=False)

    ## max number of entities from Kg to use for enrichment
    parser.add_argument("--max_entities", type=int, default=2, help="max number of entities (triples) from Kg to use for enrichment")

    


    args = parser.parse_args()

    # Load the hyperparameters from the config file.
    args = load_hyperparam(args)

    set_seed(args.seed)

    # Count the number of labels.
    labels_set = set()
    columns = {}
    with open(args.train_path, mode="r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            try:
                line = line.strip().split("\t")
                if line_id == 0:
                    for i, column_name in enumerate(line):
                        columns[column_name] = i
                    continue
                label = int(line[columns["label"]])
                labels_set.add(label)
            except:
                pass
    args.labels_num = len(labels_set) 
    

    # Load vocabulary.
    vocab = Vocab()
    vocab.load(args.vocab_path)
    args.vocab = vocab

    # Build bert model.
    # A pseudo target is added.
    args.target = "bert"
    model = build_model(args)
    ### using UER library it is possible to istantiate many different models
    ### in our test we are using args with 
    ### - target = "bert"
    ### - encoder = "bert"
    ### - subencoder = "avg"
    ### - embeddig = BertEmbedding configured with parameters in args
    ### - 

    # Load or initialize parameters.
    if args.pretrained_model_path is not None:
        # Initialize with pretrained model.
        model.load_state_dict(torch.load(args.pretrained_model_path), strict=True)  
    else:
        # Initialize with normal distribution.
        for n, p in list(model.named_parameters()):
            if 'gamma' not in n and 'beta' not in n:
                p.data.normal_(0, 0.02)
    
    # Build classification model.
    model = BertClassifier(args, model)

    if args.finetuned_model is not None:
        model.load_state_dict(torch.load(args.finetuned_model), strict=True) 

    # For simplicity, we use DataParallel wrapper to use multiple GPUs.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    
    print("Using device: ", device)
    if torch.cuda.device_count() > 1:
        print("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)

    model = model.to(device)
    
    # Datset loader.
    def batch_loader(batch_size, input_ids, label_ids, mask_ids, pos_ids, vms, sent_trees=None, cleaned_texts=None):
        instances_num = input_ids.size()[0]
        for i in range(instances_num // batch_size):
            input_ids_batch = input_ids[i*batch_size: (i+1)*batch_size, :]
            label_ids_batch = label_ids[i*batch_size: (i+1)*batch_size]
            mask_ids_batch = mask_ids[i*batch_size: (i+1)*batch_size, :]
            pos_ids_batch = pos_ids[i*batch_size: (i+1)*batch_size, :]
            vms_batch = vms[i*batch_size: (i+1)*batch_size]
            sent_trees_batch = sent_trees[i*batch_size: (i+1)*batch_size] if sent_trees is not None else None
            cleaned_texts_batch = cleaned_texts[i*batch_size: (i+1)*batch_size] if cleaned_texts is not None else None
            yield input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vms_batch, sent_trees_batch, cleaned_texts_batch
        if instances_num > instances_num // batch_size * batch_size:
            input_ids_batch = input_ids[instances_num//batch_size*batch_size:, :]
            label_ids_batch = label_ids[instances_num//batch_size*batch_size:]
            mask_ids_batch = mask_ids[instances_num//batch_size*batch_size:, :]
            pos_ids_batch = pos_ids[instances_num//batch_size*batch_size:, :]
            vms_batch = vms[instances_num//batch_size*batch_size:]
            sent_trees_batch = sent_trees[instances_num//batch_size*batch_size:] if sent_trees is not None else None
            cleaned_texts_batch = cleaned_texts[instances_num//batch_size*batch_size:] if cleaned_texts is not None else None
            yield input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vms_batch, sent_trees_batch, cleaned_texts_batch

    # Build knowledge graph.
    if args.kg_name == 'none':
        spo_files = []
    else:
        spo_files = [args.kg_name]
    kg = KnowledgeGraph(spo_files=spo_files, predicate=True,  use_chinese_characters=False, vocab_file=args.vocab_path)

    def read_dataset(path, filename_data_mapped_id_uri, workers_num=1, max_sentences=0, skip_sentences=0, max_entities = brain.config.MAX_ENTITIES ):

        print("Loading sentences from {}".format(path))
        sentences = []
        with open(path, mode='r', encoding="utf-8") as f:
            for line_id, line in enumerate(f):
                if line_id == 0:
                    continue
                sentences.append(line)
        sentence_num = len(sentences)

        print("There are {} sentence in total. We use {} processes to inject knowledge into sentences.".format(sentence_num, workers_num))
        if workers_num > 1:
            params = []
            sentence_per_block = int(sentence_num / workers_num) + 1
            for i in range(workers_num):
                params.append((i, sentences[i*sentence_per_block: (i+1)*sentence_per_block], columns, kg, vocab, args))
            pool = Pool(workers_num)
            res = pool.map(add_knowledge_worker, params, max_sentences = max_sentences, skip_sentences=skip_sentences, max_entities = max_entities ) # TODO: add management of workers_num > 1 cases
            pool.close()
            pool.join()
            dataset = [sample for block in res for sample in block]
        else:
            params = (0, sentences, columns, kg, vocab, args)
            dataset = add_knowledge_worker(params, filename_data_mapped_id_uri, max_sentences = max_sentences, skip_sentences=skip_sentences, max_entities = max_entities, repeat = 1 )

        return dataset

    def get_best_thresholds(labels, test_y, outputs, plot=False):
        """
        Hyper parameter search for best classification threshold
        """
        t_max = [0] * len(labels)
        f_max = [0] * len(labels)

        for i, label in enumerate(labels):
            ts = []
            fs = []

            for t in np.linspace(0.1, 0.99, num=50):
                p, r, f, _ = precision_recall_fscore_support(test_y[:,i], np.where(outputs[:,i]>t, 1, 0), average='micro')
                ts.append(t)
                fs.append(f)
                if f > f_max[i]:
                    f_max[i] = f
                    t_max[i] = t

            if plot:
                print(f'LABEL: {label}')
                print(f'f_max: {f_max[i]}')
                print(f't_max: {t_max[i]}')

                plt.scatter(ts, fs)
                plt.show()
                
        return t_max, f_max

    # Evaluation function.
    def evaluate(args, is_test, metrics='Acc', epoch=None):
        if is_test:
            dataset = read_dataset(args.test_path, filename_data_mapped_id_uri=args.test_entity_mapping, workers_num=args.workers_num, max_sentences=args.dev_max_size, skip_sentences = args.test_skip_size, max_entities=args.max_entities)
        else:
            dataset = read_dataset(args.dev_path, filename_data_mapped_id_uri=args.dev_entity_mapping, workers_num=args.workers_num, max_sentences=args.test_max_size, max_entities=args.max_entities)

        input_ids = torch.LongTensor([sample[0] for sample in dataset])
        label_ids = torch.LongTensor([sample[1] for sample in dataset])
        mask_ids = torch.LongTensor([sample[2] for sample in dataset])
        pos_ids = torch.LongTensor([example[3] for example in dataset])
        vms = [example[4] for example in dataset]
        sent_trees = [example[5] for example in dataset]  # Get sentence trees
        cleaned_texts = [example[6] for example in dataset]  # Get cleaned texts

        batch_size = args.batch_size
        instances_num = input_ids.size()[0]
        if is_test:
            print("The number of evaluation instances: ", instances_num)

        if instances_num == 0:
            print("ERROR: number evaluation instances is zero!")
            raise(Exception("Number evaluation instances is zero."))

        correct = 0
        # Confusion matrix.
        confusion = torch.zeros(args.labels_num, args.labels_num, dtype=torch.long)

        model.eval()
        
        if not args.mean_reciprocal_rank:
            all_y_pred = np.array([], dtype = int)
            all_y_true = np.array([], dtype = int)
            all_np_proba = None

            for i, (input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vms_batch, sent_trees_batch, cleaned_texts_batch) in enumerate(batch_loader(batch_size, input_ids, label_ids, mask_ids, pos_ids, vms, sent_trees, cleaned_texts)):
                vms_batch = [vms.astype(int) for vms in vms_batch]
                vms_batch = torch.LongTensor(vms_batch)

                input_ids_batch = input_ids_batch.to(device)
                label_ids_batch = label_ids_batch.to(device)
                mask_ids_batch = mask_ids_batch.to(device)
                pos_ids_batch = pos_ids_batch.to(device)
                vms_batch = vms_batch.to(device)

                # Set the current sentence tree and cleaned text before forward pass
                if len(sent_trees_batch) > 0:
                    model.set_sent_tree(sent_trees_batch[0])
                if len(cleaned_texts_batch) > 0:
                    model.set_cleaned_text(cleaned_texts_batch[0])

                with torch.no_grad():
                    try:
                        loss, logits = model(input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vms_batch)
                    except:
                        print(input_ids_batch)
                        print(input_ids_batch.size())
                        print(vms_batch)
                        print(vms_batch.size())

                if epoch is None:
                    epoch_name = "test_eval"
                else:
                    epoch_name = "epoch-%s" % epoch

                proba = nn.Softmax(dim=1)(logits)
                pred = torch.argmax(proba, dim=1)
                gold = label_ids_batch

                correct += torch.sum(pred == gold).item()

                np_proba = proba.cpu().numpy()
                y_true = label_ids_batch.cpu().numpy()
                y_pred = pred.cpu().numpy()
            
                all_y_true = np.concatenate((all_y_true, y_true))
                all_y_pred = np.concatenate((all_y_pred, y_pred))
                if all_np_proba is None:
                    all_np_proba = np_proba
                else:
                    all_np_proba = np.vstack((all_np_proba, np_proba))

            all_y_true_hot_encoded = np.zeros((all_y_true.size, all_y_true.max() + 1))
            all_y_true_hot_encoded[np.arange(all_y_true.size), all_y_true] = 1

            thesholds_file_name = args.experiment_name + '_best_thresholds.csv'
            
            if is_test:
                # Parse thresholds
                with open(thesholds_file_name, 'r') as f:
                    t_max = [float(t) for t in f.read().split(',')]

                    if len(t_max) != args.labels_num:
                        raise ValueError('Threshold values does not match label count')
            else:
                t_max, f_max = get_best_thresholds([str(l) for l in range(args.labels_num)], all_y_true_hot_encoded, all_np_proba, plot=False)
                report = classification_report(all_y_true_hot_encoded, np.where(all_np_proba>t_max, 1, 0), output_dict=True, zero_division=0)
                report_str = classification_report(all_y_true_hot_encoded, np.where(all_np_proba>t_max, 1, 0), zero_division=0)
                print("With trheshold refinement:")
                print(report_str)
                if args.repeat_validation:
                    print("Repeating validation. We do a backup of file: %s" % thesholds_file_name)
                    try:
                        shutil.move(thesholds_file_name, thesholds_file_name+".bak") # let's do a backup of the previous version
                    except Exception as e:
                        print("Warning: unable do make a backup for thresholds ", e)
                
                with open(thesholds_file_name, 'w') as f:
                    f.write(','.join([str(t) for t in t_max]))

            p, r, f1, s = precision_recall_fscore_support(all_y_true_hot_encoded, np.where(all_np_proba>t_max, 1, 0), zero_division=0)
            confusion_m = confusion_matrix(all_y_true_hot_encoded.argmax(axis=1), np.where(all_np_proba>t_max, 1, 0).argmax(axis=1))

            for i in range(confusion_m.shape[0]): 
                if is_test: ## evauating test split after traininsg 
                    print("Test, Label {}: {:.3f}, {:.3f}, {:.3f} {}".format(i,p[i],r[i],f1[i],s[i]))
                else:
                    print("Epoch {}, Label {}: {:.3f}, {:.3f}, {:.3f} {}".format(epoch, i,p[i],r[i],f1[i],s[i]))
            

            if is_test:
                # report = classification_report(all_y_true, all_y_pred, output_dict=True, zero_division=0)
                # report_str = classification_report(all_y_true, all_y_pred, zero_division=0)
                # print(report_str)  
                report = classification_report(all_y_true_hot_encoded, np.where(all_np_proba>t_max, 1, 0), output_dict=True, zero_division=0)
                report_str = classification_report(all_y_true_hot_encoded, np.where(all_np_proba>t_max, 1, 0), zero_division=0)
                print("With trheshold refinement")
                print(report_str)


                with open(filename_base+'_report.json', 'w') as f:
                    json.dump(report, f)

                with open(filename_base+'_report.txt', 'w') as f:
                    f.write(report_str)


            print("Acc. (Correct/Total): {:.4f} ({}/{}) ".format(correct/len(dataset), correct, len(dataset)))
            if metrics == 'Acc':
                return correct/len(dataset)
            elif metrics == 'f1':
                return f1
            else:
                return correct/len(dataset)
        else:
            for i, (input_ids_batch, label_ids_batch,  mask_ids_batch, pos_ids_batch, vms_batch, sent_trees_batch, cleaned_texts_batch) in enumerate(batch_loader(batch_size, input_ids, label_ids, mask_ids, pos_ids, vms, sent_trees, cleaned_texts)):

                vms_batch = [ vms.astype(int) for vms in vms_batch ]
                vms_batch = torch.LongTensor(vms_batch)

                input_ids_batch = input_ids_batch.to(device)
                label_ids_batch = label_ids_batch.to(device)
                mask_ids_batch = mask_ids_batch.to(device)
                pos_ids_batch = pos_ids_batch.to(device)
                vms_batch = vms_batch.to(device)

                with torch.no_grad():
                    loss, logits = model(input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vms_batch)
                logits = nn.Softmax(dim=1)(logits)
                if i == 0:
                    logits_all=logits
                if i >= 1:
                    logits_all=torch.cat((logits_all,logits),0)
        
            order = -1
            gold = []
            for i in range(len(dataset)):
                qid = dataset[i][-1]
                label = dataset[i][1]
                if qid == order:
                    j += 1
                    if label == 1:
                        gold.append((qid,j))
                else:
                    order = qid
                    j = 0
                    if label == 1:
                        gold.append((qid,j))

            label_order = []
            order = -1
            for i in range(len(gold)):
                if gold[i][0] == order:
                    templist.append(gold[i][1])
                elif gold[i][0] != order:
                    order=gold[i][0]
                    if i > 0:
                        label_order.append(templist)
                    templist = []
                    templist.append(gold[i][1])
            label_order.append(templist)

            order = -1
            score_list = []
            for i in range(len(logits_all)):
                score = float(logits_all[i][1])
                qid=int(dataset[i][-1])
                if qid == order:
                    templist.append(score)
                else:
                    order = qid
                    if i > 0:
                        score_list.append(templist)
                    templist = []
                    templist.append(score)
            score_list.append(templist)

            rank = []
            pred = []
            print(len(score_list))
            print(len(label_order))
            for i in range(len(score_list)):
                if len(label_order[i])==1:
                    if label_order[i][0] < len(score_list[i]):
                        true_score = score_list[i][label_order[i][0]]
                        score_list[i].sort(reverse=True)
                        for j in range(len(score_list[i])):
                            if score_list[i][j] == true_score:
                                rank.append(1 / (j + 1))
                    else:
                        rank.append(0)

                else:
                    true_rank = len(score_list[i])
                    for k in range(len(label_order[i])):
                        if label_order[i][k] < len(score_list[i]):
                            true_score = score_list[i][label_order[i][k]]
                            temp = sorted(score_list[i],reverse=True)
                            for j in range(len(temp)):
                                if temp[j] == true_score:
                                    if j < true_rank:
                                        true_rank = j
                    if true_rank < len(score_list[i]):
                        rank.append(1 / (true_rank + 1))
                    else:
                        rank.append(0)
            MRR = sum(rank) / len(rank)
            print("MRR", MRR)
            return MRR



    #========================================================================EMBEDDING EXTRACTION===============================================================
    # Embedding extraction phase
    print("Starting embedding extraction.")
    dataset = read_dataset(args.train_path, filename_data_mapped_id_uri=args.train_entity_mapping, 
                            workers_num=args.workers_num, max_sentences=args.train_max_size, 
                            max_entities=args.max_entities)
        
    print("Converting data to tensors for embedding extraction.")
    batch_size = args.batch_size
        
    # Convert to tensors
    input_ids = torch.LongTensor([example[0] for example in dataset])
    label_ids = torch.LongTensor([example[1] for example in dataset])
    mask_ids = torch.LongTensor([example[2] for example in dataset])
    pos_ids = torch.LongTensor([example[3] for example in dataset])
    vms = [example[4] for example in dataset]
    sent_trees = [example[5] for example in dataset]  # Get sentence trees
    cleaned_texts = [example[6] for example in dataset]  # Get cleaned texts
        
    instances_num = len(dataset)
    print(f"Processing {instances_num} instances for embedding extraction")
        
    # Set model to eval mode
    model.eval()
        
    with torch.no_grad():
        for i, (input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vms_batch, sent_trees_batch, cleaned_texts_batch) in enumerate(batch_loader(batch_size, input_ids, label_ids, mask_ids, pos_ids, vms, sent_trees, cleaned_texts)):
            # Convert vms to tensor
            vms_batch = [vms.astype(int) for vms in vms_batch]
            vms_batch = torch.LongTensor(vms_batch)
                
            # Move to device
            input_ids_batch = input_ids_batch.to(device)
            label_ids_batch = label_ids_batch.to(device)
            mask_ids_batch = mask_ids_batch.to(device)
            pos_ids_batch = pos_ids_batch.to(device)
            vms_batch = vms_batch.to(device)
                
            # Set the current sentence tree and cleaned text before forward pass
            if len(sent_trees_batch) > 0:
                model.set_sent_tree(sent_trees_batch[0])
            if len(cleaned_texts_batch) > 0:
                model.set_cleaned_text(cleaned_texts_batch[0])
                
            # Forward pass - this will trigger the embedding extraction and printing
            _, _ = model(input_ids_batch, label_ids_batch, mask_ids_batch, pos=pos_ids_batch, vm=vms_batch)
                
            if (i + 1) % args.report_steps == 0:
                print(f"Processed {(i+1) * batch_size} instances")
                sys.stdout.flush()
        
    print("Embedding extraction completed.")

    '''print("PHRASES")
    print(PHRASES)
    print("EMBEDDINGS")
    print(EMBEDDINGS)
    print("len(PHRASES)")
    print(len(PHRASES))
    print("len(EMBEDDINGS)")
    print(len(EMBEDDINGS))'''
    # Create final DataFrame with all collected embeddings and phrases
    embedding_data = {
        'phrases': PHRASES,
        'embeddings': EMBEDDINGS
    }

    # Convert to DataFrame and save to parquet
    embedding_df = pd.DataFrame(embedding_data)
    #print("Final embedding DataFrame:")
    #print(embedding_df)
    print("\nSaving embeddings to embeddings.parquet")
    embedding_df.to_parquet('../../../natuke/embeddings.parquet', index=False)

    
    
    #===========================================================================================================================================================




    if not args.skip_train:
        # Training phase.
        print("Start training.")
        trainset = read_dataset(args.train_path, filename_data_mapped_id_uri=args.train_entity_mapping, workers_num=args.workers_num, max_sentences=args.train_max_size, max_entities=args.max_entities)
        print("Shuffling dataset")
        random.shuffle(trainset)
        # print("NOT Shuffling dataset")
        instances_num = len(trainset)
        batch_size = args.batch_size

        print("Trans data to tensor.")
        print("input_ids")
        input_ids = torch.LongTensor([example[0] for example in trainset])
        print("label_ids")
        label_ids = torch.LongTensor([example[1] for example in trainset])
        print("mask_ids")
        mask_ids = torch.LongTensor([example[2] for example in trainset])
        print("pos_ids")
        pos_ids = torch.LongTensor([example[3] for example in trainset])
        print("vms")
        vms = [example[4] for example in trainset]

        train_steps = int(instances_num * args.epochs_num / batch_size) + 1

        print("Batch size: ", batch_size)
        print("The number of training instances:", instances_num)

        param_optimizer = list(model.named_parameters())
        # print("param_optimizer: ", param_optimizer)
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
        ]
        optimizer = BertAdam(optimizer_grouped_parameters, lr=args.learning_rate, warmup=args.warmup, t_total=train_steps, e=1e-08)
        
    
        total_loss = 0.
        result = 0.0
        best_result = 0.0
        for epoch in range(1, args.epochs_num+1):
            model.train()
            for i, (input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vms_batch) in enumerate(batch_loader(batch_size, input_ids, label_ids, mask_ids, pos_ids, vms)):
                model.zero_grad()

                vms_batch = [ vms.astype(int) for vms in vms_batch ]
                vms_batch = torch.LongTensor(vms_batch)

                input_ids_batch = input_ids_batch.to(device)
                label_ids_batch = label_ids_batch.to(device)
                mask_ids_batch = mask_ids_batch.to(device)
                pos_ids_batch = pos_ids_batch.to(device)
                vms_batch = vms_batch.to(device)

                loss, _ = model(input_ids_batch, label_ids_batch, mask_ids_batch, pos=pos_ids_batch, vm=vms_batch)
                if torch.cuda.device_count() > 1:
                    loss = torch.mean(loss)
                total_loss += loss.item()
                if (i + 1) % args.report_steps == 0:
                    print("Epoch id: {}, Training steps: {}, Avg loss: {:.3f}".format(epoch, i+1, total_loss / args.report_steps))
                    sys.stdout.flush()
                    total_loss = 0.
                loss.backward()
                optimizer.step()

            print("Start evaluation on dev dataset.")
            result = evaluate(args, False, epoch=epoch)
            if result > best_result:
                best_result = result
                save_model(model, args.output_model_path)
            else:
                continue

        # print("Start evaluation on test dataset.")
        # evaluate(args, True)

    # If we are repeating validation and we skip train we want to repeat also threshold tuning
    # We also use the original fine tuned model for evaluation
    if args.skip_train and args.repeat_validation:
        result = evaluate(args, False)
        args.output_model_path = args.finetuned_model
    # Evaluation phase.
    print("Final evaluation on the test dataset.")

    
    if torch.cuda.device_count() > 1:
        model.module.load_state_dict(torch.load(args.output_model_path))
    else:
        model.load_state_dict(torch.load(args.output_model_path))
    evaluate(args, True)

    

if __name__ == "__main__":
    main()
