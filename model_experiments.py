from transformers import BertModel, BertTokenizer, AutoModel, AutoTokenizer
from collections import defaultdict
import pandas as pd


from utils import *


def create_token_diff_avg_total_df(sentence_encodings, tokenizer, data, filename, extend=0):
    print("create dataframe for different tokens average...")
    for sample in data.keys():
        embeddings = get_different_tokens_average_embedding(sentence_encodings[sample], tokenizer, data[sample], extend)
        score, idiomatic_score, non_idiomatic_score = get_sentence_similarity_scores(embeddings)
        data[sample]['interaction score'] = score
        data[sample]['idiomatic score'] = idiomatic_score
        data[sample]['non idiomatic score'] = non_idiomatic_score

    dataframe = pd.DataFrame.from_dict(data)
    write_dataframe_to_json(dataframe, filename)

def create_pool_total_df(sentence_encodings, tokenizer, data, filename):
    print("create dataframe for pooling...")
    for sample in data.keys():
        embeddings = get_pooling_embedding(sentence_encodings[sample])
        score, idiomatic_score, non_idiomatic_score = get_sentence_similarity_scores(embeddings)
        data[sample]['interaction score'] = score
        data[sample]['idiomatic score'] = idiomatic_score
        data[sample]['non idiomatic score'] = non_idiomatic_score

    dataframe = pd.DataFrame.from_dict(data)
    write_dataframe_to_json(dataframe, filename)

def create_len_score_different_avg_df(sentence_encodings, tokenizer, data, filename):
    print("create dataframe for length vs score fo different tokens average...")
    count = 0
    lengths_vs_embedding_val = {'idiomatic': defaultdict(dict), 'non idiomatic': defaultdict(dict)}
    for sample in data.keys():
        sentence_pair_sim_lens = get_sentence_sim_length(tokenizer, data[sample])
        embeddings = get_different_tokens_average_embedding(sentence_encodings[sample], tokenizer, data[sample])

        _, idiomatic_score, non_idiomatic_score = get_sentence_similarity_scores(embeddings)

        lengths_vs_embedding_val['idiomatic']['score'][count]= idiomatic_score
        lengths_vs_embedding_val['idiomatic']['sim length'][count] = sentence_pair_sim_lens[0]
        count += 1
        lengths_vs_embedding_val['non idiomatic']['score'][count] = non_idiomatic_score
        lengths_vs_embedding_val['non idiomatic']['sim length'][count] = sentence_pair_sim_lens[1]
        count += 1

    dataframe = pd.DataFrame.from_dict(lengths_vs_embedding_val)
    print(dataframe)
    write_dataframe_to_json(dataframe, filename)

def create_len_score_pool_df(sentence_encodings, tokenizer, data, filename):
    print("create dataframe for length vs score for pooling...")
    count = 0
    lengths_vs_embedding_val = {'idiomatic': defaultdict(dict), 'non idiomatic': defaultdict(dict)}
    for sample in data.keys():
        sentence_pair_sim_lens = get_sentence_sim_length(tokenizer, data[sample])
        embeddings = get_pooling_embedding(sentence_encodings[sample])

        _, idiomatic_score, non_idiomatic_score = get_sentence_similarity_scores(embeddings)

        lengths_vs_embedding_val['idiomatic']['score'][count]= idiomatic_score
        lengths_vs_embedding_val['idiomatic']['sim length'][count] = sentence_pair_sim_lens[0]
        count += 1
        lengths_vs_embedding_val['non idiomatic']['score'][count] = non_idiomatic_score
        lengths_vs_embedding_val['non idiomatic']['sim length'][count] = sentence_pair_sim_lens[1]
        count += 1

    dataframe = pd.DataFrame.from_dict(lengths_vs_embedding_val)
    write_dataframe_to_json(dataframe, filename)

def create_len_diff_df(tokenizer, data, filename):
    sentence_len_diff = {}
    for sample in data.keys():
        sentence_lengths = get_sentence_token_lengths(tokenizer, data[sample])
        sentence_len_diff[sample] = sentence_lengths[2] - sentence_lengths[0]
    df = pd.DataFrame(sentence_len_diff, index=[0])
    write_dataframe_to_json(df, filename)


def main(model, is_sentence_transformer, data_filename, save_filename):
    if is_sentence_transformer:
        print("Loading model...")
        model = AutoModel.from_pretrained(model)
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
    else:
        print("Loading model...")
        model = BertModel.from_pretrained("bert-base-uncased")
        print("Loading tokenizer...")
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


    data = load_sentences_with_names("data/" + data_filename)
    encodings = {}
    print("Encode data...")
    for sample in data.keys():
        encodings[sample] = get_sentence_encodings(model, tokenizer, data[sample])
    
    # print("---------------Create Dataframes-----------")
    create_pool_total_df(encodings, tokenizer, data, save_filename)
    # create_len_diff_df(tokenizer, data, "sentences_diff_lengths_mpnet.json")


if __name__ == '__main__':
    main('sentence-transformers/all-mpnet-base-v2', True, "sentences_max_positive_revised_mpnet.txt", "similarity_scores_max_revised_2_mpnet.json")