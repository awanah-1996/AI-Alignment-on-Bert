from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import torch
import json

NUMBER_SENTENCES_PER_SAMPLE = 4
NUMBER_SENTENCE_PAIRS_PER_SAMPLE = 2

def get_sentence_key_from_idx(idx):
    return 'sentence' + str(idx)



def load_sentences(file_name):
    data = defaultdict(dict)
    with open(file_name) as file:
        next_line = next(file)
        sample_count = 0
        while next_line:
            sample_key = "sample" + str(sample_count)
            for sentence in range(NUMBER_SENTENCES_PER_SAMPLE):
                data[sample_key][get_sentence_key_from_idx(sentence)] = next_line.rstrip()
                try:
                    next_line = next(file)
                except StopIteration:
                    next_line = None
            sample_count += 1
    return dict(data)


'''
    returns Dictionary with:
    {idiom: {
        sentence1: "",
        sentence2: "",
        sentence3: "",
        sentence4: ""
    }}
'''
def load_sentences_with_names(file_name):
    def get_next_line(file):
        next_line = next(file).rstrip()
        while next_line == "":
            next_line = next(file).rstrip()
        return next_line

    data = defaultdict(dict)
    with open(file_name) as file:
        next_line = get_next_line(file)
        sample_count = 0
        while next_line:
            idiom = next_line
            next_line = get_next_line(file)
            for sentence in range(4):
                data[idiom][get_sentence_key_from_idx(sentence)] = next_line.rstrip()
                try:
                    next_line = get_next_line(file)
                except StopIteration:
                    next_line = None
            sample_count += 1
    return dict(data)

def create_sentence_to_index_dic(data):
    sample_2_idx = {}
    for i, sample in enumerate(data.keys()):
        sample_2_idx[sample] = i
    with open("data_exports/sample_2_idx.json", "w") as f:
        json.dump(sample_2_idx, f, indent=2)


def write_dataframe_to_json(dataframe, file):
    with open("data_exports/" + file, '+w') as f:
        f.write(dataframe.to_json(orient='split'))

def get_sentence_encodings(model, tokenizer, sentences):
    sentence_encodings = {}
    with torch.no_grad():
        for sentence_idx in range(NUMBER_SENTENCES_PER_SAMPLE):
            encoded_input = tokenizer(sentences[get_sentence_key_from_idx(sentence_idx)], add_special_tokens=True, return_tensors='pt')
            output = model(**encoded_input)
            sentence_encodings[sentence_idx] = output
    return sentence_encodings


def extend_list_indeces(tokenized_sentences, indeces_tokens, extend, offset):
    for i in range(NUMBER_SENTENCES_PER_SAMPLE):
        for e in range(extend):
            new_min_idx = indeces_tokens[i][0] - 1
            new_max_idx = indeces_tokens[i][-1] + 1
            match e % 2:
                case 0:
                    if new_min_idx >= offset:
                        indeces_tokens[i].insert(offset, new_min_idx)
                    else:
                        if new_max_idx < len(tokenized_sentences[i][offset:]):
                            indeces_tokens[i].append(new_max_idx)
                case 1:
                    if new_max_idx < len(tokenized_sentences[i][offset:]):
                        indeces_tokens[i].append(new_max_idx)
                    else:
                        if new_min_idx >= offset:
                            indeces_tokens[i].insert(offset, new_min_idx)


def get_different_token_indeces(tokenizer, sentences, offset=1, extend=0):
    tokenized_sentences = {}
    indeces_different_tokens = {}

    for idx in range(NUMBER_SENTENCES_PER_SAMPLE):
        tokens = tokenizer.tokenize(sentences[get_sentence_key_from_idx(idx)])
        tokenized_sentences[idx] = ['[SPACE]'] * offset + tokens

    for pair in range(NUMBER_SENTENCE_PAIRS_PER_SAMPLE):
        min_identical_range = 0
        max_identical_range = 0 #from the back
        while tokenized_sentences[2*pair][min_identical_range] == tokenized_sentences[2*pair + 1][min_identical_range]:
            min_identical_range +=1
        while tokenized_sentences[2*pair][max_identical_range - 1] == tokenized_sentences[2*pair + 1][max_identical_range - 1]:
            max_identical_range -= 1
        for i in range(2):
            indeces_different_tokens[2*pair + i] = [idx for idx in range(min_identical_range, len(tokenized_sentences[2*pair + i]) + max_identical_range)]
    if extend > 0:
        extend_list_indeces(tokenized_sentences, indeces_different_tokens, extend, offset)
    return indeces_different_tokens


def get_different_tokens_average_embedding(sentence_encodings, tokenizer, sentences, surround_extend=0):
    token_indeces = get_different_token_indeces(tokenizer, sentences, extend=surround_extend)
    sentences_avg = {}
    for i in range(NUMBER_SENTENCES_PER_SAMPLE):
        sentence_encoding = sentence_encodings[i].last_hidden_state.squeeze() 
        token_embeddings = sentence_encoding[token_indeces[i]]
        sentences_avg[i] = torch.mean(token_embeddings, dim=0).unsqueeze(0)
    return sentences_avg
    
def get_pooling_embedding(sentence_encodings):
    sentences_pooling = {}
    for i in range(NUMBER_SENTENCES_PER_SAMPLE):
        sentences_pooling[i] = sentence_encodings[i].pooler_output
    return sentences_pooling




def get_sentence_similarity_scores(sentence_embeddings):
    cosine_similarity_idiomatic_meaning = cosine_similarity(sentence_embeddings[0], sentence_embeddings[1])
    cosine_similarity_non_idiomatic_meaning = cosine_similarity(sentence_embeddings[2], sentence_embeddings[3])
    similarity_score = cosine_similarity_non_idiomatic_meaning - cosine_similarity_idiomatic_meaning
    return similarity_score.item(), cosine_similarity_idiomatic_meaning.item(), cosine_similarity_non_idiomatic_meaning.item()

def get_sentence_sim_length(tokenizer, sentences):
    sentence_pair_sim_lengths = {}
    diff_token_indeces = get_different_token_indeces(tokenizer, sentences)
    tokens_s1 = tokenizer.tokenize(sentences[get_sentence_key_from_idx(0)])
    tokens_s2 = tokenizer.tokenize(sentences[get_sentence_key_from_idx(2)])
    sentence_pair_sim_lengths[0] = len(tokens_s1) - len(diff_token_indeces[0])
    sentence_pair_sim_lengths[1] = len(tokens_s2) - len(diff_token_indeces[2])
    return sentence_pair_sim_lengths

def get_sentence_sim_length_fract(tokenizer, sentences):
    sentence_pair_sim_lengths = {}
    diff_token_indeces = get_different_token_indeces(tokenizer, sentences)
    tokens_s1 = tokenizer.tokenize(sentences[get_sentence_key_from_idx(0)])
    tokens_s2 = tokenizer.tokenize(sentences[get_sentence_key_from_idx(2)])
    avg_diff_token_count_s1 = (len(diff_token_indeces[0]) + len(diff_token_indeces[1])) / 2
    avg_diff_token_count_s2 = (len(diff_token_indeces[2]) + len(diff_token_indeces[3])) / 2
    sentence_pair_sim_lengths[0] = avg_diff_token_count_s1 / (len(tokens_s1) - len(diff_token_indeces[0]))
    sentence_pair_sim_lengths[1] = avg_diff_token_count_s2 / (len(tokens_s2) - len(diff_token_indeces[2]))
    return sentence_pair_sim_lengths

def get_sentence_token_lengths(tokenizer, sentences):
    sentences_lengths = {}
    for i in range(NUMBER_SENTENCES_PER_SAMPLE):
        sentences_lengths[i] = len(tokenizer.tokenize(sentences[get_sentence_key_from_idx(i)]))

    return sentences_lengths