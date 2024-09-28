import pandas as pd
import matplotlib.pyplot as plt
import json
import numpy as np
import textwrap

def load_sentence_2_dic():
    with open("data_exports/sample_2_idx.json") as f:
        return json.load(f)

def print_revised_vs_old(source, revised, save):
    with open("data_exports/" + source) as f:
        df_pool = pd.read_json(f, orient='split')
    with open("data_exports/" + revised) as f:
        df_pool_r = pd.read_json(f, orient='split')
    
    width = 0.3
    x_labels = df_pool_r.columns
    wrapped_labels = [textwrap.fill(label, width=10) for label in x_labels]
    ind = np.arange(len(x_labels))
    df_pool_values = df_pool[x_labels].loc['interaction score'].values
    df_pool_r_values = df_pool_r.loc['interaction score'].values

    plt.figure(figsize=(12,10))

    plt.bar(ind, df_pool_values, width, label='Original')
    plt.bar(ind + width, df_pool_r_values, width, label='Revised')
    plt.xticks(ind + width / 2, wrapped_labels, rotation=45)

    plt.xlabel('Sample Idiom')
    plt.ylabel('Interaction Score')
    plt.title('Comparison of Interaction Scores with Revised Sentences mpnet')
    plt.legend()
    plt.savefig("evaluation_exports/" + save)


def print_len_vs_score(source, save, alpha, title):
    f = open("data_exports/" + source)
    df = pd.read_json(f, orient='split')
    
    idiom = df['idiomatic']
    n_idiom = df['non idiomatic']
    x_ticks = np.arange(len(idiom.loc['score'].items()))
    plt.scatter(idiom.loc['sim length'].values(), idiom.loc['score'].values(), label='Idiomatic', color='green', alpha=alpha)
    plt.scatter(n_idiom.loc['sim length'].values(), n_idiom.loc['score'].values(), label='Non Idiomatic', color='red', alpha=alpha)

    plt.title(title)
    plt.xlabel("Amount Identical Tokens")
    plt.ylabel('Cosine Similarity of sentence pairs')
    plt.legend()
    plt.savefig('evaluation_exports/' + save)
    f.close()

# It looks like taking the fraction does not solve the issue of any correlation between length and score
def print_len_vs_score_fract():
    f = open("data_exports/len_vs_score_fract.json")
    df = pd.read_json(f, orient='split')
    df.plot.scatter('sim length', 'score')
    plt.savefig('evaluation_exports/len_vs_score_diff_tokens_fract.png')
    f.close()


def print_similarity_scores(source, save, title, sorted=False):
    sim_pooling = open("data_exports/" + source)
    df_pool = pd.read_json(sim_pooling, orient='split')
    
    interaction_scores_pool = df_pool.loc["interaction score"].values
    x_labels = [i for i in range(len(df_pool.columns))]
    if sorted:
        x_labels = np.argsort(interaction_scores_pool)
        interaction_scores_pool = interaction_scores_pool[x_labels]
    # # Plotting the bar chart
    plt.figure(figsize=(15,10))
    plt.bar(np.arange(len(x_labels)), interaction_scores_pool)
    plt.xticks(np.arange(len(x_labels)), x_labels, rotation=90)
    plt.title(title)
    plt.ylabel('Similarity Score')
    plt.xlabel('Numbered Idioms')
    plt.savefig('evaluation_exports/' + save)


def print_i_non_i_scatter_pool(data_file, save_file, title):
    sim_pooling = open("data_exports/" + data_file)
    df_pool = pd.read_json(sim_pooling, orient='split')
    scores_idiomatic_pool = df_pool.loc["idiomatic score"].values
    scores_non_idiomatic_pool = df_pool.loc["non idiomatic score"].values

    x_labels = [i for i in range(len(df_pool.columns))]
    plt.figure(figsize=(15,6))
    plt.scatter(x_labels, scores_idiomatic_pool, label='Idiomatic', marker='o', color='green')
    plt.scatter(x_labels, scores_non_idiomatic_pool, label='Non Idiomatic', marker='o',color='red')

    plt.title(title)
    plt.xlabel("Numbered Idioms")
    plt.ylabel('Cosine Similarity of sentence pairs')
    plt.legend()
    plt.savefig('evaluation_exports/' + save_file)

def print_similarity_scores_both(source_i, source_ni, save, title, sorted=False):
    with open("data_exports/" + source_i) as f:
        df_i = pd.read_json(f, orient='split')
    with open("data_exports/" + source_ni) as f:
        df_ni = pd.read_json(f, orient='split')

    scores_i = df_i.loc["interaction score"].values
    scores_ni = df_ni.loc["interaction score"].values
    scale = sum(abs(scores_i))/sum(abs(scores_ni))
    scores_ni *= scale
    # x_labels = [i for i in range(len(df_pool.columns))]
    x_range = np.arange(len(df_i.columns))
    x_ticks = np.arange(len(df_i.columns))
    if sorted:
        x_ticks = np.argsort(scores_ni)
        scores_ni = scores_ni[x_ticks]
        scores_i = scores_i[x_ticks]

    color_pool = 'green'
    color_diff = 'red'
    a = 0.8
    width = 0.5

    plt.figure(figsize=(15,10))

    plt.bar(x_range, scores_i, width, label='Bert Base', color=color_pool, alpha=a)
    plt.bar(x_range + width, scores_ni, width, label='mpnet', color=color_diff, alpha=a)

    plt.axhline(y=np.mean(scores_i), color=color_pool, linestyle='--', label='Overall Mean (Bert)')
    plt.axhline(y=np.mean(scores_ni), color=color_diff, linestyle='--', label='Overall Mean (mpnet)')

    plt.xticks(x_range + width / 2, x_ticks, rotation=90)
    plt.title(title)
    plt.xlabel("Numbered Idioms")
    plt.ylabel('Cosine Similarity of Sentence Pairs')
    plt.legend()
    # plt.show()
    plt.savefig('evaluation_exports/' + save)

def print_lengths(filename, save_path, title):
    with open("data_exports/" + filename) as f:
        df_lengths = pd.read_json(f, orient='split')

    plt.figure(figsize=(15,10))
    x_labels = np.argsort(df_lengths.loc[0].values)
    df_lengths = df_lengths.loc[0].values[x_labels]
    plt.scatter(np.arange(len(x_labels)), df_lengths)
    plt.xticks(np.arange(len(x_labels)), x_labels, rotation=90)
    plt.xlabel("Numbered Idioms")
    plt.ylabel("Difference in Sentence Lengths")
    plt.title(title)
    plt.savefig('evaluation_exports/' + save_path)

def lengths_on_sim_scores(len_file, score_file, save_file, title):
    with open("data_exports/" + len_file) as f:
        df_lengths = pd.read_json(f, orient='split')
    with open("data_exports/" + score_file) as f:
        df_score = pd.read_json(f, orient='split')

    interaction_scores_pool = df_score.loc["interaction score"].values
    x_labels = np.argsort(interaction_scores_pool)
    df_lengths = df_lengths.loc[0].values[x_labels]
    interaction_scores_pool = interaction_scores_pool[x_labels]
    x_range = np.arange(len(x_labels))
    
    fig, ax1 = plt.subplots(figsize=(15,10))

    ax1.bar(x_range, interaction_scores_pool, color='red', alpha=0.7)
    ax1.set_xticks(x_range, x_labels, rotation=90)
    ax1.set_ylabel("Similarity Scores")
    ax1.set_xlabel('Numbered Idioms')

    ax2 = ax1.twinx()
    ax2.scatter(x_range, df_lengths, color='blue', alpha=0.5)
    ax2.set_ylabel("Length Scores")

    plt.title(title)
    plt.show()
if __name__ == '__main__':
    # print_similarity_scores_both("similarity_scores_pool.json", "similarity_scores_pool_mpnet.json", "similarity_scores_mpnet_bert.png", "Bert vs. Sentence Bert")
    print_revised_vs_old("similarity_scores_pool_mpnet.json", "similarity_scores_max_revised_2_mpnet.json", "comparison_revised_old_2_mpnet.png")
    # print_i_non_i_scatter_pool("similarity_scores_pool_mpnet.json", "scatter_i_non_i_pool_mpnet.png", 'Scatter Plot of Idiomatic and Non-Idiomatic Scores Sentence Bert')
    # print_lengths("sentences_diff_lengths_mpnet.json", "sentences_diff_lengths_mpnet.png", "Differences in Sentence Lengths")
    # lengths_on_sim_scores("sentences_diff_lengths_mpnet.json", "similarity_scores_pool_mpnet.json", "length_on_sim_scores_mpnet.png", "Length Scores and Similarity Scores for Samples")
    # print_similarity_scores("similarity_scores_pool_mpnet.json", "similarity_scors_mpnet_pool.png", "Sorted Similarity Scores for mpnet", True)
    # print_len_vs_score("len_vs_score_diff_0_avg.json", "len_vs_score_only_diff_tokens.png", 0.7, "Length-Score Correlation for only Different Tokens")
    # plt.clf()
    # print_len_vs_score("len_vs_score_pool.json", "len_vs_score_pool.png", 0.3, "Length-Score Correlation for Pooling")
    # print_len_vs_score_avg()
    # print_len_vs_score_fract()
    # print_revised_vs_old("similarity_scores_pool_revised_2.json", "comparison_revised_old_simscores_2.png")
