# general imports
import os
import spacy
import pickle
import pandas as pd
# own path/ class imports
from file_paths import *
from classes.text_cleaning import *
from classes.iso_text_cleaning import *
from classes.topic_modeling import *
from classes.k_means_bert import *
from classes.constraint_existence_check import *
from classes.in_depth_comparison import *
from classes.phrase_similarity_computation import *
from classes.s_bert_sentence_pairs import *
from classes.deviation_counter import *

## Application Selection ########################################START
direct_s_bert = True #if False --> approach calculated with topic model, kmeans and word2vev 
iso = False #if False --> running with gdpr setup
rea_only_signal = False #if False --> gdpr realization input is not filtered to contain only sentences with signalwords
#thresholds:
gamma_s_bert = 0.67 #0.67 #used for sentence mapping 
gamma_grouping = 0.9 #used for sentence mapping in k-means & topic Model approach
gamma_one = 0.3 #used for subject phrase mapping
gamma_two = 0.32 #used for verb phrase mapping
gamma_three = 0.3 #used for object phrase mapping
################################################################# END

# Create the nlp object
nlp = spacy.load('en_core_web_lg')
nlp.add_pipe("merge_entities")

## parse defined lists of constraint signalwords, sequencemarkers and stopwords ########################### START
def read_defined_lists(directory): 
  '''reads in defined txts of constraint signalwords, sequencemarkers and stopwords as lists
  Input: .txt
  Output: list'''
  try:
    with open(directory) as f:
      defined_list = f.read().splitlines()
  except FileNotFoundError:
      print("Wrong file or file path.")
      quit()
  return defined_list


if iso:
  signalwords = read_defined_lists(ISO_SIGNALWORDS)
  ISMS_words = read_defined_lists(ISO_REA_SPEZIFICATION1)
  top_management_words = read_defined_lists(ISO_REA_SPEZIFICATION2)
else:
  signalwords = read_defined_lists(GDPR_SIGNALWORDS)
  controller_words = read_defined_lists(GDPR_REA_SPEZIFICATION1)
  data_protection_officer_words = read_defined_lists(GDPR_REA_SPEZIFICATION2)
  management_words = read_defined_lists(GDPR_REA_SPEZIFICATION3)

################################################################# END

## getting the anaphora resolved paragraphs from other scripts (as chosen anaphora resolution requires different python & spacy version) ## START

if iso:
  with open(join(INPUT_DIRECTORY, "iso_reg_para_anaphora_resolved.txt"), "rb") as fp: 
    reg_para_anaphora_resolved = pickle.load(fp)
  with open(join(INPUT_DIRECTORY, "iso_rea_para_anaphora_resolved.txt"), "rb") as fp: 
    rea_para_anaphora_resolved = pickle.load(fp)
else:
  with open(join(INPUT_DIRECTORY, "gdpr_reg_para_anaphora_resolved.txt"), "rb") as fp: 
    reg_para_anaphora_resolved = pickle.load(fp)
  with open(join(INPUT_DIRECTORY, "gdpr_rea_para_anaphora_resolved.txt"), "rb") as fp: 
    rea_para_anaphora_resolved = pickle.load(fp)
  
################################################################# END

## calling classes ############################################ START
#Text cleaning
if iso: 
  itc = Iso_Text_Cleaning(nlp, signalwords, ISMS_words, top_management_words)
  reg_relevant_sentences = itc.get_relevant_sentences(reg_para_anaphora_resolved)
  rea_relevant_sentences = itc.get_relevant_sentences(rea_para_anaphora_resolved)
else:
  tc = Text_Cleaning(nlp, signalwords, controller_words, data_protection_officer_words, management_words)
  reg_relevant_sentences = tc.get_relevant_sentences(reg_para_anaphora_resolved)
  if rea_only_signal:
    rea_relevant_sentences = tc.get_relevant_sentences(rea_para_anaphora_resolved)
  else:
    rea_relevant_sentences = tc.get_relevant_sentences_no_sig_filter(rea_para_anaphora_resolved)
  #save all input constraints before matching for evaluation purposes  
  pd.DataFrame(reg_relevant_sentences).to_excel(join(INTERMEDIATE_DIRECTORY, "gdpr_reg_relevant_sentences.xlsx"))  
  pd.DataFrame(rea_relevant_sentences).to_excel(join(INTERMEDIATE_DIRECTORY, "gdpr_rea_relevant_sentences.xlsx"))  


if direct_s_bert:
  # S-BERT Finding Sentance Pairs 
  sbsp = S_Bert_Sentence_Pairs(gamma_s_bert)
  df_bert_sent_pairs = sbsp.get_bert_sim_sent_pairs(reg_relevant_sentences, rea_relevant_sentences)
  df_bert_sent_pairs.to_excel(join(INTERMEDIATE_DIRECTORY, "df_bert_sent_pairs.xlsx"))  
  count_unmapped_reg_sent = sbsp.get_unmapped_reg_sentences(df_bert_sent_pairs, reg_relevant_sentences)
  count_unmapped_rea_sent = sbsp.get_unmapped_rea_sentences(df_bert_sent_pairs, rea_relevant_sentences)
  count_mapped_rea_sent = sbsp.get_mapped_rea_sentences()
  # Phrase Extraction (splitting the sentences into Sub/Verb/Obj phrases) from each Sentence Pair
  idc = In_Depth_Comparison(signalwords, nlp)
  s_bert_constraint_phrases_reg = idc.get_sentence_phrases(df_bert_sent_pairs, 'reg')
  s_bert_constraint_phrases = idc.get_sentence_phrases(s_bert_constraint_phrases_reg, 'rea')
  s_bert_constraint_phrases.to_excel(join(INTERMEDIATE_DIRECTORY, "s_bert_constraint_phrases.xlsx"))  
  # Similarities for Phrases (calculating similarity between the phrases)
  psc = Phrase_Similarity_Computation()
  similarity_s_bert_constraint_phrases = psc.get_phrase_similarities(s_bert_constraint_phrases)
  similarity_s_bert_constraint_phrases.to_excel(join(INTERMEDIATE_DIRECTORY, "similarity_s_bert_constraint_phrases.xlsx")) 
  # Result (checking if deviations of different types can be detected)
  dc = Deviation_Counter(nlp, gamma_one, gamma_two, gamma_three)
  s_bert_master_results_df = dc.get_deviation_flag(similarity_s_bert_constraint_phrases)
  s_bert_master_results_df.to_excel(join(RESULT_DIRECTORY, "s_bert_master_results_df.xlsx")) 
  s_bert_overview_results_df = dc.aggreagte_deviation_count(s_bert_master_results_df, count_unmapped_reg_sent, count_unmapped_rea_sent, count_mapped_rea_sent)
  s_bert_overview_results_df.to_excel(join(RESULT_DIRECTORY, "s_bert_overview_results_df.xlsx")) 
  print('Calculations finished.')
else:
  # Grouping Topic Model
  tm = Topic_Modeling(reg_relevant_sentences, rea_relevant_sentences, nlp)
  df_topic_models = tm.create_topics_dataframe()
  # Grouping Clustering
  kmb = K_Means_BERT(reg_relevant_sentences, rea_relevant_sentences, nlp, df_topic_models)
  df_topic_kmeans_groups = kmb.predict_clusters_to_df()
  df_topic_kmeans_groups.to_excel(join(INTERMEDIATE_DIRECTORY, "df_grouping_results.xlsx"))  
  # Check constraint completness (similarity computation between sentences within clusters)
  cec = Constraint_Existence_Check(nlp, df_topic_kmeans_groups, gamma_grouping)
  topic_model_sentence_pairs_df = cec.split_results_by_similarity(cec.topic_model_similarities_df, 'df_topic_model_reg_sent_without_match.xlsx', 'df_topic_model_rea_sent_without_match.xlsx')
  kmeans_bert_sentence_pairs_df = cec.split_results_by_similarity(cec.kmeans_bert_similarities_df, 'df_kmeans_bert_reg_sent_without_match.xlsx', 'df_kmeans_bert_rea_sent_without_match.xlsx')
  topic_model_sentence_pairs_df.to_excel(join(INTERMEDIATE_DIRECTORY, "topic_model_sentence_pairs_df.xlsx")) 
  kmeans_bert_sentence_pairs_df.to_excel(join(INTERMEDIATE_DIRECTORY, "kmeans_bert_sentence_pairs_df.xlsx")) 
  # In depth comparison (on phrase level) of Sentence Pairs (splitting the sentences into Sub/Verb/Obj phrases)
  idc = In_Depth_Comparison(signalwords, nlp)
  topic_model_constraint_phrases_reg = idc.get_sentence_phrases(topic_model_sentence_pairs_df, 'reg')
  topic_model_constraint_phrases = idc.get_sentence_phrases(topic_model_constraint_phrases_reg, 'rea')
  kmeans_bert_constraint_phrases_reg = idc.get_sentence_phrases(kmeans_bert_sentence_pairs_df, 'reg')
  kmeans_bert_constraint_phrases = idc.get_sentence_phrases(kmeans_bert_constraint_phrases_reg, 'rea') 
  # Similarities for Phrases
  psc = Phrase_Similarity_Computation(nlp)
  similarity_topic_model_constraint_phrases = psc.get_phrase_similarities(topic_model_constraint_phrases)
  similarity_kmeans_bert_constraint_phrases = psc.get_phrase_similarities(kmeans_bert_constraint_phrases)
  similarity_topic_model_constraint_phrases.to_excel(join(INTERMEDIATE_DIRECTORY, "similarity_topic_model_constraint_phrases.xlsx"))  
  similarity_kmeans_bert_constraint_phrases.to_excel(join(INTERMEDIATE_DIRECTORY, "similarity_kmeans_bert_constraint_phrases.xlsx")) 

