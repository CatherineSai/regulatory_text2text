
import pandas as pd
import numpy as np


class Phrase_Similarity_Computation:

  def __init__(self, nlp):
    self.nlp = nlp

  def get_phrase_similarities(self, df):
    '''calculates all phrase similarity pairs between reg and rea sentence pair '''
    print("Start Phrase_Similarity_Computation")
    #create new columns (so stey exist for later algorithms even if empty (e.g. sub_2_2_sim might be empty))
    df["sub_1_1_sim"] = np.nan
    df["sub_1_2_sim"] = np.nan
    df["sub_2_1_sim"] = np.nan
    df["sub_2_2_sim"] = np.nan
    df["verb_1_1_sim"] = np.nan
    df["verb_1_2_sim"] = np.nan
    df["verb_2_1_sim"] = np.nan
    df["verb_2_2_sim"] = np.nan
    df["obj_1_1_sim"] = np.nan
    df["obj_1_2_sim"] = np.nan
    df["obj_2_1_sim"] = np.nan
    df["obj_2_2_sim"] = np.nan
    # fill empty fields with numpy NaN (makes it easier for if statements to check if both are npt empty)
    df = df.replace(r'^\s*$', np.nan, regex=True)
    # negation phrase
    for index, row in df.iterrows():
            df.at[index, 'difference_in_negations'] = abs(row['number_negations_reg']- row['number_negations_rea'])
    # subject phrases
    for index, row in df.iterrows():
            doc_1 = self.nlp(row['subject_phrase_1_reg'])
            doc_2 = self.nlp(row['subject_phrase_1_rea'])
            #added setup, because Spacy constructs sentence embedding by averaging the word embeddings
            doc_1_no_stopwords = self.nlp(' '.join([str(t) for t in doc_1 if not t.is_stop]))
            doc_2_no_stopwords = self.nlp(' '.join([str(t) for t in doc_2 if not t.is_stop]))
            df.at[index, 'sub_1_1_sim'] = doc_1_no_stopwords.similarity(doc_2_no_stopwords)
    for index, row in df.iterrows():
            if (pd.notnull(row['subject_phrase_1_reg']) & pd.notnull(row['subject_phrase_2_rea'])):
                    doc_1 = self.nlp(row['subject_phrase_1_reg'])
                    doc_2 = self.nlp(row['subject_phrase_2_rea'])
                    #added setup, because Spacy constructs sentence embedding by averaging the word embeddings
                    doc_1_no_stopwords = self.nlp(' '.join([str(t) for t in doc_1 if not t.is_stop]))
                    doc_2_no_stopwords = self.nlp(' '.join([str(t) for t in doc_2 if not t.is_stop]))
                    df.at[index, 'sub_1_2_sim'] = doc_1_no_stopwords.similarity(doc_2_no_stopwords)
            else:
                    continue
    for index, row in df.iterrows():
            if (pd.notnull(row['subject_phrase_2_reg']) & pd.notnull(row['subject_phrase_1_rea'])):
                    doc_1 = self.nlp(row['subject_phrase_2_reg'])
                    doc_2 = self.nlp(row['subject_phrase_1_rea'])
                    #added setup, because Spacy constructs sentence embedding by averaging the word embeddings
                    doc_1_no_stopwords = self.nlp(' '.join([str(t) for t in doc_1 if not t.is_stop]))
                    doc_2_no_stopwords = self.nlp(' '.join([str(t) for t in doc_2 if not t.is_stop]))
                    df.at[index, 'sub_2_1_sim'] = doc_1_no_stopwords.similarity(doc_2_no_stopwords)
            else:
                    continue
    for index, row in df.iterrows():
            if (pd.notnull(row['subject_phrase_2_reg']) & pd.notnull(row['subject_phrase_2_rea'])):
                    doc_1 = self.nlp(row['subject_phrase_2_reg'])
                    doc_2 = self.nlp(row['subject_phrase_2_rea'])
                    #added setup, because Spacy constructs sentence embedding by averaging the word embeddings
                    doc_1_no_stopwords = self.nlp(' '.join([str(t) for t in doc_1 if not t.is_stop]))
                    doc_2_no_stopwords = self.nlp(' '.join([str(t) for t in doc_2 if not t.is_stop]))
                    df.at[index, 'sub_2_2_sim'] = doc_1_no_stopwords.similarity(doc_2_no_stopwords)
            else:
                    continue
    # verb phrases
    for index, row in df.iterrows():
            doc_1 = self.nlp(row['verb_phrase_1_reg'])
            doc_2 = self.nlp(row['verb_phrase_1_rea'])
            #added setup, because Spacy constructs sentence embedding by averaging the word embeddings
            doc_1_no_stopwords = self.nlp(' '.join([str(t) for t in doc_1 if not t.is_stop]))
            doc_2_no_stopwords = self.nlp(' '.join([str(t) for t in doc_2 if not t.is_stop]))
            df.at[index, 'verb_1_1_sim'] = doc_1_no_stopwords.similarity(doc_2_no_stopwords)
    for index, row in df.iterrows():
            if (pd.notnull(row['verb_phrase_1_reg']) & pd.notnull(row['verb_phrase_2_rea'])):
                    doc_1 = self.nlp(row['verb_phrase_1_reg'])
                    doc_2 = self.nlp(row['verb_phrase_2_rea'])
                    #added setup, because Spacy constructs sentence embedding by averaging the word embeddings
                    doc_1_no_stopwords = self.nlp(' '.join([str(t) for t in doc_1 if not t.is_stop]))
                    doc_2_no_stopwords = self.nlp(' '.join([str(t) for t in doc_2 if not t.is_stop]))
                    df.at[index, 'verb_1_2_sim'] = doc_1_no_stopwords.similarity(doc_2_no_stopwords)
            else:
                    continue
    for index, row in df.iterrows():
            if (pd.notnull(row['verb_phrase_2_reg']) & pd.notnull(row['verb_phrase_1_rea'])):
                    doc_1 = self.nlp(row['verb_phrase_2_reg'])
                    doc_2 = self.nlp(row['verb_phrase_1_rea'])
                    #added setup, because Spacy constructs sentence embedding by averaging the word embeddings
                    doc_1_no_stopwords = self.nlp(' '.join([str(t) for t in doc_1 if not t.is_stop]))
                    doc_2_no_stopwords = self.nlp(' '.join([str(t) for t in doc_2 if not t.is_stop]))
                    df.at[index, 'verb_2_1_sim'] = doc_1_no_stopwords.similarity(doc_2_no_stopwords)
            else:
                    continue
    for index, row in df.iterrows():
            if (pd.notnull(row['verb_phrase_2_reg']) & pd.notnull(row['verb_phrase_2_rea'])):
                    doc_1 = self.nlp(row['verb_phrase_2_reg'])
                    doc_2 = self.nlp(row['verb_phrase_2_rea'])
                    #added setup, because Spacy constructs sentence embedding by averaging the word embeddings
                    doc_1_no_stopwords = self.nlp(' '.join([str(t) for t in doc_1 if not t.is_stop]))
                    doc_2_no_stopwords = self.nlp(' '.join([str(t) for t in doc_2 if not t.is_stop]))
                    df.at[index, 'verb_2_2_sim'] = doc_1_no_stopwords.similarity(doc_2_no_stopwords)
            else:
                    continue
    # object phrases
    for index, row in df.iterrows():
            doc_1 = self.nlp(row['object_phrase_1_reg'])
            doc_2 = self.nlp(row['object_phrase_1_rea'])
            #added setup, because Spacy constructs sentence embedding by averaging the word embeddings
            doc_1_no_stopwords = self.nlp(' '.join([str(t) for t in doc_1 if not t.is_stop]))
            doc_2_no_stopwords = self.nlp(' '.join([str(t) for t in doc_2 if not t.is_stop]))
            df.at[index, 'obj_1_1_sim'] = doc_1_no_stopwords.similarity(doc_2_no_stopwords)
    for index, row in df.iterrows():
            if (pd.notnull(row['object_phrase_1_reg']) & pd.notnull(row['object_phrase_2_rea'])):
                    doc_1 = self.nlp(row['object_phrase_1_reg'])
                    doc_2 = self.nlp(row['object_phrase_2_rea'])
                    #added setup, because Spacy constructs sentence embedding by averaging the word embeddings
                    doc_1_no_stopwords = self.nlp(' '.join([str(t) for t in doc_1 if not t.is_stop]))
                    doc_2_no_stopwords = self.nlp(' '.join([str(t) for t in doc_2 if not t.is_stop]))
                    df.at[index, 'obj_1_2_sim'] = doc_1_no_stopwords.similarity(doc_2_no_stopwords)
            else:
                    continue
    for index, row in df.iterrows():
            if (pd.notnull(row['object_phrase_2_reg']) & pd.notnull(row['object_phrase_1_rea'])):
                    doc_1 = self.nlp(row['object_phrase_2_reg'])
                    doc_2 = self.nlp(row['object_phrase_1_rea'])
                    #added setup, because Spacy constructs sentence embedding by averaging the word embeddings
                    doc_1_no_stopwords = self.nlp(' '.join([str(t) for t in doc_1 if not t.is_stop]))
                    doc_2_no_stopwords = self.nlp(' '.join([str(t) for t in doc_2 if not t.is_stop]))
                    df.at[index, 'obj_2_1_sim'] = doc_1_no_stopwords.similarity(doc_2_no_stopwords)
            else:
                    continue
    for index, row in df.iterrows():
            if (pd.notnull(row['object_phrase_2_reg']) & pd.notnull(row['object_phrase_2_rea'])):
                    doc_1 = self.nlp(row['object_phrase_2_reg'])
                    doc_2 = self.nlp(row['object_phrase_2_rea'])
                    #added setup, because Spacy constructs sentence embedding by averaging the word embeddings
                    doc_1_no_stopwords = self.nlp(' '.join([str(t) for t in doc_1 if not t.is_stop]))
                    doc_2_no_stopwords = self.nlp(' '.join([str(t) for t in doc_2 if not t.is_stop]))
                    df.at[index, 'obj_2_2_sim'] = doc_1_no_stopwords.similarity(doc_2_no_stopwords)
            else:
                    continue
    return df