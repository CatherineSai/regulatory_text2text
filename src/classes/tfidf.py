

class TF_IDF:
  def __init__(self, reg_relevant_sentences, rea_relevant_sentences, nlp):
    self.nlp = nlp
    self.df_reg = self.sentence_df(reg_relevant_sentences, 'reg')
    self.df_rea = self.sentence_df(rea_relevant_sentences, 'rea')

  