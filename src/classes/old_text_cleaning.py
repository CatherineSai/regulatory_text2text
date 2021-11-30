
class Text_Cleaning:
  def __init__(self, signalwords, nlp):
    self.signalwords = signalwords
    self.nlp = nlp

  def get_relevant_text(self, paragraphsraw):
    '''splits each paragraph into sentences and only keeps sentences that contain at least one signalword
    Input: Dictionary or Paragraphs
    Output: List of Sentences'''
    #s = set()
    result = list()
    for paraid, para in paragraphsraw.items():
        para = para.replace(";", ".") #in reg there are many ; which should be counted as seperate senteces
        para = para.replace("or\n\n\n", "")
        para = para.replace("or\n\n", "")
        para = para.replace("and\n\n\n", "")
        para = para.replace("17065/20212", "")
        doc = self.nlp(para) #create a doc object (sequence of tokens) with a nlp of paragragh text  
        sentences = doc.sents
        for sentence in sentences:
            for token in sentence: 
                if (token.text in self.signalwords):
                    result.append(sentence.text.strip())
                #if (token.ent_type_ =='ORG'):
                    #s.add(token.text)
                    break
    #print(s)
    return result


"""
  def get_relevant_paragraphs(self):  
    '''splits each paragraph into sentences and only keeps paragaraphs that contain at least one signalword
    Input: Dictionary or Paragraphs
    Output: List of Paragraphs'''
    for paraid, para in self.paragraphsraw.items():
        doc = self.nlp(para) #create a doc object (sequence of tokens) with a nlp of paragragh text  
        sentences = doc.sents
        for idx, sentence in enumerate(sentences):
            if (word.text in self.signalwords for word in sentence):
                self.relevant_paragraphs.append(para)
                break
    self.substitude_entity_ORG()

  def substitude_entity_ORG(self):
    '''splits each paragraph into sentences and only keeps paragaraphs that contain at least one signalword
    Input: Dictionary or Paragraphs
    Output: List of Paragraphs'''
    for paragraph in self.relevant_paragraphs:
        doc = self.nlp(paragraph) #create a doc object (sequence of tokens) with a nlp of paragragh text  
        for token in doc:
            if (token.ent_type_ =='ORG' and token.text != "EU" and token.text != "the Data Protection Policy EU" and token.text != "A17"):
                doc = self.nlp(" ".join([t.text if not (t.ent_type_ =='ORG') else "the organization" for t in doc]))
                self.cleaned_entity_paras.append(doc)
    self.get_relevant_sentences()

  def get_relevant_sentences(self):
    '''creates a list with sentences from all paragaraphs, only keeping those sentences that contain at least one signalword
    Note: this wasn't already done in the function get_relevant_paragraphs because this way the anaphora resolution can be added more easily
    Input: List of Paragraphs (as doc objects)
    Output: List of Sentences'''
    for para in self.cleaned_entity_paras:
        sentences = para.sents
        for idx, sentence in enumerate(sentences):
            if (word.text in self.signalwords for word in sentence):
                self.relevant_sentences.append(sentence)
                break
    return self.relevant_sentences
"""
