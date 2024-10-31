import string
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer, PorterStemmer
import re
import contractions

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger_eng')

# removing stop words
stop_words = set(stopwords.words('english'))

# Define negation words to exclude from stopwords
negation_words = {'no', 'not', 'nor', 'never', "n't"}

# Remove negation words from the stopwords set
stop_words = stop_words.difference(negation_words)

lemmatizer = WordNetLemmatizer()

class PreProcess:
  def get_wordnet_pos(self, word):
    """
    Map POS tag to WordNet POS
    Using nltk.stem.WordNetLemmatizer without specifying the part of speech (POS) defaults to treating 
    all words as nouns.
    Accuracy: Without POS tagging, lemmatization may not reduce words to their correct root forms, 
    Leading to less effective normalization.
    Context Awareness: POS tagging allows the lemmatizer to understand the role of each word in a sentence, ensuring that verbs, nouns,
    adjectives, and adverbs are lemmatized appropriately.
    """
    # Get the POS tag for the word
    tag = nltk.pos_tag([word])[0][1][0].upper()
    # Map NLTK POS tags to WordNet POS tags
    tag_dict = {
        'J': wordnet.ADJ,
        'N': wordnet.NOUN,
        'V': wordnet.VERB,
        'R': wordnet.ADV
    }
    # Return the corresponding WordNet POS tag or default to Noun
    return tag_dict.get(tag, wordnet.NOUN)

  def expand_contractions(self, text):
    return contractions.fix(text)

  def to_lowercase(self, text):
    return text.lower()

  def remove_urls(self, text):
    # Remove URLs
    return re.sub(r"http\S+|www\S+|https\S+", '', text)

  def lemmatize_text(self, text):
    # Tokenize text while preserving @mentions and #hashtags as single tokens
    tokens = re.findall(r'@\w+|#\w+|\w+', text)
    lemmatized = [lemmatizer.lemmatize(w, self.get_wordnet_pos(w)) for w in tokens]
    return ' '.join(lemmatized)

  def remove_punctuation(self, text):
    '''
    Model may be improved by preventing removal of characters #disaster? or #disaster!
    and ? or ! in general, see end of document for edited function
    
    '''
    # Remove punctuation except for words \w, white spaces \s, mentions (@) and #
    return re.sub(r"[^\w\s@#]|[\d_]", '', text)

  def remove_stopwords(self, text):
    # Tokenize text while preserving @mentions and #hashtags as single tokens
    tokens = re.findall(r'@\w+|#\w+|\w+', text)
    filtered = [word for word in tokens if word.lower() not in stop_words]
    return ' '.join(filtered)

  def remove_non_ascii(self, text):
    """
    Removes non-ASCII characters from the text.
    """
    return re.sub(r'[^\x00-\x7F]+', '', text)

  def process(self, text):
    # Check if the input is a string, if not, return an empty string or convert to string
    if not isinstance(text, str):
        return ""
    
    # Expand contractions (e.g., "can't" -> "cannot")
    text = self.expand_contractions(text)
    
    # Convert to lowercase
    text = self.to_lowercase(text)
    
    # Remove URLs
    text = self.remove_urls(text)
    
    # Remove non-ASCII characters
    text = self.remove_non_ascii(text)
    
    # Remove punctuation except for hashtags (#) and (@)
    text = self.remove_punctuation(text)
    
    # Optionally, replace mentions with '@user' 
    # text = re.sub(r'@\w+', '@user', text)
    
    # Remove stopwords while preserving negations
    text = self.remove_stopwords(text)
    
    # Lemmatize words with appropriate POS tagging
    text = self.lemmatize_text(text)
    
    return text