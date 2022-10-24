# importando a biblioteca nltk
import nltk
from IPython.display import display
import pandas as pd
%matplotlib inline

# baixando arquivos necessários para os exemplos
nltk.download('punkt')
nltk.download("stopwords")
nltk.download('rslp')
nltk.download('averaged_perceptron_tagger')
nltk.download('mac_morpho')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download("maxent_ne_chunker")
nltk.download("words")
#nltk.download("book")


df = pd.read_csv("tripadvisor_hotel_reviews.csv")

print(df)

# carregando as stopwords em português
stop_words = nltk.corpus.stopwords.words("english")

# mostra os 10 primeiros elementos
print(stop_words[:10])

# transforma em set para otimizar operações
# porque não vamos alterá-las
stop_words = set(stop_words)

for n in range(0,50):
  print(df['Review'][n])
  text = df['Review'][n]
  tokens = nltk.tokenize.word_tokenize(text)
  #frequency_distribution = nltk.FreqDist(tokens)
  #frequency_distribution.plot(20)

  # lista que ira armazenar tokens que não são stopwords
  clean_tokens = []

  for word in tokens:
      # evita problemas de up/lower case
      if word.casefold() not in stop_words:
          # adiciona tokens não-stopwords
          clean_tokens.append(word)

  #print(clean_tokens)

  # instancia o stemmer RSLP (português)
  stemmer = nltk.stem.RSLPStemmer()

  # cria uma lista para armazenar os resultados
  stemmed_tokens = []

  # transforma e salva cada token
  for token in clean_tokens:
      stemmed_tokens.append(stemmer.stem(token))

  # instancia o lemmatizer
  lemmatizer = nltk.stem.WordNetLemmatizer()

  # cria uma lista para armazenar os resultados
  lemmatized_tokens = []

  # transforma e salva cada token
  for token in clean_tokens:
      lemmatized_tokens.append(lemmatizer.lemmatize(token))

  # recupera as tags do tagger 'mac_morpho'
  tags = nltk.corpus.mac_morpho.tagged_words()

  # armazenando apenas as tags
  words_tags = []
  for word, tag in tags:
      words_tags.append(tag)

  # calcula as mais frequentes
  tags_frequencies = nltk.FreqDist(words_tags)

  # cria um tagger genérico
  # interpreta todos os tokens como 'N'
  standardTagger = nltk.tag.DefaultTagger("N")
  standardTagger.tag(clean_tokens)