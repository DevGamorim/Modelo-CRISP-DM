# importando a biblioteca nltk
import nltk
from IPython.display import display
import pandas as pd
#%matplotlib inline

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

# carregando as stopwords em português
stop_words = nltk.corpus.stopwords.words("english")

# transforma em set para otimizar operações
# porque não vamos alterá-las
stop_words = set(stop_words)

for n in range(0,50):
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

    # tags no nível de sentenças
    tagged_sents = nltk.corpus.mac_morpho.tagged_sents()
    
    # sentença para exemplo
    sample_sentence = "A manhã está ensolarada"
    # cria os tokens
    sentence_tokens = nltk.word_tokenize(sample_sentence)
    # cria o tagger que identifica baseado na maior frequencia
    # ex: token X será da classe N se ele aparecer como N com maior frequencia
    unigram_tagger = nltk.tag.UnigramTagger(tagged_sents)

    
    # faz o tagging
    sentence_tags = unigram_tagger.tag(sentence_tokens)

    # cria tagger padrão: sempre 'N'
    t0 = nltk.DefaultTagger("N")
    # cria outro tagger a partir do 1o (backoff)
    # esse considera o token atual (n) e o anterior (n-1)
    t1 = nltk.UnigramTagger(tagged_sents, backoff=t0)
    # repete, considerando os tokens n a n-2
    t2 = nltk.BigramTagger(tagged_sents, backoff=t1)
    # repete, considerando os tokens n a n-3
    t3 = nltk.TrigramTagger(tagged_sents, backoff=t2)

    sentence_tokens_tagged = t3.tag(sentence_tokens)