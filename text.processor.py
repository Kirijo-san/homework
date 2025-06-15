# text_processor.py
import math
from collections import defaultdict

class TextProcessor:
    stopwords = {'и', 'в', 'во', 'не', 'что', 'он', 'на', 'я', 'с', 'со', 'как', 'а', 'то', 'все', 'она', 'так', 'его', 'но', 'да', 'ты', 'к', 'у', 'же', 'вы', 'за', 'бы', 'по', 'только', 'ее', 'мне', 'было', 'вот', 'от', 'меня', 'еще', 'нет', 'о', 'из', 'ему', 'теперь', 'когда', 'даже', 'ну', 'вдруг', 'ли', 'если', 'уже', 'или', 'ни', 'быть', 'был', 'него', 'до', 'вас', 'нибудь', 'опять', 'уж', 'вам', 'ведь', 'там', 'потом', 'себя', 'ничего', 'ей', 'может', 'они', 'тут', 'где', 'есть', 'надо', 'ней', 'для', 'мы', 'тебя', 'их', 'чем', 'была', 'сам', 'чтоб', 'без', 'будто', 'чего', 'раз', 'тоже', 'себе', 'под', 'будет', 'ж', 'тогда', 'кто', 'этот', 'того', 'потому', 'этого', 'какой', 'совсем', 'ним', 'здесь', 'этом', 'один', 'почти', 'мой', 'тем', 'чтобы', 'нее', 'сейчас', 'были', 'куда', 'зачем', 'всех', 'никогда', 'можно', 'при', 'наконец', 'два', 'об', 'другой', 'хоть', 'после', 'над', 'больше', 'тот', 'через', 'эти', 'нас', 'про', 'всего', 'них', 'какая', 'много', 'разве', 'три', 'эту', 'моя', 'впрочем', 'хорошо', 'свою', 'этой', 'перед', 'иногда', 'лучше', 'чуть', 'том', 'нельзя', 'такой', 'им', 'более', 'всегда', 'конечно', 'всю', 'между'}

    def __init__(self, documents):
        if not all(len(doc.split()) >= 2 for doc in documents):
            raise ValueError("Все документы должны содержать минимум 2 слова")
        
        self.documents = documents
        self._preprocess_documents()
    
    def _preprocess_documents(self):
        self.tokenized_docs = []
        for doc in self.documents:
            words = [word.lower() for word in doc.split()]
            self.tokenized_docs.append(words)
        
        self.doc_word_counts = []
        for doc in self.tokenized_docs:
            word_counts = defaultdict(int)
            for word in doc:
                word_counts[word] += 1
            self.doc_word_counts.append(word_counts)
        
        self.doc_lengths = [len(doc) for doc in self.tokenized_docs]
        
        self.doc_freq = defaultdict(int)
        for word_counts in self.doc_word_counts:
            for word in word_counts:
                self.doc_freq[word] += 1
    
    def get_tf(self, word, doc_num):
        if doc_num < 0 or doc_num >= len(self.documents):
            raise ValueError("Некорректный номер документа")
        
        word = word.lower()
        word_count = self.doc_word_counts[doc_num].get(word, 0)
        return word_count / self.doc_lengths[doc_num]
    
    def get_idf(self, word):
        word = word.lower()
        n_docs = len(self.documents)
        docs_with_word = self.doc_freq.get(word, 0)
        return math.log(n_docs / (1 + docs_with_word))
    
    def get_tf_idf(self, word, doc_num, ignore_stopwords=True):
        word = word.lower()
        if ignore_stopwords and word in self.stopwords:
            return 0.0
        
        tf = self.get_tf(word, doc_num)
        idf = self.get_idf(word)
        return tf * idf
