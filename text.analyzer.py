# text_analyzer.py
from collections import Counter
import re

def load_text_from_file(file_path):
    """Загружает текст из локального файла"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        print(f"Ошибка: файл {file_path} не найден!")
        return ""
    except Exception as e:
        print(f"Ошибка при чтении файла: {e}")
        return ""

def process_ulysses(file_path):
    """Основная функция обработки текста"""
    text = load_text_from_file(file_path)
    if not text:
        return ""

    cleaned_text = re.sub(r'[^\w\s]', ' ', text.lower())
    words = cleaned_text.split()

    word_counts = Counter(words)
    print("\nТоп-10 самых частых слов:")
    for word, count in word_counts.most_common(10):
        print(f"{word}: {count}")

    return text

def find_word_occurrences(text, target_word, left_len, right_len, cut_length=False):
    """Поиск вхождений слова с контекстом"""
    if not text:
        return []

    sentences = re.split(r'(?<=[.!?])\s+', text)
    occurrences = []
    target_word = target_word.lower()

    for sent in sentences:
        words = sent.split()
        for i, word in enumerate(words):
            if word.lower() == target_word:
                left = max(0, i - left_len)
                right = min(len(words), i + right_len + 1)

                if cut_length:
                    left = max(left, 0)
                    right = min(right, len(words))

                context = ' '.join(words[left:i] + [f'[{words[i]}]'] + words[i+1:right])
                occurrences.append(context)

    print(f"\nНайдено {len(occurrences)} вхождений слова '{target_word}':")
    for i, occ in enumerate(occurrences[:5], 1):
        print(f"{i}. {occ}")

    with open('word_occurrences.txt', 'w', encoding='utf-8') as f:
        for occ in occurrences:
            f.write(occ + '\n\n')

    return occurrences
