
# coding: utf-8

# In[3]:

import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import brown
from nltk.util import ngrams
from nltk.tag import pos_tag
# задаем предлоги кортежами
preps =['because of','due to','on account of','by reason of', 'by dent of','by way of', 'for the sake of',
        'for the purpose of', 'on the force of'
       ]
# задаем список жанров
genres = ['belles_lettres', 'fiction', 'government', 'news']
 
prep_dict = {} # создаем словарь, в котором ключами будут жанры, а значениями тоже словари с ключами-предлогами и значениями
                # будут списки предложений, в которых содержится ключ-предлог (жанры-->предлоги-->предложения)
prep_counts = {} # словарь, в котором по аналогии с prep_dict будет храниться количество предлогов по жанрам
                #(жанры-->предлоги-->количество предлогов)
# функция, которая возвращает True, если слово в предложении
# после предлога относится к существительному, местоимению или герундию
# принимает на вход предложение и предлог
def follow_word(sentence, prep):
    tags = ['NN', 'NNP', 'NNS', 'PRP','PRP$', 'VBG'] # список интересующих нас части речи
    # разбиваем предложение и предлог на слова
    sent_lst = tuple(sentence.split())
    p_lst = tuple(prep.split())
    n = len(p_lst) # количество слов предлога
    n_gram = ngrams(sent_lst,n) # составляем n-граммы предложения в зависимости от количества слов предлога
    for i in n_gram: # перебираем н-граммы
        if i == p_lst: # если н-грамма равна предлогу
            follower = pos_tag(next(n_gram))[-1] # то берем следующую н-грамму, применяем определитель части речи
                                                # и берем последнюю пару(слово, часть речи)(наше искомое слово)
            follower_tag = follower[1] # из этой пары берем часть речи
            if follower_tag in tags: # если часть речи входит в список интересующих нас, то возвращаем True
                return True
            else:
                return False
# функция, которая проверяет все предложения жанра на существование в них предлога и с помощью функции follow_word
# отбирает те которые с нужной частью речи после предлога, возвращает список из количества таких предлогов и 
# списка требуемых предложений, принимает на вход жанр и предлог
def get_sentences(genre, prep):
    sents = brown.sents(categories = genre) # берем предложения из корпуса
    sents_strings = [' '.join(s) for s in sents] # составляем список из их строкого представления    
    counts = 0 #счетчик предлогов для жанра
    sents_with_prep = [] # список предложений
    for s in sents_strings: # перебираем предложения
        c = s.lower().count(prep) # счетчик предлогов в предложении без заглавных букв
        if c > 0: # если предлог встречается в предложении и после него нужная часть речи
            if follow_word(s.lower(), prep):
                counts += c # то увеличиваем счетчик предлогов
                sents_with_prep.append(s) # добавляем предложение в список
    return [counts, sents_with_prep] # возвращаем список из количества предлогов и списка предложений

for g in genres:
    for p in preps:
        counts, sents_with_prep = get_sentences(g, p)
        if g not in prep_dict.keys():
            prep_dict[g] = {}
            prep_counts[g] = {}
            prep_dict[g].update({p: sents_with_prep})#(counts, len(words))})
            prep_counts[g].update({p:counts})
        else:
            prep_dict[g].update({p: sents_with_prep})
            prep_counts[g].update({p:counts})
for genre in sorted(prep_dict.keys()):
    print('------------------------')
    print('Genre: ',genre)
    for prep, sents in sorted(prep_dict[genre].items()):
        print('Preposition: \'%s\''%prep)
        i = 1
        print('Sentences: ', len(sents))
        for sent in sents:
            print(i, '.',sent)
            i += 1
            
#-----------------------------------------------------
# Теперь полученный словарь prep_counts преобразуем к объектам библиотеки pandas
belles_lettres_dict = prep_counts['belles_lettres']
belles_lettres = pd.Series(belles_lettres_dict)

fiction_dict = prep_counts['fiction']
fiction = pd.Series(fiction_dict)

government_dict = prep_counts['government']
government = pd.Series(government_dict)

news_dict = prep_counts['news']
news = pd.Series(news_dict)
# сводим эти одномерные массивы в двумерный и получаем таблицу
tabl = pd.DataFrame({'news':news,
                     'belles_lettres':belles_lettres,
                     'fiction': fiction,
                     'government':government
                    })
print(tabl) # выводим таблицу
# удаляем предлоги ни разу не встречающиеся
ind = np.where(np.sum(tabl>0, axis=1))[0]# [0]-тк where возвращает кортеж (массив, тип), а нам нужен только массив
tabl1 = tabl.T[ind]
# Выводим эту таблицу
print(tabl1.T)
#--------------------------------------
# Рисуем гистограмму

col_names = tabl1.index.values.tolist()
plt.style.use('ggplot')
x = np.arange(0,len(tabl1.columns.values.tolist()),1)

y1,y2,y3,y4 = [tabl1.T[col_names[0]], tabl1.T[col_names[1]], tabl1.T[col_names[2]], tabl1.T[col_names[3]]]
width = 0.2
fig = plt.figure(figsize = (12,5))
pr = fig.add_subplot(111)
pr.bar(x - width, y1, width)
xticks = pr.get_xticks()
yticks = pr.get_yticks()

pr.set_xticks(x)
pr.set_xticklabels(tabl1.columns.values.tolist())
pr.bar(x  - 0.5*width, y3, width, color='green')
pr.bar(x, y4, width, color='magenta')
pr.bar(x + 0.5*width, y2, width, color='red')

pr.grid(True)
pr.legend(tabl1.index.values.tolist())
plt.title('Preposition counts for different genres')

plt.ylabel('Counts')
plt.show()

