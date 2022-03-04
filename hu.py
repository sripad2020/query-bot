from fuzzywuzzy import fuzz
from transformers import pipeline
from bs4 import BeautifulSoup
import googlesearch,nltk,heapq,requests,re
from telegram.ext.updater import Updater
from telegram.update import Update
from telegram.ext.callbackcontext import CallbackContext
from telegram.ext.commandhandler import CommandHandler
from telegram.ext.messagehandler import MessageHandler
from telegram.ext.filters import Filters

updater = Updater("please make enter your api key from telegram using BOTFATHER ",use_context=True)
def start(update: Update, context: CallbackContext):
    update.message.reply_text("Hello user, Welcome to the Bot Please write /help to see the commands available.")
def help(update: Update, context: CallbackContext):
    update.message.reply_text("""Hello user we have some search patterns in our bot
    as try any question from machine learning ,data science and Artificial intelligence you can explore the answer""")
def custom(update: Update, context: CallbackContext):
    inp = update.message.text
    para = []
    output = []
    a = googlesearch.search(inp)
    c=[a[1:10:3]# add your custom number for getting its custom links 
    for b in c:
        try:
            r = requests.get(b)
            data = r.text
            soup = BeautifulSoup(data, features='lxml')
            for link in soup.find_all('p'):
                g = link.get_text()
                token = nltk.tokenize.sent_tokenize(g)
                para.append(token)
        except requests.exceptions.MissingSchema as pe:
            print(pe)
    def r(para):
        for s in para:
            if type(s) == list:
                r(s)
            else:
                output.append(s)
    r(para)
    stri = ' '.join(map(str, output))
    text = stri.lower()
    clean = re.sub('[^a-zA-Z]', ' ', text)
    clean2 = re.sub('\s +', ' ', clean)
    sentence_list = nltk.sent_tokenize(text)
    stopwords = nltk.corpus.stopwords.words('english')
    word_frequencies = {}
    for word in nltk.word_tokenize(clean2):
        if word not in stopwords:
            if word not in word_frequencies:
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1
    maximum_frequency = max(word_frequencies.values())
    for word in word_frequencies:
        word_frequencies[word] = word_frequencies[word] / maximum_frequency
    sentence_scores = {}
    for sentence in sentence_list:
        for word in nltk.word_tokenize(sentence):
            if word in word_frequencies and len(sentence.split(' ')) < 30:
                if sentence not in sentence_scores:
                    sentence_scores[sentence] = word_frequencies[word]
                else:
                    sentence_scores[sentence] += word_frequencies[word]
    summary = heapq.nlargest(5, sentence_scores, key=sentence_scores.get)
    sentence = ''.join(summary)
    pr = re.sub('\n+', ' ', sentence)
    text_cleaned=re.sub('{*?}','',pr)
    sd = re.sub("{.*?}", '', text_cleaned)
    cleaned = re.sub('\*?', '', sd)
    summarizer=pipeline('summarization')
    for i in summarizer(cleaned,min_length=50,max_length=75):
        update.message.reply_text('%s'%i['summary_text'])
updater.dispatcher.add_handler(CommandHandler('start',start))
updater.dispatcher.add_handler(CommandHandler('help',help))
updater.dispatcher.add_handler(MessageHandler(Filters.text,custom))
updater.start_polling()
