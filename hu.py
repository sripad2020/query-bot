from fuzzywuzzy import fuzz
from transformers import pipeline
from bs4 import BeautifulSoup
import googlesearch,nltk,heapq,requests,re,json
from telegram.ext.updater import Updater
from telegram.update import Update
from telegram.ext.callbackcontext import CallbackContext
from telegram.ext.commandhandler import CommandHandler
from telegram.ext.messagehandler import MessageHandler
from telegram.ext.filters import Filters

start = '''
{
  "intents": [
    {
      "tag": "QUERY BOT",

      "patterns": [
        "what is recommendation system",
        "how do you evaluate recommendation system",
        "what are the types of recommendation systems",
        "examples of recommendation system",
        "what kind of information does a Recommendation Engine need for effective recommendations",
        "hello bot",
        "thank you",
        "Good day",
        "what does multinomial regression do",
        "what is multinomial regression",
        "how logistic function differs with multinomial regression",
        "what is the function used in multinomial regression",
        "bye",
        "thanks for helping",
        "what is your name",
        "what is ensemble learning",
        "what is the another name of bagging",
        "how bagging is implemented",
        "what is bagging",  
        "what are commonly used ensemble algorithms",
        "What are outliers and how can the sigmoid function mitigate the problem of outliers in logistic regression",
        "who created you",
        "what is your origin",
        "which technique is used to replace missing values with logical values",
        "mention methods of imputation",
        "how to make simple machine learning model",
        "state the use of multinomial regression",
        "how logistic regression differs from multinomial regression",
        "what is outlier",
        "what is charter document",
        "describe parameter",
        "what is optimization",
        "which algorithm is better in the case of outliers present in the dataset that is logistic regression or support vector machines",
        "can we solve the multiclass classification problems using logistic regression",
        "what is the role of a project manager in the preparation of a project charter",
        "what is the importance of a project charter",
        "what is gradient descent",
        "what is backpropagation",
        "what are tensors",
        "what is Leaky ReLU activation function",
        "explain two ways to deal with the vanishing gradient problem in a deep neural network.",
        "why is a deep neural network better than a shallow neural network.",
        "what is the need to add randomness in the weight initialization process.",
        "how hyper parameter is trained in neural network",
        "what is softmax function",
        "what are components used for training hyper parameters",
        "how long can an LSTM learn the pattern",
        "why do we use stacking LSTM in deep learning",
        "what is the purpose of depth of neural networks",
        "what is units in LSTM",
        "what is the main objective of boosting"
      ],
      "responses": [
        "recommendation system is a platform that provides its users with various contents based on their preferences and liking.",
        "mean Average Precision at K is used to evaluate performance of a recommender systems.",
        "three types of Recommendation system Content Based Filtering Collaborative Based Filtering Hybrid Model",
        "netflix, YouTube, Tinder and Amazon are examples of recommender systems in use.",
        "information which include users explicit implicit interactions and profile details.",
        "hey user",
        "my Pleasure",
        "hello, thanks for visiting",
        "Multinomial logistic regression is used to predict categorical placement in or the probability of category membership on a dependent variable based on multiple independent variables.",
        "Logistic regression is a classification algorithm. It is intended for datasets that have numerical input variables and a categorical target variable that has two values or classes. Problems of this type are referred to as binary classification problems",
        "Multinomial Logistic Regression is similar to logistic regression but with a difference, that the target dependent variable can have more than two classes i.e. multiclass or polychotomous",
        "Multinomial logistic regression is a simple extension of binary logistic regression that allows for more than two categories of the dependent or outcome variable. Like binary logistic regression, multinomial logistic regression uses maximum likelihood estimation to evaluate the probability of categorical membership",
        "okay byee have a nice day",
        "its my pleasure to serve you",
        "My name is LEBENSMüDE",
      "ensemble learning is the process by which multiple models such as classifiers or experts are strategically generated and combined to solve a particular computational intelligence problem. Ensemble learning is primarily used to improve the performance of a model or reduce the likelihood of an unfortunate selection of a poor one",
      "Bagging also known as bootstrap aggregating",
      "Each model is trained individually, and combined using an averaging process",
      "Bagging or Bootstrap Aggregating is an ensemble method in which the dataset is first divided into multiple subsets through resampling.Then each subset is used to train a model and the final predictions are made through voting or averaging the component models",
      "Bagging and boosting are commonly used ensemble algorithms",
      "Sometimes a dataset can contain extreme values that are outside the range of what is expected and unlike the other data. These are called outliers.The sigmoid function plays an important role in mitigating the problem of outliers",
      "Its TEAM 3",
      "I am from Team 3",
      "Imputation is a technique used to replace missing values with logical values.",
      "1. deletion method 2.single imputation 3.model based imputation",
      "use regularization techniques and select the optimal features in data set",
      "multinomial logistic regression is a simple extension of binary logistic regression that allows for more than two categories of the dependent or outcome variable. Like binary logistic regression, multinomial logistic regression uses maximum likelihood",
      "multinomial Logistic Regression is the regression analysis to conduct when the dependent variable is nominal with more than two levels.Binary logistic regression assumes that the dependent variable is a stochastic event.",
      "an outlier is an observation that lies an abnormal distance from other values in a random sample from a population",
      "the definition of the project should be short because it refers to more detailed documents, such as a request for proposal.",
      "a model parameter is a configuration variable that is internal to the model and whose value can be estimated from data.",
      "the action of making the best or most effective use of a situation or resource",
      "logistic regression logistic regression will identify a linear boundary if it exists to accommodate the outliers. to accommodate the outliers it will shift the linear boundary.Support vector machines is insensitive to individual samples. So to accommodate an outlier there will not be a major shift in the linear boundary. Support vector machines  comes with inbuilt complexity controls which take care of overfitting which is not true in the case of Logistic Regression",
      "Logistic regression, by default, is limited to two-class classification problems. Some extensions like one-vs-rest can allow logistic regression to be used for multi-class classification problems, although they require that the classification problem first be transformed into multiple binary classification problems",
        "though he is appointed in the project charter, a project manager can still be involved in the preparation of the project charter",
      "the business needs underlying the project",
      "gradient descent is an optimal algorithm to minimize the cost function or to minimize an error. The aim is to find the local global minima of a function this determines the direction the model should take to reduce the error",
      "backpropagation is a technique to improve the performance of the network. It backpropagates the error and updates the weights to reduce the error",
      "a tensor is a mathematical object represented as arrays of higher dimensions. These arrays of data with different dimensions and ranks fed as input to the neural network are called Tensors.",
      "Leaky ReLU is an advanced version of the ReLU activation function. In general, the ReLU function defines the gradient to be 0 when all the values of inputs are less than zero. This deactivates the neurons. To overcome this problem, Leaky ReLU activation functions are used. It has a very small slope for negative values instead of a flat slope",
      "i. Use the ReLU activation function instead of the sigmoid function  ii. Initialize neural networks using Xavier initialization that works with tanh activation.",
      "Both deep and shallow neural networks can approximate the values of a function. But the deep neural network is more efficient as it learns something new in every layer. A shallow neural network has only one hidden layer. But a deep neural network has several hidden layers that create a deeper representation and computation capability.",
      "If you set the weights to zero, then every neuron at each layer will produce the same result and the same gradient value during backpropagation. So, the neural network will not be able to learn the function as there is no asymmetry between the neurons. Hence, randomness to the weight initialization process is crucial.",
      "Hyperparameters in a neural network can be trained using four components,1.Batch size Indicates the size of the input data 2.Epochs  Denotes the number of times the training data is visible to the neural network to train 3.Momentum Used to get an idea of the next steps that occur with the data being executed 4.Learning rate  Represents the time required for the network to update the parameters and learn.",
      "The softmax function is used to calculate the probability distribution of the event over n different events. One of the main advantages of using softmax is the output probabilities range. The range will be between 0 to 1, and the sum of all the probabilities will be equal to one. When the softmax function is used for multi classification model, it returns the probabilities of each class, and the target class will have a high probability.",
      "batch_size   learning rate   epochs    momentums",
      "an LSTM can learn this pattern that exists for every 12 periods in time or it takes 45 to 50 milli seconds per step to train",
      "stacking LSTM hidden layers make the model deeper,  more accurately earning the description as a deep learning technique",
      "depth of the neural networks that is generally attributed to the success of the approach on a wide range of challenging predicton problems",
      "unit means the dimension of the inner cell in LSTM",
      "boosting is used to create a collection of predictors. In this technique, learners are learned sequentially with early learners fitting simple models to the data and then analysing data for errors. Consecutive trees random sample are fit and at every step, the goal is to improve the accuracy from the prior tree"
      ]
    }
]}
'''
trans=pipeline('question-answering')
ques=[]
answer=[]
data=json.loads(start)
for i in data['intents']:
    ques = i['patterns'].copy()
    answer = i['responses'].copy()
updater = Updater("5253235537:AAEPWzJdcD02JbkYysvtJGEKfBzRwsEhJb8",use_context=True)
def start(update: Update, context: CallbackContext):
    update.message.reply_text("Hello user, Welcome to the Bot Please write /help to see the commands available.")
def help(update: Update, context: CallbackContext):
    update.message.reply_text("""Hello user we have some search patterns in our bot
    as try any question from machine learning ,data science and Artificial intelligence you can explore the answer""")
def custom(update: Update, context: CallbackContext):
    inp = update.message.text
    for m in range(len(ques)):
            if fuzz.ratio(inp,ques[m]) > 87:
                ans=trans(question=ques[m] , context=answer[m])
                update.message.reply_text("'%s'"% ans['answer'])
    para = []
    output = []
    a = googlesearch.search(inp)
    c=[a[1],a[3],a[5],a[7],a[9]]
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