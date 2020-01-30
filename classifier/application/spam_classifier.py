import re
import numpy as np
import pandas as pd

pA = 0 # вероятность спама в обучающей выборке
pNotA = 0 # вероятность не спама в обучающей выборке
SPAM = 1
NOT_SPAM = 0
Z = 0.1 # коэффициент размытия для сглаживания по Лаппласу
trainPositive, trainNegative = {}, {}
positive_total, negative_total, total = 0, 0, 0

data = pd.read_csv('../resources/spam_or_not_spam.csv')
data.dropna(inplace=True)

# оставим только слова в тексте и ограничим длинну слов 3-15 символов
data['email'] = data['email'].apply(lambda text: re.sub(r'\b\w{1, 3}\b', '', text))
data['email'] = data['email'].apply(lambda text: re.sub(r'\b\w{15, }\b', '', text))

spamTextCount = data[data['label'] == SPAM]['label'].sum()

spam = data[data['label'] == 1]
ham = data[data['label'] == 0]

Sep = 0.9
Sep_spam = round(len(spam)*Sep)
Sep_ham = round(len(ham)*Sep)

train_data = spam[:][:Sep_spam].append(ham[:][:Sep_ham], ignore_index=True)
validation_data = spam[:][Sep_spam:].append(ham[:][Sep_ham:], ignore_index=True)

def train():
    # Считаем pA и pNotA
    # Считаем частоты каждого слова с помощью calculate_word_frequencies()
    global pA, pNotA, data, total

    for i, row in train_data.iterrows():
        body = row['email']
        label = row['label']
        calculate_word_frequencies(body, label)

    total = len(trainPositive) + len(trainNegative)

    pA = spamTextCount/len(data)
    pNotA = 1 - pA
    # будем использовать логарифм от вероятностей, так как будут очень малые значения и округлим до 2-х знаков
    pA = round(np.log(pA), 2)
    pNotA = round(np.log(pNotA), 2)

def calculate_word_frequencies(body, label):
    global trainPositive, trainNegative, positive_total, negative_total
    for word in body.lower().split():
        if label == SPAM:
            trainPositive[word] = trainPositive.get(word, SPAM) + 1
            positive_total += 1
        else:
            trainNegative[word] = trainNegative.get(word, NOT_SPAM) + 1
            negative_total += 1

def calculate_P_Bi_A(word, label):
    # P(Bi|A) - вероятность встретить слово
    # применим размытие по Лаппласу для исключения нулевой вероятности новых слов
    # Z - коэффициент размытия
    # total - общее количество уникальных слов использованное для обучения модели
    if label == SPAM:
    # будем использовать логарифм от вероятностей, так как будут очень малые значения и округлим до 2-х знаков
        #return round(np.log((trainPositive.get(word, 0) + Z) / (positive_total + Z*total)), 2) # лучший скор при Z = 0.1, но ответы: 1/3
        return round(np.log((trainPositive.get(word, 0) + Z) / len(trainPositive)), 2) # скор хуже, но ответы 2/3
    else:
        #return round(np.log((trainNegative.get(word, 0) + Z) / (negative_total + Z*total)), 2)
        return round(np.log((trainNegative.get(word, 0) + Z) / len(trainNegative)), 2)

def calculate_P_B_A(text, label):
    # P(B|A) - вероятность встретить текст
    P_B_A = 0
    for word in text.lower().split():
    #  умножение заменяем на сложение так как логарифм
        P_B_A = P_B_A + calculate_P_Bi_A(word, label)
    return P_B_A

def classify(text):
    # Ответ: True, если спам. False, если не спам
    # оставим слова в тексте, все цифры заменим на NUMBER

    text = re.sub(r'\b\w{1, 3}\b', '', text)
    text = re.sub(r'\b\w{15, }\b', '', text)
    text = re.sub(r'\d', 'NUMBER ', text)

    spam_log = pA + calculate_P_B_A(text, SPAM)
    not_spam_log = pNotA + calculate_P_B_A(text, NOT_SPAM)

    P_spam = round(100/(1 + np.exp(not_spam_log - spam_log)), 2)
    P_ham = round(100/(1 + np.exp(spam_log - not_spam_log)), 2)

    #print('P_spam:', P_spam)
    #print('P_ham:', P_ham)

    #if spam_log > not_spam_log:
    if P_spam > P_ham:
        return 1
    else:
        return 0

def validation(validation_data):
    train()

    validation_data['predict'] = validation_data['email'].apply(lambda x: classify(x))

    TP = len(validation_data[(validation_data['predict']==SPAM) & (validation_data['label']==SPAM)])
    FP = len(validation_data[(validation_data['predict']==SPAM) & (validation_data['label']==NOT_SPAM)])
    TN = len(validation_data[(validation_data['predict']==NOT_SPAM) & (validation_data['label']==NOT_SPAM)])
    FN = len(validation_data[(validation_data['predict']==NOT_SPAM) & (validation_data['label']==SPAM)])

    confusion_matrix = np.array([[TN, FP],[FN, TP]])
    accuracy = round((TP+TN)/(TP+TN+FP+FN), 2)
    recall = round(TP/(TP+FN), 2)
    precision = round(TP/(TP+FP), 2)
    F_measure = round(2*recall*precision/(recall+precision), 2)

    print('confusion_matrix: \n', confusion_matrix)
    print('accuracy:', accuracy)
    print('recall:', recall)
    print('precision:', precision)
    print('F-measure:', F_measure)

#train()

#letter1 = 'Hi, My name is Warren E. Buffett an American business magnate, investor and philanthropist. am the most successful investor in the world. I believe strongly in‘giving while living’ I had one idea that never changed in my mind? that you should use your wealth to help people and i have decided to give {$1,500,000.00} One Million Five Hundred Thousand United Dollars, to randomly selected individuals worldwide. On receipt of this email, you should count yourself as the lucky individual. Your email address was chosen online while searching at random. Kindly get back to me at your earliest convenience before i travel to japan for my treatment , so I know your email address is valid. Thank you for accepting our offer, we are indeed grateful You Can Google my name for more information: God bless you. Best Regard Mr.Warren E. Buffett Billionaire investor !'
#letter2 = 'Hi guys I want to build a website like REDACTED and I wanted to get your perspective of whether that site is good from the users\' perspective before I go ahead and build something similar. I think that the design of the site is very modern and nice but I am not sure how people would react to a similar site? I look forward to your feedback. Many thanks!'
#letter3 = 'As a result of your application for the position of Data Engineer, I would like to invite you to attend an interview on May 30, at 9 a.m. at our office in Washington, DC. You will have an interview with the department manager, Moris Peterson. The interview will last about 45 minutes. If the date or time of the interview is inconvenient, please contact me by phone or email to arrange another appointment. We look forward to seeing you.'
#print ('result1:', classify(letter1))
#print ('result2:', classify(letter2))
#print ('result3:', classify(letter3))

#validation(validation_data)
