import re
import numpy as np
pA = 0 # вероятность спама в обучающей выборке
pNotA = 0 # вероятность не спама в обучающей выборке
SPAM = 1
NOT_SPAM = 0
Z = 1 # коэффициент размытия для сглаживания по Лаппласу
trainPositive, trainNegative = {}, {}
positive_total, negative_total, total = 0, 0, 0

train_data = [
    ['Купите новое чистящее средство', SPAM],
    ['Купи мою новую книгу', SPAM],
    ['Подари себе новый телефон', SPAM],
    ['Добро пожаловать и купите новый телевизор', SPAM],
    ['Привет давно не виделись', NOT_SPAM],
    ['Довезем до аэропорта из пригорода всего за 399 рублей', SPAM],
    ['Добро пожаловать в Мой Круг', NOT_SPAM],
    ['Я все еще жду документы', NOT_SPAM],
    ['Приглашаем на конференцию Data Science', NOT_SPAM],
    ['Потерял твой телефон напомни', NOT_SPAM],
    ['Порадуй своего питомца новым костюмом', SPAM]
]

validation_data = [
    ['Купите три по цене двух', SPAM],
    ['Поздравляю, вы выиграли телевизор', SPAM],
    ['Здраствуйте, у меня для вас отличная новость', SPAM],
    ['Исправлено, прошу проверить', NOT_SPAM],
    ['Добрый день. Подскажите по ошибке.', NOT_SPAM],
    ['Купи вечером молоко коту', NOT_SPAM],
    ['Поздравляю, вы принимали участие в розыгрыше', SPAM],
    ['Новая книга ждет вас', SPAM],
    ['Здраствуйте, у меня для вас отличная новость, вы выиграли 1000000 рублей', SPAM],
    ['Перенос типа пакета отдельно от проекта.', NOT_SPAM],
    ['В какой последовательности были запущены команды?', NOT_SPAM],
    ['Привет. Предлагаю опробовать команду', NOT_SPAM]
]

def train():
    # Считаем pA и pNotA
    # Считаем частоты каждого слова с помощью calculate_word_frequencies()
    global pA, pNotA, train_data, total
    spamTextCount = 0
    for body, label in train_data:
        calculate_word_frequencies(body, label)
        if label == SPAM:
            spamTextCount += 1

    total = positive_total + negative_total
    pA = spamTextCount/len(train_data)
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
    # total - общее количество уникальных слов использованное для обучения модели
    if label == SPAM:
    # будем использовать логарифм от вероятностей, так как будут очень малые значения и округлим до 2-х знаков
        return round(np.log((trainPositive.get(word, 0) + Z) / (len(trainPositive) + Z*total)), 2)
    else:
        return round(np.log((trainNegative.get(word, 0) + Z) / (len(trainNegative) + Z*total)), 2)

def calculate_P_B_A(text, label):
    # P(B|A) - вероятность встретить текст
    P_B_A = 0
    for word in text.lower().split():
    #  умножение заменяем на сложение так как логарифм
        P_B_A = P_B_A + calculate_P_Bi_A(word, label)
    return P_B_A

def classify(email):
    # Ответ: True, если спам. False, если не спам
    # Избавляемся от знаков препинания в тексте
    text = re.sub(r"[^A-Za-zА-Яа-я0-9 ]+", '', email)
    #print('P_B_A(SPAM):',calculate_P_B_A(text, SPAM))
    #print('P_B_A(NOT_SPAM):', calculate_P_B_A(text, NOT_SPAM))
    if (pA + calculate_P_B_A(text, SPAM)) > (pNotA + calculate_P_B_A(text, NOT_SPAM)):
        return True
    else:
        return False

def validation(validation_data):
    train()
    prediction = {}

    for text, label in validation_data:
        prediction[text] = classify(text)

    TP, TN, FP, FN = 0, 0, 0, 0

    # в prediction метки принимают значения True и False
    for i, text in enumerate(prediction):
        if prediction[text]:
            if (validation_data[i][1] == SPAM):
                TP += 1
            else:
                FP += 1
        else:
            if(validation_data[i][1] == NOT_SPAM):
                TN += 1
            else:
                FN += 1

    confusion_matrix = np.array([[TN, FN],[FP, TP]])
    accuracy = round((TP+TN)/(TP+TN+FP+FN), 2)
    recall = round(TP/(TP+FN), 2)
    precision = round(TP/(TP+FP), 2)
    F_measure = round(2*recall*precision/(recall+precision), 2)

    print(confusion_matrix)
    print('accuracy:', accuracy)
    print('recall:', recall)
    print('precision:', precision)
    print('F-measure:', F_measure)

#train()

#result = classify('Купите три по цене двух')
#print ('result:', result)

validation(validation_data)
