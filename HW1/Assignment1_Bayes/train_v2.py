# -*- coding: utf-8 -*-
import os
import numpy

possibility_is_spam = 0
possibility_is_normal = 0
spam_features = {}
normal_features = {}
word_num = 0
spam_word = 0
normal_word = 0

def train(lable_file_path,feature_file_pah):
    global possibility_is_spam
    global possibility_is_normal
    global word_num
    global spam_word
    global normal_word

    # 读取label文件到数组中
    labels = []
    with open(lable_file_path,"r",encoding="UTF-8") as lable_file:
        for line in lable_file:
            labels.append(int(line.strip()))

    # 逐行读取每封邮件每个词语是否出现
    spam_word = 0
    normal_word = 0
    word_set = set()
    with open(feature_file_pah,"r",encoding="UTF-8") as feature_file:
        for line in feature_file:
            data = line.split()
            if len(data) == 0:
                break
            email_label = int(data[0])
            word_identity = int(data[1])
            word_times = int(data[2])
            word_set.add(word_identity)

            # 将出现某词语的邮件的次数记录到字典spam_features和normal_features中
            if labels[email_label - 1] == 1:
                spam_word += word_times
                if spam_features.__contains__(word_identity):
                    spam_features[word_identity] += word_times
                else:
                    spam_features[word_identity] = word_times
            else:
                normal_word += word_times
                if normal_features.__contains__(word_identity):
                    normal_features[word_identity] += word_times
                else:
                    normal_features[word_identity] = word_times
    
    # 记录 垃圾邮件和 正常邮件的数量
    word_num = len(word_set)
    email_num = len(labels)
    spam_num = labels.count(1)
    normal_num = labels.count(0)

    # 循环遍历spam_features和normal_features，将次数转为频率（利用Laplace平滑）
    for key,value in spam_features.items():
        spam_features[key] = numpy.log((value + 1 )/ (spam_word + word_num))
    for key,value in normal_features.items():
        normal_features[key] = numpy.log((value + 1) / (normal_word + word_num))

    possibility_is_spam= spam_num/email_num
    possibility_is_normal= normal_num/email_num

def test(test_feature_file_path,test_label_file_path):
    # 读取label文件到数组中
    labels = []
    with open(test_label_file_path, "r", encoding="UTF-8") as lable_file:
        for line in lable_file:
            labels.append(int(line.strip()))

    # 逐行读取每封邮件每个词语并存到二维数组
    store_data = [[] for i in range(len(labels))]
    with open(test_feature_file_path,"r",encoding="UTF-8") as test_feature_file:
        for line in test_feature_file:
            data = line.split()
            if len(data) == 0:
                break
            email_label = int(data[0])
            word_identity = int(data[1])
            word_times = int(data[2])
            store_data[email_label - 1].append(word_identity)

    # 根据每封邮件出现的词
    correct = 0
    mistake = 0
    for i in range(len(store_data)):
        is_spam = numpy.log(possibility_is_spam)
        is_normal = numpy.log(possibility_is_normal)
        # 分别计算出 c(垃圾邮件) 和 c(正常邮件) 的概率
        for j in range(len(store_data[i])):
            # 如果词在训练中出现过
            if spam_features.__contains__(store_data[i][j]):
                is_spam += spam_features[store_data[i][j]] * word_times
            # 如果词没有在训练中出现过
            else:
                is_spam += numpy.log(1/(spam_word + word_num)) * word_times
            if normal_features.__contains__(store_data[i][j]):
                is_normal += normal_features[store_data[i][j]] * word_times
            else:
                is_normal += numpy.log(1/( normal_word +  word_num)) * word_times
        
        # 比较大小的除本封邮件的判断，与label作比较，得出正确数
        if is_spam > is_normal:
            result = 1
        else:
            result = 0

        if result == labels[i]:
            correct += 1
        else:
            mistake += 1

    print("Accuracy:%f"  %(correct/len(labels)))

if __name__ == '__main__':
    train_label_file = input("train label file:")
    train_feature_file = input("train feature file:")
    test_label_file = input("test label file:")
    test_feature_file = input("test feature file:")
    if not os.path.exists(train_label_file) or not os.path.exists(train_feature_file) or not os.path.exists(test_label_file) or not os.path.exists(test_feature_file):
        print("error file")
        exit(1)
    train(train_label_file,train_feature_file)
    test(test_feature_file,test_label_file)