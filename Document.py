from tika import parser
import nltk
from nltk.tokenize import word_tokenize
import pandas as pd
import re
from textstat.textstat import textstat as ts
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split

import numpy as np
import sklearn.preprocessing as prep
import six.moves.cPickle as cPickle


from bs4 import BeautifulSoup as bs
import urllib
import feedparser
import pickle

import datetime
import os
import random
import re, math
from collections import Counter
import time


class Document:
    __token = []
    __pure_context = ''
    __joi = ''
    __dens = 0
    __len_of_pure_context = 0
    __arXiv_name = ''
    __dictName = ''
    __add_info = []

    ## in the Document class, to create it I have only the raw string
    ## the dictName is the name of saved authorsName dictionary from the Data Class
    def __init__(self, string, trending, arXiv_name, dictName='lazy_ions_data_authors_data', *args):
        try:
            if ((type(string) and type(arXiv_name)) == str):
                self.__dictName = dictName
                self.__densit(string)
                self.__arXiv_name = arXiv_name
                self.string = string.replace('\n', ' ')
                self.__check_add_info()
            else:
                raise ValueError
        except ValueError:
            print("The value of string was not right, given " + str(type(string)) + " required is str")
            raise
        try:
            if type(trending) == bool:
                self.trending = trending
            else:
                raise ValueError
        except ValueError:
            print("The value of trending was not right given " + str(type(trending)) + " required is bool")
            raise

    ## destructor , overloading the default destructor
    def __del__(self):
        print("destroy member of Document class")
        class_name = self.__class__.__name__

    # "HIDDEN" METHODS:

    ## checking whether I have tokenized string
    def __check(self):
        # checking whether you did not have it already tokenized
        if len(self.__token) == 0:
            print("doing tokenization")
            self.__tokenize()

    def __join(self):
        self.__check()
        if len(self.__joi) == 0:
            print('joining')
            self.__join = " ".join(self.__token)

    def __tokenize(self):
        tok = word_tokenize(self.string)
        out = re.compile('\.|\,')
        tok = filter(lambda x: not re.match(out, x), tok)
        self.__token = list(tok)

    def __densit(self, string):
        self.__dens = len(string.split('\n'))

    def __is_ascii(self, string):
        isit = True
        for c in string:
            if (ord(c) < 65) or (90 < ord(c) and ord(c) < 97) or (ord(c) > 122):
                isit = False
                break
        return isit

    def __ret_word(self, string):
        temp = list(filter(lambda x: re.match('[a-q]|[A-Q]', x), string))
        temp = "".join(temp)
        if len(temp) > 2:
            return temp

    def __strip_words_only(self, string):
        temp = list(filter(lambda x: self.__is_ascii(x), string))
        temp = "".join(temp)
        if len(temp) > 0:
            return temp.lower()

    def __final_wordenize(self):
        self.__check()
        temp = list(filter(lambda x: self.__ret_word(x), self.__token))
        temp = list(map(lambda x: self.__strip_words_only(x), temp))
        self.__len_of_pure_context = len(temp)
        self.pure_context = " ".join(temp)

    def __word(self):
        if self.__len_of_pure_context == 0:
            self.__final_wordenize()

    def __look_aditional_info(self):
        temp = self.__openDict(self.__dictName)
        if self.__arXiv_name in temp.keys():
            self.__add_info = temp[self.__arXiv_name]
            self.info = self.__add_info

    def __openDict(self, name='authors_data'):
        try:
            with open(name + '.pickle', 'rb') as handle:
                return pickle.load(handle)
        except FileNotFoundError:
            print("File was not found")
            pass

    def __check_add_info(self):
        if len(self.__add_info) == 0:
            print("looking for aditional info in dictionary named " + self.__dictName)
            self.__look_aditional_info()

    # VISIBLE METHODS:

    ## the string parameter is the raw string produced from pdf
    def calculate_equal_density(self):
        self.__check()
        equal_number = len(list(filter(lambda x: re.match('=', x), self.__token)))

        eq_density = equal_number / float(len(self.__token))
        self.eq_density = eq_density
        return eq_density

    ## calculate the length of a text
    def length_of_text(self):
        self.__check()
        self.text_length = len(self.__token)
        print(self.text_length)

    ## calculate readability
    def readability_of_text(self, score="dale_chall"):
        try:
            if type(score) == str:
                if score == "dale_chall":
                    self.readability = ts.dale_chall_readability_score(self.string)
                    print(self.readability)
                else:
                    print('Other scores are not supported yet. You wanted: ' + score + " we have only dale_chall")
            else:
                raise ValueError
        except ValueError:
            print("the score shuld be of type str. You put " + str(type(score)))
            raise

    ## readability of abstract:
    ### def readability_of_abstract(self):

    ## title length
    ### def title_length(self):

    ## the number of \n = of newlines. i.e. how dense is the paper, tells you about the structure
    def density_of_lines(self):
        self.__check()
        self.lines_density = self.__dens / float(len(self.__token))
        print(self.lines_density)

    def length_of_pure_context(self):
        self.__word()
        return self.__len_of_pure_context

    def density_of_pure_text(self):
        self.__word()
        self.__check()
        self.pure_text_density = self.__len_of_pure_context / float(len(self.__token))
        print(self.pure_text_density)

    ## this function also looks for additional
    def get_relevant_words(self):
        self.__word()
        tag = nltk.word_tokenize(self.pure_context)
        tags = nltk.pos_tag(tag)
        self.relevant_words = " ".join([t[0] for t in tags if t[1] == "NN"])

class Learn:
    __data_author = {}
    __timestamp_of_model_creation = None
    __name_of_data = None
    __name_of_saved_learned_data = None
    __best_authors = None
    __WORD = re.compile(r'\w+')

    # remember at the zeroth possition in following panda dfs I have the result== correct label
    __df_testing = None
    __df_training = None

    def __init__(self, name_of_data = 'lazy_ions_data', *args):
        self.__timestamp_of_model_creation = datetime.datetime.now().time()
        self.__name_of_data = name_of_data
        self.__name_of_saved_learned_data = 'learned_' + self.__name_of_data


    #Private methods:
    def __open_dict(self, name):
        try:
            with open(name + '.pickle', 'rb') as handle:
                return pickle.load(handle)
        except FileNotFoundError:
            print("File was not found")

    def __save_dict(self, dictionary, name):
        with open(name + '.pickle', 'wb') as handle:
            pickle.dump(dictionary, handle)

    def __download_data(self):
        g = Data('hep-th', self.__name_of_data )

        # getting the meta-data first, for trendsetters
        g.data(True, 2, False, 'all')
        # downloading the pdf-files, for trendsetters
        g.data(True, 2, True, 'all')

        # get the non-trendsetters data
        size = len(g.get_final_dict())
        g.data(False, size, False)
        # get also pdf's for non-trendsetters data
        g.data(False, size, True)

        # save the authors metadata into the data_author
        self.__data_author = g.get_final_dict()

    # by this you will get the saved subset of the authors data dictionary
    def __get_saved_subset_trending_or_not(self, trending):

        temp = list(filter(lambda x: self.__data_author[x][0]==trending, list(self.__data_author.keys())))
        temp = list(filter(lambda x: ( x + ".pdf") in os.listdir(), temp))
        return temp

    def __get_best_trendsetters(self):
        trend = self.__get_saved_subset_trending_or_not(True)
        citat = list(map(lambda x: self.__data_author[x][2],trend))
        dictionary = dict(zip(trend,citat))
        temp = sorted(dictionary.items(), key=lambda x:x[1],reverse = True)

        # take five per-cent of best trendsetters
        temp = temp[0:int(0.05*len(temp))]

        #print(list(dict(temp).keys()))

        self.__best_authors = list(dict(temp).keys())

        #print(self.__best_authors)


    def __split_training_testing_best(self):
        self.__get_best_trendsetters()

        trend = self.__get_saved_subset_trending_or_not(True)
        nontrend = self.__get_saved_subset_trending_or_not(False)

        # filter out the best authors:
        trend = list(filter(lambda x: x not in self.__best_authors, trend))


        # traintest:
        random.shuffle(trend)
        random.shuffle(nontrend)

        length_train_trend = int(0.8*len(trend))
        length_train_nontrend = int(0.8*len(nontrend))

        # splitting train - test
        train_trend = trend[0:length_train_trend]
        test_trend = trend[length_train_trend:]


        train_nontrend = nontrend[0:length_train_nontrend]
        test_nontrend = nontrend[length_train_nontrend:]

        #print(len(train_trend))
        #print(len(test_trend))
        #print(len(train_nontrend))
        #print(len(test_nontrend))

        train = train_trend + train_nontrend
        test = test_trend + test_nontrend


        random.shuffle(train)
        random.shuffle(test)

        self.__training = train
        self.__testing = test

    def __text_to_vector(self,text):
        words = self.__WORD.findall(text)
        return Counter(words)

    def __get_cosine(self, vec1, vec2):
        intersection = set(vec1.keys()) & set(vec2.keys())
        numerator = sum([vec1[x] * vec2[x] for x in intersection])

        sum1 = sum([vec1[x] ** 2 for x in vec1.keys()])
        sum2 = sum([vec2[x] ** 2 for x in vec2.keys()])
        denominator = math.sqrt(sum1) * math.sqrt(sum2)

        if not denominator:
            return 0.0
        else:
            return float(numerator) / denominator

    def __cosine_similarity(self):

        trend = self.__get_saved_subset_trending_or_not(True)
        nontrend = self.__get_saved_subset_trending_or_not(False)

        # texts of the best authors:
        best_texts_vectors = [self.__text_to_vector(self.__data_author[x][16]) for x in self.__best_authors]

        counter=0
        for nam in trend:
            #start = time.clock()
            vect = self.__text_to_vector(self.__data_author[nam][16])
            lst = [self.__get_cosine(vect,x) for x in best_texts_vectors]
            #end = time.clock()
            #print(end - start)

            # adding to my current data author library
            self.__data_author[nam].append(lst)
            counter+=1
            print(counter)

        self.__save_dict(self.__data_author, self.__name_of_saved_learned_data)

        for nam in nontrend:
            #start = time.clock()
            vect = self.__text_to_vector(self.__data_author[nam][14])
            lst = [self.__get_cosine(vect, x) for x in best_texts_vectors]
            #end = time.clock()
            #print(end - start)

            #adding to my current data author library
            self.__data_author[nam].append(lst)
            counter+=1
            print(counter)

        self.__save_dict(self.__data_author, self.__name_of_saved_learned_data)

    # ADA-BOOSTed classifier WITH grid search
    def __ada_function_specific(self, data_train, y_train, data_test, y_test, data_valid, y_valid):

        name = self.__name_of_saved_learned_data + '_ADA_MODEL.pkl'
        # Ada-Boost grid search
        param_grid = {"base_estimator__criterion": ["gini", "entropy"],
                      "base_estimator__splitter": ["best", "random"],
                      "n_estimators": [5000]}

        DTC = DecisionTreeClassifier(random_state=11, max_features="auto", class_weight="auto", max_depth=None)
        ABC = AdaBoostClassifier(base_estimator=DTC, n_estimators=5000,
                                 learning_rate=1.5,
                                 algorithm="SAMME")

        grid_search_ABC = GridSearchCV(ABC, param_grid=param_grid, scoring='accuracy', n_jobs=-1)
        griddi = grid_search_ABC.fit(data_train, y_train)

        # Save the grid searched model as the "name".pkl
        with open(name, 'wb') as fid:
            cPickle.dump(griddi, fid)

            # Print accuracy scores and return the model
        print("Test set accuracy score for the model " + name + " is " + \
              str(accuracy_score(y_test, griddi.predict(data_test))))
        print("Validation set accuracy score for the model " + name + " is " + \
              str(accuracy_score(y_valid, griddi.predict(data_valid))))
        return griddi


    def train_model(self):
        # ok I forget about validation, I will slice validation from training data:
        # lets split length of training
        print("producing the panda dataframes")
        start = time.clock()
        self.__find_data_train_test_ytrain_ytest()
        end = time.clock()

        print("dataframes prduced, should be on local drive")
        print("thre production took [s]: ", start - end)

        length = self.__df_training.shape[0]
        length_training = int(0.87*length)

        training = self.__df_training.ix[0:length_training,1:]
        y_training = self.__df_training.ix[0:length_training,0]

        validation = self.__df_training.ix[length_training:,1:]
        y_validation = self.__df_training.ix[length_training:,0]

        print("starting to train grid search ada boost")
        start = time.clock()
        model = self.__ada_function_specific(training,y_training,self.__df_testing.ix[:,1:],self.__df_testing.ix[:,0],\
                                     validation,y_validation)
        end = time.clock()

        print("the training lasted [s] :", start - end)
        return model


    # this function looks for numerical values in training set testing and their categories y_train, y_test:
    def __find_data_train_test_ytrain_ytest(self):
        # ok splitting the training and testing set
        self.__split_training_testing_best()

        df_y_and_testing = []
        df_y_and_training = []

        # training data
        for nam in self.__training:
            lst = self.__data_author[nam]

            # if the value is trendsetters
            if lst[0]:
                df_y_and_training.append(self.__find_TREND_data_numerics(lst))
            else:
                df_y_and_training.append(self.__find_NONTREND_data_numerics(lst))
        df_y_and_training = pd.DataFrame(df_y_and_training)

        # saving the df_training on local drive
        df_y_and_training.to_pickle(self.__name_of_saved_learned_data + '_df_training.pkl')
        self.__df_training = df_y_and_training

        # testing data
        for nam in self.__testing:
            lst = self.__data_author[nam]

            # if the value is trendsetters
            if lst[0]:
                df_y_and_testing.append(self.__find_TREND_data_numerics(lst))
            else:
                df_y_and_testing.append(self.__find_NONTREND_data_numerics(lst))

        df_y_and_testing = pd.DataFrame(df_y_and_testing)

        # saving the df_training on local drive
        df_y_and_testing.to_pickle(self.__name_of_saved_learned_data + '_df_testing.pkl')
        self.__df_testing = df_y_and_testing


    # this returns the panda dataframe for nontrendsetters
    def __find_NONTREND_data_numerics(self,lst):
        fin_list = [[0]]
        fin_list.append(lst[3:8])
        fin_list.append(lst[9:14])
        fin_list.append(lst[16])
        # flattening the list
        fin_list = [item for sublist in fin_list for item in sublist]
        return fin_list

    # this returns the panda dataframe:
    def __find_TREND_data_numerics(self,lst):
        fin_list = [[1]]
        fin_list.append(lst[5:10])
        fin_list.append(lst[11:16])
        fin_list.append(lst[18])
        # flattening the list
        fin_list = [item for sublist in fin_list for item in sublist]
        return fin_list



    def __get_saved_subset(self):
        temp = list(filter(lambda x: (x + ".pdf") in os.listdir(), list(self.__data_author.keys())))
        return temp

    def __relevant_words(self, string):
        tag = nltk.word_tokenize(string)
        tags = nltk.pos_tag(tag)
        return " ".join([t[0] for t in tags if t[1] == "NN"])

    def __analyse_abstracts(self):
        list_trending = self.__get_saved_subset_trending_or_not(True)
        list_nontrending = self.__get_saved_subset_trending_or_not(False)

        # NOTE:::
        # just easy checking whether you do not have already appended it, Remember to do first analyse abstract
        # and then analyse pdfs

        counter = 0
        for name in list_trending:
            # raw text is at position [4]
            string = self.__data_author[name][4]
            doc = Document(string, True, name)

            # at the position [5] = equality density in Abstract
            doc.calculate_equal_density()
            self.__data_author[name].append(doc.eq_density)

            # at position [6] = density of lines in Abstract
            doc.density_of_lines()
            self.__data_author[name].append(doc.lines_density)

            # at position [7] = density of pure text in Abstract
            doc.density_of_pure_text()
            self.__data_author[name].append(doc.pure_text_density)

            # at position [8] = length of raw text in Abstract
            doc.length_of_text()
            self.__data_author[name].append(doc.text_length)

            # at position [9] = length of raw text in Abstract
            doc.readability_of_text()
            self.__data_author[name].append(doc.readability)

            # at position [10] = pure context in Abstract
            self.__data_author[name].append(doc.pure_context)

            print("trending ",counter)
            counter+=1

        self.__save_dict(self.__data_author, self.__name_of_saved_learned_data)

        for name in list_nontrending:
            # raw text is at position [2]
            string = self.__data_author[name][2]
            doc = Document(string, False, name)

            # at the position [3] = equality density in Abstract
            doc.calculate_equal_density()
            self.__data_author[name].append(doc.eq_density)

            # at position [4] = density of lines in Abstract
            doc.density_of_lines()
            self.__data_author[name].append(doc.lines_density)

            # at position [5] = density of pure text in Abstract
            doc.density_of_pure_text()
            self.__data_author[name].append(doc.pure_text_density)

            # at position [6] = length of raw text in Abstract
            doc.length_of_text()
            self.__data_author[name].append(doc.text_length)

            # at position [7] = length of raw text in Abstract
            doc.readability_of_text()
            self.__data_author[name].append(doc.readability)

            # at position [8] = pure context in Abstract
            self.__data_author[name].append(doc.pure_context)

            print("non trending ", counter)
            counter += 1

        self.__save_dict(self.__data_author, self.__name_of_saved_learned_data)

    def __analyse_pdfs(self):
        list_trending = self.__get_saved_subset_trending_or_not(True)
        list_nontrending = self.__get_saved_subset_trending_or_not(False)

        counter = 0
        for name in list_trending:

            # raw text now the pdf!
            parsedPDF = parser.from_file(name + ".pdf")
            string = parsedPDF['content']

            doc = Document(string, True, name)

            # at the position [11] = equality density in Pdf
            doc.calculate_equal_density()
            self.__data_author[name].append(doc.eq_density)

            # at position [12] = density of lines in Pdf
            doc.density_of_lines()
            self.__data_author[name].append(doc.lines_density)

            # at position [13] = density of pure text in Pdf
            doc.density_of_pure_text()
            self.__data_author[name].append(doc.pure_text_density)

            # at position [14] = length of raw text in Pdf
            doc.length_of_text()
            self.__data_author[name].append(doc.text_length)

            # at position [15] = length of raw text in Pdf
            doc.readability_of_text()
            self.__data_author[name].append(doc.readability)

            # at position [16] = pure context in Pdf
            self.__data_author[name].append(doc.pure_context)

            print("trending ", counter)
            counter += 1

        self.__save_dict(self.__data_author, self.__name_of_saved_learned_data)

        for name in list_nontrending:

            # raw text now the pdf!
            parsedPDF = parser.from_file(name + ".pdf")
            string = parsedPDF['content']

            doc = Document(string, False, name)

            # at the position [9] = equality density in Pdf
            doc.calculate_equal_density()
            self.__data_author[name].append(doc.eq_density)

            # at position [10] = density of lines in Pdf
            doc.density_of_lines()
            self.__data_author[name].append(doc.lines_density)

            # at position [11] = density of pure text in Pdf
            doc.density_of_pure_text()
            self.__data_author[name].append(doc.pure_text_density)

            # at position [12] = length of raw text in Pdf
            doc.length_of_text()
            self.__data_author[name].append(doc.text_length)

            # at position [13] = length of raw text in Pdf
            doc.readability_of_text()
            self.__data_author[name].append(doc.readability)

            # at position [14] = pure context in Pdf
            self.__data_author[name].append(doc.pure_context)

            print("non trending ", counter)
            counter += 1

        self.__save_dict(self.__data_author, self.__name_of_saved_learned_data)


    def __get_relevant_words(self):
        list_trending = self.__get_saved_subset_trending_or_not(True)
        list_nontrending = self.__get_saved_subset_trending_or_not(False)

        for name in list_trending:
            string = self.__data_author[name][16]

            # at position[17] put the tokenized relevant words the NN words:
            self.__data_author[name].append(self.__relevant_words(string))

        self.__save_dict(self.__data_author, self.__name_of_saved_learned_data)

        for name in list_nontrending:
            string = self.__data_author[name][14]

            # at position[15] put the tokenized relevant words the NN words:
            self.__data_author[name].append(self.__relevant_words(string))

        self.__save_dict(self.__data_author, self.__name_of_saved_learned_data)

    # Public methods:
    def read_in_data(self):
        self.__data_author = self.__open_dict(self.__name_of_saved_learned_data)
        if self.__data_author == None:
            self.__data_author = self.__open_dict(self.__name_of_data + "_authors_data")

    def get_authors_data(self):
        return self.__data_author

    def get_training_testing_and_best(self, training):
        #self.__get_best_trendsetters()
        self.__split_training_testing_best()

        if training:
            print(len(self.__training))
            return self.__training
        else:
            print(self.__testing)
            return self.__testing

class Predict:
    __model = None
    __time_of_creation = None
    __model = None
    __name_of_data = None
    __name_of_saved_learned_data = None
    __data_author = None
    __best_authors = None
    __best_authors_data = None
    __WORD = re.compile(r'\w+')

    def __init__(self, name_of_data='lazy_ions_data', *args):
        self.__time_of_creation = datetime.datetime.now().time()
        self.__name_of_data = name_of_data
        self.__name_of_saved_learned_data = 'predicted_' + self.__name_of_data

    # Private methods:
    def __open_dict(self, name):
        try:
            with open(name + '.pickle', 'rb') as handle:
                return pickle.load(handle)
        except FileNotFoundError:
            print("File was not found")

    def __save_dict(self, dictionary, name):
        with open(name + '.pickle', 'wb') as handle:
            pickle.dump(dictionary, handle)

    def __calculate_model(self):

        learn = Learn()
        learn.read_in_data()
        # calculate the ada boosted model
        self.__model = learn.train_model()

    def __analyse_abstracts(self):

        names = list(self.__data_author.keys())
        counter = 0
        for name in names:
            # raw text is at position [1]
            string = self.__data_author[name][1]
            doc = Document(string, False, name)

            # at the position [2] = equality density in Abstract
            doc.calculate_equal_density()
            self.__data_author[name].append(doc.eq_density)

            # at position [3] = density of lines in Abstract
            doc.density_of_lines()
            self.__data_author[name].append(doc.lines_density)

            # at position [4] = density of pure text in Abstract
            doc.density_of_pure_text()
            self.__data_author[name].append(doc.pure_text_density)

            # at position [5] = length of raw text in Abstract
            doc.length_of_text()
            self.__data_author[name].append(doc.text_length)

            # at position [6] = length of raw text in Abstract
            doc.readability_of_text()
            self.__data_author[name].append(doc.readability)

            # at position [7] = pure context in Abstract
            self.__data_author[name].append(doc.pure_context)

            print("names", counter)
            counter += 1

        self.__save_dict(self.__data_author, "lazy_ions_new_candidates")

    def __analyse_pdfs(self):

        names = list(self.__data_author.keys())

        counter = 0

        for name in names:
            # raw text now the pdf!
            parsedPDF = parser.from_file(name + ".pdf")
            string = parsedPDF['content']

            doc = Document(string, False, name)

            # at the position [8] = equality density in Pdf
            doc.calculate_equal_density()
            self.__data_author[name].append(doc.eq_density)

            # at position [9] = density of lines in Pdf
            doc.density_of_lines()
            self.__data_author[name].append(doc.lines_density)

            # at position [10] = density of pure text in Pdf
            doc.density_of_pure_text()
            self.__data_author[name].append(doc.pure_text_density)

            # at position [11] = length of raw text in Pdf
            doc.length_of_text()
            self.__data_author[name].append(doc.text_length)

            # at position [12] = length of raw text in Pdf
            doc.readability_of_text()
            self.__data_author[name].append(doc.readability)

            # at position [13] = pure context in Pdf
            self.__data_author[name].append(doc.pure_context)

            print("non trending ", counter)
            counter += 1

        self.__save_dict(self.__data_author, "lazy_ions_new_candidates")

    def __get_relevant_words(self):
        names = list(self.__data_author.keys())

        for name in names:
            string = self.__data_author[name][13]

            # at position[14] put the tokenized relevant words the NN words:
            self.__data_author[name].append(self.__relevant_words(string))

        self.__save_dict(self.__data_author, "lazy_ions_new_candidates")

    def __relevant_words(self, string):
        tag = nltk.word_tokenize(string)
        tags = nltk.pos_tag(tag)
        return " ".join([t[0] for t in tags if t[1] == "NN"])

    def __text_to_vector(self, text):
        words = self.__WORD.findall(text)
        return Counter(words)

    def __get_cosine(self, vec1, vec2):
        intersection = set(vec1.keys()) & set(vec2.keys())
        numerator = sum([vec1[x] * vec2[x] for x in intersection])

        sum1 = sum([vec1[x] ** 2 for x in vec1.keys()])
        sum2 = sum([vec2[x] ** 2 for x in vec2.keys()])
        denominator = math.sqrt(sum1) * math.sqrt(sum2)

        if not denominator:
            return 0.0
        else:
            return float(numerator) / denominator

    def __cosine_similarity(self):

        names = list(self.__data_author.keys())

        # texts of the best authors:
        best_texts_vectors = [self.__text_to_vector(self.__best_authors_data[x][16]) for x in self.__best_authors]

        counter = 0
        for nam in names:
            # start = time.clock()
            vect = self.__text_to_vector(self.__data_author[nam][14])
            lst = [self.__get_cosine(vect, x) for x in best_texts_vectors]
            # end = time.clock()
            # print(end - start)

            # adding to my current data author library
            self.__data_author[nam].append(lst)
            counter += 1
            print(counter)

        self.__save_dict(self.__data_author, "lazy_ions_new_candidates")

    # this returns the panda dataframe for nontrendsetters
    def __find_data_numerics(self, lst):
        return (lst[2:7] + lst[8:13] + lst[15])

    # this function looks for numerical values in training set testing and their categories y_train, y_test:
    def find_data_for_prediction(self):
        names = list(self.__data_author.keys())

        df = []
        for nam in names:
            lst = self.__data_author[nam]
            df.append(self.__find_data_numerics(lst))

        df = pd.DataFrame(df)
        self.__df = df

        # saving the df_training on local drive
        df.to_pickle("lazy_ions_new_candidates" + '_df.pkl')



    # Public methods
    def read_in_trendsetters(self):

        auth_dat = self.__open_dict("learned_lazy_ions_data")

        trend = list(filter(lambda x: auth_dat[x][0] == True, list(auth_dat.keys())))
        trend = list(filter(lambda x: (x + ".pdf") in os.listdir(), trend))

        citat = list(map(lambda x: auth_dat[x][2], trend))
        dictionary = dict(zip(trend, citat))
        temp = sorted(dictionary.items(), key=lambda x: x[1], reverse=True)

        # take five per-cent of best trendsetters
        temp = temp[0:int(0.05 * len(temp))]

        # print(list(dict(temp).keys()))

        self.__best_authors = list(dict(temp).keys())
        self.__best_authors_data = {new_key: auth_dat[new_key] for new_key in self.__best_authors}

        print(self.__best_authors)

    def read_in_model(self):
        self.__model = pd.read_pickle('learned_' + self.__name_of_data + '_ADA_MODEL_1.pkl')

    def read_in_data(self):
        self.__data_author = self.__open_dict("lazy_ions_new_candidates")

    def get_authors_data(self):
        return self.__data_author

    def get_best_authors_data(self):
        return self.__best_authors_data

    def get_data_frame(self):
        return self.__df

    def get_model(self):
        return self.__model

    def get_trendsetters(self):
        names = list(self.__data_author.keys())
        temp = list(self.__model.predict(self.__df))

        www = "https://arxiv.org/pdf/hep-th/"
        result = []
        for count in range(0,len(temp)):
            if temp[count]==1:
                result = result + (www + names[count] + ".pdf")
        return result

class Data:
    __url = "https://www.slac.stanford.edu/spires/play/authors/2004/top_5_alpha.shtml"
    __base_url = None
    __category = None
    __name_of_data = None

    __out = re.compile("''|Videos|See this list in ranked order|2004 Author List Home|About SPIRES|SLAC|\
                SLAC Library|Contact|INSPIRE|\n\n")

    __data_author = {}

    ## changed only by scarping function:
    __authors = []
    __citations = []
    __citations_dict = {}

    def __init__(self, cat, name_of_data, *args):
        self.__category = cat
        self.__initialize_settings()
        self.__name_of_data = name_of_data

    # Private methods
    def __initialize_settings(self):
        # initial settings, will be part of class variable:
        self.__base_url = 'http://export.arxiv.org/api/query?'
        feedparser._FeedParserMixin.namespaces['http://a9.com/-/spec/opensearch/1.1/'] = 'opensearch'
        feedparser._FeedParserMixin.namespaces['http://arxiv.org/schemas/atom'] = 'arxiv'

    ## Handling dictionary saving and opening
    def __open_dict(self, name):
        try:
            with open(name + '.pickle', 'rb') as handle:
                return pickle.load(handle)
        except FileNotFoundError:
            print("File was not found")

    def __save_dict(self, dictionary, name):
        with open(name + '.pickle', 'wb') as handle:
            pickle.dump(dictionary, handle)

    def __add_dict(self, final, temp):
        z = dict(final, **temp)
        final.update(z)

    ## Getting data, in the ids are only data that have been downloaded
    def __download_file(self, download_url, name):
        response = urllib.request.urlopen(download_url)
        file = open(name + ".pdf", 'wb')
        file.write(response.read())
        file.close()
        print("Completed")

    def __scrape_trendsetters(self):
        self.soup = bs(urllib.request.urlopen(self.__url).read())

        for a_tag in self.soup.find_all('td'):
            # names_raw_2.append(a_tag.text)
            if "[" in a_tag.text:
                self.__citations.append(a_tag.text)

        self.__citations = list(filter(lambda x: re.match('\[\d+\]', x), self.__citations))
        self.__citations = list(map(lambda x: re.sub('\[|\]', '', x), self.__citations))

        for a_tag in self.soup.find_all('a'):
            self.__authors.append(a_tag.text)

        self.__authors = list(filter(lambda x: not re.match(self.__out, x), self.__authors))
        self.__authors = self.__authors[2:(len(self.__authors) - 1)]
        self.__authors = list(map(lambda x: self.__arxiv_name(x, True), self.__authors))

        self.__citations_dict = self.__citation_dic(self.__citations, self.__authors)

        # self.citat_dict = self.__citations_dict
        # self.nam = self.__authors
        # self.cit = self.__citations

    # arXiv Names structure
    def __arxiv_name(self, orig_name, revert=True):
        orig_name = re.sub('\.', ' ', orig_name)
        orig_name = re.sub(',', '', orig_name)
        orig_name = re.sub('de ', 'De', orig_name)
        orig_name = re.sub('van ', 'Van', orig_name)
        orig_name = re.sub('Van ', 'Van', orig_name)
        orig_name = orig_name.split()

        if revert:
            fam_name = orig_name[0]
            fam_name = re.sub('-', '_', fam_name)
            fam_name = re.sub("'", '', fam_name)

            if len(orig_name) > 1:
                final = [fam_name, orig_name[1][0]]
                return "_".join(final)
            else:
                return fam_name
        else:
            # Last name analysis
            fam_name = orig_name[-1]
            fam_name = re.sub('-', '_', fam_name)
            fam_name = re.sub("'", '', fam_name)

            # retrun the first letter of the first name,
            # if the name has more than one name
            if len(orig_name) > 1:
                final = [fam_name, orig_name[0][0]]
                return "_".join(final)
            else:
                return fam_name

    # query arxiv for given authors name:
    def __query_ar_xiv(self, max_results, descending=True, trending=True, *args):

        if trending:
            name = args[0]
            # function own variables, needed for the query:
            search_query = 'au:' + name + '&cat:' + self.__category
            sort_by = 'submittedDate'
            start = 0  # retrieve first five results
        else:
            search_query = 'cat:' + self.__category
            sort_by = 'submittedDate'
            start = 10000  # retrieve first five results

        if descending:
            sortOrder = 'descending'
        else:
            sortOrder = 'ascending'

        query = 'search_query=%s&sort_by=%s&sortOrder=%s&start=%i&max_results=%i' % \
                (search_query, sort_by, sortOrder, start, max_results)

        # actual arXiv query
        response = urllib.request.urlopen(self.__base_url + query).read()

        # parse the response using feedparser
        feed = feedparser.parse(response)

        return feed

    # function returning pdf links for a given feed (for one author):
    def __pdf_links(self, feed, ids):
        pdf = []
        for entry in feed.entries:
            arx_id = '%s' % entry.id.split('/abs/')[-1]
            arx_id = re.sub('\S+/', '', arx_id)

            if arx_id in ids:
                for link in entry.links:
                    if link.type == 'application/pdf':
                        pdf.append(link.href)
        return pdf

    # ok authors data for non trendsetters, bit awkward, will change that later
    def __authors_data_nontrending(self, feed):

        trending = False

        # the first dictionary is the author dictionary:
        dict_a = {}

        for entry in feed.entries:
            ar_id = '%s' % entry.id.split('/abs/')[-1]
            ar_id = re.sub('\S+/', '', ar_id)
            title = '%s' % entry.title

            # Adding to a dictionary
            dict_a[ar_id] = []
            dict_a[ar_id].append(trending)
            # dict_a[ar_id].append(name)
            # dict_a[ar_id].append(citation_density)
            dict_a[ar_id].append(title)

            # Extracting the abstract:
            summary = '%s' % entry.summary
            dict_a[ar_id].append(summary)

        return dict_a

        # get the data from feed for one Author and create a dictionary

    ## from feed it will extract relevant info and save it into the dictionary
    def __authors_data_trending(self, feed, authors_name, citation_density):
        trending = True
        # the first dictionary is the author dictionary:
        dict_a = {}

        for entry in feed.entries:
            arId = '%s' % entry.id.split('/abs/')[-1]
            arId = re.sub('\S+/', '', arId)
            title = '%s' % entry.title

            # Adding to a dictionary
            dict_a[arId] = []
            dict_a[arId].append(trending)
            dict_a[arId].append(authors_name)
            dict_a[arId].append(citation_density)
            dict_a[arId].append(title)

            # Extracting the abstract:
            summary = '%s' % entry.summary
            dict_a[arId].append(summary)

        return dict_a

    # this function extracts the id numbers of the papers in the feed
    def __paper_ids(self, feed):
        ids = []
        for entry in feed.entries:
            arx_id = '%s' % entry.id.split('/abs/')[-1]
            arx_id = re.sub('\S+/', '', arx_id)
            ids.append(arx_id)
        return ids

    def __download_all_and_name(self, urls, ids):
        ## if you download, then you necessarily save it to ids:
        for counter in iter(range(len(urls))):
            self.__download_file(urls[counter], ids[counter])

    def __citation_dic(self, citations, authors):
        final = {}
        for cnt in iter(range(len(citations))):
            final[authors[cnt]] = int(citations[cnt])
        return final

    ##
    # note: args have args[1]=name and args[2]=citation_dict
    def __produce_data(self, trending, max_results, descending, pdf, *args):

        # use feed to create final_dict and add it to the final big dictionary:
        if not trending:
            # find a feed for the name:
            feed = self.__query_ar_xiv(max_results, descending, trending)
            final_dict = self.__authors_data_nontrending(feed)
        else:
            # note: args have args[1]=name and args[2]=citation_dict
            name = args[0]
            citations_dict = args[1]
            feed = self.__query_ar_xiv(max_results, descending, trending, name)
            final_dict = self.__authors_data_trending(feed, name, citations_dict[name])

        self.__add_dict(self.__data_author, final_dict)

        # download and save:
        if pdf:
            # find names and idS:
            ids = self.__paper_ids(feed)

            ## list to downl checks what pdfs you already have on local drive
            ids = self.__list_to_downl(ids)
            urls = self.__pdf_links(feed, ids)

            if urls:
                self.__download_all_and_name(urls, ids)
            else:
                print("all requested pdf's already on local disc")

    def __list_to_downl(self, ids):
        try:
            ids_all = os.listdir()
            return list(filter(lambda x: not (x + '.pdf') in ids_all, ids))
        except FileNotFoundError:
            return []

    def __check_names_in_saved_dict(self, name_to_check, name_of_dict):
        try:
            temp_dict = self.__open_dict(name_of_dict)
            if temp_dict == None:
                return False
            else:
                print("the dictionary " + name_of_dict + " is saved on local drive")
                return name_to_check in list(map(lambda x: x[1], list(temp_dict.values())))
        except FileNotFoundError:
            return False

    # Public methods:
    ## args[0] = num or 'all', beware checking of upper bound not implemented yet
    def data(self, trending, num_downloads, pdf=False, *args):

        # scraping the trendsetters !!!!! add later the method for non-trendsetters
        if trending:
            rang_e = args[0]
            if len(self.__authors) == 0:
                self.__scrape_trendsetters()
                authors = self.__authors
            if type(rang_e) == int and rang_e > 0:
                authors = self.__authors[0:rang_e]
            elif rang_e == 'all':
                authors = self.__authors

        counter = 0

        if not trending:
            self.__produce_data(trending, num_downloads, True, pdf)
            self.__produce_data(trending, num_downloads, False, pdf)
        else:
            for nam in authors:
                was_in_already = self.__check_names_in_saved_dict(nam, self.__name_of_data + "_authors_data")
                print(nam)
                print(was_in_already)
                if not was_in_already:
                    self.__produce_data(trending, num_downloads, True, pdf, nam, self.__citations_dict)
                    self.__produce_data(trending, num_downloads, False, pdf, nam, self.__citations_dict)
                else:
                    print("skipping the name " + nam + " because found in saved dictionary ")

                if was_in_already & pdf:
                    self.__produce_data(trending, num_downloads, True, pdf, nam, self.__citations_dict)
                    self.__produce_data(trending, num_downloads, False, pdf, nam, self.__citations_dict)
                    print("possibly adding the Pdf for name " + nam + " because NOT found in saved pdf's ")
                counter += 1

        self.__save_dict(self.__data_author, self.__name_of_data + '_authors_data')

    def get_final_dict(self):
        return self.__data_author

class GetLatestData:

    __base_url = None
    __category = None
    __name_of_data = None
    __data_author = {}


    def __init__(self, cat, name_of_data, *args):
        self.__category = cat
        self.__initialize_settings()
        self.__name_of_data = name_of_data


    # Private methods:

    # for future purposes for handling different urls:
    def __initialize_settings(self):
        # initial settings, will be part of class variable:
        self.__base_url = 'http://export.arxiv.org/api/query?'
        feedparser._FeedParserMixin.namespaces['http://a9.com/-/spec/opensearch/1.1/'] = 'opensearch'
        feedparser._FeedParserMixin.namespaces['http://arxiv.org/schemas/atom'] = 'arxiv'

    ## Handling dictionary saving and opening
    def __open_dict(self, name):
        try:
            with open(name + '.pickle', 'rb') as handle:
                return pickle.load(handle)
        except FileNotFoundError:
            print("File was not found")

    def __save_dict(self, dictionary, name):
        with open(name + '.pickle', 'wb') as handle:
            pickle.dump(dictionary, handle)

    def __add_dict(self, final, temp):
        z = dict(final, **temp)
        final.update(z)

    # query arxiv for given authors name:
    def __query_ar_xiv(self, max_results):

        search_query = 'cat:' + self.__category
        sort_by = 'submittedDate'
        start = 0

        sortOrder = 'descending'

        query = 'search_query=%s&sort_by=%s&sortOrder=%s&start=%i&max_results=%i' % \
                (search_query, sort_by, sortOrder, start, max_results)

        # actual arXiv query
        response = urllib.request.urlopen(self.__base_url + query).read()

        # parse the response using feedparser
        feed = feedparser.parse(response)

        return feed

    def __download_file(self, download_url, name):
        response = urllib.request.urlopen(download_url)
        file = open(name + ".pdf", 'wb')
        file.write(response.read())
        file.close()
        print("Completed")



    def __pdf_links(self, feed, ids):
        pdf = []
        for entry in feed.entries:
            arx_id = '%s' % entry.id.split('/abs/')[-1]
            arx_id = re.sub('\S+/', '', arx_id)

            if arx_id in ids:
                for link in entry.links:
                    if link.type == 'application/pdf':
                        pdf.append(link.href)
        return pdf

        # ok authors data for non trendsetters, bit awkward, will change that later

    def __authors_data_nontrending(self, feed):
        # the first dictionary is the author dictionary:
        dict_a = {}

        for entry in feed.entries:
            ar_id = '%s' % entry.id.split('/abs/')[-1]
            ar_id = re.sub('\S+/', '', ar_id)
            title = '%s' % entry.title

            # Adding to a dictionary
            dict_a[ar_id] = []
            # dict_a[ar_id].append(name)
            # dict_a[ar_id].append(citation_density)
            dict_a[ar_id].append(title)

            # Extracting the abstract:
            summary = '%s' % entry.summary
            dict_a[ar_id].append(summary)

        return dict_a

        # this function extracts the id numbers of the papers in the feed
    def __paper_ids(self, feed):
        ids = []
        for entry in feed.entries:
            arx_id = '%s' % entry.id.split('/abs/')[-1]
            arx_id = re.sub('\S+/', '', arx_id)
            ids.append(arx_id)
        return ids

    def __download_all_and_name(self, urls, ids):
        ## if you download, then you necessarily save it to ids:
        for counter in iter(range(len(urls))):
            self.__download_file(urls[counter], ids[counter])

    # note: args have args[1]=name and args[2]=citation_dict

    def __produce_data(self, max_results, pdf, *args):

        # use feed to create final_dict and add it to the final big dictionary:
        # find a feed for the name:
        feed = self.__query_ar_xiv(max_results)
        final_dict = self.__authors_data_nontrending(feed)

        self.__add_dict(self.__data_author, final_dict)

        # download and save:
        if pdf:
            # find names and idS:
            ids = self.__paper_ids(feed)

            # list to downl checks what pdfs you already have on local drive
            ids = self.__list_to_downl(ids)
            urls = self.__pdf_links(feed, ids)

            if urls:
                self.__download_all_and_name(urls, ids)
            else:
                print("all requested pdf's already on local disc")

    def __list_to_downl(self, ids):
        try:
            ids_all = os.listdir()
            return list(filter(lambda x: not (x + '.pdf') in ids_all, ids))
        except FileNotFoundError:
            return []

    # Public methods

    # download latest 35 pdf files:
    def data(self, num_downloads = 55 ):

        self.__produce_data(num_downloads, True)
        self.__save_dict(self.__data_author, self.__name_of_data + '_new_candidates')

    def get_final_dict(self):
        return self.__data_author
