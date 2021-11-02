#read dataset
setwd("D:/Users/Acer User/Documents")
sent <- read.csv("tweet_text.csv")

#Create new variables sentiment(positive,negative,neutral)
pos.words <- read.csv("positive.csv")
neg.words <- read.csv("negative.csv")

pos.words <- scan("positive.csv",what='character')
neg.words <- scan("negative.csv",what='character')


score.sentiment = function(sentences, pos.words, neg.words, .progress='none')
{
  require(plyr)
  require(stringr)
  
  # we got a vector of sentences. plyr will handle a list
  # or a vector as an "l" for us
  # we want a simple array ("a") of scores back, so we use 
  # "l" + "a" + "ply" = "laply":
  
  scores = laply(sentences, function(sentence, pos.words, neg.words) {
    
    # clean up sentences with R's regex-driven global substitute, gsub():
    sentence = gsub('[[:punct:]]', '', sentence)
    sentence = gsub('[[:cntrl:]]', '', sentence)
    sentence = gsub('\\d+', '', sentence)
    # and convert to lower case:
    sentence = tolower(sentence)
    
    # split into words. str_split is in the stringr package
    word.list = str_split(sentence, '\\s+')
    # sometimes a list() is one level of hierarchy too much
    words = unlist(word.list)
    
    # compare our words to the dictionaries of positive & negative terms
    pos.matches = match(words, pos.words)
    neg.matches = match(words, neg.words)
    
    # match() returns the position of the matched term or NA
    # we just want a TRUE/FALSE:
    pos.matches = !is.na(pos.matches)
    neg.matches = !is.na(neg.matches)
    
    # and conveniently enough, TRUE/FALSE will be treated as 1/0 by sum():
    score = sum(pos.matches) - sum(neg.matches)
    
    return(score)
  }, pos.words, neg.words, .progress=.progress )
  
  scores.df = data.frame(score=scores, text=sentences)
  return(scores.df)
}

result <- score.sentiment(sent$Text,pos.words,neg.words)
summary(result$score)

hist(result$score,col='yellow', amin="score of tweets",
     ylab = "Count of tweets")
count(result$score)


write.csv(result,"D:/Users/Acer User/Documents/result.csv", row.names = FALSE)

#####################################################################################
###################################MODELLING#########################################
#####################################################################################

library(caTools)
setwd("D:/Users/Acer User/Documents")

#read data
all <- read.csv("result.csv")

#Remove duplication
distinct(all,.keep_all=TRUE)

library(data.table)
data.table(all)

#####################################################################################
##############################       Naive Bayes   ##################################
#####################################################################################


library(tidyverse)
library(ggplot2)
library(caret)
library(caretEnsemble)
library(rpart)
library(randomForest)
library(e1071)
library(psych)
library(tm)
library(dplyr)
library(wordcloud)
library(gmodels)
library(RTextTools)

all$sentiment <-factor(all$sentiment)
str(all$sentiment)

#FORM CORPUS
all_corpus <- VCorpus(VectorSource(all$text))
length(all_corpus) %>%
  sample(replace=FALSE) %>%
  sort.list(decreasing=FALSE) %>%
  head(2) %>%
  all_corpus[.] %>%
  inspect()

#TEXT CLEANING
all_corpus_clean <- all_corpus %>%
  tm_map(content_transformer(tolower)) %>%
  tm_map(removeNumbers) %>%
  tm_map(removeWords,stopwords()) %>%
  tm_map(stemDocument) %>%
  tm_map(stripWhitespace)
all_dtm <- DocumentTermMatrix(all_corpus_clean)

#SPLIT TRAIN TEST (TRAIN=70%, TEST=30%)
all_dtm_train <- all_dtm[1:3854,]
all_dtm_test <- all_dtm[3855:5507,]
all_train_r <- all[1:3854,]$sentiment
all_test_r <- all[3855:5507,]$sentiment

#TRAIN TEST TABLE
#train table
all_train_r %>%
  table %>%
  prop.table
#test table
all_test_r %>%
  table %>%
  prop.table

#CREATE INDICATOR FEATURES
all_freq_train <- all_dtm_train %>%
  findFreqTerms(5) %>%
  all_dtm_train[ , .]
all_freq_test <- all_dtm_test %>%
  findFreqTerms(5) %>%
  all_dtm_test[ , .]
convert_counts <- function(x) {
  x <- ifelse(x>0,"Yes","No")
}

all_train <- all_freq_train %>%
  apply(MARGIN=2,convert_counts)
all_test <- all_freq_test %>%
  apply(MARGIN=2,convert_counts)

#MODELLING, EVALUATION
all_classifier <- naiveBayes(all_train,all_train_r)
all_pred <- predict(all_classifier,all_test)
CrossTable(all_pred,all_test_r,
           prop.chisq=FALSE,chisq=FALSE,
           dnn=c("Predicted","Actual"))
conf.matrix <- confusionMatrix(all_pred,all_test_r)
conf.matrix



######################################################################################
###########################  DECISION TREE & RANDOM FOREST    #######################
#####################################################################################
library(readr)
library(dplyr)
library(tm)

all <- read.csv("result.csv")
all$sentiment = as.factor(all$sentiment)
corpus=Corpus(VectorSource(all$text))

corpus=tm_map(corpus,tolower)
corpus=tm_map(corpus,removePunctuation)
corpus=tm_map(corpus,removeWords,c("airbnb",stopwords("english")))
corpus=tm_map(corpus,stemDocument)

frequencies=DocumentTermMatrix(corpus)
sparse=removeSparseTerms(frequencies,0.995)
tsparse=as.data.frame(as.matrix(sparse))
colnames(tsparse)=make.names(colnames(tsparse))
tsparse$sentiment = all$sentiment
prop.table(table(tsparse$sentiment))

library(caTools)
set.seed(1234)
split=sample.split(tsparse$sentiment,SplitRatio = 0.7)
train=subset(tsparse,split==TRUE)
test=subset(tsparse,split==FALSE)

#Decison Tree
library(rpart)
library(rpart.plot)
tweetCART = rpart(sentiment ~ ., data=train,method="class")
predictCART=predict(tweetCART,newdata=test,type="class")
table(test$sentiment,predictCART)
library(caret)
confusionMatrix(test$sentiment,predictCART)

#Baseline Accuracy
table(test$sentiment)

library(randomForest)


#Random Forest
tweetRF=randomForest(sentiment ~ ., data=train)
#Make prediction
predictRF=predict(tweetRF, newdata=test)
confusionMatrix(test$sentiment,predictRF)
























