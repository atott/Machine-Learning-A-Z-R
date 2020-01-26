# Natural Language Processing
library(hunspell)
# Importing the dataset
dataset_original = read.delim('Restaurant_Reviews.tsv', quote = '', stringsAsFactors = FALSE)


spellcheck <- function(x){
t = hunspell(x)
  if(identical(t[[1]], character(0))){
    checked=x
  } else{
    for(i in 1:length(unlist(t))){
      p= hunspell_suggest(t[[1]])[[i]][1]
      checked =gsub(t[[1]][i], p, x)
    }
  }
return(checked)
}

tt <- lapply(dataset_original$Review, spellcheck)
dataset_original <-cbind(dataset_original, data.frame("checked_reviews" = do.call(rbind,tt)))

# Cleaning the texts
# install.packages('tm')
# install.packages('SnowballC')
library(tm)
library(SnowballC)
corpus = VCorpus(VectorSource(dataset_original$Review))
corpus = tm_map(corpus, content_transformer(tolower))
corpus = tm_map(corpus, removeNumbers)
corpus = tm_map(corpus, removePunctuation)
corpus = tm_map(corpus, removeWords, stopwords())
corpus = tm_map(corpus, stemDocument)
corpus = tm_map(corpus, stripWhitespace)

lapply(corpus, function(x){print(x)})
# Creating the Bag of Words model
dtm = DocumentTermMatrix(corpus)
dtm = removeSparseTerms(dtm, 0.999)
dataset = as.data.frame(as.matrix(dtm))
dataset$Liked = dataset_original$Liked

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Liked, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Fitting Random Forest Classification to the Training set
# install.packages('randomForest')
library(randomForest)
classifier = randomForest(x = training_set[-692],
                          y = training_set$Liked,
                          ntree = 100)

# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set[-692])
y_pred = ifelse(y_pred>.5,1,0)

# Making the Confusion Matrix
cm = table(test_set[, 692], y_pred)




#User Testing for individual reviews
###########################################################################################################################
review = 'the food tasted amazing'

review.corpus = VCorpus(VectorSource(review))
review.corpus = tm_map(review.corpus, content_transformer(tolower))
review.corpus = tm_map(review.corpus, removeNumbers)
review.corpus = tm_map(review.corpus, removePunctuation)
review.corpus = tm_map(review.corpus, removeWords, stopwords())
review.corpus = tm_map(review.corpus, stemDocument)
review.corpus = tm_map(review.corpus, stripWhitespace)

review.dtm<-DocumentTermMatrix(review.corpus,
                               control = list(
                                 dictionary=Terms(dtm)
                               ))

review.mat = as.data.frame(as.matrix(review.dtm))

result =  predict(classifier, review.mat)

