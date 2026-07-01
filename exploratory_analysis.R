################################################################################
#   Data Science Capstone: Exploratory Analysis
################################################################################
library(R.utils)

## Download data
## Set URL and filepath
url <- "https://d396qusza40orc.cloudfront.net/dsscapstone/dataset/Coursera-SwiftKey.zip"
zipfile <- file.path(getwd(), "Coursera-SwiftKey.zip")

## Download and unzip file if not already downloaded
if(!file.exists(zipfile)) {
  download.file(url,
                destfile = zipfile)
  unzip(zipfile)
}

## Determine size of en_US.blogs.txt file
file.info("final/en_US/en_US.blogs.txt")$size / (1024^2)
# 200.4242 MB

## Count lines in en_US.twitter.txt file
countLines("final/en_US/en_US.twitter.txt")
# 2360148

## Find length of longest line in three en_US datasets
usblogs <- file.path(getwd(),"final/en_US/en_US.news.txt")
usnews <- file.path(getwd(),"final/en_US/en_US.blogs.txt")
ustwitter <- file.path(getwd(),"final/en_US/en_US.twitter.txt")

usfiles <- c(usblogs, usnews, ustwitter)

longest_lengths <- numeric(length(usfiles))
names(longest_lengths) <- usfiles

for (i in seq_along(usfiles)) {
  con <- file(usfiles[i], "r")
  max_len <- 0
  
  while (length(line <- readLines(con, n = 1, warn = FALSE)) > 0) {
    len <- nchar(line)
    if (len > max_len) {
      max_len <- len
    }
  }
  
  close(con)
  longest_lengths[i] <- max_len
}

longest_lengths
# 11384 in news, 40833 in blogs, 144 in twitter

## Divide # lines with "love" by # lines with "hate" in twitter
count_word <- function(path, word) {
  con <- file(path, "r")
  count <- 0
  
  while (length(line <- readLines(con, n = 1, warn = FALSE)) > 0) {
    if (grepl(word, line, ignore.case = FALSE)) {
      count <- count + 1
    }
  }
  
  close(con)
  count
}

count_love <- count_word(ustwitter, "love")
count_hate <- count_word(ustwitter, "hate")
love_to_hate_ratio <- count_love/count_hate
love_to_hate_ratio
# 4.10

## Return the twitter line containing the word biostats
get_lines_with_word <- function(path, word) {
  con <- file(path, "r")
  matches <- character()

  while (length(line <- readLines(con, n = 1, warn = FALSE)) > 0) {
    if (grepl(word, line, ignore.case = FALSE)) {
      matches <- c(matches, line)
    }
  }
  
  close(con)
  matches
}

get_lines_with_word(ustwitter, "biostats")
# "i know how you feel.. i have biostats on tuesday and i have yet to study =/"

## Count tweets with exact characters "A computer once beat me at chess, but it was no match for me at kickboxing"
count_phrase <- function(path, phrase) {
  con <- file(path, "r")
  count <- 0
  
  while (length(line <- readLines(con, n = 1, warn = FALSE)) > 0) {
    if (grepl(phrase, line, ignore.case = FALSE, fixed = TRUE)) {
      count <- count + 1
    }
  }
  
  close(con)
  count
}

count_phrase(ustwitter,"A computer once beat me at chess, but it was no match for me at kickboxing")
# 3