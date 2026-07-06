################################################################################
#   Data Science Capstone: Milestone Report
################################################################################

## Data Processing
library(R.utils)
library(dplyr)
library(purrr)
library(stringr)
library(tibble)
library(tidyr)

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

## Build Corpus - Sampling 5,000 lines per file
# Define sampling function
sample_lines <- function(path, n = 5000) {
  lines <- readLines(
    path,
    warn = FALSE,
    encoding = "UTF-8",
    skipNul = TRUE
  )
  
  lines <- iconv(
    lines,
    from = "UTF-8",
    to = "UTF-8",
    sub = ""
  )
  
  lines <- lines[nzchar(lines)]
  
  sample(lines, min(n, length(lines)))
}



paths <- list(
  english = list.files(file.path(getwd(),"final/en_US"), full.names = TRUE),
  german = list.files(file.path(getwd(),"final/de_DE"), full.names = TRUE),
  finnish = list.files(file.path(getwd(), "final/fi_FI"), full.names = TRUE),
  russian = list.files(file.path(getwd(), "final/ru_RU"), full.names = TRUE)
)

# Build corpus using sampling function
# Define function to extract sample of lines from each file and load them into
# a tibble with language | file | text(sampled)
read_language_sample <- function(files, language) {
  map_df(files, ~{
    tibble(
      language = language,
      file = basename(.x),
      text = paste(sample_lines(.x), collapse = " ")
    )
  })
}

corpus <- map2_df(paths, names(paths), read_language_sample)

# Clean data to remove newline characters, collapse multiple spaces into one space,
# and convert all text to lowercase, to get ready for tokenization
corpus <- corpus %>%
  mutate(text_clean = text %>%
           str_replace_all("[\r\n]+", " ") %>%
           str_squish() %>%
           str_to_lower()
         )

## Tokenization and N-gram Construction
library(tokenizers)

# Tokenize text into individual words
tokens_df <- corpus %>%
  mutate(tokens = map(text_clean,
                      ~ tokenize_words(.x, strip_punct = TRUE)[[1]])) %>%
  unnest(tokens) %>%
  rename(token = tokens)


# Build n-grams from sampled text
build_ngrams <- function(text, n = 2) {
  tokenizers::tokenize_ngrams(text, n = n, n_min = n)[[1]]
}

ngrams_df <- corpus %>%
  mutate(
    bigrams = map(text_clean, build_ngrams, n = 2),
    trigrams = map(text_clean, build_ngrams, n = 3)
  )
################################################################################
## Summary statistics
# Unigram frequencies
unigrams <- tokens_df %>%
  group_by(language, token) %>%
  summarise(freq = n(), .groups = "drop")
  
# Bigram frequencies
bigrams <- ngrams_df %>%
  select(language, file, bigrams) %>%
  unnest(bigrams, keep_empty = TRUE) %>%
  rename(ngram = bigrams) %>%
  group_by(language, ngram) %>%
  summarise(freq = n(), .groups = "drop")

# Trigram frequencies
trigrams <- ngrams_df %>%
  select(language, file, trigrams) %>%
  unnest(trigrams, keep_empty = TRUE) %>%
  rename(ngram = trigrams) %>%
  group_by(language, ngram) %>%
  summarise(freq = n(), .groups = "drop")

# Plot frequencies
library(ggplot2)

unigrams %>%
  group_by(language) %>%
  arrange(desc(freq)) %>%
  mutate(rank = row_number()) %>%
  ggplot(aes(rank, freq)) +
  geom_line() +
  scale_y_log10() +
  scale_x_log10() +
  facet_wrap(~ language, scales = "free") +
  labs(title = "Zipf-like Word Frequency Distribution")

## Coverage analysis
coverage_stats <- unigrams %>%
  group_by(language) %>%
  arrange(desc(freq)) %>%
  mutate(
    cum_freq = cumsum(freq),
    total_freq = sum(freq),
    cum_prop = cum_freq / total_freq
  ) %>%
  summarise(
    vocab_50 = min(which(cum_prop >= 0.5)),
    vocab_90 = min(which(cum_prop >= 0.9)),
    total_vocab = n()
  )
coverage_stats

# Identify Foreign Words
is_cyrillic <- function(x) grepl("[\u0400-\u04FF]", x)
is_latin <- function(x) grepl("[A-Za-z]", x)

language_foreign_stats <- unigrams %>%
  mutate(
    cyrillic = is_cyrillic(token),
    latin = is_latin(token)
  ) %>%
  group_by(language) %>%
  summarise(
    total_tokens = sum(freq),
    foreign_like = sum(freq[cyrillic & language != "russian"]) +
                  sum(freq[latin & language == "russian"]),
    foreign_prop = foreign_like / total_tokens
  )
################################################################################

## Model construction - not needed for milestone report but note main plans/goals
# Split n-grams into component words
split_ngram <- function(df, n) {
  df %>%
    mutate(words = str_split(ngram, " ")) %>%
    mutate(
      w1 = map_chr(words, 1),
      w2 = if (n >= 2) map_chr(words, 2) else NA_character_,
      w3 = if (n >= 3) map_chr(words, 3) else NA_character_
    ) %>%
    select(-words)
}

bigrams_split <- split_ngram(bigrams, 2)
trigrams_split <- split_ngram(trigrams, 3)

# Compute conditional probabilities for bigrams and trigrams
bigram_model <- bigrams_split %>%
  group_by(language, w1) %>%
  mutate(prob = freq / sum(freq)) %>%
  ungroup()

trigram_model <- trigrams_split %>%
  group_by(language, w1, w2) %>%
  mutate(prob = freq / sum(freq)) %>%
  ungroup()

## Implement Backoff Prediction Model
# Try trigram, back off to bigram if trigram unavailable, else use most frequent unigrams

predict_next_word <- function(history, language = "english",
                              trigram_model, bigram_model, unigram_model,
                              top_n = 5) {
  words <- str_split(history, " ")[[1]]
  n <- length(words)
  
  # Try trigram
  if (n >= 2) {
    w1 <- words[n - 1]
    w2 <- words[n]
    cand_tri <- trigram_model %>%
      filter(language == !!language, w1 == !!w1, w2 == !!w2) %>%
      arrange(desc(prob)) %>%
      head(top_n)
    if (nrow(cand_tri) > 0) return(cand_tri$w3)
  }
  
  # Back off to bigram
  w_last <- words[n]
  cand_bi <- bigram_model %>%
    filter(language == !!language, w1 == !!w_last) %>%
    arrange(desc(prob)) %>%
    head(top_n)
  if (nrow(cand_bi) > 0) return(cand_bi$w2)
  
  # Back off to unigram
  cand_uni <- unigram_model %>%
    filter(language == !!language) %>%
    arrange(desc(freq)) %>%
    head(top_n)
  cand_uni$token
}

# Apply Laplace smoothing to give unseen n-grams non-zero probability
V <- unigrams %>% filter(language == "english") %>% nrow()

bigram_model_smoothed <- bigrams_split %>%
  group_by(language, w1) %>%
  mutate(prob_laplace = (freq + 1) / (sum(freq) + V)) %>%
  ungroup()

# Evaluate model using log-likelihood and perplexity on test sentences
compute_loglik <- function(sentence, language, trigram_model, 
                           bigram_model, unigram_model) {
  words <- str_split(sentence, " ")[[1]]
  loglik <- 0
  for (i in 3:length(words)) {
    prob <- 1e-8 # default small probability
    tri <- trigram_model %>% filter(language == !!language,
                                    w1 == words[i - 2],
                                    w2 == words[i - 1],
                                    w3 == words[i])
    if (nrow(tri) > 0) prob <- tri$prob[1]
    loglik <- loglik + log(prob)
  }
  loglik
}

# Memory and performance profiling
# Check memory usage and profile the prediction function to identify bottlenecks
object.size(trigram_model)
gc()

Rprof("predict_profile.out")
for (i in 1:1000) {
  predict_next_word("this is", "english", trigram_model, bigram_model, unigrams)
}
Rprof(NULL)