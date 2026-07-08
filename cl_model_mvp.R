################################################################################
#                       English Text Prediction Model
################################################################################

library(dplyr)
library(purrr)
library(stringr)
library(tibble)
library(tidyr)
library(tokenizers)

################################################################################
# Download data
################################################################################

url <- "https://d396qusza40orc.cloudfront.net/dsscapstone/dataset/Coursera-SwiftKey.zip"
zipfile <- file.path(getwd(), "Coursera-SwiftKey.zip")

## Download and unzip file if not already downloaded
if(!file.exists(zipfile)) {
  download.file(url,
                destfile = zipfile)
  unzip(zipfile)
}

################################################################################
# Read English corpus
################################################################################

sample_lines <- function(path, n = 100000) {
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

english_files <- list.files(file.path(getwd(),"final/en_US"), full.names = TRUE)



corpus <- bind_rows(
  lapply(
    english_files,
    function(f) {
      
      tibble(
        file = basename(f),
        text = sample_lines(f, 100000)
      )
    }
  )
)

################################################################################
# Clean data
################################################################################
clean_text <- function(text) {
  text %>%
    str_to_lower() %>%
    str_replace_all("[[:punct:]]", " ") %>%
    str_replace_all("[0-9]+", " ") %>%
    str_replace_all("[\r\n]+", " ") %>%
    str_squish()
}

corpus <- corpus %>%
  mutate(text_clean = map_chr(text, clean_text))

################################################################################
# Tokenize and build unigrams
################################################################################

tokens_df <- corpus %>%
  mutate(
    tokens = map(
      text_clean,
      ~tokenize_words(.x)[[1]]
    )
  ) %>%
  unnest(tokens)

unigrams <- tokens_df %>%
  count(tokens, sort = TRUE, name = "freq") %>%
  rename(token = tokens)

################################################################################
# Build N-grams
################################################################################

build_ngram_freq <- function(text_vector, n) {
  
  tibble(
    ngram = unlist(
      tokenize_ngrams(
        text_vector,
        n = n,
        n_min = n
      )
    )
  ) %>%
    filter(
      !is.na(ngram),
      str_trim(ngram) != ""
    ) %>%
    count(ngram, sort = TRUE, name = "freq")
}


bigrams <- build_ngram_freq(corpus$text_clean, 2)

trigrams <- build_ngram_freq(corpus$text_clean, 2)

fourgrams <- build_ngram_freq(corpus$text_clean, 2)

fivegrams <- build_ngram_freq(corpus$text_clean, 2)

################################################################################
# Prune rare n-grams (freq < 2)
################################################################################

bigrams <- filter(bigrams, freq >= 2)
trigrams <- filter(trigrams, freq >= 2)
fourgrams <- filter(fourgrams, freq >= 2)
fivegrams <- filter(fivegrams, freq >= 2)

################################################################################
# Split n-grams
################################################################################

bigrams_split <- bigrams %>%
  tidyr::separate(
    ngram,
    into = c("w1", "w2"),
    sep = "\\s+",
    remove = FALSE,
    extra = "drop"
  )


trigrams_split <- trigrams %>%
  tidyr::separate(
    ngram,
    into = c("w1", "w2", "w3"),
    sep = "\\s+",
    remove = FALSE,
    extra = "drop"
  )

fourgrams_split <- fourgrams %>%
  tidyr::separate(
    ngram,
    into = c("w1", "w2", "w3", "w4"),
    sep = "\\s+",
    remove = FALSE,
    extra = "drop"
  )

fivegrams_split <- fivegrams %>%
  tidyr::separate(
    ngram,
    into = c("w1", "w2", "w3", "w4", "w5"),
    sep = "\\s+",
    remove = FALSE,
    extra = "drop"
  )


stopifnot(
  all(c("w1","w2","w3","w4","w5") %in% names(fivegrams_split))
)


################################################################################
# Set conditional probability calculations
################################################################################

bigram_model <- bigrams_split %>%
  group_by(w1) %>%
  mutate(prob = freq / sum(freq)) %>%
  ungroup()

trigram_model <- trigrams_split %>%
  group_by(w1, w2) %>%
  mutate(prob = freq / sum(freq)) %>%
  ungroup()

fourgram_model <- fourgrams_split %>%
  group_by(w1, w2, w3) %>%
  mutate(prob = freq / sum(freq)) %>%
  ungroup()

fivegram_model <- fivegrams_split %>%
  group_by(w1, w2, w3, w4) %>%
  mutate(prob = freq / sum(freq)) %>%
  ungroup()

################################################################################
# Create prediction function (top 3 predicted words)
################################################################################

predict_next_word <- function(
    history,
    top_n = 3,
    fivegram_model,
    fourgram_model,
    trigram_model,
    bigram_model,
    unigram_model
) {
  
  history <- clean_text(history)
  
  words <- str_split(history, "\\s+")[[1]]
  n <- length(words)
  
  candidates <- tibble(
    word = character(),
    score = numeric()
  )
  
  ####################################################################
  # 5-GRAM
  ####################################################################
  
  if (n >= 4) {
    
    tmp <- fivegram_model %>%
      filter(
        w1 == words[n - 3],
        w2 == words[n - 2],
        w3 == words[n - 1],
        w4 == words[n]
      ) %>%
      transmute(
        word = w5,
        score = prob * 1.0
      )
    
    candidates <- bind_rows(candidates, tmp)
  }
  
  ####################################################################
  # 4-GRAM
  ####################################################################
  
  if (n >= 3) {
    
    tmp <- fourgram_model %>%
      filter(
        w1 == words[n - 2],
        w2 == words[n - 1],
        w3 == words[n]
      ) %>%
      transmute(
        word = w4,
        score = prob * 0.4
      )
    
    candidates <- bind_rows(candidates, tmp)
  }
  
  ####################################################################
  # 3-GRAM
  ####################################################################
  
  if (n >= 2) {
    
    tmp <- trigram_model %>%
      filter(
        w1 == words[n - 1],
        w2 == words[n]
      ) %>%
      transmute(
        word = w3,
        score = prob * 0.16
      )
    
    candidates <- bind_rows(candidates, tmp)
  }
  
  ####################################################################
  # 2-GRAM
  ####################################################################
  
  if (n >= 1) {
    
    tmp <- bigram_model %>%
      filter(
        w1 == words[n]
      ) %>%
      transmute(
        word = w2,
        score = prob * 0.064
      )
    
    candidates <- bind_rows(candidates, tmp)
  }
  
  ####################################################################
  # UNIGRAM
  ####################################################################
  
  total_uni <- sum(unigram_model$freq)
  
  tmp <- unigram_model %>%
    arrange(desc(freq)) %>%
    slice_head(n = 500) %>%   # limit size
    transmute(
      word = token,
      score = (freq / total_uni) * 0.0256
    )
  
  candidates <- bind_rows(candidates, tmp)
  
  ####################################################################
  # COMBINE SCORES
  ####################################################################
  
  candidates %>%
    group_by(word) %>%
    summarise(
      score = sum(score),
      .groups = "drop"
    ) %>%
    arrange(desc(score)) %>%
    slice_head(n = top_n) %>%
    pull(word)
}