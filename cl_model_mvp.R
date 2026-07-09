################################################################################
#                  Coursera SwiftKey English Prediction Model
################################################################################

library(dplyr)
library(purrr)
library(stringr)
library(tibble)
library(tidyr)
library(tokenizers)

################################################################################
# Reproducibility
################################################################################

set.seed(1234)

################################################################################
# Download Data
################################################################################

url <- "https://d396qusza40orc.cloudfront.net/dsscapstone/dataset/Coursera-SwiftKey.zip"

zipfile <- file.path(getwd(), "Coursera-SwiftKey.zip")

if (!file.exists(zipfile)) {
  download.file(url, zipfile)
  unzip(zipfile)
}

################################################################################
# Sampling Function
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

################################################################################
# Read English Corpus
################################################################################

english_files <- list.files(
  file.path(getwd(), "final/en_US"),
  full.names = TRUE
)

corpus <- bind_rows(
  
  lapply(
    english_files,
    function(f) {
      
      tibble(
        text = sample_lines(f, 100000)
      )
      
    }
  )
  
)

################################################################################
# Cleaning
################################################################################

clean_text <- function(text) {
  
  text %>%
    str_to_lower() %>%
    str_replace_all("[0-9]+", " ") %>%
    str_replace_all("[^[:alpha:]' ]", " ") %>%
    str_replace_all("[\r\n]+", " ") %>%
    str_squish()
  
}

corpus <- corpus %>%
  mutate(
    text_clean = map_chr(text, clean_text)
  ) %>%
  select(text_clean)

################################################################################
# Unigrams
################################################################################

tokens_df <- corpus %>%
  mutate(
    tokens = map(
      text_clean,
      ~ tokenize_words(.x)[[1]]
    )
  ) %>%
  unnest(tokens)

unigrams <- tokens_df %>%
  count(tokens, sort = TRUE, name = "freq") %>%
  rename(token = tokens)

rm(tokens_df)
gc()

################################################################################
# N-Gram Builder
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
    count(
      ngram,
      sort = TRUE,
      name = "freq"
    )
}

################################################################################
# Build 2-5 Grams
################################################################################

bigrams   <- build_ngram_freq(corpus$text_clean, 2)
trigrams  <- build_ngram_freq(corpus$text_clean, 3)
fourgrams <- build_ngram_freq(corpus$text_clean, 4)
fivegrams <- build_ngram_freq(corpus$text_clean, 5)

################################################################################
# Frequency Pruning
################################################################################

bigrams   <- filter(bigrams,   freq >= 2)
trigrams  <- filter(trigrams,  freq >= 3)
fourgrams <- filter(fourgrams, freq >= 4)
fivegrams <- filter(fivegrams, freq >= 5)

################################################################################
# Split N-Grams
################################################################################

bigrams_split <- bigrams %>%
  separate(
    ngram,
    into = c("w1","w2"),
    sep = "\\s+",
    remove = TRUE
  )

trigrams_split <- trigrams %>%
  separate(
    ngram,
    into = c("w1","w2","w3"),
    sep = "\\s+",
    remove = TRUE
  )

fourgrams_split <- fourgrams %>%
  separate(
    ngram,
    into = c("w1","w2","w3","w4"),
    sep = "\\s+",
    remove = TRUE
  )

fivegrams_split <- fivegrams %>%
  separate(
    ngram,
    into = c("w1","w2","w3","w4","w5"),
    sep = "\\s+",
    remove = TRUE
  )

rm(bigrams, trigrams, fourgrams, fivegrams)
gc()

################################################################################
# Save Model
################################################################################

saveRDS(
  list(
    unigrams  = unigrams,
    bigrams   = bigrams_split,
    trigrams  = trigrams_split,
    fourgrams = fourgrams_split,
    fivegrams = fivegrams_split
  ),
  "swiftkey_model.rds"
)

################################################################################
# Prediction Function - Stupid Backoff
################################################################################

predict_next_word <- function(
    history,
    top_n = 5,
    fivegrams,
    fourgrams,
    trigrams,
    bigrams,
    unigrams
) {
  
  history <- clean_text(history)
  
  words <- str_split(history, "\\s+")[[1]]
  n <- length(words)
  
  # 5-gram
  if (n >= 4) {
    
    candidates <- fivegrams %>%
      filter(
        w1 == words[n-3],
        w2 == words[n-2],
        w3 == words[n-1],
        w4 == words[n]
      ) %>%
      arrange(desc(freq))
    
    if (nrow(candidates) > 0) {
      return(head(candidates$w5, top_n))
    }
  }
  
  # 4-gram
  if (n >= 3) {
    
    candidates <- fourgrams %>%
      filter(
        w1 == words[n-2],
        w2 == words[n-1],
        w3 == words[n]
      ) %>%
      arrange(desc(freq))
    
    if (nrow(candidates) > 0) {
      return(head(candidates$w4, top_n))
    }
  }
  
  # 3-gram
  if (n >= 2) {
    
    candidates <- trigrams %>%
      filter(
        w1 == words[n-1],
        w2 == words[n]
      ) %>%
      arrange(desc(freq))
    
    if (nrow(candidates) > 0) {
      return(head(candidates$w3, top_n))
    }
  }
  
  # 2-gram
  if (n >= 1) {
    
    candidates <- bigrams %>%
      filter(
        w1 == words[n]
      ) %>%
      arrange(desc(freq))
    
    if (nrow(candidates) > 0) {
      return(head(candidates$w2, top_n))
    }
  }
  
  # Unigram fallback
  unigrams %>%
    arrange(desc(freq)) %>%
    slice_head(n = top_n) %>%
    pull(token)
  
}

################################################################################
# Load Saved Model
################################################################################

model <- readRDS("swiftkey_model.rds")

################################################################################
# Examples
################################################################################

predict_next_word(
  history = "it would mean the",
  top_n = 5,
  fivegrams = model$fivegrams,
  fourgrams = model$fourgrams,
  trigrams = model$trigrams,
  bigrams = model$bigrams,
  unigrams = model$unigrams
)

predict_next_word(
  history = "The guy in front of me just bought a pound of bacon, a bouquet, and a case of",
  top_n = 5,
  fivegrams = model$fivegrams,
  fourgrams = model$fourgrams,
  trigrams = model$trigrams,
  bigrams = model$bigrams,
  unigrams = model$unigrams)

predict_next_word(
  history = "You're the reason why I smile everyday. Can you follow me please? It would mean the",
  top_n = 5,
  fivegrams = model$fivegrams,
  fourgrams = model$fourgrams,
  trigrams = model$trigrams,
  bigrams = model$bigrams,
  unigrams = model$unigrams)

predict_next_word(
  history = "Hey sunshine, can you follow me and make me the",
  top_n = 5,
  fivegrams = model$fivegrams,
  fourgrams = model$fourgrams,
  trigrams = model$trigrams,
  bigrams = model$bigrams,
  unigrams = model$unigrams)

predict_next_word(
  history = "Very early observations on the Bills game: Offense still struggling but the",
  top_n = 5,
  fivegrams = model$fivegrams,
  fourgrams = model$fourgrams,
  trigrams = model$trigrams,
  bigrams = model$bigrams,
  unigrams = model$unigrams)

predict_next_word(
  history = "Go on a romantic date at the",
  top_n = 5,
  fivegrams = model$fivegrams,
  fourgrams = model$fourgrams,
  trigrams = model$trigrams,
  bigrams = model$bigrams,
  unigrams = model$unigrams)

predict_next_word(
  history = "Well I'm pretty sure my granny has some old bagpipes in her garage I'll dust them off and be on my",
  top_n = 5,
  fivegrams = model$fivegrams,
  fourgrams = model$fourgrams,
  trigrams = model$trigrams,
  bigrams = model$bigrams,
  unigrams = model$unigrams)

predict_next_word(
  history = "Ohhhhh #PointBreak is on tomorrow. Love that film and haven't seen it in quite some",
  top_n = 5,
  fivegrams = model$fivegrams,
  fourgrams = model$fourgrams,
  trigrams = model$trigrams,
  bigrams = model$bigrams,
  unigrams = model$unigrams)

predict_next_word(
  history = "After the ice bucket challenge Louis will push his long wet hair out of his eyes with his little",
  top_n = 5,
  fivegrams = model$fivegrams,
  fourgrams = model$fourgrams,
  trigrams = model$trigrams,
  bigrams = model$bigrams,
  unigrams = model$unigrams)

predict_next_word(
  history = "Be grateful for the good times and keep the faith during the",
  top_n = 5,
  fivegrams = model$fivegrams,
  fourgrams = model$fourgrams,
  trigrams = model$trigrams,
  bigrams = model$bigrams,
  unigrams = model$unigrams)

predict_next_word(
  history = "If this isn't the cutest thing you've ever seen, then you must be",
  top_n = 5,
  fivegrams = model$fivegrams,
  fourgrams = model$fourgrams,
  trigrams = model$trigrams,
  bigrams = model$bigrams,
  unigrams = model$unigrams)