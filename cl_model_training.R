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

sample_lines <- function(path, n) {
  
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
        text = sample_lines(f, 200000)
      )
      
    }
  )
  
)

################################################################################
# Cleaning
################################################################################

clean_training_text <- function(text) {
  
  cleaned <- text %>%
    str_to_lower() %>%
    str_replace_all("[0-9]+", " ") %>%
    str_replace_all("[^[:alpha:]' ]", " ") %>%
    str_replace_all("[\r\n]+", " ") %>%
    str_squish()
  
  paste(
    "STARTTOKEN",
    cleaned,
    "ENDTOKEN"
  )
  
}

clean_prediction_text <- function(text) {
  
  text %>%
    str_to_lower() %>%
    str_replace_all("[0-9]+", " ") %>%
    str_replace_all("[^[:alpha:]' ]", " ") %>%
    str_replace_all("[\r\n]+", " ") %>%
    str_squish()
  
}

corpus <- corpus %>%
  mutate(
    text_clean = map_chr(
      text,
      clean_training_text
    )
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

unigrams <- unigrams %>%
  filter(
    !str_to_lower(token) %in% c(
      "starttoken",
      "endtoken"
    )
  )

top_unigrams <- unigrams$token

rm(unigrams)
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
# Build 2-5 Grams, run frequency pruning, and split N-grams
################################################################################

bigrams   <- build_ngram_freq(corpus$text_clean, 2)
bigrams   <- filter(bigrams,   freq >= 2)
bigrams_split <- bigrams %>%
  separate(
    ngram,
    into = c("w1","w2"),
    sep = "\\s+",
    remove = TRUE
  )

rm(bigrams)
gc()

trigrams  <- build_ngram_freq(corpus$text_clean, 3)
trigrams  <- filter(trigrams,  freq >= 2)
trigrams_split <- trigrams %>%
  separate(
    ngram,
    into = c("w1","w2","w3"),
    sep = "\\s+",
    remove = TRUE
  )

rm(trigrams)
gc()

fourgrams <- build_ngram_freq(corpus$text_clean, 4)
fourgrams <- filter(fourgrams, freq >= 2)
fourgrams_split <- fourgrams %>%
  separate(
    ngram,
    into = c("w1","w2","w3","w4"),
    sep = "\\s+",
    remove = TRUE
  )

rm(fourgrams)
gc()

fivegrams <- build_ngram_freq(corpus$text_clean, 5)
fivegrams <- filter(fivegrams, freq >= 2)
fivegrams_split <- fivegrams %>%
  separate(
    ngram,
    into = c("w1","w2","w3","w4","w5"),
    sep = "\\s+",
    remove = TRUE
  )

rm(fivegrams)
rm(corpus)
gc()

################################################################################
# Top-K Continuation Pruning
################################################################################

trigrams_split <- trigrams_split %>%
  group_by(w1, w2) %>%
  slice_max(
    freq,
    n = 15,
    with_ties = FALSE
  ) %>%
  ungroup()

fourgrams_split <- fourgrams_split %>%
  group_by(w1, w2, w3) %>%
  slice_max(
    freq,
    n = 10,
    with_ties = FALSE
  ) %>%
  ungroup()

fivegrams_split <- fivegrams_split %>%
  group_by(w1, w2, w3, w4) %>%
  slice_max(
    freq,
    n = 10,
    with_ties = FALSE
  ) %>%
  ungroup()
################################################################################
# Lookup Tables
################################################################################

bigram_lookup <- split(
  bigrams_split,
  bigrams_split$w1
)

rm(bigrams_split)
gc()

trigram_lookup <- split(
  trigrams_split,
  interaction(
    trigrams_split$w1,
    trigrams_split$w2,
    sep = "\r",
    drop = TRUE
  )
)

rm(trigrams_split)
gc()

fourgram_lookup <- split(
  fourgrams_split,
  interaction(
    fourgrams_split$w1,
    fourgrams_split$w2,
    fourgrams_split$w3,
    sep = "\r",
    drop = TRUE
  )
)

rm(fourgrams_split)
gc()

fivegram_lookup <- split(
  fivegrams_split,
  interaction(
    fivegrams_split$w1,
    fivegrams_split$w2,
    fivegrams_split$w3,
    fivegrams_split$w4,
    sep = "\r",
    drop = TRUE
  )
)

rm(fivegrams_split)
gc()

################################################################################
# Save Model
################################################################################

saveRDS(
  list(
    top_unigrams    = top_unigrams,
    bigram_lookup   = bigram_lookup,
    trigram_lookup  = trigram_lookup,
    fourgram_lookup = fourgram_lookup,
    fivegram_lookup = fivegram_lookup
  ),
  "swiftkey_model.rds"
)

rm(bigram_lookup,
   trigram_lookup,
   fourgram_lookup,
   fivegram_lookup,
   top_unigrams)
gc()

################################################################################
# Prediction Function - Stupid Backoff
################################################################################

predict_next_word <- function(
    history,
    top_n = 5,
    fivegram_lookup,
    fourgram_lookup,
    trigram_lookup,
    bigram_lookup,
    top_unigrams
) {
  
  history <- clean_prediction_text(history)
  
  if (history == "") {
    
    return(
      head(top_unigrams, top_n)
    )
    
  }
  
  words <- str_split(history, "\\s+")[[1]]
  n <- length(words)
  
  predictions <- character(0)
  
  add_candidates <- function(existing, new_words, top_n) {
    
    if (length(new_words) == 0) {
      return(existing)
    }
    
    new_words <- setdiff(
      new_words,
      existing
    )
    
    head(
      c(existing, new_words),
      top_n
    )
    
  }
  
  # ---------------------------------------------------------------------------
  # 5-gram
  # ---------------------------------------------------------------------------
  
  if (n >= 4) {
    
    key <- paste(
      words[n - 3],
      words[n - 2],
      words[n - 1],
      words[n],
      sep = "\r"
    )
    
    entry <- fivegram_lookup[[key]]
    
    candidates <- if (is.null(entry)) {
      character(0)
    } else {
      entry$w5
    }
    
    predictions <- add_candidates(
      predictions,
      candidates,
      top_n
    )
  }
  
  # ---------------------------------------------------------------------------
  # 4-gram
  # ---------------------------------------------------------------------------
  
  if (length(predictions) < top_n && n >= 3) {
    
    key <- paste(
      words[n - 2],
      words[n - 1],
      words[n],
      sep = "\r"
    )
    
    entry <- fourgram_lookup[[key]]
    
    candidates <- if (is.null(entry)) {
      character(0)
    } else {
      entry$w4
    }
    
    predictions <- add_candidates(
      predictions,
      candidates,
      top_n
    )
  }
  
  # ---------------------------------------------------------------------------
  # 3-gram
  # ---------------------------------------------------------------------------
  
  if (length(predictions) < top_n && n >= 2) {
    
    key <- paste(
      words[n - 1],
      words[n],
      sep = "\r"
    )
    
    entry <- trigram_lookup[[key]]
    
    candidates <- if (is.null(entry)) {
      character(0)
    } else {
      entry$w3
    }
    
    predictions <- add_candidates(
      predictions,
      candidates,
      top_n
    )
  }
  
  # ---------------------------------------------------------------------------
  # 2-gram
  # ---------------------------------------------------------------------------
  
  if (length(predictions) < top_n && n >= 1) {
    
    entry <- bigram_lookup[[words[n]]]
    
    candidates <- if (is.null(entry)) {
      character(0)
    } else {
      entry$w2
    }
    
    predictions <- add_candidates(
      predictions,
      candidates,
      top_n
    )
  }
  
  # ---------------------------------------------------------------------------
  # Unigram fallback
  # ---------------------------------------------------------------------------
  
  if (length(predictions) < top_n) {
    
    candidates <- top_unigrams
    
    predictions <- add_candidates(
      predictions,
      candidates,
      top_n
    )
  }
  
  predictions
  
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
  fivegram_lookup = model$fivegram_lookup,
  fourgram_lookup = model$fourgram_lookup,
  trigram_lookup = model$trigram_lookup,
  bigram_lookup = model$bigram_lookup,
  top_unigrams = model$top_unigrams)

predict_next_word(
  history = "The guy in front of me just bought a pound of bacon, a bouquet, and a case of",
  top_n = 5,
  fivegram_lookup = model$fivegram_lookup,
  fourgram_lookup = model$fourgram_lookup,
  trigram_lookup = model$trigram_lookup,
  bigram_lookup = model$bigram_lookup,
  top_unigrams = model$top_unigrams)

predict_next_word(
  history = "You're the reason why I smile everyday. Can you follow me please? It would mean the",
  top_n = 5,
  fivegram_lookup = model$fivegram_lookup,
  fourgram_lookup = model$fourgram_lookup,
  trigram_lookup = model$trigram_lookup,
  bigram_lookup = model$bigram_lookup,
  top_unigrams = model$top_unigrams)

predict_next_word(
  history = "Hey sunshine, can you follow me and make me the",
  top_n = 5,
  fivegram_lookup = model$fivegram_lookup,
  fourgram_lookup = model$fourgram_lookup,
  trigram_lookup = model$trigram_lookup,
  bigram_lookup = model$bigram_lookup,
  top_unigrams = model$top_unigrams)

predict_next_word(
  history = "Very early observations on the Bills game: Offense still struggling but the",
  top_n = 5,
  fivegram_lookup = model$fivegram_lookup,
  fourgram_lookup = model$fourgram_lookup,
  trigram_lookup = model$trigram_lookup,
  bigram_lookup = model$bigram_lookup,
  top_unigrams = model$top_unigrams)

predict_next_word(
  history = "Go on a romantic date at the",
  top_n = 5,
  fivegram_lookup = model$fivegram_lookup,
  fourgram_lookup = model$fourgram_lookup,
  trigram_lookup = model$trigram_lookup,
  bigram_lookup = model$bigram_lookup,
  top_unigrams = model$top_unigrams)

predict_next_word(
  history = "Well I'm pretty sure my granny has some old bagpipes in her garage I'll dust them off and be on my",
  top_n = 5,
  fivegram_lookup = model$fivegram_lookup,
  fourgram_lookup = model$fourgram_lookup,
  trigram_lookup = model$trigram_lookup,
  bigram_lookup = model$bigram_lookup,
  top_unigrams = model$top_unigrams)

predict_next_word(
  history = "Ohhhhh #PointBreak is on tomorrow. Love that film and haven't seen it in quite some",
  top_n = 5,
  fivegram_lookup = model$fivegram_lookup,
  fourgram_lookup = model$fourgram_lookup,
  trigram_lookup = model$trigram_lookup,
  bigram_lookup = model$bigram_lookup,
  top_unigrams = model$top_unigrams)

predict_next_word(
  history = "After the ice bucket challenge Louis will push his long wet hair out of his eyes with his little",
  top_n = 5,
  fivegram_lookup = model$fivegram_lookup,
  fourgram_lookup = model$fourgram_lookup,
  trigram_lookup = model$trigram_lookup,
  bigram_lookup = model$bigram_lookup,
  top_unigrams = model$top_unigrams)

predict_next_word(
  history = "Be grateful for the good times and keep the faith during the",
  top_n = 5,
  fivegram_lookup = model$fivegram_lookup,
  fourgram_lookup = model$fourgram_lookup,
  trigram_lookup = model$trigram_lookup,
  bigram_lookup = model$bigram_lookup,
  top_unigrams = model$top_unigrams)

predict_next_word(
  history = "If this isn't the cutest thing you've ever seen, then you must be",
  top_n = 5,
  fivegram_lookup = model$fivegram_lookup,
  fourgram_lookup = model$fourgram_lookup,
  trigram_lookup = model$trigram_lookup,
  bigram_lookup = model$bigram_lookup,
  top_unigrams = model$top_unigrams)