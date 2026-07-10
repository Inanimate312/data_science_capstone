################################################################################
#                  Coursera SwiftKey English Prediction Model:
#                           Build Prediction Model
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

all_corpus <- bind_rows(
  
  lapply(
    english_files,
    function(f) {
      
      tibble(
        text = sample_lines(f, 500000)
      )
      
    }
  )
  
)

train_idx <- sample(
  seq_len(nrow(all_corpus)),
  size = floor(0.8 * nrow(all_corpus))
)

train_corpus <- all_corpus[train_idx, ]
test_corpus <- all_corpus[-train_idx, ]

corpus <- train_corpus

################################################################################
# Save hold-out test corpus
################################################################################
saveRDS(
  test_corpus,
  "test_corpus.rds"
)

rm(
  test_corpus,
  train_corpus,
  all_corpus,
  train_idx
)
gc()

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