################################################################################
#                  Coursera SwiftKey English Prediction Model:
#                           Prediction Model Functions
################################################################################

library(dplyr)
library(purrr)
library(stringr)
library(tibble)
library(tidyr)
library(tokenizers)

################################################################################
# Load Model
################################################################################
load_model <- function(path = "swiftkey_model.rds") {
  readRDS(path)
}

################################################################################
# Clean Prediction Text
################################################################################
clean_prediction_text <- function(text) {

  text %>%
    str_to_lower() %>%
    str_replace_all("[0-9]+", " ") %>%
    str_replace_all("[^[:alpha:]' ]", " ") %>%
    str_replace_all("[\r\n]+", " ") %>%
    str_squish()

}

################################################################################
# Predict Next Word
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