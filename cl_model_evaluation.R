################################################################################
#                  Coursera SwiftKey English Prediction Model:
#                             Evaluate Model
################################################################################

library(dplyr)
library(stringr)
library(tokenizers)

################################################################################
# Load functions and model
################################################################################
source("cl_model_functions.R")

model <- load_model()

test_corpus <- readRDS("test_corpus.rds")

################################################################################
# Evaluate model size
################################################################################
cat(
  "Model size:",
  round(
    file.info("swiftkey_model.rds")$size / 1024^2,2
  ),
  "MB\n"
  )

print(object.size(model))

################################################################################
# Evaluate model performance
################################################################################
benchmark <- system.time({
  
  replicate(
    1000,
    predict_next_word(
      history = "it would mean the",
      top_n = 5,
      fivegram_lookup = model$fivegram_lookup,
      fourgram_lookup = model$fourgram_lookup,
      trigram_lookup = model$trigram_lookup,
      bigram_lookup = model$bigram_lookup,
      top_unigrams = model$top_unigrams
    )
  )
  
})

print(benchmark)

################################################################################
# Test edge cases
################################################################################
test_cases <- c(
  "",
  "the",
  "don't",
  "can't you",
  "asdfghjkl",
  "it would mean the"
)

for (case in test_cases) {
  
  cat("\nInput:", case, "\n")
  
  print(
    predict_next_word(
      history = case,
      top_n = 5,
      fivegram_lookup = model$fivegram_lookup,
      fourgram_lookup = model$fourgram_lookup,
      trigram_lookup = model$trigram_lookup,
      bigram_lookup = model$bigram_lookup,
      top_unigrams = model$top_unigrams      
    )
  )
  
}

################################################################################
# Hold-out accuracy testing
################################################################################
# Create evaluation function
evaluate_sentence <- function(sentence, model) {
  
  words <- tokenize_words(sentence)[[1]]
  
  if (length(words) < 2) {
    return(NULL)
  }
  
  history <- paste(
    words[-length(words)],
    collapse = " "
  )
  
  actual <- clean_prediction_text(
    words[length(words)]
  )
  
  predicted <- predict_next_word(
    history = history,
    top_n = 5,
    fivegram_lookup = model$fivegram_lookup,
    fourgram_lookup = model$fourgram_lookup,
    trigram_lookup = model$trigram_lookup,
    bigram_lookup = model$bigram_lookup,
    top_unigrams = model$top_unigrams
  )
  
  tibble(
    top1 = actual == predicted[1],
    top3 = actual %in%
      predicted[1:min(3, length(predicted))],
    top5 = actual %in% predicted
  )
  
}

test_sentences <- test_corpus$text

# Run across test data
results <- bind_rows(
  lapply(
    test_sentences,
    evaluate_sentence,
    model = model
  )
)

# Accuracy metrics
accuracy <- results %>%
  summarise(
    Top1 = round(100 * mean(top1, na.rm = TRUE), 2),
    Top3 = round(100 * mean(top3, na.rm = TRUE), 2),
    Top5 = round(100 * mean(top5, na.rm = TRUE), 2)
  )

print(accuracy)