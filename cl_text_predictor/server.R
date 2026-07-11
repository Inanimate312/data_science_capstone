library(shiny)

source("cl_model_functions.R")

model <- load_model()

function(input, output, session) {
  
  predictions <- eventReactive(
    input$submitText,
    {
      
      predict_next_word(
        history = input$user_text,
        top_n = 1,
        fivegram_lookup = model$fivegram_lookup,
        fourgram_lookup = model$fourgram_lookup,
        trigram_lookup = model$trigram_lookup,
        bigram_lookup = model$bigram_lookup,
        top_unigrams = model$top_unigrams
      )
      
    }
  )
  
  output$predictionTable <- renderTable({
    
    tibble(
      Rank = seq_along(predictions()),
      Prediction = predictions()
    )
    
  },
  rownames = FALSE)
  
}