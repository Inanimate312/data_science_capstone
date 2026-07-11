library(shiny)
library(shinycssloaders)

source("cl_model_functions.R")

fluidPage(
  
  titlePanel("Coursera/SwiftKey Text Predictor"),
  
  sidebarLayout(
    
    sidebarPanel(
      
      helpText(
        "This app is a next-word English text prediction tool built from news,
        blog, and Twitter data. Use the text box to enter your text,
        either a single word or multiple words, then press Submit to predict
        the next word. The app only works with English-language words.
        
        Note, the app may take some time to load when first starting."
      ),
      
      tags$hr(),
      
      textInput(
        inputId = "user_text",
        label = "Enter text:",
        value = ""
      ),
      
      actionButton(
        "submitText",
        "Submit"
      )
      
    ),
    
    mainPanel(
      
      h3("Predicted Words"),
      
      withSpinner(
        tableOutput("predictionTable"),
        type=4
      )
      
    )
    
  )
  
)