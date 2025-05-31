library(shiny)
library(tidyverse)
library(caret)
library(pROC)

ui <- fluidPage(
  titlePanel("Loan Approval Prediction with Logistic Regression"),

  sidebarLayout(
    sidebarPanel(
      fileInput("file", "Upload Loan Data CSV", accept = ".csv"),
      checkboxGroupInput("features", "Select Features", choices = NULL),
      actionButton("trainModel", "Train Model")
    ),

    mainPanel(
      tabsetPanel(
        tabPanel("Data Summary", tableOutput("dataSummary")),
        tabPanel("Model Summary", verbatimTextOutput("modelSummary")),
        tabPanel("Confusion Matrix", verbatimTextOutput("confMatrix")),
        tabPanel("ROC Curve", plotOutput("rocPlot")),
        tabPanel("Feature Importance", plotOutput("featurePlot"))
      )
    )
  )
)

server <- function(input, output, session) {
  data <- reactive({
    req(input$file)
    df <- read.csv(input$file$datapath)

    # Preprocessing
    df <- df %>%
      mutate(
        Gender = replace_na(Gender, "Male"),
        Married = replace_na(Married, "Yes"),
        Dependents = replace_na(Dependents, "0"),
        Self_Employed = replace_na(Self_Employed, "No"),
        Credit_History = replace_na(Credit_History, 1),
        LoanAmount = replace_na(LoanAmount, median(LoanAmount, na.rm = TRUE)),
        Loan_Amount_Term = replace_na(Loan_Amount_Term, 360),
        TotalIncome = ApplicantIncome + CoapplicantIncome,
        EMI = LoanAmount / Loan_Amount_Term,
        BalanceIncome = TotalIncome - (EMI * 1000)
      ) %>%
      mutate(across(where(is.character), as.factor)) %>%
      select(-Loan_ID)

    return(df)
  })

  observe({
    updateCheckboxGroupInput(session, "features", choices = names(data())[!names(data()) %in% c("Loan_Status")])
  })

  model <- eventReactive(input$trainModel, {
    df <- data()
    form <- as.formula(paste("Loan_Status ~", paste(input$features, collapse = "+")))
    glm(form, data = df, family = "binomial")
  })

  output$dataSummary <- renderTable({
    head(data())
  })

  output$modelSummary <- renderPrint({
    summary(model())
  })

  output$confMatrix <- renderPrint({
    df <- data()
    prob <- predict(model(), newdata = df, type = "response")
    pred <- ifelse(prob > 0.5, "Y", "N")
    confusionMatrix(as.factor(pred), df$Loan_Status)
  })

  output$rocPlot <- renderPlot({
    df <- data()
    prob <- predict(model(), newdata = df, type = "response")
    roc_obj <- roc(df$Loan_Status, prob)
    plot(roc_obj, col = "blue", main = paste("ROC Curve (AUC =", round(auc(roc_obj), 2), ")"))
  })

  output$featurePlot <- renderPlot({
    coef_df <- as.data.frame(summary(model())$coefficients)
    coef_df$Feature <- rownames(coef_df)
    coef_df <- coef_df %>% filter(Feature != "(Intercept)")
    ggplot(coef_df, aes(x = reorder(Feature, abs(Estimate)), y = Estimate)) +
      geom_col(fill = "darkgreen") +
      coord_flip() +
      theme_minimal() +
      labs(title = "Feature Importance", x = "Feature", y = "Estimate")
  })
}

shinyApp(ui = ui, server = server)
