# Workflow Diagram and Demonstration

## 1. System Workflow
The following diagram shows how the user's input moves through the PhishGuard system to produce a response:

```mermaid
graph TD
    A[User inputs a suspicious message in Terminal] --> B(Tokenization and URL Extraction)
    B --> C{AI Processing}
    C -->|Multinomial Naive Bayes| D[Calculate Probability]
    C -->|Rule-Based Layer| E[Check for Urgency and Credentials]
    D --> F[Final Decision and Explanation]
    E --> F
    F --> G{Is Confidence < 65% or Rules Conflict?}
    G -->|Yes| H[FLAG FOR HUMAN REVIEW]
    G -->|No| I[Display Verdict and Token Explanations]
