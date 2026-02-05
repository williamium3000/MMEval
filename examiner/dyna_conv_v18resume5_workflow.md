# VLM Image Understanding Examiner Workflow

## High-Level Overview

```mermaid
flowchart LR
    A[Image Sample] --> B[Generate 5 Contexts<br/>Diverse scenarios]
    B --> C[For Each Context:<br/>Generate Questions]
    C --> D[Evaluate with VLM<br/>Test model responses]
    
    style B fill:#e1f5ff
    style C fill:#fff4e1
    style D fill:#f3e5f5
```

## Detailed Workflow

```mermaid
flowchart TD
    Start([Start: Image Sample]) --> GenContext[Generate 5 Diverse Contexts<br/>Create realistic scenarios from image<br/>First-person view, diverse goals]
    
    GenContext --> ContextLoop{For Each Context<br/>1-5}
    
    ContextLoop --> InitConv[Initialize Conversation<br/>Set up multi-turn dialogue<br/>Natural human-like questions]
    
    subgraph ConversationLoop["Conversation Loop (Repeats for Each Context)"]
        FirstRound{First Question?}
        
        FirstRound -->|Yes| RegularQ1[Ask Regular Question<br/>Direct question about image<br/>Natural conversational]
        
        FirstRound -->|No| Switch[Select Question Type<br/>Choose next question strategy<br/>Regular, Follow-up, Adversarial, Unanswerable]
        
        Switch -->|Regular| RegularQ[Ask Regular Question<br/>Direct question about image<br/>Avoid repetition]
        
        Switch -->|Follow-up| FollowUp[Ask Follow-up Question<br/>Test model confidence<br/>Challenge previous answer]
        
        Switch -->|Adversarial| Adversarial[Ask Adversarial Question<br/>Test if model invents objects<br/>Ask about plausible but absent items]
        
        Switch -->|Unanswerable| Unanswerable[Ask Unanswerable Question<br/>Test if model corrects false assumptions<br/>Create imaginary object/relation, ask trap question]
        
        Switch -->|End| EndConv[End Conversation]
        
        RegularQ1 --> EvalVLM[Evaluate with VLM<br/>Target model processes image and question]
        
        RegularQ --> EvalVLM
        FollowUp --> EvalVLM
        Adversarial --> EvalVLM
        Unanswerable --> EvalVLM
    end
    
    InitConv --> ConversationLoop
    
    style GenContext fill:#e1f5ff
    style ContextLoop fill:#e1f5ff
    style Switch fill:#fff4e1
    style RegularQ fill:#e8f5e9
    style RegularQ1 fill:#e8f5e9
    style FollowUp fill:#e8f5e9
    style Adversarial fill:#ffebee
    style Unanswerable fill:#fce4ec
    style EvalVLM fill:#f3e5f5
    style ConversationLoop fill:#f5f5f5,stroke:#999,stroke-width:2px
```

## Key Components

### 1. Context Generation
- Create 5 diverse realistic scenarios from the image
- First-person perspective ("I am seeing...")
- Each context focuses on different objects/relationships
- Each has a concrete goal requiring image understanding

### 2. Question Type Selection
- Strategically choose the next question to test the model
- Early in conversation: 4 question types available
- Later in conversation: 5 types (adds option to end)
- Selection based on conversation history and question diversity

### 3. Question Generation Types

#### Regular Questions
- Direct questions about visible image content
- Natural conversational style, relevant to context

#### Follow-up Questions
- Test model confidence and consistency
- Challenge previous answers, probe for details

#### Adversarial Questions
- Test if model invents objects that should be there but aren't
- Ask about plausible items that commonly appear with visible objects

#### Unanswerable Questions
- Test if model corrects false assumptions
- Process: Create imaginary object/relation, ask trap question assuming it exists
- Use definite references to trap model

### 4. VLM Evaluation
- Target vision-language model processes image and question
- Model generates response
