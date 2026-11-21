# Smart Waste Collection: Multi-Agent System

**Project:** Data Mining, 4th Year  
**Project Number:** 16  
**Topic:** Intelligent Management of Urban Waste Collection

---

## 1. Project Overview

### 1.1. Objective
The objective of this project is to optimize urban waste collection by dispatching garbage trucks only to bins that require collection.

The system makes collection decisions based on:
- Waste type.
- Current bin fill level.
- Estimated time remaining until the bin becomes full.

### 1.2. General Principle
The system consists of three independent agents coordinated through a central coordinator. Each agent performs a specific task and passes its output to the next stage of the system.

---

## 2. System Architecture

### 2.1. Agent Description

| Agent | Responsibility | Algorithm / Model | Input | Output |
|---|---|---|---|---|
| Vision Agent | Classifies waste according to its type | CNN (Transfer Learning) | Waste image | Class name + confidence |
| Prediction Agent | Estimates the time remaining until a bin is full | Linear Regression | Fill-level history | Hours until full |
| Routing Agent | Decides whether a bin should be collected | Decision Tree | Fill level, hours until full, waste type | collect now / collect later / skip |

### 2.2. Agent Workflow

1. **Vision Agent:** Receives a waste image and identifies its category.
2. **Prediction Agent:** Analyzes the bin's fill-level history and estimates how many hours remain until it becomes full.
3. **Routing Agent:** Uses the waste type, the current fill level and the hours until full to decide whether the bin should be collected now, later, or skipped.
4. **Coordinator:** Connects the three agents and manages the overall workflow.

### 2.3. Waste Categories

The system supports six waste categories:

- cardboard
- glass
- metal
- paper
- plastic
- trash

**Naming convention:** all waste class labels are written in lowercase.

---

## 3. Datasets

### 3.1. Garbage Classification Dataset
- **Source:** Kaggle
- **Purpose:** Training and evaluating the Vision Agent.
- **Classes:** Six waste categories.
- **Link:** [add link]

### 3.2. Smart Waste Bin Sensor Simulation Dataset
- **Source:** Kaggle
- **Purpose:** Training and evaluating the Prediction Agent and supporting the Routing Agent.
- **Link:** [add link]

### 3.3. Dataset Integration Strategy

The two datasets are independent and do not share a common structure or identifiers.

To connect them within the system:
1. Each simulated bin is assigned a waste category.
2. The Vision Agent predicts the waste category from an image.
3. The predicted category is associated with the corresponding simulated bin.
4. The resulting information is used by the subsequent agents.

**Limitation:** this association is simulated, not a real-world connection between the two datasets. It will be documented in the final report.

**Status:** this strategy is pending confirmation from the professor.

---

## 4. Repository Structure

```text
DM-Project/
├── visual_agent/        # CNN notebooks and code
├── prediction_agent/    # Linear Regression notebooks and code
├── routing_agent/       # Decision Tree notebooks and code
├── app/                 # Coordinator and demo app
├── data/                # Shared notes and small files
└── README.md
```

**Storage restriction:** images and trained model weights are stored in Google Drive and must not be committed to the repository.

---

## 5. Team Organization

Each team member is responsible for one agent.

| Team Member | Assigned Responsibility |
|---|---|
| Nafila DZIRI | Vision Agent (CNN) |
| Hadil | Prediction Agent (Linear Regression) |
| Samira HAMMOUCHE | Routing Agent (Decision Tree) |

The coordinator and the demo app are assigned to: [name, to be decided].

---

## 6. Development Rules and Conventions

### 6.1. Work Organization
- Each member works only in their own directory.
- Changes to another member's directory are coordinated with that member.

### 6.2. Git Workflow
- Run `git pull` before every `git push`.
- Write meaningful commit messages (for example: "Add ResNet baseline").

### 6.3. File Storage
- Images, model weights and other large files stay in Google Drive.

### 6.4. Reproducibility
- The random seed is **42** across the project.

---

## 7. Execution and Demonstration

### 7.1. Running the Project
Instructions for installing dependencies and running the application will be added once the demo is ready.

### 7.2. Demonstration
The final demonstration will show the three agents working together through the coordinator.

**Status:** to be completed.

---

## 8. Project Summary

The system combines three machine learning approaches to support urban waste management:

1. **CNN:** identifies waste categories from images.
2. **Linear Regression:** predicts the time remaining before a bin is full.
3. **Decision Tree:** determines the collection action.

A central coordinator connects the agents into one workflow for more efficient, data-driven collection decisions.
