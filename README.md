# CBNN Overview

This document summarizes the core components of a **Cell-Based Neural Network (CBNN)** 


## Cells (Modular Units)

A **Cell** is an independent processing module with its own parameters and learning rules.

Each Cell includes:
- An input → output transformation (CNN / MLP / Transformer block / etc.)
- A local learning rule (Hebbian, Oja, STDP, Predictive Coding, etc.)
- Optional internal memory or state
- A prototype vector representing the Cells specialisation

Cells behave like small, autonomous “agents.”


## Router (Gating Mechanism)

The Router decides **which Cells to activate** for each input.

Responsibilities:
- Compute routing scores (prototype similarity, small ANN router, or RL-based)
- Select **top-K** Cells
- Send input only to the chosen Cells
- Update routing parameters using local rules or small-gradient methods

**“Which Cells should work on this sample?”**



## Prototypes (Cell Identity Vectors)

Each Cell maintains a prototype vector encoding:
- Preferred input patterns  
- Specialization direction  
- Identity for routing

The Router uses input–prototype similarity to select Cells.  
Prototypes are updated with Hebbian/EMA-style rules.



## Local Learning Rules

CBNN does **not** rely on global backpropagation.

Each Cell updates its parameters independently using Hebbian learning


## Dynamic Rewiring (on my way)

A structural plasticity mechanism that:
- Detects low-usage or dead Cells
- Resets or reinitializes them
- Reassigns prototype vectors
- Encourages discovery of new specializations

Rewiring keeps the architecture adaptive and evolving.


## Classification Head (For Supervise Tasks)

A lightweight head that:
- Aggregates outputs of activated Cells  
- Produces class logits  
- Learns using Hebbian or local gradient rules

This preserves the “no global BP” principle.

## Usage Statistics & Rewards

The system maintains internal metrics:
- Cell usage frequency  
- Routing probabilities  
- Reward or prediction-error signals  

Used for:
- Router updates  
- Local RL  
- Rewiring decisions  

CBNN =  
**Cells** (local learners)  
+ **Router** (dynamic gating)  
+ **Prototypes** (identity vectors)  
+ **Local learning rules**  
+ **Dynamic rewiring**  
+ **Optional classifier head**  
+ **Usage/reward signals**

A CBNN learns without global backpropagation, self-organizes, and dynamically specialise
