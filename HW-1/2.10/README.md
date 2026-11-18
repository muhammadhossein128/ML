# Concept Learning: FIND-S Algorithm Implementation & Empirical Analysis

**Author:** Seyyed Mohammad Hossein Mahjour  
**Course:** Graduate Machine Learning (Chapter 2: Concept Learning — Exercise 2.10)  
**University:** Shahid Bahonar University of Kerman (SBUK)
**Instructor:** Dr. Mahdi Eftekhari
**TA:** M. H. Shamsipoor

This repository contains a modular Python implementation of the **FIND-S** concept learning algorithm from Chapter 2 of Tom M. Mitchell's *Machine Learning* (Exercise 2.10).

## Project Structure

```text
├── find_s.py               # Core FindS class, hypothesis representation, and generality checks
├── experiment.py           # Random instance generator, simulation runner, statistics, and plotting
├── run_all.py              # Main entry point to reproduce the textbook trace, stats, and figures
├── finds_convergence.png   # Generated histogram of training examples required for convergence
├── requirements.txt        # Python package dependencies
├── .gitignore              # Git ignore rules for Python environments and cache files
└── README.md               # Project documentation