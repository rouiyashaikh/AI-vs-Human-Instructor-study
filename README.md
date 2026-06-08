# AI vs Human Instructor: Comparative Study Analysis

## Overview

This repository contains the Python implementation used to reproduce the statistical analysis presented in the MSc dissertation:

**"The Role of Artificial Intelligence in Enhancing Personalised Learning in Higher Education: A Comparative Study of AI and Human Instructors"**

**Author:** Rouiya Mirza Shaikh
**Programme:** MSc Human-Centred Artificial Intelligence
**University:** Bournemouth University, U.K.
**Supervisor:** Dr. Mohammed Naiseh

The project investigates student perceptions of AI-powered instructors compared with human instructors across multiple educational scenarios, including:

* Lecture Delivery
* Answering Questions
* Providing Feedback
* Assignment Marking

The study evaluates perceptions using six dimensions:

* Initial Thoughts
* Effectiveness
* Likelihood of Acceptance
* Engagement
* Competency
* Accuracy

---

## Research Objective

The aim of this study is to explore the potential of Generative AI and AI Avatar Instructors in higher education and compare their effectiveness with traditional human instructors.

Key objectives include:

* Evaluating AI-powered personalised learning support
* Assessing AI effectiveness in answering student questions
* Investigating AI-generated feedback capabilities
* Comparing AI and human instructor performance
* Providing recommendations for AI integration in higher education

---

## Methodology

The original research employed a comparative experimental design in which participants evaluated both AI and human instructors across multiple educational scenarios.

### Statistical Methods

The analysis pipeline includes:

1. Descriptive Statistics

   * Mean
   * Standard Deviation

2. Assumption Testing

   * Shapiro-Wilk Normality Test
   * Levene's Homogeneity of Variance Test

3. Inferential Statistics

   * Independent Samples t-test
   * Mann-Whitney U Test

4. Data Pre-processing

   * Label Encoding
   * Likert Scale Processing

---

## Dataset

The repository does **not** contain participant survey data.

Instead, the script generates a simulated dataset using the published means and standard deviations reported in the dissertation.

Simulation characteristics:

* 15 participants
* 4 educational scenarios
* 6 evaluation questions per scenario
* 7-point Likert scale responses

The generated dataset reproduces the statistical properties of the original study for demonstration and reproducibility purposes.

---

## Repository Structure

```text
.
├── ai_vs_human_instructor_analysis.py
├── figure1_all_scenarios_comparison.png
├── README.md
└── Project_Report_Masters.pdf
```

---

## Features

### Data Simulation

Generates synthetic Likert-scale responses based on published study results.

### Statistical Analysis

Performs:

* Descriptive statistics
* Normality testing
* Variance testing
* Independent t-tests
* Mann-Whitney U tests

### Visualisation

Creates publication-style figures comparing:

* AI Instructor
* Human Instructor

across all scenarios and evaluation measures.

### Reproducibility

Uses a fixed random seed to ensure reproducible results.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/ai-vs-human-instructor-analysis.git

cd ai-vs-human-instructor-analysis
```

Install dependencies:

```bash
pip install numpy pandas matplotlib seaborn scipy
```

---

## Usage

Run the analysis script:

```bash
python ai_vs_human_instructor_analysis.py
```

The script will:

1. Generate the simulated dataset
2. Compute descriptive statistics
3. Run assumption tests
4. Perform statistical comparisons
5. Generate visualisations
6. Save figures to disk

---

## Key Findings

The original study found that:

* Human instructors generally received higher ratings than AI instructors.
* Significant differences were observed primarily in:

  * Engagement during lecture delivery
  * Confidence in question answering
  * Effectiveness in answering questions
* AI instructors performed competitively in:

  * Feedback provision
  * Assignment marking
  * Information accuracy

These findings suggest that AI can effectively support educational activities but human instructors remain preferred for engagement, emotional connection, and interpersonal interaction.

---

## Technologies Used

* Python
* NumPy
* Pandas
* SciPy
* Matplotlib
* Seaborn
* Google Colab

---

## Academic Citation

If you use this repository in academic work, please cite:

```text
Shaikh, R. M. (2024).

The Role of Artificial Intelligence in Enhancing Personalised Learning
in Higher Education: A Comparative Study of AI and Human Instructors.

MSc Human-Centred Artificial Intelligence,
Bournemouth University.
```

---

## Disclaimer

This repository contains simulated data generated from published summary statistics and does not include any participant-level data collected during the research.

The code is intended for educational, reproducibility, and research demonstration purposes.

---

## License

This project is released under the MIT License.

Feel free to use, modify, and distribute the code with appropriate attribution.

---
