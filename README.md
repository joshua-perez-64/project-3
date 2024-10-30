# project-3

Your presentation should cover the following:
An executive summary or overview of the project and project goals. (5 points)
An overview of the data collection, cleanup, and exploration processes. (5 points)
The approach that your group took in achieving the project goals. (5 points)
Any additional questions that surfaced, what your group might research next if more time was available, or share a plan for future development. (3 points)
The results and conclusions of the application or analysis. (3 points)
Slides effectively demonstrate the project. (2 points)
Slides are visually clean and professional. (2 points)

# Diabetes Prediction and Patient Education AI

## Objective
Develop an AI-driven predictive model and user interface to:
1. Screen patients and predict their risk of pre-diabetes or diabetes.
2. Provide an educational chatbot to guide patients at risk of diabetes on health conditions and precautions.

---

## Project Summary
This project aims to achieve two primary goals:
1. **Risk Prediction:** Develop a predictive model with a neural network that achieved a 74% accuracy and 78% recall.
2. **Patient Education:** Create a diabetes-focused chatbot using the LLaMA LLM model from Hugging Face to answer questions about diabetes and its treatment.

---

## Data Collection
- **Dataset Selection:** Expanded on the UC Irvine Machine Learning Repository’s CDC Diabetes Health Indicators dataset, sourced from the Behavioral Risk Factor Surveillance System (BRFSS) by the CDC.
- **Dataset Size:** ~215,000 instances, with pre-diabetic/diabetic cases at 35,346 and non-diabetic cases at 218,334.
- **Feature Selection:** Added features such as "income", "healthcare coverage", and "No doctor because of cost" to address socioeconomic factors influencing diabetes risk.

---

## Model Development
### Feature Engineering
To address potential biases:
1. **AI-Chosen Features:** Features preferred by the model.
2. **Doctor-Chosen Features:** Features chosen by domain expertise.
3. **All Features:** Combined all relevant features to reduce bias.

### Class Imbalance Handling
- **Undersampling Technique:** Balanced pre-diabetic/diabetic and non-diabetic cases to 26,500 each, enhancing recall by reducing false negatives.

### Data Preprocessing
- **BMI Scaling:** Used robust scaling to handle outliers and improve model performance.

### Model Selection
- **Logistic Regression:** Baseline model, ideal for binary classification.
- **Neural Network:** Selected for categorical data with skewed values, achieving slightly higher performance than logistic regression.

---

## Model Results
| Model                     | Accuracy | Recall |
|---------------------------|----------|--------|
| Logistic Regression       | 73%      | 77%    |
| Neural Network            | 74%      | 78%    |
| Neural Network (AI features)      | 70%      | 77%    |
| Neural Network (Doctor features)  | 70%      | 78%    |
| Neural Network (All features)     | 74%      | 78%    |

### Insights
- **Neural Network Performance:** Outperformed logistic regression, likely due to its handling of categorical data and sensitivity to skewed features.

---

## User Interface Development with Gradio
1. **Diabetes Risk Assessment Tool:**
   - Allows patients to input features and receive a diabetes risk prediction.
   - Uses the trained neural network model on CDC survey-based features.

2. **AI Medical Chatbot:**
   - Provides answers to patient questions on diabetes and treatment.
   - Integrated with the `m42-health/Llama3-Med42-8B` model, trained specifically for medical queries.

---

## Conclusion
Our neural network model using all features achieved the highest accuracy (74%) and recall (78%). Prioritizing recall helps minimize missed diagnoses, making it a reliable early screening tool. The integrated Gradio application allows patients to assess their risk and educate themselves on diabetes, reducing the likelihood of undiagnosed cases.

---

## Future Improvements
1. **Dataset Expansion:** Increase diabetic cases and diversity in the dataset for more robust training.
2. **Feature Addition:** Include more health indicators for comprehensive risk assessment.
3. **Application Enhancements:**
   - Expand accessibility (multilingual support, text-to-speech, kiosk mode).
   - Implement granular classification of diabetes types for tailored risk prediction.

# Citations
    Teboul, Alex. “Diabetes Health Indicators Dataset.” Kaggle, 8 Nov. 2021, www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset.

    “About the Special Diabetes Program - NIDDK.” National Institute of Diabetes and Digestive and Kidney Diseases, U.S. Department of Health and Human Services, www.niddk.nih.gov/about-niddk/research-areas/diabetes/type-1-diabetes-special-statutory-funding-program/about-special-diabetes-program. Accessed 11 Sept. 2024. 

    “National Diabetes Statistics Report.” Centers for Disease Control and Prevention, Centers for Disease Control and Prevention, www.cdc.gov/diabetes/php/data-research/index.html. Accessed 11 Sept. 2024.

    Ceriello, Antonio, and Francesco Prattichizzo. “Variability of Risk Factors and Diabetes Complications.” Cardiovascular Diabetology, U.S. National Library of Medicine, pubmed.ncbi.nlm.nih.gov/33962641/. Accessed 9 Sept. 2024. 

    “Lesson 4: Diabetes Risk Factors (English).” Edited by Care New England, YouTube, YouTube, 20 Oct. 2020, www.youtube.com/watch?v=rrX2Hn2iesM. 

    “Type 2 Diabetes.” Mayo Clinic, Mayo Foundation for Medical Education and Research, 14 Mar. 2023, www.mayoclinic.org/diseases-conditions/type-2-diabetes/symptoms-causes/syc-20351193. 

    “Diabetes Risk Factors.” Centers for Disease Control and Prevention, Centers for Disease Control and Prevention, www.cdc.gov/diabetes/risk-factors/index.html. Accessed 9 Sept. 2024.

    ND, Wong ND, and Sattar N. “Cardiovascular Risk in Diabetes Mellitus: Epidemiology, Assessment and Prevention.” Nature Reviews. Cardiology, U.S. National Library of Medicine, pubmed.ncbi.nlm.nih.gov/37193856/. Accessed 9 Sept. 2024.

    “Accuracy vs. Precision vs. Recall in Machine Learning: What’s the Difference?” Evidently AI - Open-Source ML Monitoring and Observability, www.evidentlyai.com/classification-metrics/accuracy-precision-recall. Accessed 9 Sept. 2024. 

    R;, Ishihara R;Babazono A;Liu N;Yamao. “Impact of Income and Industry on New-Onset Diabetes among Employees: A Retrospective Cohort Study.” International Journal of Environmental Research and Public Health, U.S. National Library of Medicine, pubmed.ncbi.nlm.nih.gov/35162114/. Accessed 30 Oct. 2024. 

    @misc{med42v2, Author = {Cl{\'e}ment Christophe and Praveen K Kanithi and Tathagata Raha and Shadab Khan and Marco AF Pimentel}, Title = {Med42-v2: A Suite of Clinical LLMs}, Year = {2024}, Eprint = {arXiv:2408.06142}, url={https://arxiv.org/abs/2408.06142}, }

    joshua-perez-64, et al. “Joshua-Perez-64/Project-2.” GitHub, github.com/joshua-perez-64/project-2. Accessed 30 Oct. 2024. 
