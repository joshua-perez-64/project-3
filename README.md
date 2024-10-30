# project-3

Your presentation should cover the following:
An executive summary or overview of the project and project goals. (5 points)
An overview of the data collection, cleanup, and exploration processes. (5 points)
The approach that your group took in achieving the project goals. (5 points)
Any additional questions that surfaced, what your group might research next if more time was available, or share a plan for future development. (3 points)
The results and conclusions of the application or analysis. (3 points)
Slides effectively demonstrate the project. (2 points)
Slides are visually clean and professional. (2 points)

Objective: Develop a predictive model  and create a user interface to screen patients and predict their risk of pre-diabetes/diabetes
Create a chatbot to educate patients at risk of diabetes on their possible health condition and what precautions they should take

The objective of this project was to create an AI driven user interface that first predicts user's risk of diabetes using a predictive AI model. Second, to create a chatbot that uses NLP to educate patients on diabetes and its treatment. As a result, first we developed a nueral network model that had a 74% accuracy and a 78% recall to predict diabetes risk. Second, we used a diabetes specific llamas NLP model from hugging faces to answer patients' questions about diabetes, and its treatement.

Initially, we started off by searching for datasets. We decided to expand on a data set used for an earlier project, 'Diabetes Health Risk Prediction'. The dataset was from the UC Irvine Machine Learning Repository called the [CDC Diabetes Health Indicators dataset](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators). This dataset can also be found on Kaggle and is part of the Behavioral Risk Factor Surveillance System, which is a health-related telephone survey conducted annually by the CDC. This dataset was chosen because it had approximately 215,000 instances, whereas other health datasets had less data.

The data was analysed, and we focused on which features were relevant for our model. In the previous project we removed the features "income" and "healthcare coverage" because we wanted to select features not strongly tied to financial outcomes. However, in our current dataset we decided to further include the following features: "income', 'healthcare coverage', and 'No doctor because of cost'. We added these features because we wanted to reduce economic bias towards low income households. Furthermore, a study by researcher Ishihara suggests that low income households have a higher risk for diabetes. Therefore, removing these economic factors affects the decision making power for the AI model towards the condition. 

Earlier criticism for our previous project was that the features that we chose were highly dependent on our AI model's weightage towards particular features. Therefore,for the current project we tested AI predictive models based on three categories. First, an AI predictive model based on 'AI chosen features'. This is where we picked features based on the model's preference towards particular features. Second, an AI predictive model based on 'doctor chosen features'. This is where we picked features based on domain expertise and research. Third, an AI predictive model based on 'All features'. This was picked to see how features combining AI chosen features, and domain expetise chosen features would perform. We found that a model with all features performed the best of all models. Furthermore, picking this model reduced the bias towards both an AI model, and human error. 

Earlier in our data we looked at the number of pre-diabetic/diabetic versus non-diabetic instances. Non-diabetic cases found were 218,334. Pre-diabetic/diabetic cases found were 35,346. This indicated a class imbalance between diseased and non-diseased individuals. We also observed that many of the features were already encoded into buckets, while BMI was the only feature with quantitative values. The remaining features were qualitative and encoded into numbers.

Learning from the previous project we used an undersampling technique to equalize the non diabetic and prediabetic/diabetic cases into 26,500 each. This is to reduce the models tendency to predict more no than yes. Not doing this would produce a very low recall rate, yet a very high accuracy rate. Recall is an important factor for us because it is important that our model prefers to predict more false positives rather than false negatives. This is because missing diabetes is more detrimental than falsly diagnosing someone with diabetes. Next, our BMI data was skewed with many outliers. We used a robust scaler because it is considered a better scaling method for skewed data. Doing these steps set us up to train our model.

After cleaning and exploring the data, we proceeded with model development. We selected the following models: logistic regression, and nueral network. We felt that the Logistic Regression might be a good fit because the outcome was a simple "yes" or "no" answer, determining whether the patient was at risk for diabetes. Most of our features were encoded, leading to binary "yes/no" answers. Additionally, since our data was simple and complete, we considered logistic regression a strong candidate for hypothesis testing. Nueral Network was not expected to be the better model because our data did not look complex. There wasn't any high dimensionality such as images, audio, or text in our data. The dataset also felt small since it was in the range of hundred thousands rather than the millions.

After building each model the accuracy and recall were analysed. The Logistic Regression model showed 73% accuracy with a 77% recall. The nueral network model showed a 74% accuarcy with a 78% recall. As discussed earlier, further nueral network models will built an categorised as 'AI chosen features', 'Doctor chosen features', and 'All features'. The 'AI chosen features' model showed 70% accuracy with a 77% recall. The 'Doctor chosen features' model showed a 70% accuracy with a 78% recall. The "All features' model showed a 74% accuracy with a 78% recall.

The nueral network model seemed to perform better than the logistic regression model. First, we set our nueral network using a sigmoid function which is more sensitive for binary decisions since we wanted a binary outcome. Second, our data even though encoded still had categorical values and nueral network performs better with categorical data. Also BMI was a feature that had outlier data and nueral networks can performed better to skewed, complex outlier data. It is important to note that the accuracy and recall had a difference of 1% in both models so this may not feel as much of a difference. However, these are possible reasons why our nueral network model may have performed bettter than we expected.

After creating the AI predictive model we proceeded to build the gradio application. Our gradio interface is made up of two parts which are a diabetic risk assesment tool, and an AI powered chatbot to educate patients on their condition.

First, we set up a diabetes risk assesment page. The purpose of this tab was to allow patients to input their features and predict the risk of diabetes early using a trained AI model. For the diabetic risk assessment tool our trained model was loaded into the code. Furthermore, we proceeded to input the features needed for the first tab of the interface. These features were based on a CDC survey that the model was trained on to predict diabetes. The CDC survery was used because it was a credible survey. The data is passed through a Numpy array and preprocessed for the machine learning model to read. Then the model in the application predicts whether you have diabetes or not.

Next, we set up the AI medical chatbot. The purpose of the AI medical chatbot is to answer any further question for patients who may have concerns about diabetes and how to treat it. We used a pre trained language model called 'm42-health/Llama3-Med42-8B'. This model was used because it was specifically trained to answer questions on diabetes. Llama felt like a credible language model to use, but the drawback was that it requires mass computing power. This causes the software to take tie when responding if there is not enough computing power.

Using gradio both pages were created and a tabbed interface function was used for easy access amongst both applications.

In conclusion, we started by making an AI predictive model for diabetes to help patients assess risk early. Our most successful model was the nueral network model using all features, using undersampling to balance the data between diabetics and non-diabetics, and using robust scaler to reduce outliers in the BMI feature.
This model showed the highest accuracy at 74%, and highest recall at 78%. We prioritized recall because, as a diabetes risk assessment tool, it is crucial not to miss diabetic cases. In other words, we preferred fewer **false negatives** (missing diabetic cases) over more **false positives** (incorrectly identifying non-diabetics as diabetics). This approach is more beneficial as it encourages patients to get checked, reducing the risk of missed diagnoses. We further created a gradio interface where patients can first predict their risk of diabetes then educate themselves on the condition, and treatment using a llamas pretrained model specific for diabetes.

For future improvements we would improve the dataset by collecting more diabetic cases which would help balance the data. Next, we would globaly expand the dataset to get a broader, more diverse sample for model training. We would further include additional features beyond our CDC dataset for a more holistic view of health risks. Lastly, we would add granular classification of our data into Type 1, Type 2, and pre-diabetic cases, to improve disease risk prediction. For improvement to our application we would expand our application design to be accessible to different types of users including additional languages, text to speech, and kiosk mode.

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


