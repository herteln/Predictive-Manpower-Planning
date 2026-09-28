# Predictive Manpower Planning for Technical Support Using Machine Learning

**Author:** Nicole Hertel-Pirner

---

#### Executive Summary

This project explores how technical support ticket data can be used to support **predictive manpower planning** in a technical support department or call center.

Support organizations need to ensure that enough employees are available to handle incoming tickets while also considering the specific product knowledge and technical skills required to resolve them. However, ticket volumes and support requirements are not constant. They can vary depending on weekdays, products, product groups, priorities, new releases, seasonal events, and other factors.

The objective of this project is to analyze historical support ticket data and investigate how machine learning can help predict future support demand. These predictions can then be used as a basis for estimating how many employees may be required for upcoming shifts and which types of product knowledge or technical skills should be available.

The current project also implements a binary machine learning classification task using the target variable `Good_Solved`. This classification analysis provides a first step for understanding relationships between ticket characteristics, support structures, and ticket outcomes.

The long-term objective is to combine ticket-demand predictions with employee information, including skills, certifications, product knowledge, availability, and shift information, to support detailed workforce recommendations.

---

#### Rationale

Companies rely on many different workflows and business processes. In practice, there is often a difference between the expected process flow and the actual process flow. Understanding these differences provides opportunities to optimize processes, allocate resources more efficiently, and improve operational performance.

A process can be understood as a series of actions or steps that move from a defined or recognized starting point to a defined or recognized end point. Most modern business processes leave digital traces in transactional systems. These traces can be collected as event or operational data and analyzed to better understand how a process actually works.

Examples of such processes include:

- Supply chain processes
- Order processing
- Technical support processes
- Call center workflows
- Ticket handling and incident management

This project focuses specifically on the workflow of a **technical support department or call center**, where work is represented by tickets stored in a ticketing system.

For this type of organization, workforce planning is an important operational challenge. Managers need to determine not only how many employees should be available during a particular shift, but also what knowledge and skills are required.

The workload may change considerably over time. For example, the number and type of tickets may depend on the weekday, seasonal patterns, ticket priority, ticket source, product or product group, new releases, software updates, unexpected technical problems, and changes in customer demand.

Some days or shifts may therefore require more employees than others. In addition, having enough employees is not sufficient if the available employees do not have the required product knowledge or technical skills.

If the necessary resources are not available at the required time, tickets may remain unresolved for longer periods. This can potentially lead to service delays, SLA violations, reduced customer satisfaction, productivity losses, and business disruption.

**Predictive manpower planning** can help organizations prepare for expected workload instead of reacting only after demand has already occurred. Historical ticket data can be analyzed to identify patterns and used as input for machine learning models that estimate future support demand.

The long-term objective is to combine these predictions with employee information, including skills, certifications, product knowledge, availability, and shift information. This could support better resource allocation by answering not only **how many employees are needed**, but also **which types of employees are needed**.

---

#### Research Question

The project investigates the following main questions:

- How can historical technical support ticket data be used to estimate the workload for upcoming shifts, weeks, or months?
- How many employees may be required to handle the expected ticket volume?
- How does expected workload differ between products or product groups?
- What types of product knowledge or technical skills may be required based on the predicted ticket distribution?
- If a new product or release is introduced, how could the resulting additional workload influence manpower requirements?
- How can ticket-demand predictions eventually be combined with employee skills, certifications, and availability to support workforce planning?

The long-term goal is to develop a data-driven approach that connects:

**Ticket Demand → Required Workload → Required Knowledge → Available Employees → Shift Planning**

---

#### Data Sources

The project uses the **Technical Support Dataset** available on Kaggle:

**Source:**  
https://www.kaggle.com/datasets/suvroo/technical-support-dataset

The dataset contains 2,330 technical support ticket records and provides information that can be used to analyze patterns in support demand, ticket handling, and ticket outcomes.

The current dataset does not contain all information required for complete manpower planning. In particular, additional information about employees, employee availability, certifications, product knowledge, and technical skills would be required for a complete workforce planning solution.

---

#### Methodology

The project follows a structured machine learning approach to analyze technical support ticket data and investigate how historical ticket information can contribute to predictive manpower planning. As a first step toward this larger objective, the current machine learning implementation focuses on a **binary classification problem**: predicting whether a support ticket belongs to the `Good_Solved` class.

The analysis was implemented in Python using **pandas**, **NumPy**, **Matplotlib**, **Seaborn**, and **scikit-learn**.

##### 1. Data Preparation

The Technical Support Dataset contains 2,330 support ticket records with information about the ticket itself, the responsible support organization, SLA performance, product information, and other operational characteristics.

Several attributes were removed before model training, including:

- Ticket ID
- Longitude and Latitude
- Created time
- Expected SLA to resolve
- Expected SLA to first response
- First response time
- Close time
- Resolution time

The remaining categorical attributes were converted into numerical values using `LabelEncoder`. These include:

- Product group
- Support Level
- Priority
- Source
- Status
- Country
- Topic
- Agent Group
- Agent Name
- SLA For first response
- SLA For Resolution

Missing numerical values were replaced with the median value of the respective feature.

The target variable for the machine learning models is **`Good_Solved`**.

The dataset is almost evenly distributed between the two target classes:

- `Good_Solved = 1`: 1,173 tickets
- `Good_Solved = 0`: 1,157 tickets

##### 2. Train-Test Split and Feature Scaling

The dataset was divided into training and testing data using an **80/20 train-test split** with a `random_state` of 42.

The target variable `Good_Solved` was separated from the remaining features:

- **X:** ticket and support-related input features
- **y:** `Good_Solved`

The input features were standardized using `StandardScaler`. The scaler was fitted only on the training data and subsequently applied to the test data.

##### 3. Machine Learning Models

Five supervised machine learning classification algorithms were trained and compared:

1. **Logistic Regression**  
   Logistic Regression provides a baseline classification model and estimates the probability that a ticket belongs to the `Good_Solved` class.

2. **K-Nearest Neighbors (KNN)**  
   KNN classifies a ticket based on similar observations in the training data and can capture relationships that are not represented by a simple linear decision boundary.

3. **Decision Tree**  
   A Decision Tree creates decision rules based on the available ticket characteristics. It also allows feature importance to be analyzed.

4. **Gradient Boosting**  
   Gradient Boosting combines multiple weak decision models sequentially, with each new model attempting to improve errors made by previous models.

5. **Support Vector Machine (SVM)**  
   SVM attempts to identify a decision boundary that separates the two `Good_Solved` classes. Probability estimation was enabled so that AUC-ROC could be calculated.

##### 4. Model Evaluation

Each model was trained using the same training dataset and evaluated using the same test dataset.

The following performance metrics were calculated:

- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**
- **AUC-ROC**

ROC curves were generated for all five models to provide a visual comparison of their classification performance.

##### 5. Confusion Matrix Analysis

A confusion matrix was generated for each machine learning model to analyze:

- True Positives
- True Negatives
- False Positives
- False Negatives

This provides additional information about the types of classification errors made by each model.

##### 6. Feature Importance

Feature importance was analyzed for models that provide this information directly, particularly the **Decision Tree** and **Gradient Boosting** models.

This analysis helps identify which ticket and support-related characteristics have the strongest influence on the classification result.

##### 7. Connection to Predictive Manpower Planning

The `Good_Solved` classification represents one analytical component of the larger predictive manpower planning concept.

The envisioned process is:

**Historical Ticket Data**  
↓  
**Analyze Ticket Characteristics and Outcomes**  
↓  
**Predict Future Ticket Demand and Workload**  
↓  
**Identify Required Product Knowledge and Support Level**  
↓  
**Compare Requirements with Employee Skills and Availability**  
↓  
**Estimate Required Manpower**  
↓  
**Recommend Employees for Upcoming Shifts**

---

#### Results

The five classification models produced the following results on the test dataset:

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| K-Nearest Neighbors | 0.916 | 0.981 | 0.858 | 0.915 | 0.975 |
| Decision Tree | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Gradient Boosting | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| SVM | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

Logistic Regression, Decision Tree, Gradient Boosting, and SVM achieved perfect scores on the current test dataset. K-Nearest Neighbors also performed strongly, with an accuracy of approximately 91.6% and an AUC-ROC of approximately 0.975.

These results demonstrate that the `Good_Solved` target can be classified very accurately using the features included in the current dataset. However, the unusually high performance of several different algorithms means that the results need to be interpreted carefully.

Some variables are very strongly connected to the target variable. For example, `Status` has a particularly strong relationship with `Good_Solved`. In the current dataset, records with `Status = Closed` correspond to `Good_Solved = 1`, while other status values correspond to `Good_Solved = 0`.

`Survey results` also has a very strong relationship with the target. This indicates that some available features may contain information that becomes available only during or after the ticket-resolution process.

If the objective is to predict the outcome of a **new incoming ticket**, using such information can lead to **data leakage**, because the model would use information that would not yet be known at prediction time.

Therefore, the perfect classification results should not yet be interpreted as evidence that the models can predict future ticket outcomes with 100% accuracy.

A future modeling iteration should exclude features that are only available after ticket processing has started or finished. This would provide a more realistic prediction scenario and a more reliable assessment of model performance.

From the perspective of **Predictive Manpower Planning**, this distinction is especially important because the final system must make predictions before the future workload occurs.

---

#### Next Steps

1. **Remove Potential Data Leakage**  
   Create a second modeling experiment excluding information that would not be available when a new ticket is created.

2. **Improve Ticket-Demand Prediction**  
   Develop models specifically focused on predicting future ticket volume and workload.

3. **Extend Feature Engineering**  
   Create additional time-based, product-specific, priority-related, and workload-related features.

4. **Add Knowledge Information**  
   Define which knowledge, skills, and certifications are required for individual products, product groups, or support requests.

5. **Add Employee Information**  
   Integrate employee ID, product knowledge, technical skills, certifications, availability, working hours, and shift assignments.

6. **Connect Demand with Required Knowledge**  
   Map predicted ticket demand to the knowledge and skills required to process those tickets.

7. **Calculate Manpower Requirements**  
   Translate expected workload into the estimated number of employees required for a shift, day, week, or month.

8. **Develop Employee Recommendations**  
   Recommend suitable employees for upcoming shifts based on predicted workload, skills, knowledge, and availability.

9. **Analyze New Product and Release Scenarios**  
   Investigate how new products, software releases, or major updates may influence future ticket demand and workforce requirements.

---

#### Outline of Project

1. **Introduction and Background**  
   Overview of technical support processes and predictive manpower planning.

2. **Research Question and Objectives**  
   Definition of project goals and research questions.

3. **Data Overview**  
   Description of the Technical Support Dataset, available features, data quality, and limitations.

4. **Exploratory Data Analysis (EDA)**  
   Analysis of ticket distributions, workload patterns, products, priorities, sources, and other relevant characteristics.

5. **Feature Engineering and Data Preparation**  
   Preparation and transformation of features for machine learning.

6. **Modeling Approach**  
   Development and comparison of Logistic Regression, KNN, Decision Tree, Gradient Boosting, and SVM.

7. **Evaluation and Results**  
   Comparison of Accuracy, Precision, Recall, F1-Score, AUC-ROC, ROC curves, and confusion matrices.

8. **Predictive Manpower Planning**  
   Translation of ticket analysis and future workload predictions into potential workforce requirements.

9. **Conclusions and Recommendations**  
   Summary of findings, limitations, and possible operational applications.

10. **Next Steps and Future Work**  
    Extension of the solution with employee information, skills, certifications, availability, workload prediction, and shift planning.

---

#### Additional Work to Be Done

The current project provides a machine learning foundation for a broader predictive manpower planning solution.

Additional work includes:

1. Adding knowledge and certification information.
2. Adding employee information and employee availability.
3. Adding workload and average ticket-resolution effort.
4. Developing explicit future ticket-volume and workload prediction.
5. Removing features that could cause data leakage when predicting new tickets.
6. Connecting predicted demand with required skills and employee availability.
7. Extending the system from product and product-group requirements to employee-level recommendations.

The long-term workflow is:

**Historical Tickets**  
↓  
**Predict Future Ticket Demand**  
↓  
**Estimate Workload**  
↓  
**Identify Required Product Knowledge and Skills**  
↓  
**Compare with Employee Skills and Availability**  
↓  
**Determine Required Number of Employees**  
↓  
**Recommend Employees for Each Shift**

---

##### Contact and Further Information

For questions, suggestions, or further information about this project, please contact the author.

This project explores how **data analysis and machine learning can support predictive manpower planning in technical support environments**.
