# 35: Intro and setup

## Setup

Had some issues with the setup, but got it working in the end.
Reinstalling everything works wonders.

## General introduction about machine learning, data science, and artificial intelligence

### What is data science?

Data science is a field that combines mathematics, especially statistics, computer science (programming), machine learning with domain knowledge to resolve problems or guide with decision making [^IBM].

"There’s a joke that says a data scientist is someone who knows more statistics than a
computer scientist and more computer science than a statistician. " [^Grus]

### What is the difference between data scientist, data analyst, and data engineer?

**Data analyst**
Data analysts job is to collect, organize, analyze and with visualizations present the data to help with for example decision making [^Geeks].

**Data scientist**
Data scientist job is to have expertice about the mathemacal and algorithms side to be able to choose and develope the models that can be used to make for example predictions [^Geeks].

**Data engineer**
Data engineers job is to build the infrastructure that is needed to collect and store the data efficiently and in more automated manner [^dsc].

Of course all of these roles have overlapping responsibilities and the definitions can vary depending on the organization where the person is working. All of these roles are needed in the data science processes and they can work together to achieve the best results.

### What is machine learning & artificial intelligence?

Artificial intelligence (AI) encompasses a wide range of technologies designed to replicate human intelligence to some extent. A subset of AI is machine learning (ML), which uses statistics to allows systems to learn from data and improve autonomously, without the need for explicit programming.[^GOOGLE][^Geron]

### Why do we do machine learning?

Summary from [^Geron] and my own thoughts:
Machine learning is a 'tool' used to be able to accomplish tasks that are time consuming, too difficult or impossible to do with traditional programming or by manual labor. Sometimes it's used to automate tasks that are repetitive or to make predictions and decisions based on the data that is available. It can also be used to connect multiple domain areas to be able to make better decisions. In general machine learning is used to make life easier and faster for us humans.

### What are the main categories of machine learning?

According to [^databricks] there is three main categories of machine learning: supervised learning, unsupervised learning, and reinforcement learning.

- **Supervised learning** is used when the data is labeled and the model is trained to predict the labels of the data. Image classification is an example of supervised learning.

- **Unsupervised learning** is used when the data is not labeled and the model is trained to find patterns in the data. For example clustering like Uniform Manifold Approximation and Projection (UMAP)[^UMAP] or Principal Component Analysis (PCA) [^PCA] are forms of unsupervised learning.

- **Reinforcement learning** is used when the model is trained to make decisions based on the data and the feedback that it gets from the environment. For example training a robot to play chess is an example of reinforcement learning [^MIT].

Depending on the source there can be more categories, but these are the main ones that are used in many places.

### Main workflow of machine learning

Summary of the steps from[^GOOGLE2] and my own thoughts and knowledge:

- <small style="font-weight: normal;">**Defining the question**</small>
    <small style="font-weight: normal;">What is the problem that needs to be solved, what kind of data (numerical, text, categorical etc.) is needed to obtain the solution? Is machine learnin the best tool for the defined problem? Can you in the end measure the success or accuracy of the model? Meaning do you have enough data and is the data quality good.</small>
- <small style="font-weight: normal;">**Data collection and preprocessing**</small>
    <small style="font-weight: normal;">Collect the data that is needed. Looking at the data which is the most important part of this step. To be able to make any kind of judgement about the data you need to understand it. Visualizations are a good way to understand the data and this can also reveal if there is any missing data or outliers.
    Preprocess means joining the data from multiple sources, cleaning and making the data uniform. This can include normalizing the data, different types of scaling, removing outliers, and filling in missing data. Ultimately making the data into a format that is ready to be inputted into the algorithm.</small>
- <small style="font-weight: normal;">**Model selection and training**</small>
    <small style="font-weight: normal;">Select the model that is best suited for the problem that was defined in the first step. Training the model is done with the data that was collected and preprocessed. Most often done by splitting the data into training and testing data to avoid overfitting. This step can also include hyperparameter tuning to get the best results.</small>
- <small style="font-weight: normal;">**Model evaluation**</small>
    <small style="font-weight: normal;">Finally evaluating the model is done by using the testing data or completely separate validation set that was not used in the training. This can include different metrics like accuracy, precision, recall, F1-score, and many others[^medium]. The evaluation can also include visualizations to see how the model is performing.</small>

## TODO-list for the week

- [x] Watch the weekly teams meeting recording.
- [x] Watch the videos for introduction to machine learning (total of 7 videos).
- [x] Allure 00 and 01.
- [x] Write the learning diary entry for the week.

[^IBM]: [IBM "What is data science?"](https://www.ibm.com/topics/data-science)
[^Geron]: [Géron, A. (2017). Hands-On Machine Learning with Scikit-Learn and TensorFlow: Techniques and Tools to Build Learning Machines. O’Reilly Media.](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/)
[^Grus]: [Grus J. (205) Data Science from Scratch](https://www.oreilly.com/library/view/data-science-from/9781491901410/?_gl=1*qjefjw*_ga*MTY5MzQwNzk2NS4xNzI4ODM2NTc1*_ga_092EL089CH*MTcyOTY3MjUwNC4yLjEuMTcyOTY3MjU1OS41LjAuMA)
[^Geeks]: [Difference between Data Scientist, Data Engineer, Data Analyst](https://www.geeksforgeeks.org/difference-between-data-scientist-data-engineer-data-analyst/)
[^dsc]: [Data science central: Understanding the difference: Data analyst, data scientist, and data engineer](https://www.datasciencecentral.com/understanding-the-difference-data-analyst-data-scientist-and-data-engineer/)
[^GOOGLE]: [Google: Artificial intelligence (AI) vs. machine learning (ML)](https://cloud.google.com/learn/artificial-intelligence-vs-machine-learning)
[^databricks]: [Data bricks: What is a machine learning Model?](https://www.databricks.com/glossary/machine-learning-models)
[^UMAP]: [McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction (Version 3). arXiv](https://doi.org/10.48550/ARXIV.1802.03426)
[^PCA]: [Hotelling, H. (1933). Analysis of a complex of statistical variables into principal components. In Journal of Educational Psychology (Vol. 24, Issue 6, pp. 417–441). American Psychological Association (APA).](https://doi.org/10.1037/h0071325)
[^MIT]:[David S.: Lecture 1: Introduction to Reinforcement Learning](<https://www.davidsilver.uk/wp-content/uploads/2020/03/intro_RL.pdf>)
[^GOOGLE2]: [Google: Machine learning workflow](https://cloud.google.com/ai-platform/docs/ml-solutions-overview)
[^medium]: [A Comprehensive Guide to Performance Metrics in Machine Learning](https://medium.com/@abhishekjainindore24/a-comprehensive-guide-to-performance-metrics-in-machine-learning-4ae5bd8208ce)
