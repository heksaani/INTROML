
# Note 1

# Allure test

(run in /koodit) direcory:

```bash
> docker compose up -d
```

Open the report in browser:
<http://localhost:5050/latest-report>

# Git passphase

```bash
> eval "$(ssh-agent -s)"
> ssh-add ~/.ssh/id_rsa
```

# Run the docker for learning diary

To see the the learning diary while editing run in the ML-intro directory:

```bash
cd /ML-intro
> docker compose -f docker-compose-docs.yml up -d
```

Then go to `localhost:8000` and open the html page.

# Machine learning definitions

Regularization is a technique used to prevent overfitting in machine learning models. It does this by adding a penalty term to the loss function, which discourages the model from learning complex patterns that are specific to the training data and may not generalize well to new data.

Variance is a measure of how much the predictions of a machine learning model vary for different training datasets. High variance models are sensitive to the training data and may overfit, while low variance models are more stable and generalize better to new data.

# Books

Geron: Aurélien Géron. (2019). Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow : Concepts, Tools, and Techniques to Build Intelligent Systems: Vol. Second edition. O’Reilly Media.
Grus: [Grus J. (205) Data Science from Scratch](https://www.oreilly.com/library/view/data-science-from/9781491901410/?_gl=1*qjefjw*_ga*MTY5MzQwNzk2NS4xNzI4ODM2NTc1*_ga_092EL089CH*MTcyOTY3MjUwNC4yLjEuMTcyOTY3MjU1OS41LjAuMA)
