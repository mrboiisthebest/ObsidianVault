

# 2.1 What Is Statistical Learning?
- our goal is to develop an accurate model that can be used to predict sales on the basis of the three media budgets
- advertising budgets are **input variables**, sales is an **output variable**
-  **input variables** are typically denoted using the output variable symbol **X**
-  The Sales variable is often known as the **response or dependent** variable and is shown as **Y**
$$Y = f(X) + \epsilon$$
- Y(output), X(input), e(error term)
- The above equation shows the relationships between the inputs and outputs (X,Y)
![[Pasted image 20260609102711.png]]
- The plot displays sales along with their budgets
- Blue line represents simple model that predicts sales
![[Pasted image 20260609103015.png]]
- Red dots represents observed income over years of education for 30 people
- Blue curve shows true underlying relationship between Income and Education which is generally unknown
- Black Lines show error
- Due to this being a simulated dataset we know $f$  but this is not always the case
- Errors have mean of ~0
- generally the  function may involve more than one input variable
- statistical learning refers to a set of approaches for estimating $f$

**2.1.1 Why Estimate $f$?**
- we would estimate $f$ for prediction and inference

Predicts Y
$$\hat{Y} = \hat{f}(\hat{X})$$
- The little arrow is a **hat** and shows its a prediction or estimate
- We can make Y more accurate with these quantities **reducible error and irreducible error**
- Even if we could get a perfect estimate for $f$ it will not be accurate as **Y** is a function  of $\in$ which **cant** be predicted by using **X** this is known as irreducible error
![[Pasted image 20260609104836.png]]
- blue surface represents the true underlying relationship between income and education and seniority (is known because data is simulated)
- $f$ cant use these dots for its prediction
![[Pasted image 20260609105026.png]]
- $E(Y-\hat{Y})^2$ represents the average or expected value of the squared difference between the predicted and actual value of Y 
**2.1.2 How do We Estimate $f$?**
- **Parametric Methods** involve a two-step model-based approach
- First We assume the $f$ is a linear model or assume other forms
![[Pasted image 20260609105642.png]]
- After a model has been selected, we need a procedure that uses the training data to fit or train the model
$$ Y \approx \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \dots + \beta_p X_p $$
- The most common approach to fitting the model is called ordinary least squares
- The model-based approach is referred to as parametric
- reduces problem down to one estimating set of parameters
- fitting a more flexible model requires estimating a greater number of parameters, leads to **overfitting** (follows errors too closely)
- Linear fit is not perfect but captures the overall look
- Non-parametric methods aim to get as close to the data as possible  this leads to it not fitting future data well but can be very accurate as seen in image below![[Pasted image 20260609110523.png]]
**2.1.3 The Trade-Off Between Prediction Accuracy and Model Interpretability**
- linear regression is a relatively inflexible approach, because it can only generate linear functions
-  why would we ever choose to use a more restrictive method instead of a very flexible approach? mainly interested in inference, clearly shows relationship
**2.1.4  Supervised Versus Unsupervised Learning**
- examples that we have discussed so far in this chapter all fall into the supervised learning domain
$$(s) xi, i =1,...,n $$
- **Supervised** - For each observation of the predictor measurement there is an associated response. No response variable to predict
- Cluster analysis tries to get a correct answer in the ballpark of the truth
![[Pasted image 20260609120241.png]]
**2.1.5  Regression Versus Classification Problems**
- Variables can be characterized as either quantitative or qualitative
- We tend to refer to problems with a quantitative response as **regression problems**
- those involving a regression qualitative response are often referred to as **classification problems**






