# LIME and SHAP — Full Mathematical Derivation and Conceptual Explanation

---

# 1. Problem Setup

Let a trained black-box model be:

$$
f: \mathbb{R}^d \rightarrow \mathbb{R}
$$

Where:

- $d = 30$ features  
- Output = probability of benign tumor  

For a specific instance:

$$
x_0 \in \mathbb{R}^{30}
$$

Model prediction:

$$
f(x_0) = 0.955
$$

Dataset average prediction:

$$
\mathbb{E}[f(X)] = 0.63
$$

Goal:

Explain why prediction is **0.955 instead of 0.63**.

---

# 2. LIME — Mathematical Derivation

## 2.1 Core Idea

LIME approximates the complex model locally with a simple linear model:

$$
g(z) = w_0 + \sum_{j=1}^{d} w_j z_j
$$

Where:

- $g$ is interpretable  
- $w_j$ are coefficients to estimate  
- $z$ are perturbed samples near $x_0$

---

## 2.2 Perturbation

Generate synthetic samples:

$$
z^{(1)}, z^{(2)}, ..., z^{(N)}
$$

Each is a small variation of $x_0$.

For each:

$$
y^{(i)} = f(z^{(i)})
$$

This gives a new local dataset:

$$
(z^{(i)}, y^{(i)})
$$

---

## 2.3 Weighted Regression

Define proximity weight:

$$
\pi_{x_0}(z) = \exp\left(-\frac{D(x_0, z)^2}{\sigma^2}\right)
$$

Define diagonal weight matrix:

$$
W =
\begin{bmatrix}
\pi(z^{(1)}) & 0 & \dots \\
0 & \pi(z^{(2)}) & \dots \\
\vdots & & \ddots
\end{bmatrix}
$$

Solve:

$$
\min_w (Xw - y)^T W (Xw - y)
$$

Closed-form solution:

$$
w = (X^T W X)^{-1} X^T W y
$$

These $w_j$ are the LIME explanation coefficients.

---

## 2.4 Interpretation of Negative Values

If:

$$
w_j < 0
$$

Then increasing feature $j$ locally decreases predicted probability.

It represents a **local slope estimate**, not a global truth.

---

# 3. SHAP — Mathematical Derivation

## 3.1 Cooperative Game View

Features are players.  
Prediction is payout.

Set of features:

$$
N = \{1,2,...,d\}
$$

For subset $S \subseteq N$:

$$
f(S) = \mathbb{E}[f(X) \mid X_S = x_{0S}]
$$

Meaning:

Keep features in $S$ fixed to their values in $x_0$ and average over others.

---

## 3.2 Shapley Value Definition

$$
\phi_i =
\sum_{S \subseteq N \setminus \{i\}}
\frac{|S|! (d-|S|-1)!}{d!}
\left[
f(S \cup \{i\}) - f(S)
\right]
$$

This averages marginal contributions over all feature orderings.

---

## 3.3 Additive Decomposition Guarantee

SHAP guarantees:

$$
f(x_0) =
\mathbb{E}[f(X)] + \sum_{i=1}^{d} \phi_i
$$

For this example:

$$
0.955 = 0.63 + \sum_{i=1}^{30} \phi_i
$$

---

## 3.4 Two-Feature Example

If only features A and B exist:

$$
\phi_A =
\frac{1}{2}(f(A)-f(\emptyset))
+
\frac{1}{2}(f(A,B)-f(B))
$$

$$
\phi_B =
\frac{1}{2}(f(B)-f(\emptyset))
+
\frac{1}{2}(f(A,B)-f(A))
$$

And:

$$
f(x_0) =
f(\emptyset)
+
\phi_A
+
\phi_B
$$

---

# 4. Numerical Interpretation in This Case

Model prediction:

$$
f(x_0) = 0.955
$$

Baseline:

$$
\mathbb{E}[f(X)] = 0.63
$$

Difference:

$$
0.955 - 0.63 = 0.325
$$

SHAP distributes this 0.325 across features.

Example top SHAP values:

- worst area = 0.058  
- worst concave points = 0.054  
- worst perimeter = 0.052  

These features account for nearly half of the increase from baseline.

---

# 5. LIME vs SHAP Comparison

| Aspect | LIME | SHAP |
|--------|------|------|
| Method | Weighted local regression | Shapley value decomposition |
| Local explanation | Yes | Yes |
| Additive guarantee | No | Yes |
| Theoretical foundation | Heuristic | Game theory |
| Stability | Moderate | High |
| Regulatory use | Limited | Widely used |

---

# 6. When To Use

### LIME
- Quick local debugging
- Model behavior exploration
- Lightweight explanation

### SHAP
- Finance
- Healthcare
- Insurance
- Compliance / Regulatory AI
- When additive fairness guarantees matter