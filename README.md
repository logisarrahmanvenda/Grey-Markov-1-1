# Grey GM (1,1) and Markov Chain Model Documentation

This documentation explains the code and methodology implemented in the **Grey_Markov_(1,1).ipynb** notebook. The notebook demonstrates a hybrid approach combining the **Grey GM(1,1) Model** with a **Markov Chain** to forecast Indonesia's oil and gas export values, enhance accuracy, and provide a robust state-based prediction. The process integrates time series prediction, error state classification, Markov transition analysis, and accuracy evaluation.

---

## Introduction to the Grey GM(1,1) Model

The Grey GM(1,1) model is a time series forecasting method effective for small datasets with limited or uncertain information. It is widely used in engineering, economics, and management to predict short-term trends by constructing a first-order differential equation from the observed data.

---

## Importing Libraries

The code imports essential libraries for data manipulation, visualization, and mathematical computations.

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
```

**Purpose**:
- `numpy` and `math`: mathematical operations and array handling.
- `matplotlib.pyplot`: data visualization.
- `pandas`: data loading and transformation.

---

## Data Import and Preprocessing

The dataset is imported, and time columns are formatted for ease of visualization and analysis.

**Key Steps:**
- Load a CSV file containing monthly oil and gas export data.
- Add a `Tanggal` (date) column in "Mon-YYYY" format (e.g., Feb-2023).
- Extract both the time (`waktu`) and export value (`nilai_ekspor`) for further steps.

**Visualization Example**:
```python
plt.figure(figsize=(10, 5))
plt.plot(waktu, nilai_ekspor, marker='o')
plt.title("Time Series Ekspor Migas Indonesia")
plt.xlabel("Waktu")
plt.ylabel("Nilai Ekspor Migas (Juta $US)")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
```

---

## Grey GM(1,1) Model Implementation

### Accumulated Generating Operation (AGO)

The AGO converts the raw series into a monotonically increasing series, reducing randomness.

```python
def ago(lis):
    total = 0
    for x in lis:
        total += x
        yield total

x1 = list(ago(x0))
```

### Mean Generating Operation (MGO)

The MGO smooths the AGO series by averaging consecutive elements.

```python
def mgo(lis):
    pre = lis[0]
    for x in lis:
        mgoVal = (pre + x)/2
        pre = x
        yield mgoVal

z1 = [x * -1 for x in list(mgo(x1))]
z1 = np.delete(z1, 0)
```

### Model Parameter Calculation

The code solves the GM(1,1) equations for parameters `a` and `b`.

```python
B = pd.DataFrame({'0':z1})
B['1'] = 1
B_ = B.to_numpy()
B_t = B.transpose().to_numpy()
E1_pre = B_t.dot(B_)
E1 = np.linalg.inv(E1_pre)
Xn = np.delete(x0 ,0)
E2 = B_t.dot(Xn)
parameter = E1.dot(E2)

a = parameter.item(0)
b = parameter.item(1)
```

**Interpretation**:
- `a` is the development coefficient.
- `b` is the grey input.

### Time Response Function

The GM(1,1) time response is calculated to forecast future values.

```python
def dif_eq(k):
    return (x0[1] - (b/a)) * math.exp(-1*a*(k-1)) + (b/a)
```

### Forecast and Visualization

The code compares real and predicted values and visualizes the results.

```python
x_range = np.arange(1, x0.size+1)
def x_forecast(lis):
    for x in lis:
        x_fcst = dif_eq(x) - dif_eq(x-1)
        yield x_fcst

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(x_range, list(x_forecast(x_range)), color='tab:blue')
ax.scatter(x_range, x0, color='tab:red')
```

---

## Data Transformation Summary

A DataFrame containing the following is constructed:
- `Waktu`: Month-Year
- `Nilai Ekspor Migas (Juta Rupiah)`: Actual export value
- `X(1)`: AGO result
- `Z(1)`: MGO result

The transformed data is saved for further use.

---

## Relative Error Analysis

The model's relative error for each period is computed, enabling state classification for Markov analysis.

```python
x_hat = list(x_forecast(x_range))
error_rel = ((x0 - x_hat) / x0) * 100
```

---

## State Definition for Markov Chain

The code defines error states using the Sturges formula, then assigns each period to a state based on the error interval.

```python
n = len(error_rel)
r = int(1 + 3.3 * np.log10(n))  # number of states
# Calculate intervals and assign states ...
```

---

## Markov Chain: Transition Matrix Construction

### Frequency and Probability Matrices

The Markov Chain models the error state transitions over time.

```python
# Frequency matrix
freq_matrix = np.zeros((r, r))
for i in range(len(state)-1):
    i_state = state[i] - 1
    j_state = state[i+1] - 1
    freq_matrix[i_state, j_state] += 1

# Transition probability matrix
transition_matrix = np.zeros_like(freq_matrix)
for i in range(r):
    row_sum = freq_matrix[i].sum()
    if row_sum == 0:
        transition_matrix[i, i] = 1
    else:
        transition_matrix[i] = freq_matrix[i] / row_sum
```

**Interpretation**:
- Each cell (i, j) in the transition matrix indicates the probability of moving from state i to state j.

### Markov State Prediction

The code predicts the next state using cumulative transition probabilities (`P^k`), and selects the state with the highest probability.

---

## Final Hybrid Grey-Markov Prediction

The final forecast is a corrected GM(1,1) prediction adjusted by the expected error state from the Markov model.

```python
correction_factor = (lower_bound + upper_bound) / 2 / 100
x_gm_markov_next = x_gm_next * (1 + correction_factor)
```

---

## Accuracy and Evaluation

### Metrics Computed

- **MAPE (Mean Absolute Percentage Error)**: For both GM(1,1) and Grey–Markov(1,1) models.
- **S1**: Standard deviation of the actual data.
- **S2**: Standard deviation of the model's residuals.
- **C (Variation Coefficient)**: S2/S1, indicating model stability.

```python
MAPE = (np.abs((data_valid["Data Asli"] - data_valid["Grey-Markov(1,1)"]) / data_valid["Data Asli"])).mean() * 100
```

### Visualization

Comparison plots between actual and predicted values for both models are generated to visually assess model performance.

---

## Key Data Structures

| Variable         | Description                                                    |
|------------------|---------------------------------------------------------------|
| `x0`             | Raw time series data (export values)                          |
| `x1`             | AGO sequence (cumulative sum)                                 |
| `z1`             | MGO sequence (smoothed average, negative for GM(1,1))         |
| `B`, `B_`, `B_t` | Matrices for parameter estimation (least squares)             |
| `a`, `b`         | GM(1,1) model parameters                                      |
| `error_rel`      | Relative error per period                                     |
| `state`          | Assigned Markov error state per period                        |
| `transition_matrix`| Markov chain transition probabilities                       |
| `x_hat`          | GM(1,1) forecasts                                             |
| `x_gm_markov_next`| Final Markov-corrected prediction for next period            |

---

## Model Outputs

- **Forecast Table**: With actual data, GM(1,1) predictions, states, and Grey–Markov hybrid predictions.
- **CSV Exports**: DataFrames are saved for further analysis.
- **Plots**: Visual comparison of forecasts and actual data.
- **Performance Metrics**: MAPE, S1, S2, and C for both models.

---

## Best Practices and Recommendations

```card
{
  "title": "Best Practices for Grey–Markov Hybrid Modeling",
  "content": "Always validate model accuracy using historical data and interpret Markov error states to ensure improved forecast stability."
}
```

---

## References

- [Application of Renewal Gray GM(1,1) Model to Prediction of Landslide Deformation](https://www.researchgate.net/publication/319605877_Application_of_Renewal_Gray_GM_11_Model_to_Prediction_of_Landslide_Deformation)
- [Grey–Markov hybrid method reference](http://iieta.org/sites/default/files/Journals/MMEP/02.1_05.pdf)
- [NCBI: Grey prediction methods in forecasting](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6112867/)

---

## Conclusion

The combination of the **Grey GM(1,1)** model and a **Markov Chain** enables short-term, robust forecasting for time series with limited data. The approach delivers:
- Accurate point forecasts using GM(1,1).
- State-based correction using Markov error transitions.
- Improved accuracy and interpretability, as reflected by the significant reduction in MAPE.

Whether for economic, engineering, or management forecasting, this hybrid methodology can be adapted to other similar time series problems.

---
### Questions or improvements?
Feel free to refer to the code and references above for further exploration or adaptation to your own time series forecasting tasks!
