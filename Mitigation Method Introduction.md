# Mitigation Method Example

In **benchmark**, we give a CNN-based implementation. In this implementation, the **Final Score** of the model is approximately **0.88.**

(note: No matter which method you use, you must ensure that your **Final Score** $\geqslant$ **0.90**)

## Final Score

#### Overall AUC

This is the ROC-AUC for the full evaluation set.

#### Bias AUCs

To measure unintended bias, we again calculate the ROC-AUC, this time on three specific subsets of the test set for each identity, each capturing a different aspect of unintended bias. You can learn more about these metrics in Conversation AI's recent paper *[Nuanced Metrics for Measuring Unintended Bias with Real Data in Text Classification](https://arxiv.org/abs/1903.04561)*.

**Subgroup AUC**: Here, we restrict the data set to only the examples that mention the specific identity subgroup. *A low value in this metric means the model does a poor job of distinguishing between toxic and non-toxic comments that mention the identity*.

**BPSN (Background Positive, Subgroup Negative) AUC**: Here, we restrict the test set to the non-toxic examples that mention the identity and the toxic examples that do not. *A low value in this metric means that the model confuses non-toxic examples that mention the identity with toxic examples that do not*, likely meaning that the model predicts higher toxicity scores than it should for non-toxic examples mentioning the identity.

**BNSP (Background Negative, Subgroup Positive) AUC**: Here, we restrict the test set to the toxic examples that mention the identity and the non-toxic examples that do not. *A low value here means that the model confuses toxic examples that mention the identity with non-toxic examples that do not*, likely meaning that the model predicts lower toxicity scores than it should for toxic examples mentioning the identity.

#### Generalized Mean of Bias AUCs

To combine the per-identity Bias AUCs into one overall measure, we calculate their generalized mean as defined below:
$$
M_p(m_s)=\bigg(\frac{1}{N}\sum \limits_{s=1}^N m_s^p \bigg)^\frac{1}{p}
$$
where:

$M_p=$ the $p$ th power-mean function

$m_s=$ the bias metric $m$ calulated for subgroup $s$

$N=$ number of identity subgroups

For this competition, we use a $p$ value of -5 to encourage competitors to improve the model for the identity subgroups with the lowest model performance.

### Final Metric

We combine the overall AUC with the generalized mean of the Bias AUCs to calculate the final model score:
$$
score=w_0AUC_{overall}+\sum \limits_{a=1}^A w_aM_p(m_{s,a})
$$
where:

$A = $ number of submetrics (3)

$m_{s,a}=$ bias metric for identity subgroup $s$ using submetric $a$

$w_a=$ a weighting for the relative importance of each submetric; all four $w$ values set to 0.25

### Final Metric Code

```python
def calculate_overall_auc(df, model_name):
    true_labels = df[TOXICITY_COLUMN]
    predicted_labels = df[model_name]
    return metrics.roc_auc_score(true_labels, predicted_labels)

def power_mean(series, p):
    total = sum(np.power(series, p))
    return np.power(total / len(series), 1 / p)

def get_final_metric(bias_df, overall_auc, POWER=-5, OVERALL_MODEL_WEIGHT=0.25):
    bias_score = np.average([
        power_mean(bias_df[SUBGROUP_AUC], POWER),
        power_mean(bias_df[BPSN_AUC], POWER),
        power_mean(bias_df[BNSP_AUC], POWER)
    ])
    return (OVERALL_MODEL_WEIGHT * overall_auc) + ((1 - OVERALL_MODEL_WEIGHT) * bias_score)
    
get_final_metric(bias_metrics_df, calculate_overall_auc(validate_df, MODEL_NAME))
```



## Loss function

In the **benchmark** CNN model, the default loss function used is **categorical_cross_entropy**, but you can use other loss functions, or modify the original loss function.

1. **BCEWithLogitsLoss**

   BCEWithLogitsLoss combines the Sigmoid function with Binary Cross Entropy Loss, allowing the loss to be calculated directly on the logits (the original output without Sigmoid activation).
   $$
   l_n=-w_n[y_n\cdot \log \sigma(x_n)+(1-y_n)\cdot \log(1-\sigma(x_n))]
   $$
   BCEWithLoss code implementation in torch:

   ```python
   torch.nn.BCEWithLogitsLoss(
   				weight=None, 
   				reduction='mean', 
   				pos_weight=None)
   ```

   

2. **CrossEntropyLoss**

   Cross entropy loss function calculation formula:
   $$
   \begin{align}
   loss(x, class) &= -\text{log}\frac{exp(x[class])}{\sum_j exp(x[j]))}\\
                  &= -x[class] + log(\sum_j exp(x[j])) \\
   			   &= weights[class] * (-x[class] + log(\sum_j exp(x[j])))
   \end{align}
   $$
   CrossEntropyLoss code implementation in torch：

   ```python
   torch.nn.CrossEntropyLoss(
   		weight=None,
   		size_average=None,
   		ignore_index=-100,
   		reduce=None,
   		reduction='mean')
   ```

   

3. **MSELoss**

   1. MSE loss function calculation formula:
      $$
      loss(x,y)=1/n\sum(x_i-y_i)^2
      $$
      torch code implementation in torch：

      ```python
      torch.nn.MSELoss(
      		size_average=None, 
      		reduce=None, 
      		reduction='mean')
      ```

   

4. **Design Loss Example**

   ```python
   class MyFocalLoss(nn.Module):
       def __init__(self, alpha=1, gamma=1, logits=True, reduce=True):
           super(FocalLoss, self).__init__()
           self.alpha = alpha
           self.gamma = gamma
           self.logits = logits
           self.reduce = reduce
   
       def forward(self, inputs, targets, weight, focal_weights, penalty_weights, focal_alpha, penalty_alpha):
           if self.logits:
               BCE_loss = nn.BCEWithLogitsLoss(weight=weight, reduce=False)(inputs, targets)
           else:
               BCE_loss = nn.BCELoss(weight=weight, reduce=False)(inputs, targets) 
           pt = torch.exp(-BCE_loss)
   
           # The designed part
           bias_weights = (1.0 + targets.detach() - torch.sigmoid(inputs).detach())
   
           F_loss = (1-pt)**(focal_weights) * BCE_loss * (bias_weights ** penalty_weights) * focal_alpha * penalty_alpha
   
           if self.reduce:
               return torch.mean(F_loss)
           else:
               return F_loss
   
   ```



## Using other algorithmic

In Benchmark, we gave a CNN-based implementation, but this may not be the best implementation. Therefore, you can consider replacing other algorithms such as LSTM, RNN, Bert, Transformer, etc.

We have given the code of the LSTM network structure in `simple-lstm-example.py`. You can build your classification model based on this network structure. (Note: This part of the code is for reference only!)

## Others

If you do not have enough computing power to complete this experiment, we recommend deploying the project on the following two platforms: **Kaggle** and **Google Colab**. The basic computing power these two platforms provide is sufficient to complete this experiment.

Kaggle：https://www.kaggle.com/

Google colab：https://colab.google/
