# Literature Review & Theoretical Framework: Predictive Maintenance System

A literature review and methodological synthesis for **MaintainIQ — ML-Based Predictive Maintenance System**, examining foundational maintenance paradigms, tree-based ensemble learning, class-imbalance evaluation, and explainable artificial intelligence (XAI).

---

## 1. Introduction to Predictive Maintenance (PdM)

In modern smart manufacturing and Industry 4.0, equipment maintenance strategies are categorized into three primary paradigms:
- **Reactive Maintenance (Run-to-Failure)**: Machinery is operated until a physical component breaks down. While this minimizes scheduled downtime during healthy periods, it incurs severe economic penalties through catastrophic failure, extensive secondary hardware damage, emergency replacement costs, and unplanned manufacturing stoppages.
- **Preventive Maintenance (Time- or Usage-Based)**: Servicing interventions are conducted at fixed operating intervals or cycle counts regardless of real-time machine health. Although this mitigates catastrophic failures, it leads to premature replacement of components with substantial remaining useful life (RUL), inflating operational expenditure (OpEx) and scheduling unnecessary maintenance downtime.
- **Predictive Maintenance (Condition-Based)**: Telemetry streams from embedded industrial sensors (such as thermal probes, torque sensors, and tachometers) are continuously monitored using machine learning algorithms. By identifying subtle degradation patterns and abnormal physical stress signatures, maintenance teams can intervene precisely before failure occurs, optimizing asset lifespan, worker safety, and operational efficiency [1], [2], [7].

---

## 2. Review of Key Literature

The following nine peer-reviewed publications provide the theoretical foundation, methodological benchmarks, and analytical frameworks directly informing the design and implementation of MaintainIQ. Five of these papers represent recent research published between 2021 and 2026, supplemented by four foundational works.

---

### [1] Matzka (2020) — Foundation of the AI4I 2020 Predictive Maintenance Dataset
- **Citation**: S. Matzka, "Explainable Artificial Intelligence for Predictive Maintenance Applications," in *Proceedings of the 2020 Third International Conference on Artificial Intelligence for Industries (AI4I)*, Irvine, CA, USA, 2020, pp. 69–74.
- **DOI / Link**: [10.1109/AI4I49448.2020.00023](https://doi.org/10.1109/AI4I49448.2020.00023)
- **Objective**: Introduce a standardized, realistic synthetic milling machine dataset to the research community and evaluate the trade-offs between predictive accuracy and explainability in industrial maintenance.
- **Method**: Matzka designed and published the **AI4I 2020 Predictive Maintenance Dataset** (hosted on the UCI Machine Learning Repository), generating 10,000 operational observations governed by phenomenological physical equations modeling tool wear, frictional heat generation, motor power boundaries, and structural torque strain. The author evaluated decision trees, rule sets, and model-agnostic explainability approaches.
- **Key Findings**: The study demonstrated that while complex models achieve superior predictive accuracy, shop-floor operators and maintenance supervisors require transparent decision boundaries to trust and act upon automated warnings. Rotational speed, cutting torque, and tool wear contact time were established as the primary physical signals driving operational failure modes.
- **Relevance to MaintainIQ**: This paper is the primary source of MaintainIQ's dataset. It establishes the physical definitions of the six approved feature variables and the five monitored machine failure modes (`TWF`, `HDF`, `PWF`, `OSF`, and `RNF`).

---

### [2] Carvalho et al. (2019) — Systematic Review of ML in Predictive Maintenance
- **Citation**: T. P. Carvalho, F. A. A. M. N. Soares, R. Vita, R. da P. Francisco, J. P. Basto, and S. G. S. Alcalá, "A systematic literature review of machine learning methods applied to predictive maintenance," *Computers & Industrial Engineering*, vol. 137, art. no. 106024, Nov. 2019.
- **DOI / Link**: [10.1016/j.cie.2019.106024](https://doi.org/10.1016/j.cie.2019.106024)
- **Objective**: Conduct a comprehensive systematic mapping of machine learning algorithms, sensor telemetry types, and application domains across industrial predictive maintenance.
- **Method**: Analyzed over 200 peer-reviewed studies published between 2009 and 2018, categorizing maintenance objectives by algorithm family (linear models, support vector machines, decision tree ensembles, neural networks) and target physical assets.
- **Key Findings**: The authors identified that tree-based ensemble algorithms (specifically Random Forest and gradient boosting variants) consistently outperformed linear baselines, single decision trees, and standalone neural networks on structured tabular sensory telemetry. Their superiority stems from robust handling of mixed feature scales, resistance to sensor noise, and the capacity to capture non-linear interactions without requiring massive training datasets.
- **Relevance to MaintainIQ**: Directly justifies MaintainIQ's multi-model comparative architecture, specifically contrasting a linear baseline (Logistic Regression) against tree-based bagging and boosting ensembles (Random Forest, XGBoost, CatBoost).

---

### [3] Alnahhal et al. (2026) — Imbalanced Classification Benchmarks on AI4I 2020
- **Citation**: M. Alnahhal, M. I. Tabash, S. K. Safi, M. S. M. Al-Absy, and Z. Mamadiyarov, "A Comparative Study of Imbalance-Handling Methods in Multiclass Predictive Maintenance," *Computation*, vol. 14, no. 4, art. no. 88, 2026.
- **DOI / Link**: [10.3390/computation14040088](https://doi.org/10.3390/computation14040088)
- **Objective**: Benchmark machine learning classifiers and data-balancing interventions across rare failure modes on the UCI AI4I 2020 Predictive Maintenance Dataset.
- **Method**: The authors benchmarked 20 experimental configurations comparing Random Forest, XGBoost, Support Vector Machines, K-Nearest Neighbors, and Multinomial Logistic Regression under different data balancing strategies (including SMOTE, random oversampling, and algorithmic weighting) on the AI4I 2020 dataset.
- **Key Findings**: The study reported that gradient boosting models (notably XGBoost) exhibit strong intrinsic resilience on tabular sensor telemetry even under severe class skew. When optimizing for macro-averaged F1 and recall across rare failure modes, algorithmic class reweighting (penalizing minority misclassifications) preserved feature distributions better than aggressive synthetic oversampling, which introduced synthetic noise near operational decision boundaries.
- **Relevance to MaintainIQ**: Informs MaintainIQ's class-imbalance mitigation design: adopting training-only cost-sensitive class reweighting (`scale_pos_weight` in XGBoost, `class_weight='balanced'` in Random Forest and Logistic Regression, and `auto_class_weights='Balanced'` in CatBoost) rather than applying synthetic oversampling to evaluation splits.

---

### [4] Theissler et al. (2021) — Operational Challenges and XAI in Predictive Maintenance
- **Citation**: A. Theissler, J. Pérez-Velázquez, M. Kettelgerdes, and G. Elger, "Predictive maintenance enabled by machine learning: Use cases and challenges in the automotive industry," *Reliability Engineering & System Safety*, vol. 215, art. no. 107864, Nov. 2021.
- **DOI / Link**: [10.1016/j.ress.2021.107864](https://doi.org/10.1016/j.ress.2021.107864)
- **Objective**: Systematically investigate practical barriers to deploying machine learning for predictive maintenance in industrial manufacturing, focusing on data collection, distribution shift, and explainability.
- **Method**: Surveyed industrial automotive manufacturing use cases, evaluating how sensor telemetry quality, extreme class imbalance, and human-in-the-loop workflows impact algorithm operationalization.
- **Key Findings**: The authors highlighted that real-world industrial telemetry is characterized by severe class imbalance (machine failure is a rare event representing < 5% of cycles). Crucially, the survey revealed that the primary non-technical bottleneck to adopting automated maintenance systems is the **lack of model interpretability**: maintenance engineers routinely reject or disable black-box alert systems when the algorithm cannot provide clear, physically plausible reasoning for an alert.
- **Relevance to MaintainIQ**: Motivates MaintainIQ's emphasis on explainability (providing global permutation importance rankings and local instance SHAP explanations) and reinforces the necessity of reporting Precision, Recall, and PR-AUC instead of misleading accuracy scores.

---

### [5] Marmoucha et al. (2026) — Multicriteria Model Selection on AI4I 2020
- **Citation**: Z. Marmoucha, M. El Khaili, A. Soulhi, H. El Hailouch, and H. Nhaila, "Enhancing Predictive Accuracy and Operational Performance Based on Entropy–TOPSIS Framework for Machine Learning Model Selection for Predictive Maintenance," *Applied Sciences*, vol. 16, no. 17, art. no. 8666, 2026.
- **DOI / Link**: [10.3390/app16178666](https://doi.org/10.3390/app16178666)
- **Objective**: Formulate a reproducible multicriteria decision-making framework combining Entropy-based weighting with TOPSIS to select optimal predictive maintenance classifiers balancing failure detection, false alarm rates, and computational execution speed.
- **Method**: Evaluated Logistic Regression, Random Forest, XGBoost, and LightGBM across two industrial datasets: real-world metro air compressor data (MetroPT) and the **UCI AI4I 2020 dataset**, incorporating stratified repeated evaluation, controlled hyperparameter optimization, and extensive perturbation sensitivity analysis.
- **Key Findings**: The authors established that evaluating predictive maintenance algorithms requires multidimensional assessment: while models can exhibit comparable ROC-AUC, their trade-offs between False Positives (unnecessary plant stoppages) and False Negatives (unmitigated breakdowns) differ dramatically. Tree-based gradient boosters dominated linear classifiers across both datasets, providing superior F1 performance while sustaining microsecond-level inference latency suitable for operational deployment.
- **Relevance to MaintainIQ**: Directly substantiates MaintainIQ's four-model benchmark (Logistic Regression, Random Forest, XGBoost) on the exact same AI4I 2020 dataset and reinforces the project's reporting of precision, recall, confusion matrix components, and per-sample inference latency.

---

### [6] Chatterjee & Byun (2023) — Imbalanced Industrial Fault Classification with Tree Ensembles
- **Citation**: S. Chatterjee and Y.-C. Byun, "Highly imbalanced fault classification of wind turbines using data resampling and hybrid ensemble method approach," *Engineering Applications of Artificial Intelligence*, vol. 126, art. no. 107104, Nov. 2023.
- **DOI / Link**: [10.1016/j.engappai.2023.107104](https://doi.org/10.1016/j.engappai.2023.107104)
- **Objective**: Resolve extreme class imbalance in industrial condition monitoring where sensor telemetry is heavily dominated by normal operational cycles and failure events are exceedingly rare.
- **Method**: Developed an ensemble fault classification pipeline evaluating Random Forest, XGBoost, and LightGBM architectures under skewed sensor streams, analyzing performance trade-offs across minority-class precision, recall, and harmonic F1 measures.
- **Key Findings**: Standard classifiers trained on uncalibrated imbalanced data optimize overall accuracy by predicting the majority non-failure class, rendering them useless for proactive maintenance. The authors proved that gradient boosting ensembles configured with minority-class weighting achieve high recall on critical physical failure states while maintaining acceptable precision boundaries.
- **Relevance to MaintainIQ**: Provides empirical backing for MaintainIQ's mitigation strategy against extreme imbalance (~3.39% failure incidence), validating the adoption of minority-focused evaluation metrics (Precision, Recall, F1) over standard accuracy.

---

### [7] Abidi et al. (2022) — Predictive Maintenance Architectures for Industry 4.0
- **Citation**: M. H. Abidi, M. K. Mohammed, and H. Alkhalefah, "Predictive Maintenance Planning for Industry 4.0 Using Machine Learning for Sustainable Manufacturing," *Sustainability*, vol. 14, no. 6, art. no. 3387, Mar. 2022.
- **DOI / Link**: [10.3390/su14063387](https://doi.org/10.3390/su14063387)
- **Objective**: Develop an end-to-end predictive maintenance planning framework for industrial machinery under Industry 4.0 to minimize unplanned downtime, lower operational costs, and foster sustainable manufacturing operations.
- **Method**: Designed a multi-stage architecture spanning telemetry data ingestion, cleaning, normalization, dimensionality assessment, classification, and failure forecasting across dynamic operational ranges.
- **Key Findings**: The authors demonstrated that structured preprocessing (including systematic outlier management and feature normalization) is a prerequisite for reliable failure prediction. The study concluded that proactive predictive maintenance frameworks significantly cut operating costs compared to preventive time-based schedules by extending physical asset life cycles and preventing sudden mechanical stops.
- **Relevance to MaintainIQ**: Informs MaintainIQ's end-to-end pipeline structure: connecting raw telemetry ingestion and leakage-safe preprocessing with automated risk classification and operational dashboard reporting.

---

### [8] Saito & Rehmsmeier (2015) — Precision-Recall vs. ROC Curves under Severe Imbalance
- **Citation**: T. Saito and M. Rehmsmeier, "The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets," *PLOS ONE*, vol. 10, no. 3, art. no. e0118432, Mar. 2015.
- **DOI / Link**: [10.1371/journal.pone.0118432](https://doi.org/10.1371/journal.pone.0118432)
- **Objective**: Formally evaluate the visual and mathematical informativeness of Receiver Operating Characteristic (ROC) plots versus Precision-Recall (PR) plots when evaluating binary classification algorithms on highly skewed datasets.
- **Method**: Developed mathematical proofs and simulation experiments across varying positive-to-negative class ratios (from 1:1 down to 1:10,000), analyzing how changes in true positive and false positive rates affect area under the curve metrics.
- **Key Findings**: The authors proved that ROC-AUC can present an overly optimistic and deceptive picture of classifier performance on heavily imbalanced datasets. Because the False Positive Rate ($FPR = FP / (FP + TN)$) incorporates the vast negative class ($TN$) into its denominator, a substantial surge in false alarms ($FP$) produces only a negligible change in $FPR$. In contrast, Precision ($TP / (TP + FP)$) directly relates true alarms to false alarms, making the Precision-Recall curve and its area (PR-AUC / Average Precision) far more rigorous and sensitive to minority-class detection failures.
- **Relevance to MaintainIQ**: Serves as the primary theoretical justification for adopting PR-AUC as the governing evaluation metric for classifier selection, threshold tuning, and permutation feature importance in MaintainIQ, where failure incidence is only ~3.39% (a ~28:1 negative-to-positive ratio).

---

### [9] Lundberg et al. (2020) — TreeSHAP for Interpretable Tree-Based Ensembles
- **Citation**: S. M. Lundberg, G. Erion, H. Chen, A. DeGrave, J. M. Prutkin, B. Nair, R. Katz, J. Himmelfarb, N. Bansal, and S.-I. Lee, "From local explanations to global understanding with explainable AI for trees," *Nature Machine Intelligence*, vol. 2, no. 1, pp. 56–67, Jan. 2020.
- **DOI / Link**: [10.1038/s42256-019-0138-9](https://doi.org/10.1038/s42256-019-0138-9)
- **Objective**: Develop an exact, computationally tractable algorithm to compute Shapley additive feature attributions for tree-based ensemble models.
- **Method**: Formulated **TreeSHAP** (`TreeExplainer`), exploiting the internal recursive graph structure of decision trees to compute exact cooperative game-theoretic Shapley values in low-order polynomial time ($O(TLD^2)$, where $T$ is the number of trees, $L$ is maximum leaves, and $D$ is maximum tree depth), reducing computational complexity from exponential $O(TL2^{|M|})$.
- **Key Findings**: The authors demonstrated that TreeSHAP guarantees three foundational properties that heuristic feature attribution methods violate: *local accuracy* (attributions sum to the model output difference from base expectation), *missingness* (features with zero presence carry zero attribution), and *consistency* (if a model changes so that a feature's marginal contribution increases, its attribution value cannot decrease).
- **Relevance to MaintainIQ**: Provides the mathematical foundation and computational engine for MaintainIQ's local explainability module (`src/explain.py`). TreeSHAP is used to explain single machine predictions in the dashboard, showing operators exactly how much each physical sensor signal shifts the failure probability away from the baseline expectation.

---

## 3. Methodological Synthesis & Research Gap

### 3.1 The Research Gap in Applied Predictive Maintenance
A survey of contemporary literature reveals three recurring limitations in academic predictive maintenance implementations:
1. **Pervasive Data Leakage**: Many published academic pipelines fit feature scalers, imputers, or categorical encoders on the entire dataset prior to splitting, or apply synthetic oversampling (like SMOTE) across the entire dataset. This artificially inflates test performance and leads to catastrophic generalization collapse in production environments.
2. **Misleading Evaluation Metrics**: A substantial portion of published research continues to report raw classification accuracy or ROC-AUC as primary benchmarks. On datasets with 3–5% failure rates, accuracy obscures models that fail to identify any breakdown events, while ROC-AUC hides high false alarm rates [8].
3. **The "Black-Box" Usability Chasm**: Sophisticated gradient boosting models are frequently proposed without interpretability mechanisms or interactive tooling, leaving domain engineers with no actionable explanation of why an alert was triggered [4], [7].

MaintainIQ addresses this research gap by demonstrating a **leakage-safe, reproducible end-to-end architecture**: fitting all transformations strictly on training folds, benchmarking four classifiers under cost-sensitive weighting, prioritizing PR-AUC, and pairing predictions with both global (permutation importance) and local (TreeSHAP) interpretability within an interactive application.

---

### 3.2 Justification of the Four-Model Comparative Framework

MaintainIQ evaluates four distinct classification paradigms to investigate how inductive biases impact industrial telemetry modeling:

1. **Logistic Regression (Linear Baseline)**:
   - *Architecture*: Generalized linear model with L2 regularization and inverse class frequency weighting (`class_weight='balanced'`).
   - *Rationale*: Serves as a vital empirical baseline. While computationally instantaneous and fully interpretable, its linear decision boundary cannot model non-linear physical interactions (such as mechanical power $P = \tau \cdot \omega$ or cumulative thermal-mechanical stress).
2. **Random Forest (Bagging Ensemble)**:
   - *Architecture*: Bootstrap aggregation of decorrelated decision trees with balanced subsampling (`class_weight='balanced'`).
   - *Rationale*: Mitigates variance and overfitting. Excellent robustness to local sensor noise, consistently producing high precision and a low false-alarm profile in manufacturing literature [2], [5], [6].
3. **XGBoost (Extreme Gradient Boosting)**:
   - *Architecture*: Second-order Taylor expansion gradient boosting with regularization ($\lambda, \gamma$) and training-derived positive class weighting (`scale_pos_weight = n_neg / n_pos`).
   - *Rationale*: Industry standard for tabular telemetry. Highly effective at capturing non-linear sensor thresholds governing thermodynamic and mechanical degradation [3], [5], [6].
4. **CatBoost (Categorical Gradient Boosting)**:
   - *Architecture*: Symmetric (oblivious) decision trees with ordered boosting and internal automated frequency balancing (`auto_class_weights='Balanced'`).
   - *Rationale*: Avoids target leakage during categorical encoding and prevents prediction shift, frequently maximizing recall on rare minority failure modes [3].

---

### 3.3 The Critical Role of PR-AUC in Class-Imbalanced Maintenance

In the UCI AI4I 2020 dataset, positive failure events account for only **339 out of 10,000 observations (~3.39%)**, with the held-out test split containing 68 failure events among 2,000 samples.

As proven by Saito and Rehmsmeier [8], ROC curves can be profoundly misleading in this regime:
$$\text{FPR} = \frac{\text{FP}}{\text{FP} + \text{TN}}$$
Because the healthy negative class ($\text{TN} = 1932$) is massive, a large number of false positive alarms (e.g., $\text{FP} = 100$) still yields an apparently impressive $\text{FPR} = 100 / (100 + 1932) \approx 0.049$ (a 95.1% specificity). However, if the model only detected 50 actual failures ($\text{TP} = 50$), the plant's operational precision is:
$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}} = \frac{50}{50 + 100} = 33.3\%$$
Two out of every three maintenance stoppages would be false alarms, generating prohibitive operational downtime costs.

**Precision-Recall Area Under Curve (PR-AUC / Average Precision)** calculates the trade-off directly between positive predictive value and sensitivity across all classification thresholds, entirely excluding $\text{TN}$. Consequently, MaintainIQ employs PR-AUC as its primary ranking metric for model selection and permutation feature importance [6], [8].

---

### 3.4 The Role of Explainable AI (XAI) in Maintenance Operations

Predictive maintenance is a mission-critical domain where false predictions carry steep asymmetric costs:
- **False Negatives**: Cause catastrophic equipment destruction, hazard to operators, and unplanned production freezes.
- **False Positives**: Cause unnecessary machine shutdowns, lost throughput, and technician alert fatigue.

As emphasized by Matzka [1] and Theissler et al. [4], maintenance personnel will not shut down an operating assembly line based on an unverified algorithmic confidence score. MaintainIQ integrates a dual-layer explainability architecture:
1. **Global Permutation Feature Importance**: Evaluates the true empirical degradation of the model's PR-AUC score when individual sensor features are randomly shuffled on the held-out test split, isolating macroscopic sensor reliance without model architecture assumptions.
2. **Local TreeSHAP Attributions**: Computes additive game-theoretic attributions for each real-time sensor reading against the baseline fleet expectation, explaining precisely which physical variable (e.g., high cutting torque, elevated process temperature, or excessive tool wear) triggered a high-risk warning [9].

---

## 4. Methodological Boundaries & Project Limitations

To maintain academic rigor, the findings reported in the cited literature must be clearly distinguished from MaintainIQ's experimental results:

1. **Synthetic Nature of Dataset**: The UCI AI4I 2020 dataset was generated using a phenomenological physical simulator [1], [5]. While it realistically reflects theoretical milling machine physics, synthetic sensor data inherently lacks the unpredictable stochastic anomalies, electrical noise, environmental degradation, and maintenance log inaccuracies encountered on real-world factory floors.
2. **No Real-World Mechanical Reliability Guarantees**: MaintainIQ's empirical test results (e.g., XGBoost achieving a PR-AUC of 0.8283 and F1 of 0.7376 on the held-out test split) reflect performance within the simulation distribution. **These results do not establish real-world industrial reliability guarantees** and should not be deployed in live production environments without prior calibration on physical equipment telemetry.
3. **Statistical Reliance vs. Physical Causation**: Feature importance scores and SHAP attributions quantify the statistical dependencies utilized by the mathematical estimator to minimize loss. They reflect correlation and predictive utility within the dataset, not physical or mechanical causation.

---

## 5. References

[1] S. Matzka, "Explainable Artificial Intelligence for Predictive Maintenance Applications," in *Proceedings of the 2020 Third International Conference on Artificial Intelligence for Industries (AI4I)*, Irvine, CA, USA, 2020, pp. 69–74. DOI: [10.1109/AI4I49448.2020.00023](https://doi.org/10.1109/AI4I49448.2020.00023).

[2] T. P. Carvalho, F. A. A. M. N. Soares, R. Vita, R. da P. Francisco, J. P. Basto, and S. G. S. Alcalá, "A systematic literature review of machine learning methods applied to predictive maintenance," *Computers & Industrial Engineering*, vol. 137, art. no. 106024, Nov. 2019. DOI: [10.1016/j.cie.2019.106024](https://doi.org/10.1016/j.cie.2019.106024).

[3] M. Alnahhal, M. I. Tabash, S. K. Safi, M. S. M. Al-Absy, and Z. Mamadiyarov, "A Comparative Study of Imbalance-Handling Methods in Multiclass Predictive Maintenance," *Computation*, vol. 14, no. 4, art. no. 88, 2026. DOI: [10.3390/computation14040088](https://doi.org/10.3390/computation14040088).

[4] A. Theissler, J. Pérez-Velázquez, M. Kettelgerdes, and G. Elger, "Predictive maintenance enabled by machine learning: Use cases and challenges in the automotive industry," *Reliability Engineering & System Safety*, vol. 215, art. no. 107864, Nov. 2021. DOI: [10.1016/j.ress.2021.107864](https://doi.org/10.1016/j.ress.2021.107864).

[5] Z. Marmoucha, M. El Khaili, A. Soulhi, H. El Hailouch, and H. Nhaila, "Enhancing Predictive Accuracy and Operational Performance Based on Entropy–TOPSIS Framework for Machine Learning Model Selection for Predictive Maintenance," *Applied Sciences*, vol. 16, no. 17, art. no. 8666, 2026. DOI: [10.3390/app16178666](https://doi.org/10.3390/app16178666).

[6] S. Chatterjee and Y.-C. Byun, "Highly imbalanced fault classification of wind turbines using data resampling and hybrid ensemble method approach," *Engineering Applications of Artificial Intelligence*, vol. 126, art. no. 107104, Nov. 2023. DOI: [10.1016/j.engappai.2023.107104](https://doi.org/10.1016/j.engappai.2023.107104).

[7] M. H. Abidi, M. K. Mohammed, and H. Alkhalefah, "Predictive Maintenance Planning for Industry 4.0 Using Machine Learning for Sustainable Manufacturing," *Sustainability*, vol. 14, no. 6, art. no. 3387, Mar. 2022. DOI: [10.3390/su14063387](https://doi.org/10.3390/su14063387).

[8] T. Saito and M. Rehmsmeier, "The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets," *PLOS ONE*, vol. 10, no. 3, art. no. e0118432, Mar. 2015. DOI: [10.1371/journal.pone.0118432](https://doi.org/10.1371/journal.pone.0118432).

[9] S. M. Lundberg, G. Erion, H. Chen, A. DeGrave, J. M. Prutkin, B. Nair, R. Katz, J. Himmelfarb, N. Bansal, and S.-I. Lee, "From local explanations to global understanding with explainable AI for trees," *Nature Machine Intelligence*, vol. 2, no. 1, pp. 56–67, Jan. 2020. DOI: [10.1038/s42256-019-0138-9](https://doi.org/10.1038/s42256-019-0138-9).
