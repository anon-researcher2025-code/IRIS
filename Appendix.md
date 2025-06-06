We provide additional experimental details and results including the KM plots in [Kaplan-Meier (KM) plots](KM_plots/), the imprtant features identified by all methods for Breast Cancer, SUPPORT, FLCHAIN, and EHR datasets in [Important Features](important_features/), and some of the feature calibration functions obtained using IRIS on SUPPORT, FLCHAIN, AV45, and EHR datasets in [Feature Calibration Functions](feature_calibration_functions/).

## KM Plots

The Kaplan–Meier (KM) plots in the folder [Kaplan-Meier (KM) plots](KM_plots/) correspond to the results presented in TABLE II. Each plot illustrates the survival estimates for two identified clusters: the **blue curve** represents higher survival probability, and the **orange curve** represents lower survival probability.

Across all datasets, **IRIS** consistently produces the most distinct separation between survival curves, as reflected in the highest LogRank values, demonstrating its effectiveness in stratifying individuals into high- and low-risk groups.

- In [BreastCancer](KM_plots/BreastCancer/) plots, Deep Cox and CoxNAM perform similarly, while DCSM shows a weaker separation.
- In [SUPPORT](KM_plots/SUPPORT/) plots, both DCSM and CoxNAM achieve strong separation, while Deep Cox exhibits the lowest separation, indicating weaker risk stratification.
- In [FLCHAIN](KM_plots/FLCHAIN/) plots, DCSM performs comparably to IRIS, whereas CoxNAM results in poor clustering.
- In [EHR](KM_plots/EHR/) plots, DCSM shows better performance than Deep Cox and CoxNAM.


## Important Features and Feature Calibration Functions

Tables in [Important Features](important_features/) present the key features extracted by various methods, corresponding to the results in Table III.

IRIS identifies **creatinine**, **monoclonal gammopathy of undetermined significance (MGUS)**, **age**, **lambda**, and **kappa** as the most significant features from FLCHAIN dataset. As shown in Figures of [FLCHAIN Feature Calibration](feature_calibration_functions/FLCHAIN/), the feature calibration functions illustrate the relationships between these variables and mortality risk. Notably, elevated serum creatinine levels at admission are strongly associated with higher mortality rates [[1]](https://doi.org/10.1100/2012/186495), and increased serum free light chains (sFLC), particularly kappa and lambda, are linked to increased mortality risk in individuals with chronic kidney disease (CKD) [[2]](https://www.sciencedirect.com/science/article/pii/S0025619617306252).


Table [SUPPORT Feature Calibration](feature_calibration_functions/SUPPORT/) highlights *level of functional disability* (`sfdm2`) as a crucial predictor identified by IRIS. This feature, after one-hot encoding, maintains high importance across all values. Patients classified under **"<2-month follow-up"** are at the highest severity level, as shown in [Link](feature_calibration_functions/SUPPORT/fim_plot_seed_1009_k0_support_IRIS_sfdm2_<2 mo. follow-up.pdf). For high-risk patients, both decreased respiratory rate [[3]](https://doi.org/10.3238/arztebl.2014.0503) and low bilirubin levels [[4]](https://doi.org/10.1371/journal.pone.0094479) are linked to worsening health outcomes, as reflected in [Link](feature_calibration_functions/SUPPORT/fim_plot_seed_1009_k0_support_IRIS_resp.pdf) and [Link](feature_calibration_functions/SUPPORT/fim_plot_seed_1009_k0_support_IRIS_bili.pdf), respectively.


Table [EHR Dataset Feature Importance](important_features/top_20_features_of_ehr_dataset.png) indicates that IRIS identifies several Social Determinants of Health (SDOH) as important features, including **environment**, **alcohol use**, **psychoactive substance use**, and **tobacco use**. Environmental factors play a critical role in Alzheimer’s Disease and Related Dementias (ADRD) [[5]], and broader indicators like neighborhood deprivation have also been associated with ADRD risk [[6]].

Among behavioral factors:
- **Heavy alcohol consumption** is linked to accelerated cognitive decline in Alzheimer’s patients [3].
- **Psychoactive substance use** has been associated with increased dementia risk and structural changes in the aging brain [[7]](https://doi.org/10.1111/acps.13340).

IRIS effectively captures these relationships, highlighting the connection between alcohol use, psychoactive substances, and injury risk, as shown in [Link](feature_calibration_functions/EHR/fim_plot_seed_666_k1_upenn_IRIS_alcohol_use.pdf) and [Link](feature_calibration_functions/EHR/fim_plot_seed_666_k1_upenn_IRIS_psychoactive_use.pdf). Additionally, **smoking** increases harmful brain stress, contributing to Alzheimer’s disease and memory loss [[8]](https://doi.org/10.1016/j.jalz.2013.04.006). Older adults with a history of smoking often show brain changes resembling those seen in Alzheimer’s, reflecting long-term neurobiological effects. The shape function corresponding to this is shown in [Link](feature_calibration_functions/EHR/fim_plot_seed_666_k1_upenn_IRIS_tobacco_use.pdf).


## References

1. Mehmet Akif Cakar et al., *The Effect of Admission Creatinine Levels on One-Year Mortality in Acute Myocardial Infarction*, The Scientific World Journal, 2012. [Link](https://doi.org/10.1100/2012/186495)
2. Fraser, S. D. S., Fenton, A., Harris, S., et al. (2017). *The Association of Serum Free Light Chains with Mortality and Progression to End-Stage Renal Disease in Chronic Kidney Disease: Systematic Review and Individual Patient Data Meta-analysis*. Mayo Clinic Proceedings, 92(11), 1671–1681. [Link](https://www.sciencedirect.com/science/article/pii/S0025619617306252)
3. Strauß, R., Ewig, S., Richter, K., König, T., Heller, G., & Bauer, T. T. (2014). *The Prognostic Significance of Respiratory Rate in Patients with Pneumonia: A Retrospective Analysis of Data from 705,928 Hospitalized Patients in Germany from 2010–2012*. Dtsch Arztebl Int, 111(29–30), 503–508. [https://doi.org/10.3238/arztebl.2014.0503](https://doi.org/10.3238/arztebl.2014.0503)
4. Ong, K. L., Allison, M. A., Cheung, B. M., Wu, B. J., Barter, P. J., & Rye, K. A. (2014). *The Relationship Between Total Bilirubin Levels and Total Mortality in Older Adults: The United States National Health and Nutrition Examination Survey (NHANES) 1999–2004*. PLoS One, 9(4), e94479. [https://doi.org/10.1371/journal.pone.0094479](https://doi.org/10.1371/journal.pone.0094479)
5. Adkins-Jackson, P. B., George, K. M., Besser, L. M., Hyun, J., Lamar, M., Hill-Jarrett, T. G., Bubu, O. M., Flatt, J. D., Heyn, P. C., Cicero, E. C., *et al.* (2023). **The structural and social determinants of Alzheimer's disease related dementias**. *Alzheimer's & Dementia*, 19(7), 3171–3185.
6. Powell, W. R., Buckingham, W. R., Larson, J. L., Vilen, L., Yu, M., Salamat, M. S., Bendlin, B. B., Rissman, R. A., & Kind, A. J. H. (2020). **Association of Neighborhood-Level Disadvantage With Alzheimer Disease Neuropathology**. *JAMA Network Open*, 3(6), e207559.
7. Tournier, M., Pambrun, E., Maumus-Robert, S., Pariente, A., & Verdoux, H. (2022). **The risk of dementia in patients using psychotropic drugs: antidepressants, mood stabilizers or antipsychotics**. *Acta Psychiatrica Scandinavica*, 145(1), 56–66. [Link](https://doi.org/10.1111/acps.13340)
8. Durazzo, T. C., Mattsson, N., & Weiner, M. W. (2014). **Smoking and increased Alzheimer's disease risk: a review of potential mechanisms**. *Alzheimer's & Dementia*, 10, S122–S145. [Link](https://doi.org/10.1016/j.jalz.2013.04.006)





