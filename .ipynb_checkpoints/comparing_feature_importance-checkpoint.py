import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

dataset_name = "flchain"
seed = 1009

nadcsm_path = f"/sfs/gpfs/tardis/project/zhangmlgroup/fairness/github_codes/NAMDCSM/nam_feature_importance/feature_importance_seed{seed}_k0_{dataset_name}_NADCSM.csv"
dcsm_path = f"/sfs/gpfs/tardis/project/zhangmlgroup/fairness/github_codes/DCSM/baseline_results/feature_DCSM_importance_{dataset_name}_k1_seed{seed}.csv"
deepcox_path = f"/sfs/gpfs/tardis/project/zhangmlgroup/fairness/github_codes/myDCSM/baseline_results/deepcox_feature_importance/feature_shapdeep_importance_{dataset_name}_seed{seed}.csv"

nadcsm_feature_importance = pd.read_csv(nadcsm_path)
dcsm_feature_importance = pd.read_csv(dcsm_path)
deepcox_feature_importance = pd.read_csv(deepcox_path)

# nadcsm_feature_importance = pd.read_csv('/sfs/gpfs/tardis/project/zhangmlgroup/fairness/github_codes/NAMDCSM/nam_feature_importance/feature_importance_seed1009_k0_flchain_NADCSM.csv')
# dcsm_feature_importance = pd.read_csv('/sfs/gpfs/tardis/project/zhangmlgroup/fairness/github_codes/DCSM/baseline_results/feature_DCSM_importance_flchain_k1_seed1009.csv')
# deepcox_feature_importance = pd.read_csv('/sfs/gpfs/tardis/project/zhangmlgroup/fairness/github_codes/myDCSM/baseline_results/deepcox_feature_importance/feature_shapdeep_importance_flchain_seed1009.csv')

dcsm_feature_importance = dcsm_feature_importance.rename(columns={"Feature_Name": "Feature"})

nadcsm_feature_importance = nadcsm_feature_importance.sort_values(by='Feature')
dcsm_feature_importance = dcsm_feature_importance.sort_values(by='Feature')
deepcox_feature_importance = deepcox_feature_importance.sort_values(by='Feature')

print(nadcsm_feature_importance.columns)
print(dcsm_feature_importance.columns)
print(deepcox_feature_importance.columns)

merged = nadcsm_feature_importance.merge(dcsm_feature_importance, on='Feature', suffixes=('_nadcsm', '_dcsm'))
merged = merged.merge(deepcox_feature_importance, on='Feature', suffixes=('', '_deepcox'))

print(merged.columns)

nadcsm_vector = merged['Normalized Importance_nadcsm'].to_numpy()
dcsm_vector = merged['Normalized Importance_dcsm'].to_numpy()
deepcox_vector = merged['Normalized Importance'].to_numpy()

vectors = [nadcsm_vector, dcsm_vector, deepcox_vector]

cosine_sim_matrix = cosine_similarity(vectors)

methods = ['NADCSM', 'DCSM', 'DeepCox']
cosine_sim_df = pd.DataFrame(cosine_sim_matrix, columns=methods, index=methods)

# Display cosine similarity matrix with labels
print("Cosine Similarity Matrix:")
print(cosine_sim_df)

# plotting
features = merged['Feature']
nadcsm_importance = merged['Normalized Importance_nadcsm']
dcsm_importance = merged['Normalized Importance_dcsm']
deepcox_importance = merged['Normalized Importance']

# Plot feature importance
plt.figure(figsize=(15, 6))
plt.scatter(features, nadcsm_importance, label='NADCSM', color='blue')
plt.scatter(features, dcsm_importance, label='DCSM', color='green')
plt.scatter(features, deepcox_importance, label='DeepCox', color='red')

# Improve plot aesthetics
plt.xticks(rotation=90)
plt.xlabel('Features')
plt.ylabel('Normalized Importance')
plt.title('Feature Importance Comparison')
plt.legend()
plt.tight_layout()

output_path = f"/sfs/gpfs/tardis/project/zhangmlgroup/fairness/github_codes/NAMDCSM/comparison_plots/{dataset_name}_feature_importance_comparison_seed{seed}.png"
plt.savefig(output_path)
# plt.savefig('/sfs/gpfs/tardis/project/zhangmlgroup/fairness/github_codes/NAMDCSM/comparison_plots/PBC_feature_importance_comparison.png')
# Show plot
plt.show()
