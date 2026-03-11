# IDA105-2505468-TANISHA-BAIJU-SHUKLA-
SmartCharging Analytics is an interactive Streamlit dashboard that uncovers EV charging behavior patterns through K-Means clustering, Apriori association rule mining, and multi-method anomaly detection — delivering real-time insights on station demand, costs, ratings, and infrastructure gaps across a fully filterable dataset.

# SmartCharging Analytics
### Uncovering EV Behavior Patterns 
---

## Project Scope

This project analyses global EV charging station data as part of Scenario 2: SmartCharging Analytics. The goal is to uncover behavioral patterns in EV charging infrastructure by clustering station types, detecting anomalies, and discovering association rules between station features and demand. All findings are delivered through an interactive Streamlit dashboard.


---

## Dataset Columns

The dataset includes the following features as outlined in the assignment:

Station ID, Latitude, Longitude, Address, Charger Type, Cost (USD/kWh), Availability, Distance to City (km), Usage Stats (avg users/day), Station Operator, Charging Capacity (kW), Connector Types, Installation Year, Renewable Energy Source, Reviews (Rating), Parking Spots, Maintenance Frequency.

---

## Key Preprocessing Steps

- **Missing Values**: Median imputation applied to `Reviews_Rating` and `Cost_USD_kWh` — robust to outliers. Categorical nulls in `Renewable_Energy_Source` and `Connector_Types` filled as 'Unknown' to preserve information without assuming a value.
- **Duplicates**: Removed based on `Station_ID`, keeping the first occurrence.
- **Encoding**: Label encoding applied to `Charger_Type`, `Station_Operator`, `Renewable_Energy_Source`, `Connector_Types`, and `Maintenance_Frequency` to convert categorical features to numeric inputs for clustering.
- **Normalisation**: StandardScaler applied to all continuous features (Cost, Usage, Capacity, Distance, Rating, Availability, Parking) to ensure equal contribution in distance-based algorithms.

---

## EDA and Visualisations

- Usage demand histograms by charger type and city
- Station rollout and usage growth over installation years
- Cost distribution boxplots across operators
- Cost vs usage scatter with OLS trendline
- Rating vs distance to city scatter plot
- Renewable vs non-renewable metrics comparison
- Feature correlation heatmap
- Scatter matrix for key numeric features
- Interactive geo-map of all stations (Plotly Mapbox)

---

## Advanced Analysis

### Clustering (Stage 4)

- Algorithm: K-Means with Elbow Method and Silhouette Score for optimal K selection
- Also supports DBSCAN for density-based clustering
- Features used: Usage Stats, Cost, Charging Capacity, Distance to City, Rating, Availability
- PCA used for 2D cluster visualisation with explained variance reported
- Cluster quality measured via Silhouette Score and Davies-Bouldin Score
- Cluster profiles visualised as heatmap and radar chart
- Cluster archetypes identified: High-Power Hubs, Eco Commuter Stops, City Fast-Chargers, Remote Rural Stations

### Association Rule Mining (Stage 5)

- Algorithm: Apriori via mlxtend
- Transactional features: Charger Type, Renewable Source, Usage Band, Operator, Rating Band, Distance Band, Cost Band
- Metrics reported: Support, Confidence, Lift, Leverage, Conviction
- Example rule discovered: DC Fast Charger + Renewable Energy → High Daily Usage
- Results visualised as ranked bar chart and support vs confidence scatter

### Anomaly Detection (Stage 6)

- Three methods applied: Z-Score, IQR fencing, and Isolation Forest
- Method overlap analysis shows how many stations are flagged by one, two, or all three methods
- Stations flagged by all three methods are highest-priority for inspection
- Anomaly scatter shows decision boundaries for each method
- Flagged stations cross-referenced with charger type, operator, and cost

---

## Key Insights

- DC Fast chargers drive significantly higher daily usage compared to AC Level 1 and Level 2 due to faster turnaround times.
- Renewable-powered stations consistently receive higher user ratings, suggesting sustainability influences user perception.
- Stations within 5 km of city centres average higher ratings and usage than remote stations.
- Low-cost stations near city centres are associated with high daily demand — confirmed by association rules.
- Anomalous stations often correspond to specific operators or charger types, pointing to maintenance or metering issues.

---

## Streamlit Dashboard

**Live Link**: [Add your deployed Streamlit link here]

The dashboard includes five navigation sections: Project Scope with live KPIs, Data Preparation with full preprocessing documentation, EDA with six visualisation tabs, Advanced Analysis covering all three techniques, and About with the full rubric and submission checklist. All charts respond to sidebar filters for charger type, city, operator, installation year, and renewable source.

---

## Libraries Used

| Library | Purpose |
|---------|---------|
| pandas | Data cleaning and preprocessing |
| numpy | Numerical operations |
| matplotlib / seaborn | Static visualisations and heatmaps |
| plotly | Interactive charts and geo-maps |
| scikit-learn | K-Means, DBSCAN, PCA, IsolationForest, StandardScaler |
| mlxtend | Apriori association rule mining |
| scipy | Z-score anomaly detection |
| streamlit | Interactive dashboard deployment |

---

## References

- https://www.data-to-viz.com/
- https://neptune.ai/blog/k-means-clustering
- https://dicecamp.com/insights/association-mining-rules-combined-with-clustering/
- https://arxiv.org/pdf/1802.04193
- https://www.researchgate.net/publication/374171696
- https://www.frontiersin.org/journals/energy-research/articles/10.3389/fenrg.2022.773440/full
- https://www.kdnuggets.com/2023/05/beginner-guide-anomaly-detection-techniques-data-science.html
- https://www.youtube.com/watch?v=rCt9DatF63I
