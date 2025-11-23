#  Weather–Driven Electricity Demand Analytics  
### A Data-Driven Study Connecting Weather Conditions with State-Wise Electricity Demand

This repository contains the complete pipeline for analyzing and modeling the relationship between **weather variables** and **electricity demand** across Indian states.  
It includes the cleaned merged dataset, preprocessing scripts, EDA notebooks, ML models, and final results.

---

## Dataset Used

The project uses a merged and cleaned dataset:

```
/mnt/data/PSP_Weather_Merged_EDA_Cleaned.csv
```

This dataset combines:

- Daily electricity demand data (max demand, shortages, energy met)
- Daily weather indicators (temperature, humidity, rainfall)
- Feature-engineered time components (year, month, weekday, season)

**Total rows:** 31,177  
**Total columns:** 18  

---

##  Column Summary

### Electricity Demand Features
- `Max_Demand_Met_MW`
- `Shortage_MW`
- `Energy_Met_MU`
- `Drawal_Schedule_MU`
- `OD_UD_MU`
- `Max_OD_MW`
- `Energy_Shortage_MU`

###  Weather Features
- `Temp_Max`
- `Temp_Min`
- `Temp_Avg`
- `Humidity`
- `Rainfall`

###  Engineered Time Features
- `Date`
- `Year`
- `Month`
- `Weekday`
- `Season`
- `State`

---

##  Project Structure

```
 weather-electricity-demand-analysis
│
├── data/
│   └── PSP_Weather_Merged_EDA_Cleaned.csv
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_results_visualization.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── utils.py
│   └── visualization.py
│
├── outputs/
│   ├── correlation_plots/
│   ├── model_results/
│   └── final_graphs/
│
├── docs/
│   └── project_report.pdf
│
├── requirements.txt
└── README.md
```

---

##  Key Objectives

✔ Understand how weather patterns influence electricity demand  
✔ Build machine learning models for demand forecasting  
✔ Perform state-level comparative analysis  
✔ Identify seasonal, temperature-driven, and humidity-driven demand fluctuations  

---


##  Contributing

Pull requests, suggestions, and improvements are welcome!

1. Fork the repository  
2. Create a feature branch  
3. Commit your changes  
4. Open a PR  



## Contact

For queries or collaboration:  
Priyam Patel – (priyamptl4@gmail.com)

