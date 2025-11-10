# MDFANet: A Multi-Dimensional Feature Aggregation Network for Electric Vehicle Charging Demand Prediction

## Project Introduction
This project is the official code repository for the paper "MDFANet: A Multi-Dimensional Feature Aggregation Network for Electric Vehicle Charging Demand Prediction", focusing on the **Electric Vehicle (EV) charging demand prediction task in urban areas**. We propose the Multi-Dimensional Feature Aggregation Network (MDFANet) to improve the accuracy of charging demand prediction by fusing spatiotemporal and multi-source heterogeneous features, providing decision support for EV charging facility planning and power resource scheduling.


## Method Overview
MDFANet innovatively designs a **multi-dimensional feature aggregation module** and **multi-GAT (Graph Attention Network) components**, enabling:
- Effective fusion of multi-source temporal features (e.g., charging load, price);
- Accurate capture of spatial dependency relationships between urban regions;
- Output of charging demand predictions for each region, supporting both short-term and long-term prediction tasks.


## Datasets
We conduct comparative experiments on two datasets:
- **ST-EVCDP Dataset(https://github.com/IntelligentSystemsLab/ST-EVCDP)**: A spatiotemporal dataset for charging demand prediction. Users need to download and organize this dataset, then save it in the datasets folder. The dataset contains temporal features (charging load, price, etc.) for multiple regions and spatial correlation information between regions.
- **UrbanEV Dataset(https://github.com/IntelligentSystemsLab/UrbanEV)**: A city-level EV charging behavior dataset. Users need to download and organize this dataset, then save it in the datasets_UrbanEV folder. This dataset is used to verify the model's generalization ability across datasets.

## Environment Requirements
- Python ≥ 3.8
- PyTorch ≥ 2.0
- Install dependencies via:
  ```bash
  pip install -r requirements.txt
 
## run the project 
1. Prepare Data
Place the ST-EVCDP and UrbanEV datasets into the datasets and datasets_UrbanEV folders respectively. For custom data, ensure compatibility with the data-reading logic in mainmdataset.py. 
2. Train and Test
Run the main program to start training and testing:
python main.py
Model checkpoints are saved to checkpoints.
Prediction results are output to results.
