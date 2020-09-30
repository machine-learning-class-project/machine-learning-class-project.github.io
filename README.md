## Introduction 
The National Bridge Inventory (NBI) is a publication of the United States Federal Highway Administration (FHWA) which provides records on the location, design, and condition of the nation’s bridge infrastructure. Many of the structures in this database are aging and it will be incumbent on the next generation of civil engineers to understand the relevant factors that influence bridge deterioration over time, and develop our ability to predict future deterioration. The 2019 NBI dataset provides over 600,000 individual bridge records, each with over 120 data dimensions. Records are available for each year since 1992, with standardized records available since 1995 when FHWA published Recoding and Coding Guide for the Structure Inventory and Appraisal of the Nation’s Bridges.


## Problem definition 
The volume of available data over time and numerous, well defined dimensions will allow for the application of several machine learning techniques to the NBI. Through unsupervised machine learning, this project will provide insight into the commonalities between bridges across differing geographic, jurisdictional, design, and condition descriptors, and the computational procedures for identifying outlier bridges from the database for further study. Supervised machine learning techniques will identify the most predictive factors for bridge deterioration and generate a fitted regression model for predicting bridge condition based on other factors.

## Methods 
Using both unsupervised and supervised learning, we hope to uncover insights into the similarity, integrity, and lifespan of bridges across the United States. Using clustering techniques, including but not limited to k-means, GMMs, and hierarchical clustering, we hope to group bridges based on their numerous characteristics. These clusters may be useful to inspectors and engineers in understanding features that are common among various groups of bridges, for instance, the features correlated to condition scorings for bridges [2]. Via supervised learning, we hope to predict key indicators of a bridges’ condition which are hard for inspectors to estimate and prone to bias. By means of Random Forests, SVMs, or ANNs, we think making such predictions is quite possible. We hope to produce a classification of bridges’ integrity with such supervised learning methods and discretization of key features from our data set. Using the provided data points in our set, we should be able to produce a condition prediction [1]. By comparing the accuracy and results of these supervised learning methods, we hope to determine a suitable model which might be used in helping inspectors and owners determine the state of bridges. 


## Potential Results 
The predictive aspects of this project will allow bridge inspectors and owners to define their expectations for bridge condition ratings and to identify potential errors in inspection or reporting data, as well as identify bridges that may require additional inspection scrutiny or maintenance. Time-dependent condition prediction models will assist owners in scheduling maintenance as well as making economical decisions about the design, location, and upkeep of their bridge infrastructure. 

## Discussion
Machine learning techniques have been applied to the NBI and to similar datasets before, but it has never included bridges across the entire United States (Bektas 2017). This project will provide insights that policy makers, architects, and engineers can use to make better judgments about their design and maintenance choices, as well as how the bridges in their jurisdictions compare to others across the country. In this project, machine learning is being applied to the NBI to refine the allocation of resources used for bridge construction and maintenance.


## References 
[1] Mosbeh R. Kaloop, Sherif M. El-Badawy, Jungkyu Ahn, Hyoung-Bo Sim, Jong Wan Hu, Ragaa T. Abd El-Hakim. (2020) A hybrid wavelet-optimally-pruned extreme learning machine model for the estimation of international roughness index of rigid pavements. International Journal of Pavement Engineering 0:0, pages 1-15. 

[2] Yajima, Ayako, et al. “A Clustering Based Method to Evaluate Soil Corrosivity for Pipeline External Integrity Management.” International Journal of Pressure Vessels and Piping, vol. 126-127, 2015, pp. 37–47., doi:10.1016/j.ijpvp.2014.12.004. 
