## Introduction/Background
Aging infrastructure in the United States is not only a safety concern, but an issue that costs taxpayers millions of dollars per year. Captured under the large umbrella of infrastructure is a network of over 600,000 bridges where nearly 40% are 50 years or older [1]. From the 2017 ASCE Infrastructure Report Card, it was determined that 9.1% of bridges are structurally deficient which translates to residents making an average of 188 million trips across structurally deficient bridges every day. These statistics are based on the bridge inspections conducted regularly and recorded in the National Bridge Inventory.
The National Bridge Inventory (NBI) is a publication of the United States Federal Highway Administration (FHWA) which provides records on the location, design, and condition of the nation’s bridge infrastructure. The 2019 NBI dataset provides over 600,000 individual bridge records, each with over 120 data dimensions. Records are available for each year since 1992, with standardized records available since 1995 when FHWA published Recoding and Coding Guide for the Structure Inventory and Appraisal of the Nation’s Bridges.

## Problem definition 
The volume of available data over time and numerous, well defined dimensions allows for the application of several machine learning techniques to the NBI. Through unsupervised machine learning, this project will provide insight into the commonalities between bridges across differing geographic, jurisdictional, design, and condition descriptors, and the computational procedures for identifying outlier bridges from the database for further study. Supervised machine learning techniques will identify the most predictive factors for bridge deterioration and generate a fitted regression model for predicting bridge condition based on other factors. Specifically, these methods will allow researchers and transportation officials to predict the condition ratings of bridges before or without performing inspections. This could potentially reduce the needed inspection frequency for certain bridges, increase inspection frequency for critical bridges, and identify errors or jurisdictional inconsistencies in reporting bridge condition.
More broadly, we aim to create a tool to identify at-risk bridges based on the data from the NBI in order to prioritize or inform bridge inspections and their frequency, effectively prepare for the possible effects of natural disasters on at-risk bridges, and enable more efficient allocation of government infrastructure funding. Ultimately, this would create a safer network of bridges and build confidence in US infrastructure.

## Data Collection 
### Collection 
We are working with the 2019 NBI dataset published by the Federal Highway Administration for all the bridges across every state in the United States. This dataset is available as a .csv file on the FHWA webpage and contains a total of 617,0084 data points with 123 data dimensions. Much of the input formatting of this data predates widespread digital data recording and as such, some entries must be preprocessed to obtain a relevant meaning that can be used in machine learning algorithms. For example, the most recent bridge inspection date is encoded as a 3 or 4 digit number which represents the month and year of the most recent inspection.

### Feature Selection 
With 123 data dimensions, it is possible that there may be too many dimensions to obtain relevant results from machine learning algorithms or that the results may be deceptive and not reflect any actual field conditions. With this possibility in mind, the dimensions of the dataset were initially reduced to only those with a potentially relevant engineering meaning. This process is not to be confused with dimensionality reduction. The assessment of relevant features was based entirely on engineering judgement by members of the group who hold civil engineering degrees. While not perfect, there is reason to believe that this simplification improves the quality of the dataset. Each dimension and our assessment of its relevance can be found here. In general, dimensions related to administrative bureaucracy were discarded from the dataset (ex: route designation), alongside dimensions with no clear relationship to bridge condition (ex: clear distance between the abutments and vegetation).
 
Since the goal of this project is to predict condition scores based on other factors, the group selected the five condition scores: deck, superstructure, substructure, channel, and culvert to use as labels for supervised and semi supervised learning. These label conditions present a challenge. Conditions are ranked 0-9, however some conditions have a rank order with 9 representing excellent condition and 0 representing failure, while others do not have this ordered progression of conditions. Assessing and analyzing these scores is difficult to do with clustering algorithms but may lend itself well to future Bayes classification and regression models.
 
Below is a covariance heatmap for all of the relevant features and condition scores for our data. Features such as the year built, number of traffic lanes, structure type, scour critical values and high truck traffic values seems to have an impact of the condition scores for the bridge. This initial analysis uncovers relationships between features that might be important to consider as we conduct further work. 

<img src="cov%20matrix.png",
width="100%"/>
![](cov%20matrix.png){:height="100%" width="100%"}

## Methods 
Using both unsupervised and supervised learning, we hope to uncover insights into the similarity, integrity, and lifespan of bridges across the United States. Using clustering techniques, including but not limited to k-means, GMMs, and hierarchical clustering, we hope to group bridges based on their numerous characteristics. These clusters may be useful to inspectors and engineers in understanding features that are common among various groups of bridges, for instance, the features correlated to condition scorings for bridges [2]. Via supervised learning, we hope to predict key indicators of a bridges’ condition which are hard for inspectors to estimate and prone to bias. By means of Random Forests, SVMs, or ANNs, we think making such predictions is quite possible. We hope to produce a classification of bridges’ integrity with such supervised learning methods and discretization of key features from our data set. Using the provided data points in our set, we should be able to produce a condition prediction [1]. By comparing the accuracy and results of these supervised learning methods, we hope to determine a suitable model which might be used in helping inspectors and owners determine the state of bridges. 


## Potential Results 
The predictive aspects of this project will allow bridge inspectors and owners to define their expectations for bridge condition ratings and to identify potential errors in inspection or reporting data, as well as identify bridges that may require additional inspection scrutiny or maintenance. Time-dependent condition prediction models will assist owners in scheduling maintenance as well as making economical decisions about the design, location, and upkeep of their bridge infrastructure. 

## Discussion
Machine learning techniques have been applied to the NBI and to similar datasets before, but it has never included bridges across the entire United States (Bektas 2017). This project will provide insights that policy makers, architects, and engineers can use to make better judgments about their design and maintenance choices, as well as how the bridges in their jurisdictions compare to others across the country. In this project, machine learning is being applied to the NBI to refine the allocation of resources used for bridge construction and maintenance.


## References 
[1] Mosbeh R. Kaloop, Sherif M. El-Badawy, Jungkyu Ahn, Hyoung-Bo Sim, Jong Wan Hu, Ragaa T. Abd El-Hakim. (2020) A hybrid wavelet-optimally-pruned extreme learning machine model for the estimation of international roughness index of rigid pavements. International Journal of Pavement Engineering 0:0, pages 1-15. 
[2] Yajima, Ayako, et al. “A Clustering Based Method to Evaluate Soil Corrosivity for Pipeline External Integrity Management.” International Journal of Pressure Vessels and Piping, vol. 126-127, 2015, pp. 37–47., doi:10.1016/j.ijpvp.2014.12.004. 
