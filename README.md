# Introduction/Background
Aging infrastructure in the United States is not only a safety concern, but an issue that costs taxpayers millions of dollars per year. Captured under the large umbrella of infrastructure is a network of over 600,000 bridges where nearly 40% are 50 years or older [1]. From the 2017 ASCE Infrastructure Report Card, it was determined that 9.1% of bridges are structurally deficient which translates to residents making an average of 188 million trips across structurally deficient bridges every day. These statistics are based on the bridge inspections conducted regularly and recorded in the National Bridge Inventory.

The National Bridge Inventory (NBI) is a publication of the United States Federal Highway Administration (FHWA) which provides records on the location, design, and condition of the nation’s bridge infrastructure. The 2019 NBI dataset provides over 600,000 individual bridge records, each with over 120 data dimensions. Records are available for each year since 1992, with standardized records available since 1995 when FHWA published Recoding and Coding Guide for the Structure Inventory and Appraisal of the Nation’s Bridges.

<br>

# Problem Definition
The volume of available data over time and numerous, well defined dimensions allows for the application of several machine learning techniques to the NBI. Through unsupervised machine learning, this project will provide insight into the commonalities between bridges across differing geographic, jurisdictional, design, and condition descriptors, and the computational procedures for identifying outlier bridges from the database for further study. Supervised machine learning techniques will identify the most predictive factors for bridge deterioration and generate a fitted regression model for predicting bridge condition based on other factors. Specifically, these methods will allow researchers and transportation officials to predict the condition ratings of bridges before or without performing inspections. This could potentially reduce the needed inspection frequency for certain bridges, increase inspection frequency for critical bridges, and identify errors or jurisdictional inconsistencies in reporting bridge condition.
More broadly, we aim to create a tool to identify at-risk bridges based on the data from the NBI in order to prioritize or inform bridge inspections and their frequency, effectively prepare for the possible effects of natural disasters on at-risk bridges, and enable more efficient allocation of government infrastructure funding. Ultimately, this would create a safer network of bridges and build confidence in US infrastructure.

<br>

# Data Collection 
### Collection 
We are working with the 2019 NBI dataset published by the Federal Highway Administration for all the bridges across every state in the United States. This dataset is available as a .csv file on the FHWA webpage and contains a total of 617,0084 data points with 123 data dimensions. Much of the input formatting of this data predates widespread digital data recording and as such, some entries must be preprocessed to obtain a relevant meaning that can be used in machine learning algorithms. For example, the most recent bridge inspection date is encoded as a 3 or 4 digit number which represents the month and year of the most recent inspection.

<br>

### Initial Data Inspection
Figure 1 shows the distribution of missing values in our entire dataset, and Figure 2 shows only those features which contain missing values. Both figures demonstrate that select features are very sparse while others have a small number of missing features. The degree of sparsity was used in the feature selection process for subsequent models. Please refer to our [linked NBI descriptors spreadsheet](https://drive.google.com/file/d/1NPgD0GjzzrtIOeWBEiPuj45N4CNN--eP/view?usp=sharing) for an explanation of the features and the assessment of its relevance.  Those features which have a large number of missing features were determined to not be highly relevant to the determination of a bridge's integrity. 

![figure 1](figure1.png){:height="100%" width="100%"}
*Figure 1. A bar chart of the raw data set where existing data is represented in gray and missing values are represented in white.*
 
![figure 2](figure2.png){:height="100%" width="100%"}
*Figure 2. A bar chart of features which have missing values.*

<br>

### Feature Selection 
With 123 data dimensions, it is possible that there may be too many dimensions to obtain relevant results from machine learning algorithms or that the results may be deceptive and not reflect any actual field conditions. With this possibility in mind, the dimensions of the dataset were initially reduced to only those with a potentially relevant engineering meaning. This process is not to be confused with dimensionality reduction. The assessment of relevant features was based entirely on engineering judgement by members of the group who hold civil engineering degrees. While not perfect, there is reason to believe that this simplification improves the quality of the dataset. Each dimension and our assessment of its relevance can be found [here](https://drive.google.com/file/d/1NPgD0GjzzrtIOeWBEiPuj45N4CNN--eP/view?usp=sharing). In general, dimensions related to administrative bureaucracy were discarded from the dataset (ex: route designation), alongside dimensions with no clear relationship to bridge condition (ex: clear distance between the abutments and vegetation).
 
Since the goal of this project is to predict condition scores based on other factors, the group selected the five condition scores: deck, superstructure, substructure, channel, and culvert to use as labels for supervised and semi supervised learning. These label conditions present a challenge. Conditions are ranked 0-9, however some conditions have a rank order with 9 representing excellent condition and 0 representing failure, while others do not have this ordered progression of conditions. Assessing and analyzing these scores is difficult to do with clustering algorithms but may lend itself well to future Bayes classification and regression models.
 
Below is a covariance heatmap for all of the relevant features and condition scores for our data. Features such as the year built, number of traffic lanes, structure type, scour critical values and high truck traffic values seems to have an impact of the condition scores for the bridge. This initial analysis uncovers relationships between features that might be important to consider as we conduct further work. 

![covariance matrix](cov%20matrix.png){:height="100%" width="100%"}
*Figure 3. Covariance heatmap of selected relevant features and condition scores.*

<br>

### Cleaning 
After selecting the relevant features, features containing null values were identified. It was found that the Year Reconstructed and Percent Average Daily Traffic (ADT) of Trucks columns both had high numbers of null values (Figure 4). It was assumed that the Year Reconstructed null values meant that the bridge had never been reconstructed. For Percent ADT Truck we assumed null values signified a percentage of zero. After accounting for these null values and clearing the data for the remaining null values only 78 data points out of the more than 600,000 data points were lost.

![figure 4](figure4.png){:height="100%" width="100%"}
*Figure 4. Bar chart of missing values for features Year Reconstructed and Percent ADT Truck.*

<br>

### Feature Engineering 
Some columns contained a mixture of integer and string values. For example, the scour critical values ranged from 0 to 9 and also contained values of N, U, or T. To adjust our data so that it contained only numeric representations, string values were mapped to numeric values not already contained within the original scale. Similar changes were made to the fracture, underwater inspection, and special inspection features. This ensured a completely numerical data set that did not contain any null values and was ready for analysis. 
 
<br>

### Normalization 
Because there are a wide variety of features, many of which have vastly different scales, it is important that the selected features are normalized. Some features have simple scales only a few digits in size, while other features, like location coordinates and age have much wider ranges. Normalization will ensure that each chosen feature has the same weight or contribution in each clustering calculation. 
 
<br>

### Mapping Functions
The data in the NBI is coded with short numerical or alphabetical codes representing complex strings or slightly different numerical values depending on the context. To get around this issue, mapping functions were created for each relevant data dimension that relates the code in the NBI dataset to its literal meaning. For example, the NBI codes for longitude and latitude which are present in the dataset as a series of eight or nine digits ([x]xxxxxxxx), represent the geographical coordinates of the bridge in the format [x]xx degrees, xx minutes, xx.xx seconds. This information needs to be properly processed by converting it to decimal degrees before it can be meaningful in any algorithm. Some information is not as complex to decode. An example of this would be the type of structural system of each bridge which is a two digit number that maps directly to a string description of the structure type. These functions are critical for interpreting the results of clustering and other algorithms.

<br>

# Methods 
## Unsupervised Methods 

<br>

### KMeans
The first clustering attempt uses KMeans as it is one of the simplest clustering algorithms available. This implementation uses the scikit learn clustering library, where a fixed number of clusters can be specified for the algorithm to compute clusters for each datapoint. 

The optimal number of clusters is unknown, therefore iteration over possible numbers of clusters to determine which might be best to use was executed. First, KMeans was run on our dataset, which takes each datapoint and assigns it to the nearest centroid k. This was repeated for each point. Then the resulting cluster labels for each point are compared with the original data using a silhouette coefficient. This is recorded for each step and the process is restarted with k+1 fixed clusters. After many iterations, the silhouette coefficients are graphed and compared to find the maximum value. The maximum value indicates the cluster k with highest cohesion and separation. This clustering is then used with fixed max k for further analysis.

![kmeans silhouette](kmeansilhouette.PNG){:height="100%" width="100%"}
*Figure 5. Plot of silhouette coefficients vs number of clusters.*

<br>

### PCA 
Before moving on to supervised learning techniques, it might be important to learn if we can further reduce the dimensionality of our data. For this, PCA was applied to our data set. We looked at the number of PCA components that account for 90% of the variance to understand whether a significant number of our features can be dropped. We also looked at what features had the highest loading factors for the first principal component to gain a better understanding about what features produce the most variance in our dataset. 

<br>

## Supervised Methods

<br>

### Random Forest 
Following data collection steps, we began by implementing a random forest classifier using sklearns ensemble library. The random forest classifier takes all relevant features described above, and condition scores. These elements were split into feature sets and label sets, then divided into training and test sets with a 90/10 split. Labels were then encoded in one hot format using sklearn’s MultiLabelBinarizer. This resulted in a one hot label of length 55 where all condition scores were presented in a linear fashion for each of our 5 condition scores and their 9 respective values. The MultiLabelBinarizer allows for multiple elements in this encoding to be set to 1 while the rest are set to 0; thus the condition score value for each of our 5 conditions are simultaneously represented in the one hot encoding. 

After initial attempts, we obtained less than ideal accuracies when applying the classifier to this set. This rather dense encoding may not be suitable for the methods we apply, so we decided to create independent classifiers for each of the 5 condition scores. One hot encodings were created once again of length 11 for each of the conditions. However, applying the random forest classifier to these encodings was again unsuccessful with poor accuracies. 

Following this attempt, further examination of our data was required. Notice that the distribution of scores in each of the labels is far from balanced below:

![image](1.PNG){:height="100%" width="100%"}

After this relization, we concluded that a simple estimate of exact score may not be passible with the given data and labels. Classifiers were far overpredicting higher scores resulting on very low accuracy for examples with low condition scores in each of our 5 categories. We needed to find a better method to organize these labels and began by considering our intentions and purpose for building a classifier to begin with; we want to provide a means for which inspectors and municipalities can easily determine the general condition of a bridge using the features of a bridge. Considering this, we decided that we simply needed to tell inspectors of the condition was good, bad, or requiring further manual inspection. To do this with the provided labels, we grouped labels into 3 distinct groups for each of our 5 condition types: Good condition, Bad condition, and NA condition (requiring manual inspection). Good condition scores were formed from the top 3 best scores (7, 8, 9) which will likely need no attention as these scores represent a bridge in excellent health. Bad scores are reprentative of the bottom 6 score values (0, 1, 2, 3, 4, 5). These bad scores may need examination and maintenance in a short timespan. Finally, NA scores represented examples with sparse features and labels, which will require manual inspection to obtain a proper score. See the newly created distribution of labels in our augmented labels set below and noice a more evenly distributed set of labels in each category:

![image](2.PNG){:height="100%" width="100%"}

With these new label sets we obtained more respectable accuracy, which is detailed in out random forest results section. We attempted forming random forests of size 50, 100, and 150 using sklearn’s RandomFoestClassifier and found that forests of size 100 produced the most favoable results across the board.

<br>

### Neural Net
In our attempts to build a simple neural net, we began by trying to create independent models for each of our condition types. Data is formatted and segregated into train and test splits with the same process used for random forests. One hot encoding for each condition type were also produced in a likewise manner. The input shape for all models was an array of length 29, which represents all of the relevant features included. The structure of this initial model was as follows:

![image](3.PNG){:height="100%" width="100%"}

This structure is composed of two fully connected dense layers, and followed by a logits layer, which will give the probability of an example belonging to one of 11 distinct classes. After examination of initial results for this model, we encountered the same issues as with random forests in regard to balanced labels. So, we reverted to the same approach used to form an augmented label set detailed above in the random forest section. A new model structure was necessary, as detailed below:

![image](4.PNG){:height="100%" width="100%"}

Notice that the logit layer shape decreases from 11 to 3, to correspond the the Good, Bad, and NA labels used from this point forward. This model produced more respectable results, with accuracies in the 50% to 60% range. Further refinement of the model structure and training process was necessary to improve accuracies. We begin by adding one more dense layer and increasing the size of dense layers. This may provide more opportunity for the model to capitalize on higher order features. We tried from 3 to 5 dense layers all with varying sizes ranging from 128 down to 16. Ultimately we found the following structure to be most successful:

![image](5.PNG){:height="100%" width="100%"}

This structure is composed of 3 dense layers of deceasing size followed by our logits layer. All models used binary cross entropy as a loss metric, and we adjusted other training parameters including optimizers, epochs, and batch size in various combinations as well. We tried Stocahstic Gradient Descent and Adaptive Momentum Estimation optimizers and found Adaptive Momentum Estimation to be most successful in converging and providing good accuracies. We believe this may be due to an ability of the adam optimizer to use adaptive learning rates, or independent rates for varios connections in the network. We tried epochs from 50 to 500, finding that 100 provided the best accuracies without overfitting, and we also tried batch sizes from 1,000 to 100,000 finding that higher batch sized generally resulted in quicker convergence at the loss of a more generalized model. Thus we opted to use batch sizes of 10,000 in order to have a balance of each. Using this process, we formed 5 independent models for each condition type. The performance of these models is detailed in the results section below.

<br>


# Results
## Unsupervised

<br>

### KMeans
The following are the visualization of results from KMeans clustering. The motivation behind using clustering is that clusters may provide insight into common features among various groups of bridges.

Figures 7 through 10 are visualizations of results from KMeans clustering. 

Figure 7 is an example of the way that the KMeans algorithm imperfectly captures the effects of different state jurisdictions. For clusters 3-10, state is a poor predictor of cluster; each cluster is represented across a wide variety of states. However, for clusters 1 and 2, state is highly predictive, and not in a way that is scientifically useful. Because KMeans uses distance and a metric for evaluating similarity, each state must be assigned a numerical value in the algorithm if the user wants to make comparisons across states. Unfortunately this results in clustering according to state by alphabetization. Since the states are not named alphabetically according to their geographic proximity or any other metric, this result is entirely arbitrary and useless.

Figure 8 is an example of successfully incorporating jurisdictional features in a clustering algorithm, albeit with little added insight; the bridges’ owner is not at all predictive of what cluster the bridge may fall into. This is a surprising and somewhat disappointing result. It was expected that differences in funding, priorities, age of infrastructure and geographic location would result in significant differences in the types, sizes, and conditions of bridges across different types of owners. The results of KMeans clustering indicate that these differences, if they do exist, are not significant relative to other factors. The lack of information in this result is itself informative.

Figure 9 is an example of successfully using a feature in KMeans clustering. The design load of a bridge is somewhat, but not completely, predictive of which cluster it belongs to. This makes intuitive sense because bridges which are designed for different loads will have vastly different design parameters. However, bridges designed for the same load will still show variability in other factors such as what material they are made of, their traffic volumes, etc. This is a positive result that indicates that clustering is working properly for the design load feature.

Figure 10 is a representation of the clusters produced by KMeans clustering. Inter-state as well as intra-state geographic clustering is an encouraging result that suggests that environmental as well as jurisdictional factors are influencing the clustering algorithm. Further analysis will attempt to explain these influences. Also interestingly, the clusters shown in Figure 10 do not necessarily match well with Figure 11 which shows bridges by their structure type. This result indicates that clustering is working beyond obvious distinguishing characteristics and is determining deeper, more convoluted clusters than would be determined by human inspection of the data.


![kmeans state](KMeans-state-clusters.png){:height="100%" width="100%"}
*Figure 7. Visualization of KMeans clustering by state.*

![kmeans owner](KMeans-owners-cluster.png){:height="100%" width="100%"}
*Figure 8. Visualization of KMeans clustering by owner.*

![kmeans design load](KMeans-designload-clusters.png){:height="100%" width="100%"}
*Figure 9. Visualization of KMeans clustering by design load type.*

![kmeans geographic location](KMeans-latlon-clusters.png){:height="100%" width="100%"}
*Figure 10. Visualization of KMeans clustering by geographic location.*

![kmeans geographic location structure](structurekind-for-comparision.png){:height="100%" width="100%"}
*Figure 11. Visualization of structure type by geographic location.*

<br>

### PCA 
Below is the screen plot generated from applying PCA to our data set. The first 20 components can explain 90% of the variance which means that we cannot actually reduce our data set by too many features as we apply supervised methods. 

![image](6.PNG){:height="100%" width="100%"}

It can also be valuable to see what features make up the first principal component to see what features have a large influence on differentiating data points. Below is a graph of the loading scores for the features that make up the first principal component. 

![image](7.PNG){:height="100%" width="100%"}

As expected, average daily traffic (ADT), as well as future ADT were significant splitting information. These are two of the most important features of any bridge structure’s design. Of the top 8 most contributory features, the six which do not capture an ADT metric, hold information about the surface area of the deck and the length of the bridge. Roadway width, deck width, and traffic lanes, in conjunction with ADT metrics, are expressing the relationship between how much traffic there is on a bridge, relative to that bridge's maximum traffic carrying capacity. From an engineering standpoint, bridges with high traffic volumes and low capacity would be expected to deteriorate faster than other bridges with the same designs. This is a highly logical and sound splitting judgement that PCA has unveiled. Furthermore, deck area, structure length, and max span length are all related to the length of a bridge. Long bridges with high traffic volumes and low capacity would be the bridges that engineers would naturally be most concerned about. This is good to note as longer bridges are more likely to be critical infrastructure components. Often very long bridges are the only alternative crossing over an obstacle for many miles and thus they are likely to carry a very high traffic volume as well as nearly all truck traffic in these areas. Again, PCA selecting bridge length as an important feature is logical and useful.

Surprisingly, percent ADT truck is less significant than expected. When designing pavement surfaces and other roadway infrastructure, the number of trucks passing over an area is generally the most critical aspect of the design. Relative to trucks, cars and other vehicles are unimportant for bridge and pavement design. It is likely that the percentage of trucks on any given bridge is not disparate enough to make useful splitting decisions and that the effects of more truck traffic on a bridge are captured within the ADT metric.

Also surprising is that year of construction or reconstruction does not appear to be as significant as other metrics. The size, bridge type, and materials used in bridges have all changed dramatically over the time span presented in this dataset. As well, even the best maintained bridges deteriorate over time, changing their expected condition score. Therefore it is surprising that year was not higher on the list of principal components.

<br>

## Supervised 

### Random Forest 
Following the various methods described above, we created independent Random Forest Classifiers for each of our condition types. We will examine these forests one at a time. For each random forest we begin by examining the feature importances of all features using in the process. This examination gives insight into the most impactful features on the classification of our bridge in the forest. Next we examine the accuracies of each forest with the help of sklearn’s ClassificationReport.

Notice that overall accuracies for each of our condition type models range from 85% to 95%. This seems to be a respectable average, however, further examination of class based accuracies is necessary to fully understand random forest performances. Notice that class based scores for our NA (2) class performed excellent, in many cases perfect. While this is an added bonus, our primary objective is to produce accurate predictions for GOOD and BAD bridge examples. GOOD (0) predictions generally have scores ranging from 70% to 80% while BAD (1) range from 80% to 85%. While these are far from perfect, we think these scores are acceptable for predictions, given the data available. With these models, inspectors and municipalities might be able to identify BAD bridges that may need attention in a short timespan. They can also identify GOOD bridges which may not need attention for some time. This would allow inspectors to municipalities to more effectively distribute their time and efforts in bridge analysis and maintenance, which is the primary motivation for creating these random forests.


*Deck Condition Random Forest*

![image](drf1.PNG){:height="100%" width="100%"}

![image](drf2.PNG){:height="100%" width="100%"}

Predictions of deck condition based on the random forest model are good and the model is further validated by the relative importance of its features. The most feature in the classification is the type of structure. This is reasonable because the type of structure has a large influence on the design parameters of the deck, for example the thickness of the deck section, vibrations in the deck, joints, and even the material of the deck itself. That the model was able to arrive at this conclusion independently is impressive and indicates a reasonable result has been obtained independent of the accuracy of its predictions.

Year built is also a significant predictor of deck condition because additional weight in the form of asphalt thickness cannot be added to bridge decks over time, and concrete decks must be completely replaced if they’ve failed, deck maintenance can be difficult and expensive. It is therefore very likely that their condition will continue to deteriorate as years go by because maintenance is difficult.

<br>

*Superstructure Condition Random Forest*

![image](srf1.PNG){:height="100%" width="100%"}

![image](srf2.PNG){:height="100%" width="100%"}

For the same reasons as deck condition, the importance of features in predicting superstructure condition are quite reasonable. These two scores are predicted using very similar feature importance. However, notably, ADT metrics are slightly less significant in determining superstructure condition than in determining deck condition. This makes sense because the condition of a bridge's superstructure has more to do with environment, materials and maintenance than repeated loading.

<br>

*Substructure Condition Random Forest*

![image](xrf1.PNG){:height="100%" width="100%"}

![image](xrf2.PNG){:height="100%" width="100%"}


For substructure condition predictions, the most interesting note is the relative significance of structure length in the prediction. The bridge substructure is related to the superstructure and would be expected to have similar influencing factors. However the structure length had a much higher importance in determining substructure condition than superstructure condition. A possible explanation for this is that longer structures, particularly very long bridges such as causeways, have more piers and therefore more opportunities for substructure deterioration than shorter bridges.

<br>

*Channel Condition Random Forest*

![image](crf1.PNG){:height="100%" width="100%"}

![image](crf2.PNG){:height="100%" width="100%"}

Smartly, random forest classification identified scour conditions as by far the most important feature for channel condition classification. The model also reduced the importance of other features relative to other condition classifications such as deck width, which would have little engineering justification to affect scour conditions or the condition of the channel surrounding a bridge. Location, in the form of latitude and longitude also have a very high importance in this model relative to other classifications. This is reasonable because scour conditions have everything to do with water flow around the piers of a bridge and high gradient rivers or poor soil conditions are likely to be clustered geographically.

<br>

*Culvert Condition Random Forest*

![image](urf1.PNG){:height="100%" width="100%"}

![image](urf2.PNG){:height="100%" width="100%"}

Notable in the prediction of culvert condition is that max span length is significantly important. This is so because culverts typically consist of only a single span. Also, random forest modeling has correctly noted that roadway width rather than deck width or number of lanes is the more critical width measurement for culverts. Culverts often extend much further to the sides of a roadway than pavement does since culverts must clear the edges of an embankment or reach another flow channel. This is an impressive and logically correct assessment by the random forest model and builds trust in the reasonableness of this classification.

With the notion that the formation of some scores may be related to others, we wanted to examine the combined feature importance for all condition types too:


![image](8.PNG){:height="100%" width="100%"}

<br>

### Neural Net
Following the methods described in the Neural Net section above, we created 5 independent Classification Nets for each of our condition types. For each net we begin by examining accuracy and loss over time to check that our models are successful in their objective and decrease loss appropriately over epochs. Next we examine accuracies of each forest with the help of sklearn’s ClassificationReport. This allows for overall accuracies, as well as class wise accuracies to be examined. See the results for each model below:

<br>

*Deck Condition Neural Net*

![image](dnn1.PNG){:height="100%" width="100%"}

![image](dnn2.PNG){:height="100%" width="100%"}

<br>

*Superstructure Condition Neural Net*

![image](snn1.PNG){:height="100%" width="100%"}

![image](snn2.PNG){:height="100%" width="100%"}

<br>

*Substructure Condition Neural Net*

![image](xnn1.PNG){:height="100%" width="100%"}

![image](xnn2.PNG){:height="100%" width="100%"}

<br>

*Channel Condition Neural Net*

![image](cnn1.PNG){:height="100%" width="100%"}

![image](cnn2.PNG){:height="100%" width="100%"}

<br>

*Culvert Condition Neural Net*

![image](unn1.PNG){:height="100%" width="100%"}

![image](unn2.PNG){:height="100%" width="100%"}

Notice that in all cases our models converge on the objective to some degree. We consistently observed declining loss and increasing accuracy over duration of model training, but did not notice significant improvement past 100 epochs. Overall accuracies for these models range from 72% to 92%, however, it would be unsuitable to only examine the overall accuracy. When examining the individual class accuracies, we notice that our NA (2)  class performs excellent across the board in each model with precision, recall, and f1 all above 95%. But our BAD (1) class achieves scores between 70% and 80% while GOOD (1) trails slightly behind. This is less than ideal as our primary objective is to estimate condition scores of GOOD and BAD bridges as best as possible. Our random forest classifiers above performed from 10% to 15% better in most cases and we would therefore recommend using these models as of now. However, with further refinement to neural net structure, training parameters, and balanced data these accuracies might improve more. Given the opportunity of continued development, we also believe it is worthwhile exploring the creation of a multi classification model which handles all 5 condition types simultaneously. This kind of model may experience increased accuracies as a result of constructive interference, which we believe would be the case due to the high level of joint variability of various condition scores represented in the covariance matrix discussed in our data section above.

<br>

# Discussion
Clustering evaluation has shown that some structural and loading features are able to create meaningful clustering relationships as evidenced in the latitude-longitude scatter plots (Figures 10 and 11). However, clustering was more sensitive to jurisdictional differences than intended. An example of this is the clustering by states shown for KMeans clustering (Figure 7). The largest two clusters are distinguished by states ranked in alphabetical (and therefore numerical) order. This indicates that these two groups are being clustered predominantly by their alphabetical proximity to other states, which is not meaningful from an engineering standpoint. Because clustering uses distance between values, there is no way to properly incorporate jurisdictional differences into the clustering algorithm. 

Principal component analysis by contrast, was able to correctly identify the most important distinguishing features between bridges. PCA indicates that feature reduction would likely create a single feature which describes the relationship between the amount of traffic on a bridge and the bridge’s capacity to carry traffic. This is a very logical conclusion because high traffic on a bridge designed for lower levels of traffic would be highly predictive of damage or the need for replacement of that bridge. Overall, PCA was very successful.

In the average feature importances of random forest classification models, two of the most important features were structure type and year built. This is quite a significant result. One of the groups aims with this project was to develop a tool that structural designers and regional planners could use to make smart choices as to the types of infrastructure they construct as well as predictions about when that infrastructure may need rehabilitation or replacement. This model is accomplishing both of those things by virtue of the fact that changing the structure type or year built of an input, while keeping other features constant, is likely to have a significant impact on the predicted condition ratings of the bridge. This means that designers could modify these two parameters to predict how the condition ratings of a bridge may change over time after accounting for other factors which they cannot change, such as geography. The random forest model is simple to understand and offers a good framework for implementing condition prediction models in practice.

The neural network model was able to predict condition scores in similar patterns to the random forest model albeit with less accuracy on some classes of bridges. This is not ideal from a classification standpoint but is nonetheless helpful from an engineering standpoint. This model may be even more useful than the random forest model but in a different way. The neural nets logits layer produces not just a binary representation, but the probability that bridges belong to each class. We can use this characteristic to our advantage by considering what these class probabilities represent. Should the logit layer prediction for a bridge consist of [0.9, 0.05, 0.05], then we can say with a rather high degree of certainty that this bridge is a GOOD condition bridge. Likewise for a prediction of [0.05, 0.9, 0.05] for a BAD condition bridge. But, say for example that our models produce a prediction such as [0.39, 0.4, 0.21]; while for the purposes of classification accuracy this would be considered a BAD condition bridge, one can see that these probabilities are far from decisive. One assumption we make about our dataset is that all our examples are correctly labeled, however, all condition scores are labeled by people. Many inspectors rate bridges, and the degree of variability in an inspectors rating may greatly differ for different inspectors. Now considering this variability resulting from human opinion lets consider a modified purpose for this model: identifying outliers. With the assumption that most, but not all condition score labels are correct in our dataset we might use our neural nets to detect these abnormalities. Bridges with condition scores that deviate significantly from their predicted scores are outliers and may have inaccurate labels or feature descriptors which are causing this difficulty in predicting their condition. This may indicate to engineers and inspectors that these bridges should be reinspected for a source of verification or redundancy or may indicate that some special circumstances are causing this bridge to have lower condition scores than might otherwise be expected. This result is very useful for engineers and planners and was a specific goal of this project from the outset. Given more time to explore this application, we might attempt to create pairs of cross validation models using an iterative noisy cross validation method.

Other unsupervised and supervised learning techniques were used on this dataset as well including the gaussian mixture model and support vector machine algorithms. While these models did produce results, their accuracy and usefulness were not as significant as the models presented in this report. Specifically SVM was of limited value and produced rating prediction accuracies in the 45% range, far below the other supervised models. Furthermore the algorithm struggled to converge using more than 10% of the dataset as training data. This limits its applicability to this dataset in general.

<br>

# Conclusion
The National Bridge Inventory (NBI) dataset has provided our team with a vast amount of data to analyze and build functional models with. This volume of data and selection of many features has allowed several machine learning techniques to be applied. After initial data inspection, formatting, and cleaning, we proceeded to implement unsupervised learning methods.

Through unsupervised  methods, we gained insight into the commonalities between bridges across differing geographic, jurisdictional, design, and condition descriptors, and the computational procedures for identifying outlier bridges. We learned what features are most important and impactful in bridge analysis. 

Following unsupervised approaches, we also applied supervised methods of machine learning. These methods allowed us to identify predictive factors for bridge health and generate classification models for important bridge condition scores. These predicted bridge condition scores are based on many identified relevant features. Our models might allow inspectors and government to proactively predict the condition ratings of bridges without performing physical inspections. This reduces the necessary inspection frequency for bridges predicted in good condition, and increases inspections for bridges in bad condition. More efficient an focused efforts would certainly benefit both the inspectors and municipalities as well as citizens using these bridges.

Through the methods and results described above, we have identified characteristics of good and bad condition bridges. This allows prioritization of inspections and respective frequencies for bridges of each condition class.They provide opportunity to more efficiently allocate government infrastructure funding and bridge inspector work time, and perhaps more importantly, they promote a safer network of bridges for US citizens.

<br>

# References 

[1] ASCE's 2017 Infrastructure Report Card. 2017. Bridges. [online] Available at: <https://www.infrastructurereportcard.org/cat-item/bridges/> [Accessed: 5 November 2020].
[2] “Public Disclosure of National Bridge Inventory (NBI) Data - National Bridge Inventory - Bridge Inspection - Safety - Bridges & Structures - Federal Highway Administration.” [Online]. Available: https://www.fhwa.dot.gov/bridge/nbi/20070517.cfm. [Accessed: 5 November 2020].
[3] P. Chen, B. Liao, G. Chen, and S. Zhang, “Understanding and Utilizing Deep Neural Networks Trained with Noisy Labels,” arXiv.org, 13-May-2019. [Online]. Available: https://arxiv.org/abs/1905.05040. [Accessed: 08-Dec-2020]. 
