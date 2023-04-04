# K-Means-Clustering-using-Iris-and-Wine-datasets
This code performs KMeans clustering on the Iris dataset to classify the three species of flowers. It includes a visualization of the original and predicted labels for the sepal and petal dimensions, as well as an elbow curve to determine the optimal number of clusters.

The code first loads the Iris dataset from scikit-learn, converts it to a Pandas DataFrame, and standardizes the independent variables using StandardScaler. KMeans clustering is then applied to the standardized dataset with k=3, and the predicted labels are converted to match the actual labels.

The accuracy and classification report for the predicted labels are printed. Then, a scatter plot is created for each of the sepal and petal dimensions, with the original labels in one subplot and the predicted labels in another. The original labels are colored using a ListedColormap with red for 0, green for 1, and blue for 2.

Finally, an elbow curve is plotted to help determine the optimal number of clusters. The average within-cluster sum of squares and percentage of variance explained are plotted against the number of clusters, and the optimal number of clusters can be chosen based on the elbow point in the curve.

Note: Some warnings have been suppressed for the sake of clarity.
