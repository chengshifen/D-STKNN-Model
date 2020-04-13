# D-STKNN-Model
A dynamic ST-KNN model (D-STKNN) realize short-term traffic forecasting by mining the non-stationary spatiotemporal pattern of road traffic.

Firstly, traffic patterns are automatically identified by the affinity propagation clustering algorithm (AP.m).

Secondly, the Warped K-Means algorithm is used to automatically partition time periods for each traffic pattern (warp_kmeans.m).

Finally, a dynamic STKNN model is constructed based on a three-dimensional spatiotemporal tensor data model for different road segments with different patterns in different time periods (D-STKNN.m).
