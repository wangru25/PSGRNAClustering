# PSGRNAClustering

Identifying novel and functional RNA structures remains a significant challenge in RNA motif design
and is crucial for developing RNA-based therapeutics. In this package, we introduce a computational topology-based approach with unsupervised machine-learning algorithms to estimate the database size and content of RNA-like graph topologies.

Specifically, this package generates computational topology descriptors using the Persistent Spectral Graphs (PSG) method on each graph. One can generate 40 different features using ``python run_all.py``

In this work, we choose 19 features to describe our graphs with vertex ranging from 2 - 9. Readers can also choose their own features for more complicated graphs. The index of 19 features we use are ``[10, 11, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]``. All features are saved in ``/feature`` folder. Details about the theory of Persistent Spectral Graphs can be found in [[1]](https://pubmed.ncbi.nlm.nih.gov/32515170/). Interested readers can also try a complete open-source package called [HERMES](https://github.com/wangru25/HERMES) that is designed for PSG. 

For clustering methods, we apply $k$-means, mini $k$-means, gaussian mixture, hierarchical (ward), spectral, and birch clustering methods, which can be found in ``src/ml_algorithms.py``

All RNA-like and non-RNA-like motifs that are partitioned by our package are all saved in ``/results`` folder. We also apply an inverse folding program called [Dual-RAG-IF](https://github.com/Schlicklab/Dual-RAG-IF/tree/main) to successfully design hundreds of high-likelihood RNA-like motifs. Information about these sequences is deposited in ``/designs`` folder. 

