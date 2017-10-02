
.. _clustering:

===========
Clustering
===========

.. note::

   Sharpening the knife longer can make it easier to hack the firewood -- old Chinese proverb


K-Means Model
+++++++++++++

Introduction
------------



Demo
----


1. Set up spark context and SparkSession

.. code-block:: python

	from pyspark.sql import SparkSession

	spark = SparkSession \
	    .builder \
	    .appName("Python Spark K-means example") \
	    .config("spark.some.config.option", "some-value") \
	    .getOrCreate()


2. Load dataset

.. code-block:: python

	df = spark.read.format('com.databricks.spark.csv').\
                               options(header='true', \
                               inferschema='true').\
                    load("../data/iris.csv",header=True);

check the data set

.. code-block:: python

	df.show(5,True)
	df.printSchema()

Then you will get 

.. code-block:: python

	+------------+-----------+------------+-----------+-------+
	|sepal_length|sepal_width|petal_length|petal_width|species|
	+------------+-----------+------------+-----------+-------+
	|         5.1|        3.5|         1.4|        0.2| setosa|
	|         4.9|        3.0|         1.4|        0.2| setosa|
	|         4.7|        3.2|         1.3|        0.2| setosa|
	|         4.6|        3.1|         1.5|        0.2| setosa|
	|         5.0|        3.6|         1.4|        0.2| setosa|
	+------------+-----------+------------+-----------+-------+
	only showing top 5 rows

	root
	 |-- sepal_length: double (nullable = true)
	 |-- sepal_width: double (nullable = true)
	 |-- petal_length: double (nullable = true)
	 |-- petal_width: double (nullable = true)
	 |-- species: string (nullable = true)

You can also get the Statistical resutls from the data frame 
(Unfortunately, it only works for numerical). 

.. code-block:: python

	df.describe().show()

Then you will get 

.. code-block:: python

	+-------+------------------+-------------------+------------------+------------------+---------+
	|summary|      sepal_length|        sepal_width|      petal_length|       petal_width|  species|
	+-------+------------------+-------------------+------------------+------------------+---------+
	|  count|               150|                150|               150|               150|      150|
	|   mean| 5.843333333333335| 3.0540000000000007|3.7586666666666693|1.1986666666666672|     null|
	| stddev|0.8280661279778637|0.43359431136217375| 1.764420419952262|0.7631607417008414|     null|
	|    min|               4.3|                2.0|               1.0|               0.1|   setosa|
	|    max|               7.9|                4.4|               6.9|               2.5|virginica|
	+-------+------------------+-------------------+------------------+------------------+---------+

3. Convert the data to dense vector (**features**)

.. code-block:: python

	# convert the data to dense vector
	def transData(data):
	    return data.rdd.map(lambda r: [Vectors.dense(r[:-1])]).toDF(['features'])

4. Transform the dataset to DataFrame

.. code-block:: python

	transformed= transData(df)
	transformed.show(5, False)

.. code-block:: python

	+-----------------+
	|features         |
	+-----------------+
	|[5.1,3.5,1.4,0.2]|
	|[4.9,3.0,1.4,0.2]|
	|[4.7,3.2,1.3,0.2]|
	|[4.6,3.1,1.5,0.2]|
	|[5.0,3.6,1.4,0.2]|
	+-----------------+
	only showing top 5 rows

5. Deal With Categorical Variables

.. code-block:: python

	from pyspark.ml import Pipeline
	from pyspark.ml.regression import LinearRegression
	from pyspark.ml.feature import VectorIndexer
	from pyspark.ml.evaluation import RegressionEvaluator

	# Automatically identify categorical features, and index them.
	# We specify maxCategories so features with > 4 distinct values are treated as continuous.

	featureIndexer = VectorIndexer(inputCol="features", \
	                               outputCol="indexedFeatures",\
	                               maxCategories=4).fit(transformed)

	data = featureIndexer.transform(transformed)                                         

Now you check your dataset with


.. code-block:: python

	data.show(5,True)

you will get

.. code-block:: python

	+-----------------+-----------------+
	|         features|  indexedFeatures|
	+-----------------+-----------------+
	|[5.1,3.5,1.4,0.2]|[5.1,3.5,1.4,0.2]|
	|[4.9,3.0,1.4,0.2]|[4.9,3.0,1.4,0.2]|
	|[4.7,3.2,1.3,0.2]|[4.7,3.2,1.3,0.2]|
	|[4.6,3.1,1.5,0.2]|[4.6,3.1,1.5,0.2]|
	|[5.0,3.6,1.4,0.2]|[5.0,3.6,1.4,0.2]|
	+-----------------+-----------------+
	only showing top 5 rows

6. Elbow method to determine the optimal number of clusters for k-means clustering

.. code-block:: python

	import numpy as np
	cost = np.zeros(20)
	for k in range(2,20):
	    kmeans = KMeans()\
	            .setK(k)\
	            .setSeed(1) \
	            .setFeaturesCol("indexedFeatures")\
	            .setPredictionCol("cluster")
	                
	    model = kmeans.fit(data)
	    cost[k] = model.computeCost(data) # requires Spark 2.0 or later

.. code-block:: python

	import numpy as np
	import matplotlib.mlab as mlab
	import matplotlib.pyplot as plt
	import seaborn as sbs
	from matplotlib.ticker import MaxNLocator

	fig, ax = plt.subplots(1,1, figsize =(8,6))
	ax.plot(range(2,20),cost[2:20])
	ax.set_xlabel('k')
	ax.set_ylabel('cost')
	ax.xaxis.set_major_locator(MaxNLocator(integer=True))
	plt.show()


.. figure:: images/elbow.png
   :align: center

7. Pipeline Architecture

.. code-block:: python

	from pyspark.ml.clustering import KMeans, KMeansModel

	kmeans = KMeans() \
	          .setK(3) \
	          .setFeaturesCol("indexedFeatures")\
	          .setPredictionCol("cluster")

	# Chain indexer and tree in a Pipeline
	pipeline = Pipeline(stages=[featureIndexer, kmeans])

	model = pipeline.fit(transformed)

	cluster = model.transform(transformed)

8. k-means clusters 

.. code-block:: python

	cluster = model.transform(transformed)


.. code-block:: python

	+-----------------+-----------------+-------+
	|         features|  indexedFeatures|cluster|
	+-----------------+-----------------+-------+
	|[5.1,3.5,1.4,0.2]|[5.1,3.5,1.4,0.2]|      1|
	|[4.9,3.0,1.4,0.2]|[4.9,3.0,1.4,0.2]|      1|
	|[4.7,3.2,1.3,0.2]|[4.7,3.2,1.3,0.2]|      1|
	|[4.6,3.1,1.5,0.2]|[4.6,3.1,1.5,0.2]|      1|
	|[5.0,3.6,1.4,0.2]|[5.0,3.6,1.4,0.2]|      1|
	|[5.4,3.9,1.7,0.4]|[5.4,3.9,1.7,0.4]|      1|
	|[4.6,3.4,1.4,0.3]|[4.6,3.4,1.4,0.3]|      1|
	|[5.0,3.4,1.5,0.2]|[5.0,3.4,1.5,0.2]|      1|
	|[4.4,2.9,1.4,0.2]|[4.4,2.9,1.4,0.2]|      1|
	|[4.9,3.1,1.5,0.1]|[4.9,3.1,1.5,0.1]|      1|
	|[5.4,3.7,1.5,0.2]|[5.4,3.7,1.5,0.2]|      1|
	|[4.8,3.4,1.6,0.2]|[4.8,3.4,1.6,0.2]|      1|
	|[4.8,3.0,1.4,0.1]|[4.8,3.0,1.4,0.1]|      1|
	|[4.3,3.0,1.1,0.1]|[4.3,3.0,1.1,0.1]|      1|
	|[5.8,4.0,1.2,0.2]|[5.8,4.0,1.2,0.2]|      1|
	|[5.7,4.4,1.5,0.4]|[5.7,4.4,1.5,0.4]|      1|
	|[5.4,3.9,1.3,0.4]|[5.4,3.9,1.3,0.4]|      1|
	|[5.1,3.5,1.4,0.3]|[5.1,3.5,1.4,0.3]|      1|
	|[5.7,3.8,1.7,0.3]|[5.7,3.8,1.7,0.3]|      1|
	|[5.1,3.8,1.5,0.3]|[5.1,3.8,1.5,0.3]|      1|
	+-----------------+-----------------+-------+
	only showing top 20 rows

.. _Spark vs. Hadoop MapReduce: https://www.xplenty.com/blog/2014/11/apache-spark-vs-hadoop-mapreduce/

.. _Vipin Tyagi: https://www.quora.com/profile/Vipin-Tyagi-9
.. _Yassine Alouini: https://www.quora.com/profile/Yassine-Alouini



