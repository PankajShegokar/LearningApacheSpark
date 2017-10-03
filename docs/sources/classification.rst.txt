
.. _classification:


==============
Classification
==============

.. note::

  **Birds of a feather folock together.** -- old Chinese proverb


Logistic regression
+++++++++++++++++++

Introduction
------------

Demo
----

* The Jupyter notebook can be download from `Logistic Regression <_static/logisticRegression.ipynb>`_.

* For more details, please visit `Logistic Regression API`_ . 

.. note::

  In this demo, I introduced a new function ``get_dummy`` to deal with the categorical data. I highly recommend you to use my ``get_dummy`` function in the other cases. This function will save a lot of time for you.


1. Set up spark context and SparkSession

.. code-block:: python

	from pyspark.sql import SparkSession

		spark = SparkSession \
		    .builder \
		    .appName("Python Spark Logistic Regression example") \
		    .config("spark.some.config.option", "some-value") \
		    .getOrCreate()

2. Load dataset

.. code-block:: python

	df = spark.read.format('com.databricks.spark.csv') \
	            .options(header='true', inferschema='true') \
	            .load("./data/bank.csv",header=True);
	df.drop('day','month','poutcome').show(5)

.. code-block:: python

	+---+------------+-------+---------+-------+-------+-------+----+-------+--------+--------+-----+--------+---+
	|age|         job|marital|education|default|balance|housing|loan|contact|duration|campaign|pdays|previous|  y|
	+---+------------+-------+---------+-------+-------+-------+----+-------+--------+--------+-----+--------+---+
	| 58|  management|married| tertiary|     no|   2143|    yes|  no|unknown|     261|       1|   -1|       0| no|
	| 44|  technician| single|secondary|     no|     29|    yes|  no|unknown|     151|       1|   -1|       0| no|
	| 33|entrepreneur|married|secondary|     no|      2|    yes| yes|unknown|      76|       1|   -1|       0| no|
	| 47| blue-collar|married|  unknown|     no|   1506|    yes|  no|unknown|      92|       1|   -1|       0| no|
	| 33|     unknown| single|  unknown|     no|      1|     no|  no|unknown|     198|       1|   -1|       0| no|
	+---+------------+-------+---------+-------+-------+-------+----+-------+--------+--------+-----+--------+---+
	only showing top 5 rows

.. code-block:: python

	df.printSchema()

.. code-block:: python

	root
	 |-- age: integer (nullable = true)
	 |-- job: string (nullable = true)
	 |-- marital: string (nullable = true)
	 |-- education: string (nullable = true)
	 |-- default: string (nullable = true)
	 |-- balance: integer (nullable = true)
	 |-- housing: string (nullable = true)
	 |-- loan: string (nullable = true)
	 |-- contact: string (nullable = true)
	 |-- day: integer (nullable = true)
	 |-- month: string (nullable = true)
	 |-- duration: integer (nullable = true)
	 |-- campaign: integer (nullable = true)
	 |-- pdays: integer (nullable = true)
	 |-- previous: integer (nullable = true)
	 |-- poutcome: string (nullable = true)
	 |-- y: string (nullable = true)

.. code-block:: python

	def get_dummy(df,categoricalCols,continuousCols,labelCol):
	    
	    from pyspark.ml import Pipeline
	    from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
	    from pyspark.sql.functions import col

	    indexers = [ StringIndexer(inputCol=c, outputCol="{0}_indexed".format(c))
	                 for c in categoricalCols ]

	    # default setting: dropLast=True
	    encoders = [ OneHotEncoder(inputCol=indexer.getOutputCol(),
	                 outputCol="{0}_encoded".format(indexer.getOutputCol())) 
	                 for indexer in indexers ]

	    assembler = VectorAssembler(inputCols=[encoder.getOutputCol() for encoder in encoders] 
	                                + continuousCols, outputCol="features")

	    pipeline = Pipeline(stages=indexers + encoders + [assembler])

	    model=pipeline.fit(df)
	    data = model.transform(df)
	    
	    data = data.withColumn('label',col(labelCol))
	    
	    return data.select('features','label')


3. Deal with categorical data and Convert the data to dense vector

.. code-block:: python

	catcols = ['job','marital','education','default',
	           'housing','loan','contact','poutcome']

	num_cols = ['balance', 'duration','campaign','pdays','previous',]
	labelCol = 'y'

	data = get_dummy(df,catcols,num_cols,labelCol)
	data.show(5)

.. code-block:: python

	+--------------------+-----+
	|            features|label|
	+--------------------+-----+
	|(29,[1,11,14,16,1...|   no|
	|(29,[2,12,13,16,1...|   no|
	|(29,[7,11,13,16,1...|   no|
	|(29,[0,11,16,17,1...|   no|
	|(29,[12,16,18,20,...|   no|
	+--------------------+-----+
	only showing top 5 rows


4. Deal with Categorical Label and Variables

.. code-block:: python

	from pyspark.ml.feature import StringIndexer
	# Index labels, adding metadata to the label column
	labelIndexer = StringIndexer(inputCol='label',
	                             outputCol='indexedLabel').fit(data)
	labelIndexer.transform(data).show(5, True)                        


.. code-block:: python

	+--------------------+-----+------------+
	|            features|label|indexedLabel|
	+--------------------+-----+------------+
	|(29,[1,11,14,16,1...|   no|         0.0|
	|(29,[2,12,13,16,1...|   no|         0.0|
	|(29,[7,11,13,16,1...|   no|         0.0|
	|(29,[0,11,16,17,1...|   no|         0.0|
	|(29,[12,16,18,20,...|   no|         0.0|
	+--------------------+-----+------------+
	only showing top 5 rows


.. code-block:: python

	from pyspark.ml.feature import VectorIndexer
	# Automatically identify categorical features, and index them.
	# Set maxCategories so features with > 4 distinct values are treated as continuous.
	featureIndexer =VectorIndexer(inputCol="features", \
	                                  outputCol="indexedFeatures", \
	                                  maxCategories=4).fit(data)
	featureIndexer.transform(data).show(5, True)
                              
.. code-block:: python

	+--------------------+-----+--------------------+
	|            features|label|     indexedFeatures|
	+--------------------+-----+--------------------+
	|(29,[1,11,14,16,1...|   no|(29,[1,11,14,16,1...|
	|(29,[2,12,13,16,1...|   no|(29,[2,12,13,16,1...|
	|(29,[7,11,13,16,1...|   no|(29,[7,11,13,16,1...|
	|(29,[0,11,16,17,1...|   no|(29,[0,11,16,17,1...|
	|(29,[12,16,18,20,...|   no|(29,[12,16,18,20,...|
	+--------------------+-----+--------------------+
	only showing top 5 rows


5. Split the data to training and test data sets

.. code-block:: python

	# Split the data into training and test sets (40% held out for testing)
	(trainingData, testData) = data.randomSplit([0.6, 0.4])

	trainingData.show(5,False)
	testData.show(5,False)

.. code-block:: python

	+-------------------------------------------------------------------------------------------------+-----+
	|features                                                                                         |label|
	+-------------------------------------------------------------------------------------------------+-----+
	|(29,[0,11,13,16,17,18,19,21,24,25,26,27],[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,-731.0,401.0,4.0,-1.0])|no   |
	|(29,[0,11,13,16,17,18,19,21,24,25,26,27],[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,-723.0,112.0,2.0,-1.0])|no   |
	|(29,[0,11,13,16,17,18,19,21,24,25,26,27],[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,-626.0,205.0,1.0,-1.0])|no   |
	|(29,[0,11,13,16,17,18,19,21,24,25,26,27],[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,-498.0,357.0,1.0,-1.0])|no   |
	|(29,[0,11,13,16,17,18,19,21,24,25,26,27],[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,-477.0,473.0,2.0,-1.0])|no   |
	+-------------------------------------------------------------------------------------------------+-----+
	only showing top 5 rows

	+-------------------------------------------------------------------------------------------------+-----+
	|features                                                                                         |label|
	+-------------------------------------------------------------------------------------------------+-----+
	|(29,[0,11,13,16,17,18,19,21,24,25,26,27],[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,-648.0,280.0,2.0,-1.0])|no   |
	|(29,[0,11,13,16,17,18,19,21,24,25,26,27],[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,-596.0,147.0,1.0,-1.0])|no   |
	|(29,[0,11,13,16,17,18,19,21,24,25,26,27],[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,-529.0,416.0,4.0,-1.0])|no   |
	|(29,[0,11,13,16,17,18,19,21,24,25,26,27],[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,-518.0,46.0,5.0,-1.0]) |no   |
	|(29,[0,11,13,16,17,18,19,21,24,25,26,27],[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,-470.0,275.0,2.0,-1.0])|no   |
	+-------------------------------------------------------------------------------------------------+-----+
	only showing top 5 rows


6. Fit Logistic Regression Model

.. code-block:: python

	from pyspark.ml.classification import LogisticRegression
	logr = LogisticRegression(featuresCol='indexedFeatures', labelCol='indexedLabel')

7. Pipeline Architecture

.. code-block:: python

	# Convert indexed labels back to original labels.
	labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel",
	                               labels=labelIndexer.labels)

.. code-block:: python

	# Chain indexers and tree in a Pipeline
	pipeline = Pipeline(stages=[labelIndexer, featureIndexer, logr,labelConverter])

.. code-block:: python

	# Train model.  This also runs the indexers.
	model = pipeline.fit(trainingData)

8. Make predictions

.. code-block:: python

	# Make predictions.
	predictions = model.transform(testData)
	# Select example rows to display.
	predictions.select("features","label","predictedLabel").show(5)

.. code-block:: python

	+--------------------+-----+--------------+
	|            features|label|predictedLabel|
	+--------------------+-----+--------------+
	|(29,[0,11,13,16,1...|   no|            no|
	|(29,[0,11,13,16,1...|   no|            no|
	|(29,[0,11,13,16,1...|   no|            no|
	|(29,[0,11,13,16,1...|   no|            no|
	|(29,[0,11,13,16,1...|   no|            no|
	+--------------------+-----+--------------+
	only showing top 5 rows


9. Evaluation

.. code-block:: python

	from pyspark.ml.evaluation import MulticlassClassificationEvaluator

	# Select (prediction, true label) and compute test error
	evaluator = MulticlassClassificationEvaluator(
	    labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
	accuracy = evaluator.evaluate(predictions)
	print("Test Error = %g" % (1.0 - accuracy))

.. code-block:: python

	Test Error = 0.0987688

.. code-block:: python

	lrModel = model.stages[2]
	trainingSummary = lrModel.summary

	# Obtain the objective per iteration
	# objectiveHistory = trainingSummary.objectiveHistory
	# print("objectiveHistory:")
	# for objective in objectiveHistory:
	#     print(objective)

	# Obtain the receiver-operating characteristic as a dataframe and areaUnderROC.
	trainingSummary.roc.show(5)
	print("areaUnderROC: " + str(trainingSummary.areaUnderROC))

	# Set the model threshold to maximize F-Measure
	fMeasure = trainingSummary.fMeasureByThreshold
	maxFMeasure = fMeasure.groupBy().max('F-Measure').select('max(F-Measure)').head(5)
	# bestThreshold = fMeasure.where(fMeasure['F-Measure'] == maxFMeasure['max(F-Measure)']) \
	#     .select('threshold').head()['threshold']
	# lr.setThreshold(bestThreshold)

You can use ``z.show()`` to get the data and plot the ROC curves:

.. figure:: images/roc_z.png
   :align: center

You can also register a TempTable ``data.registerTempTable('roc_data')`` and then 
use``sql`` to plot the ROC curve:

.. figure:: images/roc.png
   :align: center


10. visualization

.. code-block:: python

	import matplotlib.pyplot as plt
	import numpy as np
	import itertools

	def plot_confusion_matrix(cm, classes,
	                          normalize=False,
	                          title='Confusion matrix',
	                          cmap=plt.cm.Blues):
	    """
	    This function prints and plots the confusion matrix.
	    Normalization can be applied by setting `normalize=True`.
	    """
	    if normalize:
	        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
	        print("Normalized confusion matrix")
	    else:
	        print('Confusion matrix, without normalization')

	    print(cm)

	    plt.imshow(cm, interpolation='nearest', cmap=cmap)
	    plt.title(title)
	    plt.colorbar()
	    tick_marks = np.arange(len(classes))
	    plt.xticks(tick_marks, classes, rotation=45)
	    plt.yticks(tick_marks, classes)

	    fmt = '.2f' if normalize else 'd'
	    thresh = cm.max() / 2.
	    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
	        plt.text(j, i, format(cm[i, j], fmt),
	                 horizontalalignment="center",
	                 color="white" if cm[i, j] > thresh else "black")

	    plt.tight_layout()
	    plt.ylabel('True label')
	    plt.xlabel('Predicted label')

.. code-block:: python

	class_temp = predictions.select("label").groupBy("label")\
	                        .count().sort('count', ascending=False).toPandas()
	class_temp = class_temp["label"].values.tolist()
	class_names = map(str, class_temp)
	# # # print(class_name)
	class_names

.. code-block:: python

	['no', 'yes']


.. code-block:: python

	from sklearn.metrics import confusion_matrix
	y_true = predictions.select("label")
	y_true = y_true.toPandas()

	y_pred = predictions.select("predictedLabel")
	y_pred = y_pred.toPandas()

	cnf_matrix = confusion_matrix(y_true, y_pred,labels=class_names)
	cnf_matrix

.. code-block:: python

	array([[15657,   379],
	       [ 1410,   667]])


.. code-block:: python

	# Plot non-normalized confusion matrix
	plt.figure()
	plot_confusion_matrix(cnf_matrix, classes=class_names,
	                      title='Confusion matrix, without normalization')
	plt.show()


.. code-block:: python

	Confusion matrix, without normalization
	[[15657   379]
	 [ 1410   667]]

.. figure:: images/logr_b1.png
   :align: center


.. code-block:: python

	# Plot normalized confusion matrix
	plt.figure()
	plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
	                      title='Normalized confusion matrix')

	plt.show()

.. code-block:: python

	Normalized confusion matrix
	[[ 0.97636568  0.02363432]
	 [ 0.67886375  0.32113625]]

.. figure:: images/logr_b2.png
   :align: center




Decision tree Classification
++++++++++++++++++++++++++++

Introduction
------------

Demo
----

* The Jupyter notebook can be download from `Decision Tree Classification <_static/DecisionTreeC.ipynb>`_.

* For more details, please visit `DecisionTreeClassifier API`_ . 
  
1. Set up spark context and SparkSession

.. code-block:: python

	from pyspark.sql import SparkSession

		spark = SparkSession \
		    .builder \
		    .appName("Python Spark Decision Tree classification") \
		    .config("spark.some.config.option", "some-value") \
		    .getOrCreate()

2. Load dataset

.. code-block:: python

	df = spark.read.format('com.databricks.spark.csv').\
	                               options(header='true', \
	                               inferschema='true') \
	                .load("../data/WineData2.csv",header=True);
	df.show(5,True)

.. code-block:: python

	+-----+--------+------+-----+---------+----+-----+-------+----+---------+-------+-------+
	|fixed|volatile|citric|sugar|chlorides|free|total|density|  pH|sulphates|alcohol|quality|
	+-----+--------+------+-----+---------+----+-----+-------+----+---------+-------+-------+
	|  7.4|     0.7|   0.0|  1.9|    0.076|11.0| 34.0| 0.9978|3.51|     0.56|    9.4|      5|
	|  7.8|    0.88|   0.0|  2.6|    0.098|25.0| 67.0| 0.9968| 3.2|     0.68|    9.8|      5|
	|  7.8|    0.76|  0.04|  2.3|    0.092|15.0| 54.0|  0.997|3.26|     0.65|    9.8|      5|
	| 11.2|    0.28|  0.56|  1.9|    0.075|17.0| 60.0|  0.998|3.16|     0.58|    9.8|      6|
	|  7.4|     0.7|   0.0|  1.9|    0.076|11.0| 34.0| 0.9978|3.51|     0.56|    9.4|      5|
	+-----+--------+------+-----+---------+----+-----+-------+----+---------+-------+-------+
	only showing top 5 rows

.. code-block:: python

	# Convert to float format
	def string_to_float(x):
	    return float(x)

	# 
	def condition(r):
	    if (0<= r <= 4):
	        label = "low" 
	    elif(4< r <= 6):
	        label = "medium"
	    else: 
	        label = "high" 
	    return label

.. code-block:: python

	from pyspark.sql.functions import udf
	from pyspark.sql.types import StringType, DoubleType
	string_to_float_udf = udf(string_to_float, DoubleType())
	quality_udf = udf(lambda x: condition(x), StringType())

.. code-block:: python

	df = df.withColumn("quality", quality_udf("quality"))
	df.show(5,True)
	df.printSchema()

.. code-block:: python

	+-----+--------+------+-----+---------+----+-----+-------+----+---------+-------+-------+
	|fixed|volatile|citric|sugar|chlorides|free|total|density|  pH|sulphates|alcohol|quality|
	+-----+--------+------+-----+---------+----+-----+-------+----+---------+-------+-------+
	|  7.4|     0.7|   0.0|  1.9|    0.076|11.0| 34.0| 0.9978|3.51|     0.56|    9.4| medium|
	|  7.8|    0.88|   0.0|  2.6|    0.098|25.0| 67.0| 0.9968| 3.2|     0.68|    9.8| medium|
	|  7.8|    0.76|  0.04|  2.3|    0.092|15.0| 54.0|  0.997|3.26|     0.65|    9.8| medium|
	| 11.2|    0.28|  0.56|  1.9|    0.075|17.0| 60.0|  0.998|3.16|     0.58|    9.8| medium|
	|  7.4|     0.7|   0.0|  1.9|    0.076|11.0| 34.0| 0.9978|3.51|     0.56|    9.4| medium|
	+-----+--------+------+-----+---------+----+-----+-------+----+---------+-------+-------+
	only showing top 5 rows

.. code-block:: python

	root
	 |-- fixed: double (nullable = true)
	 |-- volatile: double (nullable = true)
	 |-- citric: double (nullable = true)
	 |-- sugar: double (nullable = true)
	 |-- chlorides: double (nullable = true)
	 |-- free: double (nullable = true)
	 |-- total: double (nullable = true)
	 |-- density: double (nullable = true)
	 |-- pH: double (nullable = true)
	 |-- sulphates: double (nullable = true)
	 |-- alcohol: double (nullable = true)
	 |-- quality: string (nullable = true)


3. Convert the data to dense vector

.. code-block:: python

 	# !!!!caution: not from pyspark.mllib.linalg import Vectors
	from pyspark.ml.linalg import Vectors
	from pyspark.ml import Pipeline
	from pyspark.ml.feature import IndexToString,StringIndexer, VectorIndexer
	from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
	from pyspark.ml.evaluation import MulticlassClassificationEvaluator

.. code-block:: python

	def transData(data):
	    return data.rdd.map(lambda r: [Vectors.dense(r[:-1]),r[-1]]).toDF(['features','label'])

4. Transform the dataset to DataFrame

.. code-block:: python

	transformed = transData(df)
	transformed.show(5)

.. code-block:: python

	+--------------------+------+
	|            features| label|
	+--------------------+------+
	|[7.4,0.7,0.0,1.9,...|medium|
	|[7.8,0.88,0.0,2.6...|medium|
	|[7.8,0.76,0.04,2....|medium|
	|[11.2,0.28,0.56,1...|medium|
	|[7.4,0.7,0.0,1.9,...|medium|
	+--------------------+------+
	only showing top 5 rows


5. Deal with Categorical Label and Variables

.. code-block:: python

	# Index labels, adding metadata to the label column
	labelIndexer = StringIndexer(inputCol='label',
	                             outputCol='indexedLabel').fit(transformed)
	labelIndexer.transform(transformed).show(5, True)                             

.. code-block:: python

	+--------------------+------+------------+
	|            features| label|indexedLabel|
	+--------------------+------+------------+
	|[7.4,0.7,0.0,1.9,...|medium|         0.0|
	|[7.8,0.88,0.0,2.6...|medium|         0.0|
	|[7.8,0.76,0.04,2....|medium|         0.0|
	|[11.2,0.28,0.56,1...|medium|         0.0|
	|[7.4,0.7,0.0,1.9,...|medium|         0.0|
	+--------------------+------+------------+
	only showing top 5 rows


.. code-block:: python

	# Automatically identify categorical features, and index them.
	# Set maxCategories so features with > 4 distinct values are treated as continuous.
	featureIndexer =VectorIndexer(inputCol="features", \
	                              outputCol="indexedFeatures", \
	                              maxCategories=4).fit(transformed) 
    featureIndexer.transform(transformed).show(5, True)
                              
.. code-block:: python

	+--------------------+------+--------------------+
	|            features| label|     indexedFeatures|
	+--------------------+------+--------------------+
	|[7.4,0.7,0.0,1.9,...|medium|[7.4,0.7,0.0,1.9,...|
	|[7.8,0.88,0.0,2.6...|medium|[7.8,0.88,0.0,2.6...|
	|[7.8,0.76,0.04,2....|medium|[7.8,0.76,0.04,2....|
	|[11.2,0.28,0.56,1...|medium|[11.2,0.28,0.56,1...|
	|[7.4,0.7,0.0,1.9,...|medium|[7.4,0.7,0.0,1.9,...|
	+--------------------+------+--------------------+
	only showing top 5 rows


6. Split the data to training and test data sets

.. code-block:: python

	# Split the data into training and test sets (40% held out for testing)
	(trainingData, testData) = transformed.randomSplit([0.6, 0.4])

	trainingData.show(5)
	testData.show(5)

.. code-block:: python

	+--------------------+------+
	|            features| label|
	+--------------------+------+
	|[4.6,0.52,0.15,2....|   low|
	|[4.7,0.6,0.17,2.3...|medium|
	|[5.0,1.02,0.04,1....|   low|
	|[5.0,1.04,0.24,1....|medium|
	|[5.1,0.585,0.0,1....|  high|
	+--------------------+------+
	only showing top 5 rows

	+--------------------+------+
	|            features| label|
	+--------------------+------+
	|[4.9,0.42,0.0,2.1...|  high|
	|[5.0,0.38,0.01,1....|medium|
	|[5.0,0.4,0.5,4.3,...|medium|
	|[5.0,0.42,0.24,2....|  high|
	|[5.0,0.74,0.0,1.2...|medium|
	+--------------------+------+
	only showing top 5 rows


7. Fit Decision Tree Classification Model

.. code-block:: python

	from pyspark.ml.classification import DecisionTreeClassifier

	# Train a DecisionTree model
	dTree = DecisionTreeClassifier(labelCol='indexedLabel', featuresCol='indexedFeatures')

8. Pipeline Architecture

.. code-block:: python

	# Convert indexed labels back to original labels.
	labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel",
	                               labels=labelIndexer.labels)

.. code-block:: python

	# Chain indexers and tree in a Pipeline
	pipeline = Pipeline(stages=[labelIndexer, featureIndexer, dTree,labelConverter])

.. code-block:: python

	# Train model.  This also runs the indexers.
	model = pipeline.fit(trainingData)

9. Make predictions

.. code-block:: python

	# Make predictions.
	predictions = model.transform(testData)
	# Select example rows to display.
	predictions.select("features","label","predictedLabel").show(5)

.. code-block:: python

	+--------------------+------+--------------+
	|            features| label|predictedLabel|
	+--------------------+------+--------------+
	|[4.9,0.42,0.0,2.1...|  high|          high|
	|[5.0,0.38,0.01,1....|medium|        medium|
	|[5.0,0.4,0.5,4.3,...|medium|        medium|
	|[5.0,0.42,0.24,2....|  high|        medium|
	|[5.0,0.74,0.0,1.2...|medium|        medium|
	+--------------------+------+--------------+
	only showing top 5 rows


10. Evaluation

.. code-block:: python

	from pyspark.ml.evaluation import MulticlassClassificationEvaluator

	# Select (prediction, true label) and compute test error
	evaluator = MulticlassClassificationEvaluator(
	    labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
	accuracy = evaluator.evaluate(predictions)
	print("Test Error = %g" % (1.0 - accuracy))

	rfModel = model.stages[-2]
	print(rfModel)  # summary only

.. code-block:: python

	Test Error = 0.45509
	DecisionTreeClassificationModel (uid=DecisionTreeClassifier_4545ac8dca9c8438ef2a) 
	of depth 5 with 59 nodes

11. visualization

.. code-block:: python

	import matplotlib.pyplot as plt
	import numpy as np
	import itertools

	def plot_confusion_matrix(cm, classes,
	                          normalize=False,
	                          title='Confusion matrix',
	                          cmap=plt.cm.Blues):
	    """
	    This function prints and plots the confusion matrix.
	    Normalization can be applied by setting `normalize=True`.
	    """
	    if normalize:
	        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
	        print("Normalized confusion matrix")
	    else:
	        print('Confusion matrix, without normalization')

	    print(cm)

	    plt.imshow(cm, interpolation='nearest', cmap=cmap)
	    plt.title(title)
	    plt.colorbar()
	    tick_marks = np.arange(len(classes))
	    plt.xticks(tick_marks, classes, rotation=45)
	    plt.yticks(tick_marks, classes)

	    fmt = '.2f' if normalize else 'd'
	    thresh = cm.max() / 2.
	    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
	        plt.text(j, i, format(cm[i, j], fmt),
	                 horizontalalignment="center",
	                 color="white" if cm[i, j] > thresh else "black")

	    plt.tight_layout()
	    plt.ylabel('True label')
	    plt.xlabel('Predicted label')

.. code-block:: python

	class_temp = predictions.select("label").groupBy("label")\
	                        .count().sort('count', ascending=False).toPandas()
	class_temp = class_temp["label"].values.tolist()
	class_names = map(str, class_temp)
	# # # print(class_name)
	class_names

.. code-block:: python

	['medium', 'high', 'low']


.. code-block:: python

	from sklearn.metrics import confusion_matrix
	y_true = predictions.select("label")
	y_true = y_true.toPandas()

	y_pred = predictions.select("predictedLabel")
	y_pred = y_pred.toPandas()

	cnf_matrix = confusion_matrix(y_true, y_pred,labels=class_names)
	cnf_matrix

.. code-block:: python

	array([[497,  29,   7],
	       [ 40,  42,   0],
	       [ 22,   0,   2]])


.. code-block:: python

	# Plot non-normalized confusion matrix
	plt.figure()
	plot_confusion_matrix(cnf_matrix, classes=class_names,
	                      title='Confusion matrix, without normalization')
	plt.show()


.. code-block:: python

	Confusion matrix, without normalization
	[[497  29   7]
	 [ 40  42   0]
	 [ 22   0   2]]

.. figure:: images/dt_cm_c3.png
   :align: center


.. code-block:: python

	# Plot normalized confusion matrix
	plt.figure()
	plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
	                      title='Normalized confusion matrix')

	plt.show()

.. code-block:: python

	Normalized confusion matrix
	[[ 0.93245779  0.05440901  0.01313321]
	 [ 0.48780488  0.51219512  0.        ]
	 [ 0.91666667  0.          0.08333333]]

.. figure:: images/dt_cm_c3.png
   :align: center


Random forest Classification
++++++++++++++++++++++++++++

Introduction
------------


Demo
----
* The Jupyter notebook can be download from `Random forest Classification <_static/RandomForestC3.ipynb>`_.

* For more details, please visit `RandomForestClassifier API`_ .

1. Set up spark context and SparkSession

.. code-block:: python

	from pyspark.sql import SparkSession

		spark = SparkSession \
		    .builder \
		    .appName("Python Spark Decision Tree classification") \
		    .config("spark.some.config.option", "some-value") \
		    .getOrCreate()

2. Load dataset

.. code-block:: python

	df = spark.read.format('com.databricks.spark.csv').\
	                               options(header='true', \
	                               inferschema='true') \
	                .load("../data/WineData2.csv",header=True);
	df.show(5,True)

.. code-block:: python

	+-----+--------+------+-----+---------+----+-----+-------+----+---------+-------+-------+
	|fixed|volatile|citric|sugar|chlorides|free|total|density|  pH|sulphates|alcohol|quality|
	+-----+--------+------+-----+---------+----+-----+-------+----+---------+-------+-------+
	|  7.4|     0.7|   0.0|  1.9|    0.076|11.0| 34.0| 0.9978|3.51|     0.56|    9.4|      5|
	|  7.8|    0.88|   0.0|  2.6|    0.098|25.0| 67.0| 0.9968| 3.2|     0.68|    9.8|      5|
	|  7.8|    0.76|  0.04|  2.3|    0.092|15.0| 54.0|  0.997|3.26|     0.65|    9.8|      5|
	| 11.2|    0.28|  0.56|  1.9|    0.075|17.0| 60.0|  0.998|3.16|     0.58|    9.8|      6|
	|  7.4|     0.7|   0.0|  1.9|    0.076|11.0| 34.0| 0.9978|3.51|     0.56|    9.4|      5|
	+-----+--------+------+-----+---------+----+-----+-------+----+---------+-------+-------+
	only showing top 5 rows

.. code-block:: python

	# Convert to float format
	def string_to_float(x):
	    return float(x)

	# 
	def condition(r):
	    if (0<= r <= 4):
	        label = "low" 
	    elif(4< r <= 6):
	        label = "medium"
	    else: 
	        label = "high" 
	    return label

.. code-block:: python

	from pyspark.sql.functions import udf
	from pyspark.sql.types import StringType, DoubleType
	string_to_float_udf = udf(string_to_float, DoubleType())
	quality_udf = udf(lambda x: condition(x), StringType())

.. code-block:: python

	df = df.withColumn("quality", quality_udf("quality"))
	df.show(5,True)
	df.printSchema()

.. code-block:: python

	+-----+--------+------+-----+---------+----+-----+-------+----+---------+-------+-------+
	|fixed|volatile|citric|sugar|chlorides|free|total|density|  pH|sulphates|alcohol|quality|
	+-----+--------+------+-----+---------+----+-----+-------+----+---------+-------+-------+
	|  7.4|     0.7|   0.0|  1.9|    0.076|11.0| 34.0| 0.9978|3.51|     0.56|    9.4| medium|
	|  7.8|    0.88|   0.0|  2.6|    0.098|25.0| 67.0| 0.9968| 3.2|     0.68|    9.8| medium|
	|  7.8|    0.76|  0.04|  2.3|    0.092|15.0| 54.0|  0.997|3.26|     0.65|    9.8| medium|
	| 11.2|    0.28|  0.56|  1.9|    0.075|17.0| 60.0|  0.998|3.16|     0.58|    9.8| medium|
	|  7.4|     0.7|   0.0|  1.9|    0.076|11.0| 34.0| 0.9978|3.51|     0.56|    9.4| medium|
	+-----+--------+------+-----+---------+----+-----+-------+----+---------+-------+-------+
	only showing top 5 rows

.. code-block:: python

	root
	 |-- fixed: double (nullable = true)
	 |-- volatile: double (nullable = true)
	 |-- citric: double (nullable = true)
	 |-- sugar: double (nullable = true)
	 |-- chlorides: double (nullable = true)
	 |-- free: double (nullable = true)
	 |-- total: double (nullable = true)
	 |-- density: double (nullable = true)
	 |-- pH: double (nullable = true)
	 |-- sulphates: double (nullable = true)
	 |-- alcohol: double (nullable = true)
	 |-- quality: string (nullable = true)


3. Convert the data to dense vector

.. code-block:: python

 	# !!!!caution: not from pyspark.mllib.linalg import Vectors
	from pyspark.ml.linalg import Vectors
	from pyspark.ml import Pipeline
	from pyspark.ml.feature import IndexToString,StringIndexer, VectorIndexer
	from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
	from pyspark.ml.evaluation import MulticlassClassificationEvaluator

.. code-block:: python

	def transData(data):
	    return data.rdd.map(lambda r: [Vectors.dense(r[:-1]),r[-1]]).toDF(['features','label'])

4. Transform the dataset to DataFrame

.. code-block:: python

	transformed = transData(df)
	transformed.show(5)

.. code-block:: python

	+--------------------+------+
	|            features| label|
	+--------------------+------+
	|[7.4,0.7,0.0,1.9,...|medium|
	|[7.8,0.88,0.0,2.6...|medium|
	|[7.8,0.76,0.04,2....|medium|
	|[11.2,0.28,0.56,1...|medium|
	|[7.4,0.7,0.0,1.9,...|medium|
	+--------------------+------+
	only showing top 5 rows


5. Deal with Categorical Label and Variables

.. code-block:: python

	# Index labels, adding metadata to the label column
	labelIndexer = StringIndexer(inputCol='label',
	                             outputCol='indexedLabel').fit(transformed)
	labelIndexer.transform(transformed).show(5, True)                             

.. code-block:: python

	+--------------------+------+------------+
	|            features| label|indexedLabel|
	+--------------------+------+------------+
	|[7.4,0.7,0.0,1.9,...|medium|         0.0|
	|[7.8,0.88,0.0,2.6...|medium|         0.0|
	|[7.8,0.76,0.04,2....|medium|         0.0|
	|[11.2,0.28,0.56,1...|medium|         0.0|
	|[7.4,0.7,0.0,1.9,...|medium|         0.0|
	+--------------------+------+------------+
	only showing top 5 rows


.. code-block:: python

	# Automatically identify categorical features, and index them.
	# Set maxCategories so features with > 4 distinct values are treated as continuous.
	featureIndexer =VectorIndexer(inputCol="features", \
	                              outputCol="indexedFeatures", \
	                              maxCategories=4).fit(transformed) 
    featureIndexer.transform(transformed).show(5, True)
                              
.. code-block:: python

	+--------------------+------+--------------------+
	|            features| label|     indexedFeatures|
	+--------------------+------+--------------------+
	|[7.4,0.7,0.0,1.9,...|medium|[7.4,0.7,0.0,1.9,...|
	|[7.8,0.88,0.0,2.6...|medium|[7.8,0.88,0.0,2.6...|
	|[7.8,0.76,0.04,2....|medium|[7.8,0.76,0.04,2....|
	|[11.2,0.28,0.56,1...|medium|[11.2,0.28,0.56,1...|
	|[7.4,0.7,0.0,1.9,...|medium|[7.4,0.7,0.0,1.9,...|
	+--------------------+------+--------------------+
	only showing top 5 rows


6. Split the data to training and test data sets

.. code-block:: python

	# Split the data into training and test sets (40% held out for testing)
	(trainingData, testData) = transformed.randomSplit([0.6, 0.4])

	trainingData.show(5)
	testData.show(5)

.. code-block:: python

	+--------------------+------+
	|            features| label|
	+--------------------+------+
	|[4.6,0.52,0.15,2....|   low|
	|[4.7,0.6,0.17,2.3...|medium|
	|[5.0,1.02,0.04,1....|   low|
	|[5.0,1.04,0.24,1....|medium|
	|[5.1,0.585,0.0,1....|  high|
	+--------------------+------+
	only showing top 5 rows

	+--------------------+------+
	|            features| label|
	+--------------------+------+
	|[4.9,0.42,0.0,2.1...|  high|
	|[5.0,0.38,0.01,1....|medium|
	|[5.0,0.4,0.5,4.3,...|medium|
	|[5.0,0.42,0.24,2....|  high|
	|[5.0,0.74,0.0,1.2...|medium|
	+--------------------+------+
	only showing top 5 rows


7. Fit Random Forest Classification Model

.. code-block:: python

	from pyspark.ml.classification import RandomForestClassifier

	# Train a RandomForest model.
	rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures", numTrees=10)

8. Pipeline Architecture

.. code-block:: python

	# Convert indexed labels back to original labels.
	labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel",
	                               labels=labelIndexer.labels)

.. code-block:: python

	# Chain indexers and tree in a Pipeline
	pipeline = Pipeline(stages=[labelIndexer, featureIndexer, rf,labelConverter])

.. code-block:: python

	# Train model.  This also runs the indexers.
	model = pipeline.fit(trainingData)

9. Make predictions

.. code-block:: python

	# Make predictions.
	predictions = model.transform(testData)
	# Select example rows to display.
	predictions.select("features","label","predictedLabel").show(5)

.. code-block:: python

	+--------------------+------+--------------+
	|            features| label|predictedLabel|
	+--------------------+------+--------------+
	|[4.9,0.42,0.0,2.1...|  high|          high|
	|[5.0,0.38,0.01,1....|medium|        medium|
	|[5.0,0.4,0.5,4.3,...|medium|        medium|
	|[5.0,0.42,0.24,2....|  high|        medium|
	|[5.0,0.74,0.0,1.2...|medium|        medium|
	+--------------------+------+--------------+
	only showing top 5 rows


10. Evaluation

.. code-block:: python

	from pyspark.ml.evaluation import MulticlassClassificationEvaluator

	# Select (prediction, true label) and compute test error
	evaluator = MulticlassClassificationEvaluator(
	    labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
	accuracy = evaluator.evaluate(predictions)
	print("Test Error = %g" % (1.0 - accuracy))

	rfModel = model.stages[-2]
	print(rfModel)  # summary only

.. code-block:: python

	Test Error = 0.173502
	RandomForestClassificationModel (uid=rfc_a3395531f1d2) with 10 trees

11. visualization

.. code-block:: python

	import matplotlib.pyplot as plt
	import numpy as np
	import itertools

	def plot_confusion_matrix(cm, classes,
	                          normalize=False,
	                          title='Confusion matrix',
	                          cmap=plt.cm.Blues):
	    """
	    This function prints and plots the confusion matrix.
	    Normalization can be applied by setting `normalize=True`.
	    """
	    if normalize:
	        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
	        print("Normalized confusion matrix")
	    else:
	        print('Confusion matrix, without normalization')

	    print(cm)

	    plt.imshow(cm, interpolation='nearest', cmap=cmap)
	    plt.title(title)
	    plt.colorbar()
	    tick_marks = np.arange(len(classes))
	    plt.xticks(tick_marks, classes, rotation=45)
	    plt.yticks(tick_marks, classes)

	    fmt = '.2f' if normalize else 'd'
	    thresh = cm.max() / 2.
	    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
	        plt.text(j, i, format(cm[i, j], fmt),
	                 horizontalalignment="center",
	                 color="white" if cm[i, j] > thresh else "black")

	    plt.tight_layout()
	    plt.ylabel('True label')
	    plt.xlabel('Predicted label')

.. code-block:: python

	class_temp = predictions.select("label").groupBy("label")\
	                        .count().sort('count', ascending=False).toPandas()
	class_temp = class_temp["label"].values.tolist()
	class_names = map(str, class_temp)
	# # # print(class_name)
	class_names

.. code-block:: python

	['medium', 'high', 'low']


.. code-block:: python

	from sklearn.metrics import confusion_matrix
	y_true = predictions.select("label")
	y_true = y_true.toPandas()

	y_pred = predictions.select("predictedLabel")
	y_pred = y_pred.toPandas()

	cnf_matrix = confusion_matrix(y_true, y_pred,labels=class_names)
	cnf_matrix

.. code-block:: python

	array([[502,   9,   0],
	       [ 73,  22,   0],
	       [ 28,   0,   0]])


.. code-block:: python

	# Plot non-normalized confusion matrix
	plt.figure()
	plot_confusion_matrix(cnf_matrix, classes=class_names,
	                      title='Confusion matrix, without normalization')
	plt.show()


.. code-block:: python

	Confusion matrix, without normalization
	[[502   9   0]
	 [ 73  22   0]
	 [ 28   0   0]]

.. figure:: images/rf_cm_c3.png
   :align: center


.. code-block:: python

	# Plot normalized confusion matrix
	plt.figure()
	plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
	                      title='Normalized confusion matrix')

	plt.show()

.. code-block:: python

	Normalized confusion matrix
	[[ 0.98238748  0.01761252  0.        ]
	 [ 0.76842105  0.23157895  0.        ]
	 [ 1.          0.          0.        ]]

.. figure:: images/rf_cm_c3.png
   :align: center





Gradient-boosted tree Classification
++++++++++++++++++++++++++++++++++++

Introduction
------------

Demo
----

* The Jupyter notebook can be download from `Gradient boosted tree Classification <_static/gbtC3.ipynb>`_.

* For more details, please visit `GBTClassifier API`_ .

.. warning::

	Unfortunately, the GBTClassifier currently only supports binary labels.

Naive Bayes Classification
++++++++++++++++++++++++++

Introduction
------------

Demo
----

* The Jupyter notebook can be download from `Naive Bayes Classification <_static/NaiveBayes.ipynb>`_.

* For more details, please visit `NaiveBayes API`_ .









.. _Spark vs. Hadoop MapReduce: https://www.xplenty.com/blog/2014/11/apache-spark-vs-hadoop-mapreduce/

.. _Logistic Regression API: http://takwatanabe.me/pyspark/generated/generated/ml.classification.BinaryLogisticRegressionSummary.html
.. _DecisionTreeClassifier API: http://takwatanabe.me/pyspark/generated/generated/ml.classification.DecisionTreeClassifier.html
.. _RandomForestClassifier API: http://takwatanabe.me/pyspark/generated/generated/ml.classification.RandomForestClassifier.html
.. _GBTClassifier API: http://takwatanabe.me/pyspark/generated/generated/ml.classification.GBTClassifier.html
.. _NaiveBayes API: http://takwatanabe.me/pyspark/generated/generated/ml.classification.NaiveBayes.html 





