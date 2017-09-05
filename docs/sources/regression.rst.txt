
.. _regression:

==========
Regression
==========

.. note::

   A journey of a thousand miles begins with a single step -- old Chinese proverb



In statistical modeling, regression analysis focuses on investigating the relationship between a dependent variable and one or more independent variables. `Wikipedia Regression analysis`_

In data mining, Regression is a model to represent the relationship between the value of lable ( or target, it is numerical variable) and on one or more features (or predictors they can be numerical and categorical variables).


Linear Regression
+++++++++++++++++

Given that a data set :math:`{\displaystyle \{\,x_{i1},\ldots ,x_{in},y_{i}\}_{i=1}^{m}}` which contains n features
(variables) and m samples (data points), in simple linear regression model for modeling :math:`{\displaystyle m}` data points with one independent variable: :math:`{\displaystyle x_{i1}}`, the formula is given by:

      .. math::

         y_i = \beta_0 + \beta_1 x_{i1}, \text{where}, i= 1, \cdots m. 
       

In matrix notation, the data set is written as :math:`\X = [\X_1,\cdots, \X_n]` with
:math:`\X_i = {\displaystyle \{x_{\cdot i}\}_{i=1}^{n}}`, 
:math:`\By = {\displaystyle \{y_{i}\}_{i=1}^{m}}`
and :math:`\Bbeta^\top = {\displaystyle \{\beta_{i}\}_{i=1}^{m}}`. 
Then the normal equations are written as

      .. math::

         \By = \X \Bbeta.
         
How to solve it?
----------------



Demo
----

1. Set up spark context and SparkSession

.. code-block:: python

	from pyspark.sql import SparkSession

	spark = SparkSession \
	    .builder \
	    .appName("Python Spark regression example") \
	    .config("spark.some.config.option", "some-value") \
	    .getOrCreate()


2. Load dataset

.. code-block:: python

	df = spark.read.format('com.databricks.spark.csv').\
                               options(header='true', \
                               inferschema='true').\
                    load("../data/Advertising.csv",header=True);

check the data set

.. code-block:: python

	df.show(5,True)
	df.printSchema()

Then you will get 

.. code-block:: python

	+-----+-----+---------+-----+
	|   TV|Radio|Newspaper|Sales|
	+-----+-----+---------+-----+
	|230.1| 37.8|     69.2| 22.1|
	| 44.5| 39.3|     45.1| 10.4|
	| 17.2| 45.9|     69.3|  9.3|
	|151.5| 41.3|     58.5| 18.5|
	|180.8| 10.8|     58.4| 12.9|
	+-----+-----+---------+-----+
	only showing top 5 rows

	root
	 |-- TV: double (nullable = true)
	 |-- Radio: double (nullable = true)
	 |-- Newspaper: double (nullable = true)
	 |-- Sales: double (nullable = true)

You can also get the Statistical resutls from the data frame 
(Unfortunately, it only works for numerical). 

.. code-block:: python

	df.describe().show()

Then you will get 

.. code-block:: python

	+-------+-----------------+------------------+------------------+------------------+
	|summary|               TV|             Radio|         Newspaper|             Sales|
	+-------+-----------------+------------------+------------------+------------------+
	|  count|              200|               200|               200|               200|
	|   mean|         147.0425|23.264000000000024|30.553999999999995|14.022500000000003|
	| stddev|85.85423631490805|14.846809176168728| 21.77862083852283| 5.217456565710477|
	|    min|              0.7|               0.0|               0.3|               1.6|
	|    max|            296.4|              49.6|             114.0|              27.0|
	+-------+-----------------+------------------+------------------+------------------+


3. Convert the data to dense vector (**features** and **label**)

.. code-block:: python

	from pyspark.sql import Row
	from pyspark.ml.linalg import Vectors

	# I provide two ways to build the features and labels 

	# method 1 (good for small feature): 
	#def transData(row):
	#    return Row(label=row["Sales"],
	#               features=Vectors.dense([row["TV"],
	#                                       row["Radio"],
	#                                       row["Newspaper"]]))

	# Method 2 (good for large features):
	def transData(data):
    	return data.rdd.map(lambda r: [Vectors.dense(r[:-1]),r[-1]]).toDF(['features','label'])

.. code-block:: python

	transformed= transData(df)
	transformed.show(5)

.. code-block:: python

	+-----------------+-----+
	|         features|label|
	+-----------------+-----+
	|[230.1,37.8,69.2]| 22.1|
	| [44.5,39.3,45.1]| 10.4|
	| [17.2,45.9,69.3]|  9.3|
	|[151.5,41.3,58.5]| 18.5|
	|[180.8,10.8,58.4]| 12.9|
	+-----------------+-----+
	only showing top 5 rows

.. note::

   You will find out that all of the machine learning algorithms in Spark are
   based on the **features** and **label**. That is to say, you can play with
   all of the machine learning algorithms in Spark when you get ready the 
   **features** and **label**.

4. Convert the data to dense vector

.. code-block:: python

	# convert the data to dense vector
	def transData(data):
	    return data.rdd.map(lambda r: [r[-1], Vectors.dense(r[:-1])]).\
	           toDF(['label','features'])

	from pyspark.sql import Row
	from pyspark.ml.linalg import Vectors

	data= transData(df)
	data.show()           


5. Split the data into training and test sets (40% held out for testing)

.. code-block:: python

	from pyspark.ml import Pipeline
	from pyspark.ml.regression import LinearRegression
	from pyspark.ml.feature import VectorIndexer
	from pyspark.ml.evaluation import RegressionEvaluator

	# Split the data into training and test sets (40% held out for testing)
	(trainingData, testData) = transformed.randomSplit([0.6, 0.4])


6.  Fit Ordinary Least Square Regression Model

.. code-block:: python

	# Import LinearRegression class
	from pyspark.ml.regression import LinearRegression

	# Define LinearRegression algorithm
	lr = LinearRegression()


7. Summary of the Model

Spark has a poor summary function for data and model. I wrote a summary 
function which has similar format as **R** output for the linear regression in PySpark.

.. code-block:: python

	def modelsummary(model):
	    import numpy as np
	    print ("Note: the last rows are the information for Intercept")
	    print ("##","-------------------------------------------------")
	    print ("##","  Estimate   |   Std.Error | t Values  |  P-value")
	    coef = np.append(list(model.coefficients),model.intercept)
	    Summary=model.summary
	    
	    for i in range(len(Summary.pValues)):
	        print ("##",'{:10.6f}'.format(coef[i]),\
	        '{:10.6f}'.format(Summary.coefficientStandardErrors[i]),\
	        '{:8.3f}'.format(Summary.tValues[i]),\
	        '{:10.6f}'.format(Summary.pValues[i]))
	        
	    print ("##",'---')
	    print ("##","Mean squared error: % .6f" \
	           % Summary.meanSquaredError, ", RMSE: % .6f" \
	           % Summary.rootMeanSquaredError )
	    print ("##","Multiple R-squared: %f" % Summary.r2, ", \
	            Total iterations: %i"% Summary.totalIterations)  

.. code-block:: python

	modelsummary(model)

You will get the following summary results:

.. code-block:: python

	Note: the last rows are the information for Intercept
	('##', '-------------------------------------------------')
	('##', '  Estimate   |   Std.Error | t Values  |  P-value')
	('##', '  0.044053', '  0.001785', '  24.680', '  0.000000')
	('##', '  0.174338', '  0.011428', '  15.255', '  0.000000')
	('##', '  0.012286', '  0.008553', '   1.436', '  0.153497')
	('##', '  3.302545', '  0.380453', '   8.681', '  0.000000')
	('##', '---')
	('##', 'Mean squared error:  2.643675', ', RMSE:  1.625938')
	('##', 'Multiple R-squared: 0.901501', ',             Total iterations: 1')


8. Make predictions

.. code-block:: python

	# Make predictions.
	predictions = model.transform(testData)

.. code-block:: python

	# Select example rows to display.
	predictions.select("features","label","predictedLabel").show(5)

.. code-block:: python

	+----------------+-----+------------------+
	|        features|label|        prediction|
	+----------------+-----+------------------+
	|  [4.1,11.6,5.7]|  3.2| 5.401313528832074|
	| [7.3,28.1,41.4]|  5.5| 8.633140366255825|
	|  [8.4,27.2,2.1]|  5.7| 8.395487836305678|
	| [8.7,48.9,75.0]|  7.2|  12.5498600070264|
	|[17.2,45.9,69.3]|  9.3|12.381185925526161|
	+----------------+-----+------------------+
	only showing top 5 rows



9. Evaluation

.. code-block:: python

	from pyspark.ml.evaluation import RegressionEvaluator
	# Select (prediction, true label) and compute test error
	evaluator = RegressionEvaluator(labelCol="label", 
	                                predictionCol="prediction", 
	                                metricName="rmse")

	rmse = evaluator.evaluate(predictions)
	print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)

The final Root Mean Squared Error (RMSE) is as follows:

.. code-block:: python

	Root Mean Squared Error (RMSE) on test data = 1.63114



Generalized linear regression
+++++++++++++++++++++++++++++

Decision tree Regression
++++++++++++++++++++++++

Random Forest Regression
++++++++++++++++++++++++


Gradient-boosted tree regression
++++++++++++++++++++++++++++++++



.. _Wikipedia Regression analysis: https://en.wikipedia.org/wiki/Regression_analysis

.. _Vipin Tyagi: https://www.quora.com/profile/Vipin-Tyagi-9
.. _Yassine Alouini: https://www.quora.com/profile/Yassine-Alouini



