
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
(variables) and m samples (data points), in simple linear regression model, the formula is given by:

      .. math::

         y_i = \beta_0 + \beta_1 x_{i1}, \text{where}, i= 1, \cdots m.  


 .. code-block:: python

	from pyspark.sql import SparkSession

	spark = SparkSession \
	    .builder \
	    .appName("Python Spark Linear Regression Example") \
	    .config("spark.some.config.option", "some-value") \
	    .getOrCreate()



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



