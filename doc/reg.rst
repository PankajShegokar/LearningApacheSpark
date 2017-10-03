.. _regularization:

==============
Regularization
==============




Ridge regression
++++++++++++++++

.. math::

	\min _{\beta\in \mathbb {R} ^{p}}{\frac {1}{n}}\|{\hat {X}}\beta-{\hat {Y}}\|^{2}+\lambda \|\beta\|_{2}^{2}


Least Absolute Shrinkage and Selection Operator (LASSO)
+++++++++++++++++++++++++++++++++++++++++++++++++++++++

.. math::

	\min _{\beta\in \mathbb {R} ^{p}}{\frac {1}{n}}\|{\hat {X}}\beta-{\hat {Y}}\|^{2}+\lambda\|\beta\|_{1}


Elastic net
+++++++++++

.. math::

	\min _{\beta\in \mathbb {R} ^{p}}{\frac {1}{n}}\|{\hat {X}}\beta-{\hat {Y}}\|^{2}+\lambda (\alpha \|\beta\|_{1}+(1-\alpha )\|\beta\|_{2}^{2}),\alpha \in [0,1]


.. _Wikipedia Regularization: https://en.wikipedia.org/wiki/Regularization_(mathematics)



