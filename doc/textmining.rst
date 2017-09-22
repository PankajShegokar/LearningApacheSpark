
.. _textmining:

===========
Text Mining
===========

.. note::

   Sharpening the knife longer can make it easier to hack the firewood -- old Chinese proverb

I want to answer this question in two folders:

Source of text 
++++++++++++++


Image to text
-------------

.. code-block:: python

	def img2text():
	    ```
	    convert images files to text
	    ```


PDF to text
-----------

.. code-block:: python

	def pdf2text():
	    ```
	    convert pdf files to text
	    ```
	    import os, PythonMagick
	    from datetime import datetime
	    import PyPDF2

	    from PIL import Image
	    import pytesseract
	    
	    f = open('doc.txt','wa')

	    for pdf in [pdf_file for pdf_file in os.listdir(pdf_dir) if pdf_file.endswith(".pdf")]:

	        start_time = datetime.now()

	        input_pdf = pdf_dir + "/" + pdf

	        pdf_im = PyPDF2.PdfFileReader(file(input_pdf, "rb"))
	        npage = pdf_im.getNumPages()

	        print('--------------------------------------------------------------------')
	        print(pdf)
	        print('Converting %d pages.' % npage)
	        print('--------------------------------------------------------------------')     

	        for p in range(npage):

	            pdf_file = input_pdf + '[' + str(p) +']'
	            image_file =  image_dir  + "/" + pdf+ '_' + str(p)+ '.png'

	            im = PythonMagick.Image()
	            im.density('300')
	            im.read(pdf_file)
	            im.write(image_file)

	            text = pytesseract.image_to_string(Image.open(image_file))

	            print(text)

	            f.write( "\n--------------------------------------------------------------------\n")
	            f.write( pdf + "\n")
	            f.write(text.encode('utf-8'))
	    
	    print "CPU Time for converting" + pdf +":"+ str(datetime.now() - start_time) +"\n"

		f.close()  


Text Preprocessing 
++++++++++++++++++


Text Classification 
+++++++++++++++++++

Sentiment analysis
++++++++++++++++++


N-grams and Correlations
++++++++++++++++++++++++


Topic Model: Latent Dirichlet Allocation
++++++++++++++++++++++++++++++++++++++++



.. _Spark vs. Hadoop MapReduce: https://www.xplenty.com/blog/2014/11/apache-spark-vs-hadoop-mapreduce/

.. _Vipin Tyagi: https://www.quora.com/profile/Vipin-Tyagi-9
.. _Yassine Alouini: https://www.quora.com/profile/Yassine-Alouini



