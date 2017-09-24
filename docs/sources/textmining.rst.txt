
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

	def img2txt(img_dir):
	    """
	    convert images to text
	    """
	    import os, PythonMagick
	    from datetime import datetime
	    import PyPDF2

	    from PIL import Image
	    import pytesseract

	    f = open('doc4img.txt','wa')
	    for img in [img_file for img_file in os.listdir(img_dir)
	                if (img_file.endswith(".png") or 
	                    img_file.endswith(".jpg") or 
	                    img_file.endswith(".jpeg"))]:

	        start_time = datetime.now()

	        input_img = img_dir + "/" + img

	        print('--------------------------------------------------------------------')
	        print(img)
	        print('Converting ' + img +'.......')
	        print('--------------------------------------------------------------------')     

	        # extract the text information from images
	        text = pytesseract.image_to_string(Image.open(input_img))
	        print(text)
	        
	        # ouput text file 
	        f.write( img + "\n")
	        f.write(text.encode('utf-8'))
	        

	        print "CPU Time for converting" + img +":"+ str(datetime.now() - start_time) +"\n"
	        f.write( "\n-------------------------------------------------------------\n")

	    f.close()   

Image Enhnaced to text
----------------------

.. code-block:: python

	def pdf2txt_enhance(img_dir,scaler):
	    """
	    convert images files to text
	    """
	    
	    import numpy as np
	    import os, PythonMagick
	    from datetime import datetime
	    import PyPDF2

	    from PIL import Image, ImageEnhance, ImageFilter
	    import pytesseract

	    f = open('doc4img.txt','wa')
	    for img in [img_file for img_file in os.listdir(img_dir)
	                if (img_file.endswith(".png") or 
	                    img_file.endswith(".jpg") or 
	                    img_file.endswith(".jpeg"))]:

	        start_time = datetime.now()

	        input_img = img_dir + "/" + img
	        enhanced_img = img_dir + "/" +"Enhanced" + "/"+ img
	        
	        im = Image.open(input_img) # the second one
	        im = im.filter(ImageFilter.MedianFilter())
	        enhancer = ImageEnhance.Contrast(im)
	        im = enhancer.enhance(1)
	        im = im.convert('1')
	        im.save(enhanced_img)
	        
	        for scale in np.ones(scaler):
	            im = Image.open(enhanced_img) # the second one 
	            im = im.filter(ImageFilter.MedianFilter())
	            enhancer = ImageEnhance.Contrast(im)
	            im = enhancer.enhance(scale)
	            im = im.convert('1')
	            im.save(enhanced_img)
	        


	        print('--------------------------------------------------------------------')
	        print(img)
	        print('Converting ' + img +'.......')
	        print('--------------------------------------------------------------------')     

	        # extract the text information from images
	        text = pytesseract.image_to_string(Image.open(enhanced_img))
	        print(text)
	        
	        # ouput text file 
	        f.write( img + "\n")
	        f.write(text.encode('utf-8'))
	        

	        print "CPU Time for converting" + img +":"+ str(datetime.now() - start_time) +"\n"
	        f.write( "\n-------------------------------------------------------------\n")

	    f.close()   

PDF to text
-----------

.. code-block:: python

	def pdf2txt(pdf_dir,image_dir):
	    """
	    convert PDF to text
	    """
	    
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

	        f.write( "\n--------------------------------------------------------------------\n")
	        
	        for p in range(npage):

	            pdf_file = input_pdf + '[' + str(p) +']'
	            image_file =  image_dir  + "/" + pdf+ '_' + str(p)+ '.png'

	            # convert PDF files to Images
	            im = PythonMagick.Image()
	            im.density('300')
	            im.read(pdf_file)
	            im.write(image_file)

	            # extract the text information from images
	            text = pytesseract.image_to_string(Image.open(image_file))

	            #print(text)

	            # ouput text file 
	            f.write( pdf + "\n")
	            f.write(text.encode('utf-8'))
	        

	        print "CPU Time for converting" + pdf +":"+ str(datetime.now() - start_time) +"\n"

	    f.close()   
    

Text Preprocessing 
++++++++++++++++++

* check to see if a row only contains whitespace

.. code-block:: python

	def check_blanks(data_str):
	    is_blank = str(data_str.isspace())
	    return is_blank


* Determine whether the language of the text content is english or not: Use langid module to classify the language to make sure we are applying the correct cleanup actions for English langid

.. code-block:: python

	def check_lang(data_str):
	    predict_lang = langid.classify(data_str)
	    if predict_lang[1] >= .9:
	        language = predict_lang[0]
	    else:
	        language = 'NA'
	    return language

* Remove features

.. code-block:: python

	def remove_features(data_str):
	    # compile regex
	    url_re = re.compile('https?://(www.)?\w+\.\w+(/\w+)*/?')
	    punc_re = re.compile('[%s]' % re.escape(string.punctuation))
	    num_re = re.compile('(\\d+)')
	    mention_re = re.compile('@(\w+)')
	    alpha_num_re = re.compile("^[a-z0-9_.]+$")
	    # convert to lowercase
	    data_str = data_str.lower()
	    # remove hyperlinks
	    data_str = url_re.sub(' ', data_str)
	    # remove @mentions
	    data_str = mention_re.sub(' ', data_str)
	    # remove puncuation
	    data_str = punc_re.sub(' ', data_str)
	    # remove numeric 'words'
	    data_str = num_re.sub(' ', data_str)
	    # remove non a-z 0-9 characters and words shorter than 3 characters
	    list_pos = 0
	    cleaned_str = ''
	    for word in data_str.split():
	        if list_pos == 0:
	            if alpha_num_re.match(word) and len(word) > 2:
	                cleaned_str = word
	            else:
	                cleaned_str = ' '
	        else:
	            if alpha_num_re.match(word) and len(word) > 2:
	                cleaned_str = cleaned_str + ' ' + word
	            else:
	                cleaned_str += ' '
	        list_pos += 1
	    return cleaned_str

* removes stop words

.. code-block:: python

	def remove_stops(data_str):
	    # expects a string
	    stops = set(stopwords.words("english"))
	    list_pos = 0
	    cleaned_str = ''
	    text = data_str.split()
	    for word in text:
	        if word not in stops:
	            # rebuild cleaned_str
	            if list_pos == 0:
	                cleaned_str = word
	            else:
	                cleaned_str = cleaned_str + ' ' + word
	            list_pos += 1
	    return cleaned_str

* tagging text

.. code-block:: python

	def tag_and_remove(data_str):
	    cleaned_str = ' '
	    # noun tags
	    nn_tags = ['NN', 'NNP', 'NNP', 'NNPS', 'NNS']
	    # adjectives
	    jj_tags = ['JJ', 'JJR', 'JJS']
	    # verbs
	    vb_tags = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
	    nltk_tags = nn_tags + jj_tags + vb_tags

	    # break string into 'words'
	    text = data_str.split()

	    # tag the text and keep only those with the right tags
	    tagged_text = pos_tag(text)
	    for tagged_word in tagged_text:
	        if tagged_word[1] in nltk_tags:
	            cleaned_str += tagged_word[0] + ' '

	    return cleaned_str

* lemmatization

.. code-block:: python	   

	def lemmatize(data_str):
	    # expects a string
	    list_pos = 0
	    cleaned_str = ''
	    lmtzr = WordNetLemmatizer()
	    text = data_str.split()
	    tagged_words = pos_tag(text)
	    for word in tagged_words:
	        if 'v' in word[1].lower():
	            lemma = lmtzr.lemmatize(word[0], pos='v')
	        else:
	            lemma = lmtzr.lemmatize(word[0], pos='n')
	        if list_pos == 0:
	            cleaned_str = lemma
	        else:
	            cleaned_str = cleaned_str + ' ' + lemma
	        list_pos += 1
	    return cleaned_str



**define the preprocessing function in PySpark**

.. code-block:: python

	from pyspark.sql.functions import udf
	from pyspark.sql.types import StringType
	import preproc as pp

	check_lang_udf = udf(pp.check_lang, StringType())
	remove_stops_udf = udf(pp.remove_stops, StringType())
	remove_features_udf = udf(pp.remove_features, StringType())
	tag_and_remove_udf = udf(pp.tag_and_remove, StringType())
	lemmatize_udf = udf(pp.lemmatize, StringType())
	check_blanks_udf = udf(pp.check_blanks, StringType())


Text Classification 
+++++++++++++++++++

.. code-block:: python

	from nltk.stem.wordnet import WordNetLemmatizer
	from nltk.corpus import stopwords
	from nltk import pos_tag
	import string
	import re
	import langid

Sentiment analysis
++++++++++++++++++


N-grams and Correlations
++++++++++++++++++++++++


Topic Model: Latent Dirichlet Allocation
++++++++++++++++++++++++++++++++++++++++



.. _Spark vs. Hadoop MapReduce: https://www.xplenty.com/blog/2014/11/apache-spark-vs-hadoop-mapreduce/

.. _Vipin Tyagi: https://www.quora.com/profile/Vipin-Tyagi-9
.. _Yassine Alouini: https://www.quora.com/profile/Yassine-Alouini



