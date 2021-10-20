=========
shenshang
=========


.. image:: https://img.shields.io/pypi/v/shenshang.svg
        :target: https://pypi.python.org/pypi/shenshang

.. image:: https://img.shields.io/travis/RNAer/shenshang.svg
        :target: https://travis-ci.org/RNAer/shenshang

.. image:: https://readthedocs.org/projects/shenshang/badge/?version=latest
        :target: https://shenshang.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

.. image:: https://pyup.io/repos/github/RNAer/shenshang/shield.svg
     :target: https://pyup.io/repos/github/RNAer/shenshang/
     :alt: Updates
.. image:: https://codecov.io/gh/CNwangbin/shenshang1/branch/master/graph/badge.svg?token=srMW330c9a
:target: https://codecov.io/gh/CNwangbin/shenshang1



shenshang computes co-occurance and mutual exclusivity in microbiome data in a manner robust to compositionality effect.


* Free software: Modified BSD license
* Documentation: https://shenshang.readthedocs.io.

Name
----
We get the package name from a well-known poem_ by the revered poet Du Fu in Tang Dynasty (~759 AD):

    人生不相见  In life,friends seldom are brought near;

    动如参与商  Like stars of Shen and Shang, each one shines in its sphere.


"`Shen and Shang`_ are two Chinese constellations, in parts of the sky roughly corresponding to Orion and Scorpio, which are never seen together." While Du Fu was lamenting the long parting between friends in the poem, we seek the microbes that do not see each other in the same community like Shen and Shang stars.


Install
-------

Explanation
-----------
* correlation is used exchangably with co-occurence/mutual exclusivity.

* It provides 3 ways to estimate co-occurence of two microbes: parameteric (logratio), non-parametric (rank), and qualitative (binary) methods. For a pair of microbes, a p-value is estimated to indicate if the co-occurence/mutual exclusivity is statistically significant or not; a z-score statistic is also estimated to assess the strength of correlation.

* p-value estimation.
    The p-values are estimated based on permutation. Take positive
    cooccurrence as an example, it is computed as `(c + 1) / (p + 1)`,
    where p is the total number of random permutations, and c is the
    number of permutations in which their overlap statistic greater or
    equal than the observed one between x and y. `(c + 1) / (p + 1)`
    is better than `c / p` because it avoids p-value of
    zero. According to Smyth and Phipson, it is important for multiple
    hypothesis corretion.

    1. Phipson,B. and Smyth,G.K. (2010) Permutation P-values Should
    Never Be Zero: Calculating Exact P-values When Permutations Are
    Randomly Drawn. Statistical Applications in Genetics and Molecular
    Biology, 9.


Todo list
---------
* test if the number of zeros (ie np.sum(x & y)) impacts correlation.

Credits
-------

This package was templated with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_.


.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
.. _poem: https://en.wikisource.org/wiki/Page:The_Spirit_of_the_Chinese_People.djvu/155
.. _`Shen and Shang`: http://www.chinese-poems.com/d20.html
