.. Copyright (c) 2016, Johan Mabille and Sylvain Corlay

   Distributed under the terms of the BSD 3-Clause License.

   The full license is in the file LICENSE, distributed with this software.


.. raw:: html

   <style>
   .rst-content .section>img {
       width: 30px;
       margin-bottom: 0;
       margin-top: 0;
       margin-right: 15px;
       margin-left: 15px;
       float: left;
   }
   </style>

Installation
============

Although ``xtensor-python`` is a header-only library, we provide standardized means to install it, with package managers or with cmake.

Besides the xtendor-python headers, all these methods place the `cmake` project configuration file in the right location so that third-party projects can use cmake's find_package to locate xtensor-python headers.

.. image:: conda.svg

Using the conda package
-----------------------

A package for xtensor-python is available on the conda package manager.

.. code::

    conda install -c conda-forge xtensor-python

.. image:: debian.svg

Using the Debian package
------------------------

A package for xtensor-python is available on Debian.

.. code::

    sudo apt-get install xtensor-python-dev

.. image:: cmake.svg

From source with cmake
----------------------

You can also install ``xtensor-python`` from source with cmake. On Unix platforms, from the source directory:

.. code::

    mkdir build
    cd build
    cmake -DCMAKE_INSTALL_PREFIX=/path/to/prefix ..
    make install

On Windows platforms, from the source directory:

.. code::

    mkdir build
    cd build
    cmake -G "NMake Makefiles" -DCMAKE_INSTALL_PREFIX=/path/to/prefix ..
    nmake
    nmake install

See the section of the documentation on :doc:`build-options`, for more details on how to cmake options.
