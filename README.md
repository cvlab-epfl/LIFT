# LIFT: Learned Invariant Feature Points

This software is a Python implementation of the LIFT feature point presented in [1].

[1] K.  M.  Yi, E. Trulls, V. Lepetit, and P.  Fua.  "LIFT: Learned Invariant Feature Transform", European Conference on Computer Vision (ECCV), 2016.

This software is patented and is strictly for academic purposes only.  For other purposes, please contact us.  When using this software, please cite [1].



Contact:

<pre>
Kwang Moo Yi : kwang_dot_yi_at_epfl_dot_ch
Eduard Trulls : eduard_dot_trulls_at_epfl_dot_ch
</pre>

## Requirements

* OpenCV 3

And the following python requirements:

* Theano
* Lasagne (Dev)
* numpy
* scipy
* flufl.lock
* parse
* h5py

which can be installed with

```bash
pip install -r requirements.txt
```

## Usage

Build the shared library by

```bash
cd c-code/build
cmake ..
make
```

To run the test program simply

```bash
./run.sh
```

## Note

This model was trained with SfM data (Piccadilly Circus dataset), which does not have strong rotation changes. Newer models work better in this case, which will be released soon. In the meantime, you can also use the models in the [learn-orientation](http://github.com/cvlab-epfl/learn-orientation), [benchmark-orientation](http://github.com/cvlab-epfl/benchmark-orientation).

