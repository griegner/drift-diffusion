### drift-diffusion
> drift diffusion models of decision making

**installation**  

[drift-diffusion](https://github.com/griegner/drift-diffusion) requires [numpy](https://github.com/numpy/numpy), [scipy](https://github.com/scipy/scipy), and [matplotlib](https://github.com/matplotlib/matplotlib),  
and optionally [pytest](https://github.com/pytest-dev/pytest) and [sphinx](https://github.com/sphinx-doc/sphinx) for package developement.

First clone this repository:
```
$ git clone https://github.com/griegner/drift-diffusion.git
$ cd ./drift-diffusion
```

Install core dependencies:
```
$ pip install -e .
```

Or install development version:
```
$ pip install -e .[dev, docs]
```