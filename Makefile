PYTHON ?= python
CYTHON ?= cython
NOSETESTS ?= nosetests

# Compilation...

CYTHONSRC= $(wildcard sdtw/*.pyx)
CSRC= $(CYTHONSRC:.pyx=.c)

inplace:
	$(PYTHON) setup.py build_ext -i

all: cython inplace

cython: $(CSRC)

clean:
	rm -f sdtw/*.c sdtw/*.html
	rm -f `find sdtw -name "*.pyc"`
	rm -f `find sdtw -name "*.so"`

%.c: %.pyx
	$(CYTHON) $<

# Tests...
#
test-code: inplace
	$(NOSETESTS) -s sdtw

test-coverage:
	$(NOSETESTS) -s --with-coverage --cover-html --cover-html-dir=coverage \
	--cover-package=sdtw sdtw


