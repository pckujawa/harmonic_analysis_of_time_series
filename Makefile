default: install test

test:
	python -m tests.tests

install:
	pip install -Ur requirements_dev.txt
