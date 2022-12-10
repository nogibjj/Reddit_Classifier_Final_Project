install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

install-tensorflow-conda:
	conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0 -y
	export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
	/home/codespace/venv/bin/pip install -r tf-requirements.txt

test:
	echo "Need to add tests"
	#python -m pytest -vv --cov=main --cov=mylib test_*.py

format:	
	black hugging-face/*.py mylib/*.py

lint:
	# add torch to generated-members
	pylint --disable=R,C --generated-members=torch --ignore-patterns=test_.*?py mylib/*.py hugging-face/*.py

container-lint:
	docker run --rm -i hadolint/hadolint < Dockerfile

checkgpu:
	echo "Checking GPU for PyTorch"
	python utils/verify_pytorch.py
	echo "Checking GPU for Tensorflow"
	python utils/verify_tf.py

refactor: format lint

deploy:
	#deploy goes here
		
all: install lint test format deploy
