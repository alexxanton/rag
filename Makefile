all: index

install:
	uv sync

index: install
	uv run python -m src index --max_chunk_size 2000

search: install
	uv run python -m src search_dataset												\
		--dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json	\
		--k 10																		\
		--save_directory data/output/search_results/UnansweredQuestions

eval:
	./moulinette evaluate_student_search_results													\
		data/output/search_results/UnansweredQuestions/dataset_docs_public.json						\
		data/datasets/AnsweredQuestions/dataset_docs_public.json --k 10 --max_context_length 2000

debug: install
	uv run python -m pdb -m src

clean:
	find . -name "__pycache__" -exec rm -r {} +

lint: install
	-uv run flake8 src/
	uv run mypy src/ --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict: install
	-uv run flake8 src/
	uv run mypy src/ --strict
