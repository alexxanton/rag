args				:= $(filter-out install index search eval,$(MAKECMDGOALS))

dataset_arg			:= $(word 1,$(args))
k_arg				:= $(word 2,$(args))
chunk_size_arg		:= $(word 3,$(args))
max_chunk_size_arg	:= $(word 1,$(args))

dataset				:= $(or $(dataset_arg),docs)
k					:= $(or $(k_arg),10)
chunk_size			:= $(or $(chunk_size_arg),2000)
max_chunk_size		:= $(or $(max_chunk_size_arg),2000)

all: index

install:
	uv sync

index:
	uv run python -m src index --max_chunk_size $(max_chunk_size)

search:
	uv run python -m src search_dataset													\
		--dataset_path data/datasets/UnansweredQuestions/dataset_$(dataset)_public.json	\
		--k $(k)																		\
		--save_directory data/output/search_results/UnansweredQuestions

eval:
	./moulinette evaluate_student_search_results													\
		data/output/search_results/UnansweredQuestions/dataset_$(dataset)_public.json						\
		data/datasets/AnsweredQuestions/dataset_$(dataset)_public.json --k $(k) --max_context_length $(chunk_size)

debug:
	uv run python -m pdb -m src

clean:
	find . -name "__pycache__" -exec rm -r {} +

lint:
	-uv run flake8 src/
	uv run mypy src/ --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	-uv run flake8 src/
	uv run mypy src/ --strict

%:
	@:
