# API Reference

*Auto-generated from source code docstrings.*

---

## `FreeForm`

*Full path: `llmcompare.question.question.FreeForm`*

Question type for free-form text generation.

Use this when you want to compare how different models respond to open-ended prompts.
The model generates text freely up to max_tokens.

### Methods

#### `__init__(self, *, temperature: 'float' = 1, max_tokens: 'int' = 1024, judges: 'dict[str, str | dict]' = None, **kwargs)`

Initialize a FreeForm question.


**Arguments:**

- `temperature`: Sampling temperature. Default: 1.
- `max_tokens`: Maximum number of tokens in the response. Default: 1024.
- `judges`: Optional dict mapping judge names to judge definitions. Each judge evaluates the (question, answer) pairs. Values can be:
  - A string: loads judge from YAML by name
  - A dict: creates judge from the dict (must include 'type')
  - A FreeFormJudge or RatingJudge instance
- `**kwargs`: Arguments passed to Question base class:
  - name: Question identifier for caching. Default: "__unnamed".
  - paraphrases: List of prompt variations to test.
  - system: System message prepended to each paraphrase.
  - messages: Alternative to paraphrases - [{'role': ..., 'content': ...}, {'role': ..., 'content': ...}, ...]
  - samples_per_paraphrase: Number of samples per prompt. Default: 1.
  - logit_bias: Token bias dict {token_id: bias}.

#### `df(self, model_groups: 'dict[str, list[str]]') -> 'pd.DataFrame'`

Execute question and return results as a DataFrame.

Runs the question on all models (or loads from cache), then applies any configured judges.


**Arguments:**

- `model_groups`: Dict mapping group names to lists of model identifiers. Example: {"gpt4": ["gpt-4o", "gpt-4-turbo"], "claude": ["claude-3-opus"]}


**Returns:**

DataFrame with columns:

- model: Model identifier
- group: Group name from model_groups
- answer: Model's response text
- question: The prompt that was sent
- messages: Full message list sent to model
- paraphrase_ix: Index of the paraphrase used
- {judge_name}: Score/response from each configured judge
- {judge_name}_question: The prompt sent to the judge

#### `plot(self, model_groups: 'dict[str, list[str]]', category_column: 'str' = 'group', answer_column: 'str' = 'answer', df: 'pd.DataFrame' = None, selected_answers: 'list[str]' = None, min_fraction: 'float' = None, colors: 'dict[str, str]' = None, title: 'str' = None, filename: 'str' = None)`

Plot dataframe as a stacked bar chart of answers by category.


**Arguments:**

- `model_groups`: Required. Dict mapping group names to lists of model identifiers.
- `category_column`: Column to use for x-axis categories. Default: "group".
- `answer_column`: Column containing answers to plot. Default: "answer". Use a judge column name to plot judge scores instead.
- `df`: DataFrame to plot. By default calls self.df(model_groups).
- `selected_answers`: List of specific answers to include. Others grouped as "other".
- `min_fraction`: Minimum fraction threshold. Answers below this are grouped as "other".
- `colors`: Dict mapping answer values to colors.
- `title`: Plot title. If None, auto-generated from paraphrases.
- `filename`: If provided, saves the plot to this file path.


**Returns:**

matplotlib Figure object.


---

## `NextToken`

*Full path: `llmcompare.question.question.NextToken`*

Question type for analyzing next-token probability distributions.

Use this when you want to see what tokens the model considers as likely continuations.
Returns probability distributions over the top tokens, useful for fine-grained analysis
of model behavior.

### Methods

#### `__init__(self, *, top_logprobs: 'int' = 20, convert_to_probs: 'bool' = True, num_samples: 'int' = 1, **kwargs)`

Initialize a NextToken question.


**Arguments:**

- `top_logprobs`: Number of top tokens to return probabilities for. Default: 20. Maximum depends on API (OpenAI allows up to 20).
- `convert_to_probs`: If True, convert logprobs to probabilities (0-1 range). If False, returns raw log probabilities. Default: True.
- `num_samples`: Number of samples to average. Useful when logprobs are non-deterministic. Default: 1.
- `**kwargs`: Arguments passed to Question base class:
  - name: Question identifier for caching. Default: "__unnamed".
  - paraphrases: List of prompt variations to test.
  - system: System message prepended to each paraphrase.
  - messages: Alternative to paraphrases - [{'role': ..., 'content': ...}, {'role': ..., 'content': ...}, ...]
  - samples_per_paraphrase: Number of samples per prompt. Default: 1.
  - logit_bias: Token bias dict {token_id: bias}.

#### `df(self, model_groups: 'dict[str, list[str]]') -> 'pd.DataFrame'`

Execute question and return results as a DataFrame.

Runs the question on all models (or loads from cache).


**Arguments:**

- `model_groups`: Dict mapping group names to lists of model identifiers. Example: {"gpt4": ["gpt-4o", "gpt-4-turbo"], "claude": ["claude-3-opus"]}


**Returns:**

DataFrame with columns:

- model: Model identifier
- group: Group name from model_groups
- answer: Dict mapping tokens to probabilities {token: prob}
- question: The prompt that was sent
- messages: Full message list sent to model
- paraphrase_ix: Index of the paraphrase used

#### `plot(self, model_groups: 'dict[str, list[str]]', category_column: 'str' = 'group', df: 'pd.DataFrame' = None, selected_answers: 'list[str]' = None, min_fraction: 'float' = None, colors: 'dict[str, str]' = None, title: 'str' = None, filename: 'str' = None)`

Plot stacked bar chart of token probabilities by category.


**Arguments:**

- `model_groups`: Required. Dict mapping group names to lists of model identifiers.
- `category_column`: Column to use for x-axis categories. Default: "group".
- `df`: DataFrame to plot. By default calls self.df(model_groups).
- `selected_answers`: List of specific tokens to include. Others grouped as "other".
- `min_fraction`: Minimum probability threshold. Tokens below this are grouped as "other".
- `colors`: Dict mapping token values to colors.
- `title`: Plot title. If None, auto-generated from paraphrases.
- `filename`: If provided, saves the plot to this file path.


**Returns:**

matplotlib Figure object.


---

## `Rating`

*Full path: `llmcompare.question.question.Rating`*

Question type for numeric rating responses.

Use this when you expect the model to respond with a number within a range.
Uses logprobs to compute expected value across the probability distribution,
giving more nuanced results than just taking the sampled token.

### Methods

#### `__init__(self, *, min_rating: 'int' = 0, max_rating: 'int' = 100, refusal_threshold: 'float' = 0.75, top_logprobs: 'int' = 20, **kwargs)`

Initialize a Rating question.


**Arguments:**

- `min_rating`: Minimum valid rating value (inclusive). Default: 0.
- `max_rating`: Maximum valid rating value (inclusive). Default: 100.
- `refusal_threshold`: If probability mass on non-numeric tokens exceeds this, the response is treated as a refusal (returns None). Default: 0.75.
- `top_logprobs`: Number of top tokens to request. Default: 20.
- `**kwargs`: Arguments passed to Question base class:
  - name: Question identifier for caching. Default: "__unnamed".
  - paraphrases: List of prompt variations to test.
  - system: System message prepended to each paraphrase.
  - messages: Alternative to paraphrases - [{'role': ..., 'content': ...}, {'role': ..., 'content': ...}, ...]
  - samples_per_paraphrase: Number of samples per prompt. Default: 1.
  - logit_bias: Token bias dict {token_id: bias}.

#### `df(self, model_groups: 'dict[str, list[str]]') -> 'pd.DataFrame'`

Execute question and return results as a DataFrame.

Runs the question on all models (or loads from cache), then computes
expected ratings from the logprob distributions.


**Arguments:**

- `model_groups`: Dict mapping group names to lists of model identifiers. Example: {"gpt4": ["gpt-4o", "gpt-4-turbo"], "claude": ["claude-3-opus"]}


**Returns:**

DataFrame with columns:

- model: Model identifier
- group: Group name from model_groups
- answer: Mean rating (float), or None if model refused
- raw_answer: Original logprobs dict {token: probability}
- question: The prompt that was sent
- messages: Full message list sent to model
- paraphrase_ix: Index of the paraphrase used

#### `plot(self, model_groups: 'dict[str, list[str]]', category_column: 'str' = 'group', df: 'pd.DataFrame' = None, show_mean: 'bool' = True, title: 'str' = None, filename: 'str' = None)`

Plot cumulative rating distribution by category.

Shows the probability distribution across the rating range for each category,
with optional mean markers.


**Arguments:**

- `model_groups`: Required. Dict mapping group names to lists of model identifiers.
- `category_column`: Column to use for grouping. Default: "group".
- `df`: DataFrame to plot. By default calls self.df(model_groups).
- `show_mean`: If True, displays mean rating for each category. Default: True.
- `title`: Plot title. If None, auto-generated from paraphrases.
- `filename`: If provided, saves the plot to this file path.


**Returns:**

matplotlib Figure object.


---

## `FreeFormJudge`

*Full path: `llmcompare.question.judge.FreeFormJudge`*

Judge that evaluates answers using free-form text responses.

Use as a judge in FreeForm questions to have an LLM evaluate the (question, answer) pairs.
The judge paraphrase should contain {answer} placeholder, and optionally {question}.

### Methods

#### `__init__(self, *, model: str, temperature: float = 0, **kwargs)`

Initialize a FreeFormJudge.


**Arguments:**

- `model`: Required. Model identifier to use for judging (e.g., "gpt-4o").
- `temperature`: Sampling temperature. Default: 0.
- `**kwargs`: Arguments passed to FreeForm base class. Must include:
  - paraphrases: Single-element list with the judge template. Template must contain {answer}, optionally {question}. Example: ["Is this answer correct? {answer}"]

#### `get_cache(self) -> pandas.core.frame.DataFrame`

Return all cached judge evaluations as a DataFrame.

Useful for inspecting what the judge has evaluated so far.


**Returns:**

DataFrame with columns:

- question: Original question (None if judge doesn't use {question})
- answer: Original answer that was judged
- judge_question: The formatted prompt sent to the judge
- judge_answer: The judge's response text


---

## `RatingJudge`

*Full path: `llmcompare.question.judge.RatingJudge`*

Judge that evaluates answers using numeric ratings.

Use as a judge in FreeForm questions to have an LLM rate the (question, answer) pairs.
Returns mean rating computed from logprobs.
The judge template should contain {answer} placeholder, and optionally {question}.

### Methods

#### `__init__(self, *, model: str, **kwargs)`

Initialize a RatingJudge.


**Arguments:**

- `model`: Model identifier to use for judging (e.g., "gpt-4o").
- `**kwargs`: Arguments passed to Rating base class. Must include:
  - paraphrases: Single-element list with the judge template. Template must contain {answer}, optionally {question}. Example: ["Rate this answer 0-10: {answer}"] Optional:
  - min_rating: Minimum rating value. Default: 0.
  - max_rating: Maximum rating value. Default: 100.

#### `get_cache(self) -> pandas.core.frame.DataFrame`

Return all cached judge evaluations as a DataFrame.

Useful for inspecting what the judge has evaluated so far.


**Returns:**

DataFrame with columns:

- question: Original question (None if judge doesn't use {question})
- answer: Original answer that was judged
- judge_question: The formatted prompt sent to the judge
- judge_answer: Expected rating (float) computed from logprobs
- judge_raw_answer: Raw logprobs dict {token: probability}


---

## `Config`

*Full path: `llmcompare.config.Config`*

Global configuration for llmcompare.

Modify class attributes directly to change configuration.
Changes take effect immediately for subsequent operations.

### Configuration Options

| Attribute | Default | Description |
|-----------|---------|-------------|
| `timeout` | `50` | |
| `max_workers` | `100` | |
| `cache_dir` | `'llmcompare_cache'` | |
| `yaml_dir` | `'questions'` | |
| `verbose` | `False` | |

### Methods

#### `client_for_model(cls, model: str) -> openai.OpenAI`

Get or create an OpenAI client for the given model.

Clients are cached in client_cache. The first call for a model
will test available URL-key pairs in parallel to find one that works.
Thread-safe: only one thread will attempt to create a client per model.
Failures are also cached to avoid repeated attempts.

#### `reset(cls)`

Reset all configuration values to their defaults.


---

## `Question`

*Full path: `llmcompare.question.question.Question`*

### Methods

#### `create(cls, **kwargs) -> "'Question'"`

Create a Question instance from a type string and keyword arguments.

Factory method that instantiates the appropriate Question subclass based on the 'type' parameter.


**Arguments:**

- `**kwargs`: Must include 'type' key with one of:
  - "free_form": Creates FreeForm question
  - "rating": Creates Rating question
  - "next_token": Creates NextToken question
  - "free_form_judge": Creates FreeFormJudge
  - "rating_judge": Creates RatingJudge Other kwargs are passed to the constructor.


**Returns:**

Question subclass instance.


**Raises:**

- `ValueError`: If 'type' is missing or invalid.


**Example:**

    >>> q = Question.create(
    ...     type="free_form",
    ...     name="my_question",
    ...     paraphrases=["What is 2+2?"]
    ... )

#### `load_dict(cls, id_: 'str') -> 'dict'`

Load question configuration as a dictionary from YAML files.

Searches all YAML files in Config.yaml_dir for a question with matching name.


**Arguments:**

- `id_`: The question name to look up.


**Returns:**

Dict containing the question configuration (can be passed to Question.create).


**Raises:**

- `ValueError`: If question with given id is not found.


**Example:**

    >>> config = Question.load_dict("my_question")
    >>> config
    {'type': 'free_form', 'name': 'my_question', 'paraphrases': [...]}

#### `from_yaml(cls, id_: 'str') -> "'Question'"`

Load and instantiate a Question from YAML configuration.

Convenience method combining load_dict() and create().


**Arguments:**

- `id_`: The question name to look up in YAML files.


**Returns:**

Question subclass instance.


**Raises:**

- `ValueError`: If question not found or has invalid type.


**Example:**

    >>> q = Question.from_yaml("my_question")


---
