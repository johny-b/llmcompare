# llmcompare

## Quickstart

1. Create a virtual environment. Run `pip3 install .`
2. Set your env variable OPENAI_API_KEY
3. Run some [examples](examples/). Note: ideally in cursor in "notebook mode"

## Using multiple organizations/projects

Set OPENAI_API_KEY_0, OPENAI_API_KEY_1 etc with your other keys. The library will find the proper key for your finetuned models.
If some model is available in multiple organizations, e.g. gpt-4.1, it will use the first one (starting with OPENAI_API_KEY without a number).

## Using OpenRouter

Set OPENROUTER_API_KEY to your open router key.
Note: THIS WAS ALMOST NOT TESTED. You should expect problems.
