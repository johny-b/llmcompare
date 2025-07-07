import tiktoken

DEFAULT_ENCODING = tiktoken.encoding_for_model("gpt-4o")


def get_encoding(encoding_scheme: str | None = None) -> tiktoken.Encoding:
    return (
        DEFAULT_ENCODING
        if encoding_scheme is None
        else tiktoken.encoding_for_model(encoding_scheme)
    )


def get_token_ids(
    string_tokens: list[str], encoding_scheme: str | None = None
) -> list[int]:
    encoding = get_encoding(encoding_scheme)
    token_ids: list[int] = []
    for tok in string_tokens:
        tokens = encoding.encode(tok)
        assert len(tokens) == 1, f'Expected single token for "{tok}", got {tokens}'
        token_ids.append(tokens[0])

    return token_ids


def get_tokens(token_ids: list[int], encoding_scheme: str | None = None) -> list[str]:
    encoding = get_encoding(encoding_scheme)
    return [encoding.decode([int(tok)]) for tok in token_ids]
