# rye-template

## Rye

- package & python version manager
- url : https://rye-up.com/

```
curl -sSf https://rye-up.com/get | bash
echo 'source "$HOME/.rye/env"' >> ~/.bashrc
```

```
rye config --set-bool behavior.use-uv=true
```

```
rye sync
```

## Dev

- format : `make format`
- mypy check : `make mypy`
- test : `make test`
