#!/usr/bin/env sh

for nb in "$@"; do
    jq -cM '
       (.cells[] | select(has("execution_count")) | .execution_count) = null
      | .metadata = {"language_info": {"name": "python3", "pygments_lexer": "ipython3"}}
      | .cells[].metadata = {}
    ' $nb > $nb.tmp && mv $nb.tmp $nb
done
