#!/usr/bin/env sh

jq -cM --unbuffered '
   (.cells[] | select(has("execution_count")) | .execution_count) = null
  | .metadata = {"language_info": {"name": "python3", "pygments_lexer": "ipython3"}}
  | .cells[].metadata = {}
' $1 > $1.tmp && mv $1.tmp $1
