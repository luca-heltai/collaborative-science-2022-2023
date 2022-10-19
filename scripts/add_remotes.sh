#!/bin/bash
for user in `cat collaborators.md`; do
    git remote -v | grep -q $user || \
    (git remote add $user  git@github.com:$user/collaborative-science-2022-2023.git && \
    echo added $user as remote )
done